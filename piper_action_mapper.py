"""
Action Mapper: MetaWorld (Sawyer) → PiPER Real Robot
=====================================================

Handles the coordinate transformation and scaling between MetaWorld's
simulation action space and PiPER's real-world control space.

MetaWorld Sawyer:
  - Action: [dx, dy, dz, gripper] ∈ [-1, 1]
  - End-effector workspace (meters):
      x: [-0.2, 0.2], y: [0.55, 0.75], z: [0.05, 0.3]
  - action_scale = 1/100 → each action unit ≈ 0.01m = 10mm

PiPER:
  - End-effector workspace (millimeters): configurable
  - Pose control: X(mm), Y(mm), Z(mm), RX(deg), RY(deg), RZ(deg)
  - Gripper: position in meters

Mapping strategies:
  1. "direct"     — Simple scale: sim delta (m) → real delta (mm)
  2. "calibrated" — With coordinate alignment (sim frame → real frame)
  3. "identity"   — Direct pass-through (for custom setups)
"""

import numpy as np
from typing import Dict, Optional, Tuple


class ActionMapper:
    """Base action mapper interface."""

    def sim_to_real(self, sim_action: np.ndarray, cur_real_pose: np.ndarray) -> np.ndarray:
        """Map simulation action to real-robot target pose.

        Args:
            sim_action: MetaWorld action [dx, dy, dz, gripper] in [-1, 1]
            cur_real_pose: Current PiPER end-effector pose [X, Y, Z, RX, RY, RZ] (mm, deg)

        Returns:
            Target PiPER pose [X, Y, Z, RX, RY, RZ] (mm, deg)
        """
        raise NotImplementedError

    def real_to_sim_pose(self, real_pose: np.ndarray) -> np.ndarray:
        """Convert real-robot pose to simulation coordinate frame.

        Args:
            real_pose: PiPER pose [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]

        Returns:
            Sim pose [X_m, Y_m, Z_m] (meters)
        """
        raise NotImplementedError


class DirectMapper(ActionMapper):
    """Simple direct mapping: scale sim deltas to real deltas.

    Assumes the real robot and sim have the same coordinate orientation.
    Only the position deltas are scaled; orientation is preserved.

    Configurable:
      - pos_scale: mm per sim action unit (default: 2.0, i.e., [-1,1] → ±2mm)
      - gripper_range: max gripper opening in meters
    """

    def __init__(self, pos_scale: float = 2.0, gripper_range: float = 0.08):
        self.pos_scale = pos_scale
        self.gripper_range = gripper_range

    def sim_to_real(self, sim_action: np.ndarray, cur_real_pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """Map sim action to real target pose + gripper position.

        Returns:
            (target_pose [X,Y,Z,RX,RY,RZ] in mm/deg, gripper_pos in meters)
        """
        dx_mm = sim_action[0] * self.pos_scale
        dy_mm = sim_action[1] * self.pos_scale
        dz_mm = sim_action[2] * self.pos_scale

        target_pose = cur_real_pose.copy()
        target_pose[0] += dx_mm
        target_pose[1] += dy_mm
        target_pose[2] += dz_mm
        # Orientation unchanged

        gripper_pos = (sim_action[3] + 1.0) / 2.0 * self.gripper_range
        return target_pose, gripper_pos

    def real_to_sim_pose(self, real_pose: np.ndarray) -> np.ndarray:
        """Convert mm to meters."""
        return real_pose[:3] / 1000.0


class CalibratedMapper(ActionMapper):
    """Calibrated mapping with coordinate alignment.

    Uses a homography / rigid transform to align the simulation coordinate
    frame with the real robot frame. This is necessary when:
      - The robot base is not aligned with the sim's world frame
      - The sim's axes don't match the real axes
      - There's a scale difference due to different arm lengths

    Calibration process:
      1. Move robot to several known positions
      2. Record corresponding sim positions
      3. Compute the rigid transformation (R, t, s)

    Configurable:
      - R: 3x3 rotation matrix (sim → real)
      - t: 3x1 translation vector (sim → real, in mm)
      - s: scale factor (sim meters → real mm)
      - pos_scale: mm per action unit for deltas
    """

    def __init__(
        self,
        R: np.ndarray = None,
        t: np.ndarray = None,
        s: float = 1000.0,   # 1 sim meter = 1000 real mm (default)
        pos_scale: float = 2.0,
        gripper_range: float = 0.08,
        # Default: identity mapping (sim X→real X, sim Y→real Y, sim Z→real Z)
        # For Sawyer→Piper, common alignment:
        #   sim X (left-right)  → real X
        #   sim Y (forward-back) → real Y
        #   sim Z (up-down)    → real Z
    ):
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.s = s
        self.pos_scale = pos_scale
        self.gripper_range = gripper_range

    def sim_pos_to_real(self, sim_pos_m: np.ndarray) -> np.ndarray:
        """Convert sim position (meters) to real position (mm)."""
        return self.R @ (sim_pos_m * self.s) + self.t

    def sim_delta_to_real(self, sim_delta: np.ndarray) -> np.ndarray:
        """Convert sim action delta to real delta (mm)."""
        # Delta only needs rotation and scale, no translation
        return self.R @ (sim_delta * self.pos_scale)

    def sim_to_real(self, sim_action: np.ndarray, cur_real_pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """Map sim action to real target pose + gripper position."""
        sim_delta = sim_action[:3]
        real_delta_mm = self.sim_delta_to_real(sim_delta)

        target_pose = cur_real_pose.copy()
        target_pose[0] += real_delta_mm[0]
        target_pose[1] += real_delta_mm[1]
        target_pose[2] += real_delta_mm[2]

        gripper_pos = (sim_action[3] + 1.0) / 2.0 * self.gripper_range
        return target_pose, gripper_pos

    def real_to_sim_pose(self, real_pose: np.ndarray) -> np.ndarray:
        """Convert real pose (mm) to sim pose (meters)."""
        real_pos_mm = real_pose[:3]
        sim_pos_m = self.R.T @ (real_pos_mm - self.t) / self.s
        return sim_pos_m

    @staticmethod
    def calibrate(
        sim_points: np.ndarray,
        real_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calibrate the rigid transformation from sim to real coordinates.

        Uses SVD-based Procrustes analysis to find R, t, s that minimizes:
            ||real_points - (s * R @ sim_points + t)||^2

        Args:
            sim_points: (N, 3) array of N points in sim frame (meters)
            real_points: (N, 3) array of N points in real frame (mm)

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
            s: scale factor
        """
        assert sim_points.shape == real_points.shape
        assert sim_points.shape[1] == 3

        sim_center = sim_points.mean(axis=0)
        real_center = real_points.mean(axis=0)

        sim_centered = sim_points - sim_center
        real_centered = real_points - real_center

        # Compute scale
        sim_norm = np.linalg.norm(sim_centered)
        real_norm = np.linalg.norm(real_centered)
        s = real_norm / sim_norm if sim_norm > 1e-8 else 1000.0

        # Compute rotation using SVD
        H = real_centered.T @ sim_centered
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # Compute translation
        t = real_center - s * R @ sim_center

        return R, t, s


def create_mapper(
    mode: str = "direct",
    **kwargs
) -> ActionMapper:
    """Factory function to create an action mapper.

    Args:
        mode: "direct", "calibrated", or "identity"
        **kwargs: mapper-specific parameters

    Returns:
        ActionMapper instance
    """
    if mode == "direct":
        return DirectMapper(**kwargs)
    elif mode == "calibrated":
        return CalibratedMapper(**kwargs)
    elif mode == "identity":
        return DirectMapper(pos_scale=1.0, **kwargs)
    else:
        raise ValueError(f"Unknown mapper mode: {mode}")
