"""
Hand-Eye Calibration Script for PiPER + RealSense
===================================================

Performs eye-on-base calibration: camera is fixed, robot moves.

Usage:
    python calibrate_hand_eye.py --can_port can0 --num_samples 10 --output calibration.npy

Procedure:
    1. Print a checkerboard and fix it on the table (visible to camera)
    2. Move the robot to different poses above the checkerboard
    3. At each pose, press ENTER to record the sample
    4. After enough samples (≥3, recommend 10+), calibration runs automatically

Checkerboard default: 8x6 inner corners, 25mm square size
    → Print from: https://github.com/opencv/opencv/blob/master/doc/pattern.png
"""

import argparse
import numpy as np
from piper_sdk import C_PiperInterface_V2
from realsense_camera import RealSenseCamera, RealSenseCalibrator
from piper_env import PiperRobot


def main():
    parser = argparse.ArgumentParser(description="PiPER + RealSense Hand-Eye Calibration")
    parser.add_argument("--can_port", type=str, default="can0", help="CAN port")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of calibration samples")
    parser.add_argument("--checker_rows", type=int, default=8, help="Checkerboard inner corners (rows)")
    parser.add_argument("--checker_cols", type=int, default=6, help="Checkerboard inner corners (cols)")
    parser.add_argument("--square_size", type=float, default=0.025, help="Checkerboard square size (meters)")
    parser.add_argument("--output", type=str, default="calibration.npy", help="Output .npy file")
    args = parser.parse_args()

    # Connect camera
    print("=" * 50)
    print("Step 1: Connecting RealSense camera...")
    camera = RealSenseCamera(
        serial_number=args.serial,
        target_size=(640, 480),  # full res for calibration
    )
    camera.connect()

    # Connect robot
    print("=" * 50)
    print("Step 2: Connecting PiPER robot...")
    robot = PiperRobot(args.can_port)
    robot.connect()
    robot.enable()

    # Create calibrator
    calibrator = RealSenseCalibrator(
        camera=camera,
        robot=robot,
        checkerboard_size=(args.checker_rows, args.checker_cols),
        square_size=args.square_size,
    )

    # Collect samples
    print("=" * 50)
    print(f"Step 3: Collecting {args.num_samples} calibration samples")
    print("Move the robot to different poses where the checkerboard is visible.")
    print("Press ENTER to capture a sample, or type 'skip' to skip, 'done' to finish early.")
    print()

    while calibrator.num_samples < args.num_samples:
        print(f"\n--- Sample {calibrator.num_samples + 1}/{args.num_samples} ---")
        print("Current robot pose:", robot.get_end_pose())
        user_input = input("Press ENTER to capture (or 'skip'/'done'): ").strip().lower()

        if user_input == 'skip':
            continue
        elif user_input == 'done':
            break

        ok = calibrator.collect_sample()
        if not ok:
            print("  → Checkerboard not detected! Make sure it's visible and try again.")

    # Run calibration
    if calibrator.num_samples < 3:
        print(f"\nNot enough samples ({calibrator.num_samples}). Need at least 3.")
        camera.disconnect()
        robot.disable()
        return

    print("=" * 50)
    print(f"Step 4: Computing calibration from {calibrator.num_samples} samples...")
    try:
        X = calibrator.calibrate()
        np.save(args.output, X)
        print(f"\nCalibration saved to: {args.output}")
        print("\nTo use it, set in your config:")
        print(f"  real.calibration_file: {args.output}")
    except Exception as e:
        print(f"\nCalibration failed: {e}")

    # Cleanup
    camera.disconnect()
    robot.disable()
    print("\nDone!")


if __name__ == "__main__":
    main()
