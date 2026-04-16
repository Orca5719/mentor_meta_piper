"""
Test Script: Spacemouse Intervention on Piper Robot
====================================================
Simple test: Piper arm slowly descends from a starting height.
Use the Spacemouse to intervene and override the descending motion.
Press Ctrl+C to stop.

This validates:
  - SpacemouseReader correctly reads device input
  - Intervention detection (dead_zone)
  - Action override: Spacemouse input replaces policy action
  - is_intervened flag is correctly set
  - Piper arm responds to both policy and intervention actions
"""

import time
import numpy as np
from piper_env import PiperRobot
from spacemouse_reader import SpacemouseReader


def main():
    # --- Configuration ---
    HOME_POSE = np.array([57.0, 0.0, 300.0, 0.0, 85.0, 0.0])  # Start high
    DESCEND_SPEED = 0.5    # mm per step downward (policy action)
    STEP_INTERVAL = 0.05   # seconds between steps (~20Hz)
    MAX_STEPS = 500        # total steps before reset
    DEAD_ZONE = 0.1        # Spacemouse dead zone

    # --- Initialize Piper ---
    print("[Test] Connecting to Piper robot...")
    robot = PiperRobot("can0")
    robot.connect()
    robot.enable()
    time.sleep(1.0)

    # Move to home pose
    print(f"[Test] Moving to home pose: {HOME_POSE}")
    robot.move_to_pose(*HOME_POSE, speed=30)
    time.sleep(2.0)

    # --- Initialize Spacemouse ---
    print("[Test] Initializing Spacemouse...")
    reader = SpacemouseReader(dead_zone=DEAD_ZONE, action_scale=2.0)
    reader.start()
    time.sleep(1.0)

    # --- Test Loop ---
    print("=" * 50)
    print("TEST START: Robot will slowly descend.")
    print("Use Spacemouse to intervene and override!")
    print("Press Ctrl+C to stop.")
    print("=" * 50)

    current_pose = HOME_POSE.copy()
    intervention_count = 0
    total_count = 0

    try:
        for step in range(MAX_STEPS):
            # --- Policy action: descend slowly ---
            policy_action = np.array([0.0, 0.0, -DESCEND_SPEED, 0.0])  # [dx, dy, dz, gripper]

            # --- Check Spacemouse intervention ---
            sm_action, is_intervening = reader.get_action()

            if is_intervening and sm_action is not None:
                # Override: use Spacemouse 7D → 4D [dx, dy, dz, gripper]
                override_action = sm_action[[0, 1, 2, 6]]
                action = override_action
                is_intervened = True
                intervention_count += 1
            else:
                action = policy_action
                is_intervened = False

            total_count += 1

            # --- Execute action on Piper ---
            cur_pose = robot.get_end_pose()

            # Compute target pose
            target_pose = cur_pose.copy()
            target_pose[0] += action[0]  # dx
            target_pose[1] += action[1]  # dy
            target_pose[2] += action[2]  # dz

            # Workspace clipping
            target_pose[0] = np.clip(target_pose[0], -200, 200)
            target_pose[1] = np.clip(target_pose[1], -200, 200)
            target_pose[2] = np.clip(target_pose[2], 50, 400)

            robot.move_to_pose(*target_pose, speed=50)

            # Gripper
            gripper_pos = (action[3] + 1.0) / 2.0 * 0.08
            robot.set_gripper(gripper_pos)

            # --- Print status ---
            status = "INTERVENED" if is_intervened else "policy    "
            print(
                f"[Step {step:4d}] {status} | "
                f"action=[{action[0]:+5.2f},{action[1]:+5.2f},{action[2]:+5.2f},{action[3]:+5.2f}] | "
                f"pose=[{cur_pose[0]:6.1f},{cur_pose[1]:6.1f},{cur_pose[2]:6.1f}] | "
                f"interventions={intervention_count}/{total_count}"
            )

            time.sleep(STEP_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user.")

    # --- Summary ---
    print("\n" + "=" * 50)
    print(f"TEST COMPLETE")
    print(f"  Total steps:       {total_count}")
    print(f"  Intervention steps: {intervention_count}")
    print(f"  Intervention rate:  {intervention_count/max(total_count,1)*100:.1f}%")
    print("=" * 50)

    # --- Cleanup ---
    reader.stop()
    print("[Test] Spacemouse reader stopped.")

    # Move back to home pose first, then disable
    print("[Test] Returning to home pose before disable...")
    robot.move_to_pose(*HOME_POSE, speed=30)
    time.sleep(2.0)
    print("[Test] Home pose reached. Now disabling...")

    robot.disable()
    print("[Test] Robot disabled. Done.")


if __name__ == "__main__":
    main()
