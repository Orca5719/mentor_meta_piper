"""
Replay Buffer Trajectory Replay Script
========================================
Replay recorded trajectories from pretrain buffer (.npz files) on the
PiPER real robot, or inspect/validate the data without a robot.

Supports three modes:
  1. inspect  — Print statistics and validate .npz files (no robot needed)
  2. replay   — Replay actions on the real PiPER robot
  3. visualize — Save recorded observation images side-by-side with live camera

Usage:
    # Inspect buffer (no robot needed)
    python replay_buffer_trajectory.py inspect --buffer_dir /path/to/buffer

    # Replay on real robot
    python replay_buffer_trajectory.py replay --buffer_dir /path/to/buffer \
        --can_port can0 --speed 30 --action_scale 2.0

    # Visualize: show recorded images alongside live camera
    python replay_buffer_trajectory.py visualize --buffer_dir /path/to/buffer \
        --can_port can0

    # Replay specific episodes
    python replay_buffer_trajectory.py replay --buffer_dir /path/to/buffer \
        --episode_indices 0 2 5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Data loading helpers (reused from replay_buffer.py)
# ---------------------------------------------------------------------------

def load_episode(fn):
    """Load a single episode from .npz file."""
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    return episode


def episode_len(episode):
    """Get episode length (subtract 1 for the dummy first transition)."""
    return next(iter(episode.values())).shape[0] - 1


def discover_episodes(buffer_dir):
    """Discover and sort all .npz episode files in a buffer directory.

    Returns list of (filepath, episode_index, episode_length) tuples.
    """
    buffer_path = Path(buffer_dir)
    if not buffer_path.exists():
        raise FileNotFoundError(f"Buffer directory not found: {buffer_dir}")

    episodes = []
    for fn in sorted(buffer_path.glob('*.npz')):
        try:
            parts = fn.stem.split('_')
            # Format: {timestamp}_{index}_{length}.npz
            if len(parts) >= 3:
                eps_idx = int(parts[-2])
                eps_len = int(parts[-1])
            else:
                eps_idx = 0
                eps_len = -1  # unknown, will be computed
            episodes.append((fn, eps_idx, eps_len))
        except (ValueError, IndexError):
            # Non-standard filename, still include it
            episodes.append((fn, 0, -1))

    return episodes


# ---------------------------------------------------------------------------
# Mode: inspect
# ---------------------------------------------------------------------------

def inspect_buffer(buffer_dir, verbose=False, save_images=False):
    """Inspect buffer: print statistics and validate data integrity."""
    episodes = discover_episodes(buffer_dir)
    if len(episodes) == 0:
        print(f"[ERROR] No .npz files found in {buffer_dir}")
        return

    print("=" * 60)
    print(f"Buffer Inspection: {buffer_dir}")
    print(f"Total episode files: {len(episodes)}")
    print("=" * 60)

    total_transitions = 0
    all_keys = set()
    has_observation = False
    has_action = False
    has_reward = False
    has_is_intervened = False
    has_state = False

    action_dims = []
    observation_shapes = []
    reward_sums = []
    episode_lengths = []
    intervention_counts = []
    corrupted = []

    output_dir = Path(buffer_dir) / "inspection_images"
    if save_images:
        output_dir.mkdir(exist_ok=True)

    for i, (fn, eps_idx, eps_len) in enumerate(episodes):
        try:
            ep = load_episode(fn)
        except Exception as e:
            corrupted.append((fn, str(e)))
            continue

        keys = set(ep.keys())
        all_keys.update(keys)

        elen = episode_len(ep)
        episode_lengths.append(elen)
        total_transitions += elen

        # Check key fields
        if 'observation' in ep:
            has_observation = True
            observation_shapes.append(ep['observation'].shape)
        if 'action' in ep:
            has_action = True
            action_dims.append(ep['action'].shape)
        if 'reward' in ep:
            has_reward = True
            reward_sums.append(ep['reward'][1:].sum())  # skip dummy first
        if 'is_intervened' in ep:
            has_is_intervened = True
            intervention_counts.append(int(ep['is_intervened'][1:].sum()))
        if 'state' in ep:
            has_state = True

        # Save first observation image from each episode
        if save_images and 'observation' in ep and cv2 is not None:
            obs = ep['observation'][0]
            # observation might be (C, H, W) or (H, W, C)
            if obs.ndim == 3 and obs.shape[0] in (3, 9):
                # (C, H, W) -> (H, W, C)
                img = np.transpose(obs, (1, 2, 0))
                if img.shape[2] == 9:
                    # frame-stacked, just save the last 3 channels
                    img = img[:, :, -3:]
            elif obs.ndim == 3:
                img = obs
            else:
                img = None

            if img is not None:
                img_path = output_dir / f"ep{eps_idx:04d}_step0.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Verbose: print per-episode info
        if verbose:
            info_parts = [f"  [{i:3d}] {fn.name}  len={elen}"]
            if 'reward' in ep:
                info_parts.append(f"reward_sum={ep['reward'][1:].sum():.2f}")
            if 'is_intervened' in ep:
                n_int = int(ep['is_intervened'][1:].sum())
                info_parts.append(f"interventions={n_int}/{elen}")
            if 'action' in ep:
                actions = ep['action'][1:]
                info_parts.append(
                    f"action_range=[{actions.min():.3f}, {actions.max():.3f}]")
            print("  ".join(info_parts))

    # Summary
    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Total episodes:      {len(episodes)}")
    print(f"  Total transitions:   {total_transitions}")
    print(f"  Episode length:      min={min(episode_lengths)}, "
          f"max={max(episode_lengths)}, "
          f"mean={np.mean(episode_lengths):.1f}")
    print(f"  Keys in episodes:    {sorted(all_keys)}")
    print(f"  Has 'observation':   {has_observation}")
    print(f"  Has 'action':        {has_action}")
    print(f"  Has 'reward':        {has_reward}")
    print(f"  Has 'is_intervened': {has_is_intervened}")
    print(f"  Has 'state':         {has_state}")

    if has_action and action_dims:
        unique_shapes = set(action_dims)
        print(f"  Action shapes:       {unique_shapes}")

    if has_observation and observation_shapes:
        unique_obs_shapes = set(observation_shapes)
        print(f"  Observation shapes:  {unique_obs_shapes}")

    if has_reward and reward_sums:
        print(f"  Reward sum:          min={min(reward_sums):.2f}, "
              f"max={max(reward_sums):.2f}, "
              f"mean={np.mean(reward_sums):.2f}")

    if has_is_intervened and intervention_counts:
        total_int = sum(intervention_counts)
        print(f"  Interventions:       {total_int}/{total_transitions} "
              f"({total_int/max(total_transitions,1)*100:.1f}%)")
        intervened_eps = sum(1 for c in intervention_counts if c > 0)
        print(f"  Intervened episodes: {intervened_eps}/{len(episodes)}")

    if corrupted:
        print(f"\n  CORRUPTED FILES ({len(corrupted)}):")
        for fn, err in corrupted:
            print(f"    {fn.name}: {err}")

    if save_images:
        print(f"\n  Inspection images saved to: {output_dir}")

    print()
    return {
        'total_episodes': len(episodes),
        'total_transitions': total_transitions,
        'keys': sorted(all_keys),
        'corrupted': len(corrupted),
    }


# ---------------------------------------------------------------------------
# Mode: replay on real robot
# ---------------------------------------------------------------------------

def replay_on_robot(buffer_dir, can_port, speed, action_scale,
                     episode_indices=None, step_delay=0.05,
                     max_steps=None, dry_run=False):
    """Replay recorded actions on the PiPER real robot.

    For each selected episode:
      1. Move robot to home pose
      2. Step through the recorded actions one by one
      3. Print per-step info (action, pose, reward, intervened)
    """
    # Import robot interface
    try:
        from piper_env import PiperRobot
    except ImportError:
        print("[ERROR] Cannot import PiperRobot. Make sure piper_sdk is installed.")
        return

    # Load episodes
    episodes = discover_episodes(buffer_dir)
    if len(episodes) == 0:
        print(f"[ERROR] No .npz files found in {buffer_dir}")
        return

    # Filter by indices
    if episode_indices is not None:
        selected = []
        for idx in episode_indices:
            if 0 <= idx < len(episodes):
                selected.append(episodes[idx])
            else:
                print(f"[WARN] Episode index {idx} out of range (0-{len(episodes)-1})")
        episodes = selected

    if len(episodes) == 0:
        print("[ERROR] No episodes to replay after filtering")
        return

    # Connect robot
    print("[Replay] Connecting to PiPER robot...")
    robot = PiperRobot(can_port)
    robot.connect()
    robot.enable()
    time.sleep(1.0)

    print(f"[Replay] Will replay {len(episodes)} episode(s)")
    print(f"[Replay] Speed={speed}, action_scale={action_scale}, step_delay={step_delay}s")
    print()

    try:
        for ep_i, (fn, eps_idx, eps_len) in enumerate(episodes):
            ep = load_episode(fn)
            elen = episode_len(ep)

            if 'action' not in ep:
                print(f"[WARN] Episode {fn.name} has no 'action' field, skipping")
                continue

            actions = ep['action']
            rewards = ep.get('reward', np.zeros(elen + 1))
            is_intervened = ep.get('is_intervened', np.zeros(elen + 1))

            # Determine action dimension mapping
            action_shape = actions.shape
            print("=" * 60)
            print(f"Episode {ep_i+1}/{len(episodes)}: {fn.name}")
            print(f"  Length: {elen}, Action shape: {action_shape}")
            if is_intervened[1:].sum() > 0:
                n_int = int(is_intervened[1:].sum())
                print(f"  Intervened steps: {n_int}/{elen}")
            print()

            # Move to home pose
            print("  Moving to home pose...")
            robot.move_to_pose(57.0, 0.0, 215.0, 0.0, 85.0, 0.0, speed=30)
            time.sleep(2.0)
            robot.set_gripper(0.0)
            time.sleep(0.5)

            if dry_run:
                print("  [DRY RUN] Skipping actual motion")
                continue

            # Replay each step (skip the dummy first transition at index 0)
            steps_to_replay = elen if max_steps is None else min(elen, max_steps)
            for step in range(steps_to_replay):
                action = actions[step + 1]  # +1 to skip dummy first
                reward = rewards[step + 1]
                intervened = is_intervened[step + 1]

                # Get current pose
                cur_pose = robot.get_end_pose()

                # Map action to pose delta
                # Action format: [dx, dy, dz, gripper] ∈ [-1, 1]
                if len(action) >= 4:
                    dx = action[0] * action_scale  # mm
                    dy = action[1] * action_scale
                    dz = action[2] * action_scale

                    tgt_x = np.clip(cur_pose[0] + dx, -200, 200)
                    tgt_y = np.clip(cur_pose[1] + dy, -200, 200)
                    tgt_z = np.clip(cur_pose[2] + dz, 50, 400)

                    # Keep current orientation
                    robot.move_to_pose(
                        tgt_x, tgt_y, tgt_z,
                        cur_pose[3], cur_pose[4], cur_pose[5],
                        speed=speed
                    )

                    # Gripper
                    gripper_pos = (action[3] + 1.0) / 2.0 * 0.08
                    robot.set_gripper(gripper_pos)

                # Safety check
                if not robot.check_safety():
                    print("\n[SAFETY] Robot safety check failed! Stopping replay.")
                    raise RuntimeError("Safety check failed")

                # Print status
                int_flag = "INT" if intervened > 0.5 else "   "
                print(
                    f"  Step {step+1:3d}/{steps_to_replay} "
                    f"{int_flag} "
                    f"action=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},{action[3]:+.3f}] "
                    f"pose=[{cur_pose[0]:6.1f},{cur_pose[1]:6.1f},{cur_pose[2]:6.1f}] "
                    f"reward={reward:.2f}"
                )

                time.sleep(step_delay)

            print(f"  Episode replay complete.")
            print()

    except KeyboardInterrupt:
        print("\n[Replay] Interrupted by user")
    finally:
        # Move home then disable
        print("[Replay] Returning to home pose...")
        try:
            robot.move_to_pose(57.0, 0.0, 215.0, 0.0, 85.0, 0.0, speed=30)
            time.sleep(2.0)
        except Exception:
            pass
        robot.disable()
        print("[Replay] Robot disabled. Done.")


# ---------------------------------------------------------------------------
# Mode: visualize (recorded images + live camera side by side)
# ---------------------------------------------------------------------------

def visualize_replay(buffer_dir, can_port, episode_indices=None,
                      step_delay=0.1, max_steps=None, save_video=False):
    """Replay on robot while showing recorded images alongside live camera feed."""
    try:
        from piper_env import PiperRobot
        from realsense_camera import RealSenseCamera
    except ImportError as e:
        print(f"[ERROR] Missing import: {e}")
        return

    if cv2 is None:
        print("[ERROR] OpenCV is required for visualization. Install: pip install opencv-python")
        return

    # Load episodes
    episodes = discover_episodes(buffer_dir)
    if len(episodes) == 0:
        print(f"[ERROR] No .npz files found in {buffer_dir}")
        return

    if episode_indices is not None:
        selected = []
        for idx in episode_indices:
            if 0 <= idx < len(episodes):
                selected.append(episodes[idx])
        episodes = selected

    # Connect robot and camera
    print("[Visualize] Connecting PiPER robot and RealSense camera...")
    robot = PiperRobot(can_port)
    robot.connect()
    robot.enable()
    time.sleep(1.0)

    camera = RealSenseCamera(target_size=(84, 84))
    camera.connect()
    time.sleep(1.0)

    action_scale = 2.0
    speed = 50

    try:
        for ep_i, (fn, eps_idx, eps_len) in enumerate(episodes):
            ep = load_episode(fn)
            elen = episode_len(ep)

            if 'action' not in ep or 'observation' not in ep:
                print(f"[WARN] Episode {fn.name} missing action/observation, skipping")
                continue

            actions = ep['action']
            observations = ep['observation']
            rewards = ep.get('reward', np.zeros(elen + 1))
            is_intervened = ep.get('is_intervened', np.zeros(elen + 1))

            print(f"\nEpisode {ep_i+1}: {fn.name} (len={elen})")

            # Move to home
            robot.move_to_pose(57.0, 0.0, 215.0, 0.0, 85.0, 0.0, speed=30)
            time.sleep(2.0)
            robot.set_gripper(0.0)

            steps_to_replay = elen if max_steps is None else min(elen, max_steps)

            for step in range(steps_to_replay):
                action = actions[step + 1]
                recorded_obs = observations[step]  # observation at this step

                # Execute action on robot
                if len(action) >= 4:
                    cur_pose = robot.get_end_pose()
                    dx = action[0] * action_scale
                    dy = action[1] * action_scale
                    dz = action[2] * action_scale
                    tgt_x = np.clip(cur_pose[0] + dx, -200, 200)
                    tgt_y = np.clip(cur_pose[1] + dy, -200, 200)
                    tgt_z = np.clip(cur_pose[2] + dz, 50, 400)
                    robot.move_to_pose(tgt_x, tgt_y, tgt_z,
                                       cur_pose[3], cur_pose[4], cur_pose[5],
                                       speed=speed)
                    gripper_pos = (action[3] + 1.0) / 2.0 * 0.08
                    robot.set_gripper(gripper_pos)

                # Capture live image
                live_img = camera.capture()

                # Prepare recorded image for display
                if recorded_obs.ndim == 3 and recorded_obs.shape[0] in (3, 9):
                    rec_img = np.transpose(recorded_obs, (1, 2, 0))
                    if rec_img.shape[2] == 9:
                        rec_img = rec_img[:, :, -3:]
                elif recorded_obs.ndim == 3:
                    rec_img = recorded_obs
                elif recorded_obs.ndim == 2:
                    rec_img = np.stack([recorded_obs] * 3, axis=-1)
                else:
                    rec_img = np.zeros((84, 84, 3), dtype=np.uint8)

                # Resize for display
                display_size = (252, 252)  # 3x
                rec_display = cv2.resize(rec_img, display_size)
                live_display = cv2.resize(live_img, display_size)

                # Side by side
                combined = np.hstack([rec_display, live_display])

                # Labels
                cv2.putText(combined, "Recorded", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "Live", (display_size[0] + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Step info
                int_flag = "INT" if is_intervened[step + 1] > 0.5 else ""
                info_text = (f"Step {step+1}/{steps_to_replay} "
                             f"reward={rewards[step+1]:.2f} {int_flag}")
                cv2.putText(combined, info_text, (10, display_size[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Replay: Recorded (left) vs Live (right)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                elif key == ord(' '):
                    # Pause until another key
                    cv2.waitKey(0)

                time.sleep(step_delay)

    except KeyboardInterrupt:
        print("\n[Visualize] Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        try:
            robot.move_to_pose(57.0, 0.0, 215.0, 0.0, 85.0, 0.0, speed=30)
            time.sleep(2.0)
        except Exception:
            pass
        camera.disconnect()
        robot.disable()
        print("[Visualize] Done.")


# ---------------------------------------------------------------------------
# Mode: export images from buffer (no robot needed)
# ---------------------------------------------------------------------------

def export_images(buffer_dir, output_dir=None, episode_indices=None,
                   max_episodes=None, max_steps_per_episode=None):
    """Export observation images from buffer episodes as PNG files."""
    if cv2 is None:
        print("[ERROR] OpenCV is required. Install: pip install opencv-python")
        return

    episodes = discover_episodes(buffer_dir)
    if len(episodes) == 0:
        print(f"[ERROR] No .npz files found in {buffer_dir}")
        return

    if episode_indices is not None:
        episodes = [episodes[i] for i in episode_indices if i < len(episodes)]

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    if output_dir is None:
        output_dir = Path(buffer_dir) / "exported_images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Exporting images to {output_dir}...")

    for ep_i, (fn, eps_idx, eps_len) in enumerate(episodes):
        ep = load_episode(fn)
        elen = episode_len(ep)

        if 'observation' not in ep:
            continue

        observations = ep['observation']
        actions = ep.get('action', np.zeros((elen + 1, 4)))
        rewards = ep.get('reward', np.zeros(elen + 1))
        is_intervened = ep.get('is_intervened', np.zeros(elen + 1))

        ep_dir = output_dir / f"episode_{eps_idx:04d}"
        ep_dir.mkdir(exist_ok=True)

        steps = elen if max_steps_per_episode is None else min(elen, max_steps_per_episode)

        for step in range(steps):
            obs = observations[step]

            # Convert observation to image
            if obs.ndim == 3 and obs.shape[0] in (3, 9):
                img = np.transpose(obs, (1, 2, 0))
                if img.shape[2] == 9:
                    img = img[:, :, -3:]
            elif obs.ndim == 3:
                img = obs
            elif obs.ndim == 2:
                img = np.stack([obs] * 3, axis=-1)
            else:
                continue

            # Scale up for visibility
            img_display = cv2.resize(img, (252, 252))

            # Add info overlay
            action_str = ""
            if step + 1 < len(actions):
                a = actions[step + 1]
                action_str = f"a=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]"
            reward_str = f"r={rewards[step+1]:.2f}" if step + 1 < len(rewards) else ""
            int_str = "INT" if step + 1 < len(is_intervened) and is_intervened[step + 1] > 0.5 else ""

            cv2.putText(img_display, f"step={step} {action_str}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(img_display, f"{reward_str} {int_str}", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            img_path = ep_dir / f"step_{step:04d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))

        print(f"  Episode {eps_idx}: {steps} images saved")

    print(f"Done. Images exported to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay and inspect pretrain buffer trajectories for PiPER robot")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # --- inspect ---
    p_inspect = subparsers.add_parser('inspect',
        help='Inspect buffer: print statistics and validate data (no robot needed)')
    p_inspect.add_argument('--buffer_dir', type=str, required=True,
        help='Path to buffer directory containing .npz files')
    p_inspect.add_argument('--verbose', '-v', action='store_true',
        help='Print per-episode details')
    p_inspect.add_argument('--save_images', action='store_true',
        help='Save first observation image from each episode')

    # --- replay ---
    p_replay = subparsers.add_parser('replay',
        help='Replay recorded actions on the PiPER real robot')
    p_replay.add_argument('--buffer_dir', type=str, required=True,
        help='Path to buffer directory containing .npz files')
    p_replay.add_argument('--can_port', type=str, default='can0',
        help='CAN port for PiPER robot (default: can0)')
    p_replay.add_argument('--speed', type=int, default=30,
        help='Robot motion speed 0-100 (default: 30, slower for safety)')
    p_replay.add_argument('--action_scale', type=float, default=2.0,
        help='mm per action unit (default: 2.0)')
    p_replay.add_argument('--episode_indices', type=int, nargs='+', default=None,
        help='Specific episode indices to replay (0-based). Default: all')
    p_replay.add_argument('--step_delay', type=float, default=0.05,
        help='Delay between steps in seconds (default: 0.05)')
    p_replay.add_argument('--max_steps', type=int, default=None,
        help='Max steps per episode to replay (default: all)')
    p_replay.add_argument('--dry_run', action='store_true',
        help='Load data but do not execute robot motions')

    # --- visualize ---
    p_visualize = subparsers.add_parser('visualize',
        help='Replay with live camera view alongside recorded observations')
    p_visualize.add_argument('--buffer_dir', type=str, required=True,
        help='Path to buffer directory containing .npz files')
    p_visualize.add_argument('--can_port', type=str, default='can0',
        help='CAN port for PiPER robot')
    p_visualize.add_argument('--episode_indices', type=int, nargs='+', default=None,
        help='Specific episode indices (0-based)')
    p_visualize.add_argument('--step_delay', type=float, default=0.1,
        help='Delay between steps (default: 0.1)')
    p_visualize.add_argument('--max_steps', type=int, default=None,
        help='Max steps per episode (default: all)')

    # --- export ---
    p_export = subparsers.add_parser('export',
        help='Export observation images from buffer as PNG files (no robot needed)')
    p_export.add_argument('--buffer_dir', type=str, required=True,
        help='Path to buffer directory containing .npz files')
    p_export.add_argument('--output_dir', type=str, default=None,
        help='Output directory for images (default: buffer_dir/exported_images)')
    p_export.add_argument('--episode_indices', type=int, nargs='+', default=None,
        help='Specific episode indices (0-based)')
    p_export.add_argument('--max_episodes', type=int, default=None,
        help='Max number of episodes to export')
    p_export.add_argument('--max_steps', type=int, default=None,
        help='Max steps per episode to export')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    if args.mode == 'inspect':
        inspect_buffer(args.buffer_dir, verbose=args.verbose,
                       save_images=args.save_images)

    elif args.mode == 'replay':
        replay_on_robot(
            args.buffer_dir,
            can_port=args.can_port,
            speed=args.speed,
            action_scale=args.action_scale,
            episode_indices=args.episode_indices,
            step_delay=args.step_delay,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )

    elif args.mode == 'visualize':
        visualize_replay(
            args.buffer_dir,
            can_port=args.can_port,
            episode_indices=args.episode_indices,
            step_delay=args.step_delay,
            max_steps=args.max_steps,
        )

    elif args.mode == 'export':
        export_images(
            args.buffer_dir,
            output_dir=args.output_dir,
            episode_indices=args.episode_indices,
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps,
        )


if __name__ == '__main__':
    main()
