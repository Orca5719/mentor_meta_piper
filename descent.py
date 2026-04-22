import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import sys
import time
import cv2
import numpy as np
import torch
import threading
from collections import deque, namedtuple
from pathlib import Path
from dm_env import StepType, specs
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from replay_buffer import ReplayBufferStorage, make_replay_loader
    import utils
    from logger import Logger
    from piper_env import PiperRobot
    from realsense_camera import RealSenseCamera
    from spacemouse_reader import SpacemouseReader
    import hydra
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    sys.exit(1)

_TimeStepBase = namedtuple('_TimeStepBase', [
    'observation', 'action', 'reward', 'discount', 'first', 'is_last', 'is_intervened'
])

class TimeStep(_TimeStepBase):
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)

    def last(self):
        return self.is_last


class PiperRobotTrainer:
    def __init__(self):
        print("="*60)
        print("     Piper 机械臂实时训练 (兼容 mentor piper_env)")
        print("="*60)

        self.work_dir = Path.cwd()

        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.frame_stack = 3
        self.batch_size = 256
        self.update_every_episodes = 10
        self.save_interval = 1000
        self.seed_steps = 1000

        self.camera = None
        self.piper_arm = None
        self.agent = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._global_step = 0
        self._global_episode = 0
        self.last_save_step = -9999

        self.frames_queue = deque(maxlen=3)
        self.replay_storage = None
        self.replay_loader = None
        self.replay_iter = None

        self._reward = 0.0
        self.episode_reward = 0.0
        self.episode_step = 0
        self._last_action = None
        self._obs_spec = None
        self._act_spec = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = True

        # 机械臂初始位姿 (mm, deg 浮点单位，与 piper_env.py 一致)
        self.HOME_X, self.HOME_Y, self.HOME_Z = 300.614, -12.185, 282.341
        self.HOME_RX, self.HOME_RY, self.HOME_RZ = -179.351, 23.933, 177.934
        self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
        self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
        self.gripper_open = True

        self._buffer_dir = Path.cwd() / 'buffer_robot'

        # 动作参数 (mm 浮点单位)
        self.action_scale = 2.0   # mm per action unit，与 piper_env.py 的 action_scale 一致
        self.action_interval = 0.08  # 最小动作间隔(秒)

        # 安全范围 (mm)
        self.WS_X_MIN, self.WS_X_MAX = 150.0, 450.0
        self.WS_Y_MIN, self.WS_Y_MAX = -150.0, 150.0
        self.WS_Z_MIN, self.WS_Z_MAX = 150.0, 350.0

        self.random_amplitude = 0.6
        self.random_drift_prob = 0.2
        self.last_random_direction = np.zeros(3)

        # 输出目录
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path.cwd() / "piper_outputs" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 训练输出目录: {self.output_dir.resolve()}")

        # 3D鼠标
        self.DEAD_ZONE = 0.15
        self.SPACE_MOUSE_ACTION_SCALE = 1.0
        self.spacemouse_reader = None
        self.is_intervening = False

        # 夹爪
        self.random_gripper_state = 1.0
        self.last_gripper_change_step = 0
        self.gripper_change_interval = 100

        # 防卡死机制
        self.last_action_time = time.time()
        self.arm_command_lock = threading.Lock()
        self.heartbeat_timeout = 2.0
        self.last_heartbeat = time.time()

        # 闭环位置反馈
        self.use_closed_loop = True

    def _init_specs(self):
        """直接硬编码spec，与 piper_env.py 的 piper_wrapper 输出一致。"""
        self._obs_spec = specs.BoundedArray(
            shape=(9, self.IMG_HEIGHT, self.IMG_WIDTH),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        self._act_spec = specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )

    def _init_replay_storage(self):
        """初始化ReplayBuffer，spec与 train_mw.py 一致（5个字段，含is_intervened）。"""
        self._init_specs()

        data_specs = (
            specs.Array(self._obs_spec.shape, self._obs_spec.dtype, 'observation'),
            specs.Array(self._act_spec.shape, self._act_spec.dtype, 'action'),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
            specs.Array((1,), np.float32, 'is_intervened'),
        )

        self._buffer_dir.mkdir(exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, self._buffer_dir)

        self.replay_loader, _ = make_replay_loader(
            self._buffer_dir, max_size=100000, batch_size=self.batch_size, num_workers=4,
            save_snapshot=False, nstep=3, discount=0.99
        )
        self.replay_iter = iter(self.replay_loader)

        print("✅ 回放缓冲区初始化完成")

    def init_hardware(self):
        """初始化硬件，全部使用本地 piper_env.py / realsense_camera.py 接口。"""
        print("初始化硬件...")

        # 相机：使用 RealSenseCamera
        self.camera = RealSenseCamera(
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH)
        )
        self.camera.connect()

        # 机械臂：使用 piper_env.PiperRobot
        self.piper_arm = PiperRobot("can0")
        self.piper_arm.connect()
        self.piper_arm.enable()

        # 清除所有错误状态
        time.sleep(0.5)

        # 移动到初始位姿
        self.piper_arm.move_to_pose(
            self.HOME_X, self.HOME_Y, self.HOME_Z,
            self.HOME_RX, self.HOME_RY, self.HOME_RZ,
            speed=50
        )
        self.piper_arm.set_gripper(0.08)  # 打开夹爪
        time.sleep(2.0)

        # 读取实际位置，同步软件位置
        self._sync_actual_position()

        self._init_replay_storage()

        # 3D鼠标：使用 SpacemouseReader（非阻塞）
        self.spacemouse_reader = SpacemouseReader(
            dead_zone=self.DEAD_ZONE,
            action_scale=self.SPACE_MOUSE_ACTION_SCALE
        )
        self.spacemouse_reader.start()
        time.sleep(1.0)

        # 图像线程
        threading.Thread(target=self._image_thread, daemon=True).start()
        time.sleep(0.5)

        # 机械臂状态监控线程
        threading.Thread(target=self._arm_status_monitor_thread, daemon=True).start()

        print("✅ 硬件初始化完成")
        print()

    def _sync_actual_position(self):
        """读取机械臂实际位置，同步到软件变量（mm 浮点单位）"""
        with self.arm_command_lock:
            try:
                actual_pose = self.piper_arm.get_end_pose()  # 返回 [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]
                if actual_pose is not None and len(actual_pose) >= 6:
                    self.X = actual_pose[0]
                    self.Y = actual_pose[1]
                    self.Z = actual_pose[2]
                    self.RX = actual_pose[3]
                    self.RY = actual_pose[4]
                    self.RZ = actual_pose[5]
                    print(f"✅ 位置同步成功: X={self.X:.1f}, Y={self.Y:.1f}, Z={self.Z:.1f}")
            except Exception as e:
                print(f"⚠️  位置同步失败: {e}")

    def _arm_status_monitor_thread(self):
        """后台监控机械臂状态，自动复位保护"""
        while self._running:
            try:
                with self.arm_command_lock:
                    if not self.piper_arm.check_safety():
                        print("\n❌ 机械臂触发保护！")
                        print("正在自动复位...")

                        # PiperRobot.check_safety() 已经在严重情况下调用了 emergency_stop
                        # 需要重新使能
                        try:
                            self.piper_arm.enable()
                        except:
                            pass
                        time.sleep(0.2)

                        # 同步实际位置
                        self._sync_actual_position()

                        print("✅ 机械臂自动复位完成")

                time.sleep(0.5)
            except Exception as e:
                time.sleep(1.0)

    def _image_thread(self):
        while self._running:
            try:
                frame = self.camera.capture()  # RealSenseCamera.capture() 已 resize，返回 RGB
                if frame is not None:
                    with self._lock:
                        self.frames_queue.append(frame)
                        self._latest_frame = frame.copy()
            except Exception as e:
                print(f"[图像线程错误] {e}")
            time.sleep(0.005)

    def get_stacked_obs(self):
        with self._lock:
            frames = list(self.frames_queue)

        while len(frames) < 3:
            frames.append(frames[0] if frames else np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8))

        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))

        if self._obs_spec.dtype == np.uint8:
            return stacked.astype(np.uint8)
        return (stacked.astype(np.float32) / 255.0)

    def apply_action(self, action):
        """
        安全版动作执行（mm 浮点单位）：
        1. 闭环位置控制
        2. 速度限制
        3. 工作空间裁剪
        """
        now = time.time()
        if now - self.last_action_time < self.action_interval:
            return
        self.last_action_time = now
        self.last_heartbeat = now

        with self.arm_command_lock:
            try:
                # 先同步实际位置（闭环控制）
                if self.use_closed_loop:
                    actual_pose = self.piper_arm.get_end_pose()
                    if actual_pose is not None and len(actual_pose) >= 6:
                        self.X = actual_pose[0]
                        self.Y = actual_pose[1]
                        self.Z = actual_pose[2]

                # 计算目标位置 (action ∈ [-1,1] * action_scale mm)
                dx = action[0] * self.action_scale
                dy = action[1] * self.action_scale
                dz = action[2] * self.action_scale

                # 工作空间裁剪
                target_X = np.clip(self.X + dx, self.WS_X_MIN, self.WS_X_MAX)
                target_Y = np.clip(self.Y + dy, self.WS_Y_MIN, self.WS_Y_MAX)
                target_Z = np.clip(self.Z + dz, self.WS_Z_MIN, self.WS_Z_MAX)

                # 夹爪控制
                if action[3] > 0:
                    if not self.gripper_open:
                        self.gripper_open = True
                        self.piper_arm.set_gripper(0.08)
                else:
                    if self.gripper_open:
                        self.gripper_open = False
                        self.piper_arm.set_gripper(0.0)

                # 发送位置指令
                self.piper_arm.move_to_pose(
                    target_X, target_Y, target_Z,
                    self.RX, self.RY, self.RZ,
                    speed=50
                )

                # 更新软件位置
                self.X = target_X
                self.Y = target_Y
                self.Z = target_Z

            except Exception as e:
                print(f"[机械臂指令错误] {e}")

    def get_action(self, obs):
        # 3D鼠标优先
        sm_action, is_intervening = self.spacemouse_reader.get_action()
        self.is_intervening = is_intervening

        if is_intervening and sm_action is not None:
            dx = sm_action[0] if abs(sm_action[0]) > self.DEAD_ZONE else 0.0
            dy = sm_action[1] if abs(sm_action[1]) > self.DEAD_ZONE else 0.0
            dz = sm_action[2] if abs(sm_action[2]) > self.DEAD_ZONE else 0.0

            sm_gripper = sm_action[6]
            if abs(sm_gripper) > self.DEAD_ZONE:
                gripper_ctrl = 1.0 if sm_gripper > 0 else -1.0
            else:
                gripper_ctrl = 1.0 if self.gripper_open else -1.0

            override_action = np.array([
                dx * self.SPACE_MOUSE_ACTION_SCALE,
                dy * self.SPACE_MOUSE_ACTION_SCALE,
                dz * self.SPACE_MOUSE_ACTION_SCALE,
                gripper_ctrl
            ], dtype=self._act_spec.dtype)
            override_action = np.clip(override_action, self._act_spec.minimum, self._act_spec.maximum)
            return override_action, True

        # 随机策略
        if self.agent is None or self._global_step < self.seed_steps:
            action = np.zeros(self._act_spec.shape, dtype=self._act_spec.dtype)

            if np.random.random() < self.random_drift_prob or np.linalg.norm(self.last_random_direction) == 0:
                direction = np.random.uniform(-1, 1, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction
            else:
                direction = self.last_random_direction
                direction += np.random.normal(0, 0.15, 3)
                direction = direction / np.linalg.norm(direction)
                self.last_random_direction = direction

            action[:3] = direction * self.random_amplitude

            if self._global_step - self.last_gripper_change_step >= self.gripper_change_interval:
                self.random_gripper_state = np.random.choice([1.0, -1.0])
                self.last_gripper_change_step = self._global_step

            action[3] = self.random_gripper_state
            return action, False

        # 智能体策略
        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self._global_step, eval_mode=False)
        return action, False

    def update_policy(self, num_updates=100):
        if self.agent is None or self.replay_loader is None:
            return
        if self._global_step < self.seed_steps:
            return

        try:
            print(f"\n开始更新策略，执行 {num_updates} 次梯度更新...")
            for i in range(num_updates):
                metrics = self.agent.update(self.replay_iter, self._global_step)
                if i % 20 == 0:
                    print(f"  进度: {i+1}/{num_updates}", end='\r')
            print(f"\n✅ 策略更新完成")
            return metrics
        except Exception as e:
            print(f"❌ 策略更新失败: {e}")
            return None

    def save_snapshot(self):
        if self.agent is None:
            return
        snapshot_path = self.output_dir / f'snapshot_robot_{self._global_step}.pt'
        payload = {
            'agent': self.agent,
            '_global_step': self._global_step,
            '_global_episode': self._global_episode
        }
        torch.save(payload, snapshot_path)
        print(f"\n✅ 模型保存: {snapshot_path.name}")

    def load_snapshot(self, snapshot_path):
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.exists():
            print(f"⚠️  快照文件不存在: {snapshot_path}")
            return False

        try:
            payload = torch.load(snapshot_path, map_location=self.device, weights_only=False)
            if 'actor_state_dict' in payload and 'agent' not in payload:
                print("加载预训练Actor权重...")
                if hasattr(self.agent, 'actor'):
                    self.agent.actor.load_state_dict(payload['actor_state_dict'])
                    self.agent.actor.to(self.device)
                    print("✅ Actor权重加载成功")
            else:
                for k, v in payload.items():
                    if k in self.__dict__:
                        self.__dict__[k] = v
                        if k == 'agent':
                            self.agent.to(self.device)
            print("✅ 快照加载成功")
            return True
        except Exception as e:
            print(f"❌ 加载快照失败: {e}")
            return False

    def visualize(self, obs, reward, episode_step):
        if self._latest_frame is None:
            return True

        frame = self._latest_frame.copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        y_pos = 30
        line_spacing = 25

        if self._global_step < self.seed_steps:
            cv2.putText(frame_bgr, "SEEDING...", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            y_pos += line_spacing
            cv2.putText(frame_bgr, f"Seed: {self._global_step}/{self.seed_steps}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        else:
            cv2.putText(frame_bgr, "TRAINING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += line_spacing

        if self.is_intervening:
            cv2.putText(frame_bgr, "INTERVENING", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame_bgr, "Model Control", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, f"Step: {self._global_step}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Episode: {self._global_episode + 1}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame_bgr, f"Reward: {self.episode_reward:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame_bgr, "SPACE=+10, s=Save, q=Quit", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Piper Robot Training", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self._reward += 10.0
        elif key == ord('s'):
            self.save_snapshot()
        elif key == ord('q'):
            return False

        return True

    def train(self, num_episodes=100, max_steps_per_episode=200):
        print("\n开始训练...")
        print("="*60)

        episodes_bar = tqdm(range(num_episodes), desc="Episodes", unit="episode")

        try:
            for episode in episodes_bar:
                self._global_episode = episode
                self.episode_step = 0
                self.episode_reward = 0.0

                # 复位
                self.X, self.Y, self.Z = self.HOME_X, self.HOME_Y, self.HOME_Z
                self.RX, self.RY, self.RZ = self.HOME_RX, self.HOME_RY, self.HOME_RZ
                self.gripper_open = True

                with self.arm_command_lock:
                    self.piper_arm.move_to_pose(
                        self.X, self.Y, self.Z,
                        self.RX, self.RY, self.RZ,
                        speed=50
                    )
                    self.piper_arm.set_gripper(0.08)
                time.sleep(2.0)

                # 复位后同步位置
                self._sync_actual_position()

                self.frames_queue.clear()
                for _ in range(self.frame_stack):
                    frame = self.camera.capture()
                    if frame is not None:
                        self.frames_queue.append(frame)

                obs_prev = self.get_stacked_obs()

                step_bar = tqdm(range(max_steps_per_episode), desc=f"Episode {episode+1}", unit="step", leave=False)
                for step in step_bar:
                    self.episode_step = step
                    self._global_step += 1

                    # 动作
                    action, is_intervened = self.get_action(obs_prev)
                    self._last_action = action
                    self.apply_action(action)

                    # 观测
                    new_frame = self.camera.capture()
                    if new_frame is not None:
                        self.frames_queue.append(new_frame)
                    obs_curr = self.get_stacked_obs()

                    # 奖励
                    reward = self._reward
                    if self._reward != 0:
                        self._reward = 0
                    self.episode_reward += reward

                    # 缓冲区
                    ts = TimeStep(
                        observation=obs_prev,
                        action=action,
                        reward=np.array([reward], dtype=np.float32),
                        discount=np.array([1.0], dtype=np.float32),
                        first=(step == 0),
                        is_last=False,
                        is_intervened=np.array([1.0 if is_intervened else 0.0], dtype=np.float32),
                    )
                    self.replay_storage.add(ts)

                    # 保存
                    if self._global_step - self.last_save_step >= self.save_interval:
                        self.last_save_step = self._global_step
                        self.save_snapshot()

                    obs_prev = obs_curr

                    step_bar.set_postfix({
                        'Global Step': self._global_step,
                        'Reward': f"{self.episode_reward:.1f}",
                        'Intervene': self.is_intervening
                    })

                    # 可视化
                    if not self.visualize(obs_curr, reward, step):
                        step_bar.close()
                        episodes_bar.close()
                        print("\n用户退出训练")
                        return

                step_bar.close()

                # Episode结束
                ts_last = TimeStep(
                    observation=self.get_stacked_obs(),
                    action=self._last_action,
                    reward=np.array([0.0], dtype=np.float32),
                    discount=np.array([0.0], dtype=np.float32),
                    first=False,
                    is_last=True,
                    is_intervened=np.array([0.0], dtype=np.float32),
                )
                self.replay_storage.add(ts_last)

                # 策略更新
                if (episode + 1) % self.update_every_episodes == 0 and self._global_step >= self.seed_steps:
                    self.update_policy(num_updates=100)

                episodes_bar.set_postfix({
                    'Last Reward': f"{self.episode_reward:.1f}",
                    'Buffer Size': len(self.replay_storage),
                    'Global Step': self._global_step
                })

        except KeyboardInterrupt:
            print("\n用户中断训练")
        finally:
            episodes_bar.close()
            self.cleanup()

    def cleanup(self):
        print("\n清理资源...")
        self._running = False
        time.sleep(0.5)
        cv2.destroyAllWindows()

        if self.spacemouse_reader is not None:
            self.spacemouse_reader.stop()

        if self.piper_arm is not None:
            try:
                with self.arm_command_lock:
                    self.piper_arm.emergency_stop()
                    time.sleep(0.2)
                    self.piper_arm.disable()
            except:
                pass

        if self.camera is not None:
            self.camera.disconnect()

        print("✅ 训练结束，资源已清理")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Piper 机械臂实时训练 (兼容 mentor piper_env)')
    parser.add_argument('--snapshot', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练 episode 数量')
    parser.add_argument('--steps', type=int, default=1500, help='每个 episode 最大步数')

    args = parser.parse_args()

    try:
        with hydra.initialize(config_path='cfgs', version_base=None):
            cfg = hydra.compose(config_name='config')
    except:
        print("⚠️  无法加载Hydra配置，使用随机策略")
        cfg = None

    trainer = PiperRobotTrainer()
    trainer.init_hardware()

    if cfg is not None:
        try:
            import agents.mentor_mw as mentor_mw
            obs_spec = trainer._obs_spec
            act_spec = trainer._act_spec
            cfg.agent.obs_shape = obs_spec.shape
            cfg.agent.action_shape = act_spec.shape
            trainer.agent = hydra.utils.instantiate(cfg.agent)
            trainer.agent = trainer.agent.to(trainer.device)
            print("✅ Agent初始化成功")
        except Exception as e:
            print(f"⚠️  Agent初始化失败: {e}")
            trainer.agent = None

    if args.snapshot:
        trainer.load_snapshot(args.snapshot)

    trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.steps)


if __name__ == '__main__':
    main()
