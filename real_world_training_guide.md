# PiPER 真实世界训练指南

本指南说明如何在 PiPER 真实机械臂上运行 MENTOR 训练流程，实现从仿真到现实的迁移。

---

## 1. 硬件准备

| 设备 | 说明 |
|------|------|
| PiPER 6-DOF 机械臂 | 松灵/AgileX，通过 CAN 总线连接 |
| Intel RealSense D400 系列 | D415/D435/D455 均可，用于视觉观测 |
| 标定板 | 8×6 内角，25mm 方格（用于手眼标定） |
| 工控机/PC | Windows/Linux，需有 CAN 接口 + USB 3.0 |

## 2. 软件依赖安装

```bash
# 基础 MENTOR 环境
conda env create -f conda_env.yml
conda activate mentor

# PiPER SDK（需与 mentor-main 同级目录下的 piper_sdk）
pip install piper_sdk

# RealSense
pip install pyrealsense2 opencv-python

# 验证安装
python -c "from piper_sdk import C_PiperInterface_V2; print('piper_sdk OK')"
python -c "import pyrealsense2 as rs; print('pyrealsense2 OK')"
```

## 3. 手眼标定

训练前需完成手眼标定，获取相机到机械臂基座的变换矩阵。

### 3.1 准备工作

1. 将标定板固定在桌面上，确保相机能清晰看到
2. 机械臂上电，CAN 线连接
3. 确认 RealSense 相机已连接

### 3.2 运行标定脚本

```bash
python calibrate_hand_eye.py \
    --can_port can0 \
    --num_samples 10 \
    --checker_rows 8 \
    --checker_cols 6 \
    --square_size 0.025 \
    --output calibration.npy
```

### 3.3 标定流程

1. 脚本启动后会自动连接相机和机械臂
2. 手动将机械臂移动到不同位姿（标定板始终在视野内）
3. 每到位一个位姿，按 **ENTER** 记录样本
4. 检测失败时输入 `skip` 跳过，输入 `done` 提前结束
5. 收集 ≥3 个样本后自动计算标定结果
6. 标定文件保存为 `calibration.npy`

> **提示**：建议收集 10+ 个样本，覆盖不同角度和位置，以提高标定精度。

## 4. 配置文件

真实世界训练使用 `cfgs/task/piper_real.yaml` 配置文件，关键参数：

```yaml
real_mode: true

real:
  can_port: can0                    # CAN 端口
  action_scale: 2.0                 # 动作缩放：每个动作单位 = 2mm 位移
  gripper_range: 0.08               # 夹爪最大开合 (米)
  speed: 50                         # 运动速度 (0-100)
  home_pose: [57.0, 0.0, 215.0, 0.0, 85.0, 0.0]  # 初始位姿 [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]

  workspace:                        # 工作空间边界 (mm)
    x_min: -200
    x_max: 200
    y_min: -200
    y_max: 200
    z_min: 50
    z_max: 400

  use_realsense: true               # 启用 RealSense 相机
  realsense_serial: null            # null = 自动检测，或填序列号
  realsense_resolution: [640, 480]
  realsense_fps: 30
  calibration_file: null            # 标定文件路径，如 "calibration.npy"
  manual_reward: true               # 键盘手动奖励

# 训练参数（真实世界通常更保守）
num_train_frames: 100000
num_seed_frames: 500
eval_every_frames: 5000
num_eval_episodes: 3
episode_length: 250
```

### 4.1 参数调整建议

| 参数 | 说明 | 调整建议 |
|------|------|---------|
| `action_scale` | 动作缩放因子 | 起步用 1.0（更安全），熟悉后增到 2.0-3.0 |
| `speed` | 机械臂运动速度 | 调试时用 20-30，正式训练可提至 50 |
| `workspace` | 安全工作空间 | 根据实际桌面大小调整，**宁小勿大** |
| `home_pose` | 初始位姿 | 确保在安全位置，朝向任务区域 |
| `num_seed_frames` | 随机探索帧数 | 真实世界建议 200-500，避免太多随机动作 |

## 5. 训练流程

### 5.1 从零开始训练

```bash
python train_mw.py task_name=assembly real_mode=true real=piper_real
```

### 5.2 从仿真预训练模型迁移（推荐）

真实世界数据珍贵，建议先在仿真中预训练，再迁移到真实：

```bash
# 步骤 1：在仿真中训练
python train_mw.py task_name=assembly

# 步骤 2：用仿真模型 + 真实机器人继续训练
python train_mw.py task_name=assembly real_mode=true real=piper_real load_from_id=true load_id=50000
```

### 5.3 加载预训练 Buffer

如果已有之前的经验回放数据：

```bash
python train_mw.py task_name=assembly real_mode=true real=piper_real \
    pretrain_buffer_dir=/path/to/prev_experiment/buffer
```

### 5.4 指定标定文件

```bash
python train_mw.py task_name=assembly real_mode=true real=piper_real \
    real.calibration_file=calibration.npy
```

## 6. 训练中的操作

### 6.1 手动奖励

训练运行时，终端会启动键盘监听线程：

| 按键 | 功能 |
|------|------|
| **SPACE** | 给予 reward +10 并标记 success = True |
| **Q** | 紧急停止（E-Stop） |

**操作方式**：观察机械臂动作，当它完成目标（如抓取、放置等）时按下空格键给奖励。这是真实世界训练的核心反馈机制。

### 6.2 安全注意事项

- 训练前确保 `workspace` 参数正确设置了安全边界
- 始终保持手在键盘旁，随时准备按 **Q** 紧急停止
- `action_scale` 从小值开始，确认行为正常后再增大
- 首次运行建议用低速 (`speed: 20`) 和小动作范围 (`action_scale: 1.0`)
- 确保机械臂周围没有障碍物和人员

## 7. 系统架构

```
Agent (MENTOR MoE)
    │ action [dx, dy, dz, gripper] ∈ [-1, 1]
    ▼
PiperEnv
    │ 1. action_scale 缩放: [-1,1] → [-2mm, +2mm]
    │ 2. workspace 裁剪: 限制到安全范围
    │ 3. MOVEP 模式: 末端位姿增量控制
    ▼
PiPER Robot (via piper_sdk)
    │ CAN 总线
    ▼
机械臂运动

RealSense Camera
    │ 640×480 @ 30fps
    │ → center crop → resize 84×84
    ▼
PiperEnv._capture_image()
    │ frame_stack=3 → (9, 84, 84) 观测
    ▼
Agent

KeyboardRewardListener (后台线程)
    │ SPACE → reward=10, success=True
    │ Q → emergency_stop
    ▼
PiperEnv.step() 返回 (obs, reward, success)
```

## 8. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `[PiperRobot] Connect failed` | CAN 线未连接 / 端口错误 | 检查 CAN 接口，确认 `can_port` 正确 |
| `[PiperRobot] Enable timeout` | 机械臂未上电 / E-Stop 被按下 | 检查电源，释放急停按钮 |
| `[RealSense] No device detected` | USB 未连接 / 权限问题 | 检查 USB 3.0 连接，Linux 下检查 udev 规则 |
| `[Calibrator] Checkerboard not detected` | 光线不足 / 角度太大 | 改善光照，正对棋盘格 |
| 机械臂动作过大/过猛 | `action_scale` 过大 | 降低 `action_scale`（如 0.5-1.0） |
| 机械臂不动 | `speed` 为 0 / E-Stop 状态 | 检查 speed 参数，重新 enable |
| 训练中突然停止 | 安全检查失败 / 键盘 Q 被误触 | 检查终端输出，重新运行 |
