# xunjian_vla_workspace

<p align="left">
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/MuJoCo-3.10.0-orange.svg" alt="mujoco">
  <img src="https://img.shields.io/badge/policy-%CF%800%20(OpenPI)-green.svg" alt="openpi">
  <img src="https://img.shields.io/badge/dataset-LeRobot-yellow.svg" alt="lerobot">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="license">
</p>

基于 MuJoCo 仿真与 π0 (OpenPI) 的机械臂 VLA 任务代码：**把桌面上的一字螺丝刀抓起来，放进旁边的收纳盒**，并在策略输出之上叠加 APF 避障与末端力控两层安全约束。

```
场景定义 → 数据采集 → 格式转换 → (OpenPI 训练 π0) → 部署推理与消融实验
```

---

## Installation

```bash
# 1. 创建环境
conda create -n dummyx python=3.11 -y
conda activate dummyx

# 2. 基础依赖
pip install "numpy==1.26.4" "mujoco==3.10.0" opencv-python tqdm tyro

# 3. 视频编解码（PyPI 的 av wheel 已撤，走 conda-forge）
conda install -c conda-forge av "ffmpeg<8" -y

# 4. LeRobot（--no-deps 防止顶掉上面的版本 pin）
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 0cf8648 && pip install --no-deps -e . && cd ..

# 5. OpenPI 客户端
pip install -e path/to/openpi/packages/openpi-client

# 6. 数据集缓存路径：只保留 HF_LEROBOT_HOME
unset LEROBOT_HOME
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot
```

> π0 的训练与推理服务在 openpi 自己的环境中运行，本环境只负责采集、转换与部署三类脚本。

---

## Project Structure

```
xunjian_vla_workspace/
├── scenes/          # 仿真场景 XML
├── models/          # 机械臂本体 XML
├── assets/objects/  # 贴图与网格资源
├── common/
│   └── robot_spec.py   # 采集端与部署端的共享约定
├── scripts/
├── datasets/        # 采集产出
└── recordings/      # 实验录像与统计
```

---

## Main Pipeline

XML 场景、采集、转换、部署四个文件是同一任务的一整套流程代码，成套使用、成套修改。

| 文件 | 说明 |
|---|---|
| `scenes/xunjian_arm_scene.xml` | 仿真场景。包含 6 自由度机械臂（含平行夹爪）、桌子与三个功能色区、收纳盒、螺丝刀、干扰用海绵，以及全局相机和腕部相机。 |
| `scripts/auto_grasp_screwdriver.py` | 螺丝刀抓取放置任务的 VLA 仿真数据采集脚本，自动跑轨迹并录制图像、关节数据与语言指令。 |
| `scripts/convert_xunjiandummyx_to_lerobot.py` | 数据格式转换脚本，把采集到的原始数据打包成 OpenPI 可直接训练的 LeRobot 数据集。 |
| `scripts/deploy_screwdriver_client.py` | 推理部署与消融实验脚本，连接 policy server 驱动仿真，并叠加 APF 避障与力控安全层，批量跑工况出指标。 |

### Usage

```bash
# 1. 采集
python scripts/auto_grasp_screwdriver.py --target 150 --headless

# 2. 转换
python scripts/convert_xunjiandummyx_to_lerobot.py \
    --data_dir datasets/screwdriver_cleanup --repo_id local/dummyx_screwdriver

# 3. 起 policy server（openpi 环境）
python scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_dummyx_lora --policy.dir checkpoints/.../30000

# 4. 部署评测
python scripts/deploy_screwdriver_client.py --case 1 --num_episodes 100
```

---

## Utilities

| 文件 | 说明 |
|---|---|
| `scripts/test_policy_replay.py` | 开环回放自检脚本，用训练集真实帧对比模型输出与真值，排查推理链路问题。 |
| `scripts/visualize_camera.py` | 相机可视化脚本，实时并排显示全局与腕部两路画面。 |
| `scripts/read_data.py` | 数据检查脚本，把单条轨迹的 npz 导出成 Excel 便于核对。 |

---

## Notes

- 采集端与部署端必须一致的量统一定义在 `common/robot_spec.py`，改约定只改这一个文件。
- 场景需要 MuJoCo ≥ 3.1.3（实际使用 3.10.0），不要为兼容旧版删掉执行器的 `dampratio`。
- 场景中的动态障碍物默认处于注释状态，跑有障碍工况前需先取消注释。
- 更换 checkpoint 时注意 `asset_id` 与训练所用数据集保持一致，否则归一化会出错。
