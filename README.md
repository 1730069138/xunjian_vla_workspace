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

## Quick Index

- 采集: `scripts/collect/auto_grasp_screwdriver.py`
- 转换: `scripts/convert/convert_xunjiandummyx_to_lerobot.py`
- 部署: `scripts/deploy/deploy_screwdriver_client.py`
- 调试回放: `scripts/debug/test_policy_replay.py`
- 相机预览: `scripts/debug/visualize_camera.py`
- 数据检查: `scripts/debug/read_data.py`
- 资产检查: `tools/check_scene_paths.py`
- 历史归档: `scripts/archive/`

### Fast Commands

先进入项目环境：

```bash
conda activate dummyx
```

采集和部署需要离屏渲染时，可先执行 `export MUJOCO_GL=egl`。部署评测与开环回放还需要先启动 OpenPI policy server（默认 `localhost:8000`）。

```bash
# 1. 检查场景资源路径
python3 tools/check_scene_paths.py scenes/xunjian_arm_scene.xml

# 2. 采集
python3 scripts/collect/auto_grasp_screwdriver.py --target 150 --headless

# 3. 转换
python3 scripts/convert/convert_xunjiandummyx_to_lerobot.py \
    --data_dir datasets/screwdriver_cleanup --repo_id local/dummyx_screwdriver

# 4. 部署评测
python3 scripts/deploy/deploy_screwdriver_client.py --case 1 --num_episodes 100

# 5. 开环回放自检
python3 scripts/debug/test_policy_replay.py
```

---

## Installation

```bash
# 1. 创建环境
conda create -n dummyx python=3.11 -y
conda activate dummyx

# 2. 基础依赖
pip install "numpy==1.26.4" "mujoco==3.10.0" opencv-python tqdm tyro

# 3. 视频编解码
conda install -c conda-forge av "ffmpeg<8" -y

# 4. LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 0cf8648 && pip install --no-deps -e . && cd ..

# 5. OpenPI 客户端
pip install -e path/to/openpi/packages/openpi-client

# 6. 数据集缓存路径
unset LEROBOT_HOME
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot
```

最小可运行条件就是这几项。π0 的训练与推理服务在 OpenPI 自己的环境中运行，本环境只负责采集、转换与部署三类脚本。

---

## Project Structure

```
xunjian_vla_workspace/
├── scenes/          # 仿真场景 XML
├── models/          # 机械臂本体 XML
├── assets/          # 静态资源
│   ├── textures/
│   ├── meshes/
│   └── objects/
├── common/
│   └── robot_spec.py   # 采集端与部署端的共享约定
├── scripts/
│   ├── collect/
│   ├── convert/
│   ├── deploy/
│   └── debug/
├── tools/
├── datasets/        # 采集产出
└── recordings/      # 实验录像与统计
```

---

## Main Pipeline

XML 场景、采集、转换、部署四个文件是同一任务的一整套流程代码，成套使用、成套修改。

| 文件 | 说明 |
|---|---|
| `scenes/xunjian_arm_scene.xml` | 仿真场景。包含 6 自由度机械臂（含平行夹爪）、桌子与三个功能色区、收纳盒、螺丝刀、干扰用海绵，以及全局相机和腕部相机。 |
| `scripts/collect/auto_grasp_screwdriver.py` | 螺丝刀抓取放置任务的 VLA 采集脚本，自动跑轨迹并录制图像、关节数据与语言指令。 |
| `scripts/convert/convert_xunjiandummyx_to_lerobot.py` | 数据格式转换脚本，把采集到的原始数据打包成 OpenPI 可直接训练的 LeRobot 数据集。 |
| `scripts/deploy/deploy_screwdriver_client.py` | 推理部署与消融实验脚本，连接 policy server 驱动仿真，并叠加 APF 避障与力控安全层。 |

### Usage

```bash
# 1. 采集
python3 scripts/collect/auto_grasp_screwdriver.py --target 150 --headless

# 2. 转换
python3 scripts/convert/convert_xunjiandummyx_to_lerobot.py \
    --data_dir datasets/screwdriver_cleanup --repo_id local/dummyx_screwdriver

# 3. 起 policy server（OpenPI 环境）
python3 scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_dummyx_lora --policy.dir checkpoints/.../30000

# 4. 部署评测
python3 scripts/deploy/deploy_screwdriver_client.py --case 1 --num_episodes 100
```

---

## Utilities

| 文件 | 说明 |
|---|---|
| `scripts/debug/test_policy_replay.py` | 开环回放自检脚本，用训练集真实帧对比模型输出与真值。 |
| `scripts/debug/visualize_camera.py` | 相机可视化脚本，实时并排显示全局与腕部两路画面。 |
| `scripts/debug/read_data.py` | 数据检查脚本，把单条轨迹的 npz 导出成 Excel 便于核对。 |
| `tools/check_scene_paths.py` | 解析 MuJoCo 资产路径并检查缺失文件的工具。 |

## Archive

`scripts/archive/` 里放的是历史备份脚本，不参与当前主流程；如果你只想跑现行代码，优先看 `scripts/collect/`、`scripts/convert/`、`scripts/deploy/`、`scripts/debug/`。

### Archive Map

| 分类 | 代表文件 | 说明 |
|---|---|---|
| 历史采集 | `auto_grasp_screwdriver.py`、`auto_grasp_screwdriver2.py`、`auto_grasp_screwdriver3.py`、`data_collector（新任务单次固定位置）.py` | 旧采集流程与实验性版本 |
| 历史转换 | `convert_screwdriver_to_lerobot_oldenv.py`、`convert_screwdriver_to_lerobot_oldenv copy.py` | 旧数据转换逻辑 |
| 历史部署 | `deploy_screwdriver_client2.py` | 旧部署流程 |
| 历史训练 | `train.py`、`compute_norm_stats.py`、`config.py` | 旧训练与统计配置 |
| 其他留档 | `复盘笔记.txt`、`auto_grasp_screwdriver（claude版）.zip` | 复盘与压缩归档 |

### Dev Note

常用的 Git 提交流程是：

```bash
git status
git add .
git commit -m "更新项目文件"
git push
```

## One-Page Map

| 区域 | 该看什么 | 典型用途 |
|---|---|---|
| `common/` | `robot_spec.py` | 采集端和部署端共享约定，只改这里就能统一两边行为 |
| `scenes/` | XML 场景 | 改机械臂、桌面、相机、障碍物、资产引用 |
| `models/` | 机器人模型 | 改本体结构、关节、执行器 |
| `assets/` | meshes / textures / objects | 场景资源与贴图 |
| `scripts/collect/` | 采集脚本 | 生成数据集原始轨迹 |
| `scripts/convert/` | 转换脚本 | 把原始轨迹打成 LeRobot 格式 |
| `scripts/deploy/` | 部署脚本 | policy server 推理、消融、评测 |
| `scripts/debug/` | 调试脚本 | 回放、自检、可视化、数据核对 |
| `tools/` | 仓库工具 | 场景资产路径检查 |
| `scripts/archive/` | 历史脚本 | 只留档，不作为当前入口 |

---

## Notes

- 采集端与部署端必须一致的量统一定义在 `common/robot_spec.py`，改约定只改这一个文件。
- 场景需要 MuJoCo ≥ 3.1.3（实际使用 3.10.0），不要为兼容旧版删掉执行器的 `dampratio`。
- 场景中的动态障碍物默认处于注释状态，跑有障碍工况前需先取消注释。
- 更换 checkpoint 时注意 `asset_id` 与训练所用数据集保持一致，否则归一化会出错。
