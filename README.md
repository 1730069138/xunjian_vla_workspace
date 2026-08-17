# xunjian_vla_workspace — 螺丝刀抓取放置任务代码说明

基于 MuJoCo 仿真 + π0 (OpenPI) 的 VLA 任务：机械臂把桌面上的一字螺丝刀抓起来，放进旁边的收纳盒。

下面的 XML、采集、转换、部署四个文件是**同一任务的一整套流程代码**，成套使用、成套修改；其余为围绕这条链路的辅助工具。

---

## 一、主流程（一套）

| 文件 | 说明 |
|---|---|
| `scenes/xunjian_arm_scene.xml` | 仿真场景定义。桌面 + 绿色目标区 / 红色障碍区 / 蓝色收纳区、`plasticbox` 收纳盒、`real_screwdriver` 螺丝刀、`sponge` 干扰物、全局相机 `global_cam` 与腕部相机 `d415_rgb`；通过 `include` 引入 `models/xunjian_arm.xml`（6 自由度臂 + 平行夹爪）。用的是相对路径，需要 `models/`、`assets/`、`objects/` 同级目录。 |
| `scripts/auto_grasp_screwdriver.py` | 脚本化专家数据采集器。随机摆放螺丝刀后自动完成"抓取→搬运→投放"，先在影子仿真里试跑一遍、成功了才回放并以 50Hz 完整录制双相机图像 + `qpos/actions` + 语言指令，输出到 `datasets/screwdriver_cleanup/ep_*/`；含 5% 概率的故意失误与纠错数据。<br>`python scripts/auto_grasp_screwdriver.py --target 150 [--headless] [--seed N]` |
| `scripts/convert_xunjiandummyx_to_lerobot.py` | 把采集到的 `ep_*` 目录打包成 OpenPI 可训练的 LeRobot 数据集（双相机 256×256、state/action 各 8 维、fps=50），写到 `$HF_LEROBOT_HOME/local/dummyx_screwdriver`；兼容新旧版 LeRobot 的导入路径与 `consolidate`。<br>`python convert_xunjiandummyx_to_lerobot.py --data_dir ... --repo_id ...` |
| `scripts/deploy_screwdriver_client.py` | 推理部署 + 消融实验主程序。通过 WebSocket 连 OpenPI policy server 取动作块驱动仿真，并在 VLA 输出之上叠加 APF 避障与末端力控（导纳/阻抗虚拟墙）两层安全约束；按 8 个工况（有无障碍 × 有无 APF × 有无力控）批量跑回合，统计成功率、碰撞率、峰值力与接触冲量，支持断点续跑与录像。<br>`python scripts/deploy_screwdriver_client.py --case 1 --num_episodes 100 --control_mode both` |

**跑通顺序**：采集 → 转换 →（OpenPI 侧训练 π0 LoRA）→ 起 policy server → 部署评测。

> 采集端与部署端必须一致的量（夹爪编解码、`Q_INIT`、控制频率、相机名与镜像、`vopt` 渲染选项、语言指令集合）统一放在 `common/robot_spec.py`，两端都从它导入，不要在各自文件里重复定义。

---

## 二、辅助脚本

| 文件 | 说明 |
|---|---|
| `scripts/test_policy_replay.py` | 开环回放自检。把训练集里的真实帧直接喂给正在运行的 policy server，将返回的动作块与 npz 真值逐维对比，能快速暴露动作维度被截断、臂关节 MAE 偏大、夹爪极性反了或塌成常数等问题；不启动物理仿真。 |
| `scripts/visualize_camera.py` | 场景与相机的可视化小工具。加载场景后开一个 MuJoCo 被动窗口，同时把全局相机和腕部相机各渲染成 640×480 横向拼接、用 OpenCV 显示，按 `q` 退出。 |
| `scripts/read_data.py` | 数据检查工具。把某条轨迹的 `joint_data.npz` 导出成带样式的 Excel（Summary 摘要页 + 每个数组一页），方便肉眼核对 qpos / actions 的数值范围。npz 路径写死在 `__main__` 里，用时改一下。 |

---

## 三、已知注意事项

- **MuJoCo 版本**：场景用了 `position` 执行器的 `dampratio` 属性、代码里用了 `mjv_connector`，都要求 MuJoCo ≥ 3.1.3；采集数据集时用的是 **3.10.0**，运行环境请对齐该版本。不要为了兼容旧版去掉 `dampratio` —— 那会改变伺服阻尼，也就改变了策略训练时的跟踪特性。
- **动态障碍物默认是关的**：场景 XML 里 `dynamic_pillar` / `pillar_joint` 目前处于注释状态，而部署端找不到该关节时会静默跳过 —— 要跑有障碍的 Case 3/4/7/8 记得先取消注释，否则会悄悄降级成无障碍工况且不报错。
- **LeRobot 环境**：转换脚本依赖的 lerobot 版本对 `LEROBOT_HOME` 环境变量会直接抛错，必须 unset，只保留 `HF_LEROBOT_HOME`。
- **checkpoint 与 norm_stats 要配套**：换 checkpoint 时注意 `asset_id` 指向的数据集要和该权重训练时用的一致，否则归一化会出错。
