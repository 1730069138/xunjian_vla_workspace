"""
将本地的 dummyx 仿真数据集 (多文件夹/JPG/NPZ 格式)
转换为 OpenPI 官方支持的 LeRobot (Parquet) 格式。

【本版说明 — 修订版】
- 导入方式、API 用法、核心数据处理逻辑与旧版 convert_dummyx_to_lerobot_v2.py 完全一致，
  仅把默认数据源换成新任务的 screwdriver_cleanup 数据集。
  => 适用于旧的虚拟环境 (lerobot.common.datasets 路径)。
- 夹爪两指的反号约定 (joint8 = -joint9, 例如 开=(-0.02, 0.02) / 合=(0.02, -0.02))
  原样透传，不做任何符号翻转、缩放或归一化，8 维向量按采集时的原始数值入库。

【相对上一版新增】
1. 严格校验 qpos / actions / 图片三者帧数一致，不一致直接报错而不是静默截断
2. glob 只取目录，命名异常的 ep_* 跳过并告警
3. fps 改为 CLI 参数并强制显式确认（不再默默假设 50Hz）
4. 空 episode 跳过，避免写出损坏分片
5. 可选清理 openpi 的 norm_stats assets 目录 (--purge_norm_stats)，
   并在结束时无条件打印重跑 compute_norm_stats 的提醒
6. 结束时打印 state / action 各维统计量 + 夹爪维取值分布与开合事件占比
7. instruction.txt / npz 缺失时给出可读的错误信息
8. push_videos 改为 False（features 声明的是 image 而非 video）
"""

import os
import glob
import cv2
import numpy as np
import shutil
from pathlib import Path
import tqdm
import tyro

# 旧环境的导入路径，HF_LEROBOT_HOME 同样由 lerobot 库提供（不自行拼 ~/.cache 路径，
# 避免与库内 HF_HOME 配置不一致导致写到两个不同目录）
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def _count_images(ep_dir: str, cam: str) -> int:
    return len(glob.glob(os.path.join(ep_dir, cam, "*.jpg")))


def _load_episode(ep_dir: str):
    """读取单条 episode，并做完整性校验。返回 (instruction, qpos, actions, n)。"""
    ep_name = os.path.basename(ep_dir)

    instr_path = os.path.join(ep_dir, "instruction.txt")
    if not os.path.isfile(instr_path):
        raise FileNotFoundError(f"[{ep_name}] 缺少 instruction.txt: {instr_path}")
    with open(instr_path, "r", encoding="utf-8") as f:
        task_instruction = f.read().strip()
    if not task_instruction:
        raise ValueError(f"[{ep_name}] instruction.txt 内容为空")

    npz_path = os.path.join(ep_dir, "joint_data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"[{ep_name}] 缺少 joint_data.npz: {npz_path}")
    joint_data = np.load(npz_path)

    for key in ("qpos", "actions"):
        if key not in joint_data:
            raise KeyError(f"[{ep_name}] joint_data.npz 中缺少字段 '{key}'，实际字段: {list(joint_data.keys())}")

    qpos = joint_data["qpos"]        # (N, 8)  夹爪两维为反号原值，不做处理
    actions = joint_data["actions"]  # (N, 8)  同上

    # ---- 校验 1: 维度 ----
    if qpos.ndim != 2 or qpos.shape[1] != 8:
        raise ValueError(f"[{ep_name}] qpos 形状应为 (N, 8)，实际 {qpos.shape}")
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"[{ep_name}] actions 形状应为 (N, 8)，实际 {actions.shape}")

    # ---- 校验 2: qpos 与 actions 帧数必须一致（旧版这里会静默截断尾部）----
    if qpos.shape[0] != actions.shape[0]:
        raise ValueError(
            f"[{ep_name}] qpos 有 {qpos.shape[0]} 帧，actions 有 {actions.shape[0]} 帧，两者不一致。\n"
            f"  旧版脚本会以 qpos 为准静默截断，导致尾部(RELEASE/归位阶段)丢失监督信号。"
        )

    n = qpos.shape[0]

    # ---- 校验 3: 两路相机的图片数量必须与 npz 帧数一致（旧版只能抓到"图片少了"）----
    n_fixed = _count_images(ep_dir, "cam_fixed")
    n_wrist = _count_images(ep_dir, "cam_wrist")
    if n_fixed != n or n_wrist != n:
        raise ValueError(
            f"[{ep_name}] 帧数不一致: npz={n}, cam_fixed={n_fixed}, cam_wrist={n_wrist}。\n"
            f"  三者必须相等，否则会有帧被静默丢弃或错位。"
        )

    # ---- 校验 4: 数值健全性 ----
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"[{ep_name}] qpos 或 actions 中存在 NaN / Inf")

    return task_instruction, qpos, actions, n


def _report_stats(all_qpos: np.ndarray, all_actions: np.ndarray, fps: int):
    """打印各维统计量，重点检查夹爪维的方差与开合事件占比。"""
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    print("\n" + "=" * 78)
    print("📊 数据集统计（用于核对归一化与夹爪信号强度）")
    print("=" * 78)

    for name, arr in (("observation.state (qpos)", all_qpos), ("action (ctrl)", all_actions)):
        print(f"\n--- {name}  shape={arr.shape} ---")
        print(f"  mean : {arr.mean(axis=0)}")
        print(f"  std  : {arr.std(axis=0)}")
        print(f"  min  : {arr.min(axis=0)}")
        print(f"  max  : {arr.max(axis=0)}")
        near_zero = np.where(arr.std(axis=0) < 1e-6)[0]
        if len(near_zero) > 0:
            print(f"  ⚠️  以下维度方差近似为 0，归一化时会被除以极小值: {near_zero.tolist()}")

    # 夹爪两维（6、7）的专项检查
    print("\n--- 夹爪维专项 (action[:, 6] / action[:, 7]) ---")
    for d in (6, 7):
        col = all_actions[:, d]
        uniq = np.unique(np.round(col, 6))
        print(f"  dim {d}: 唯一值 {len(uniq)} 个" + (f" -> {uniq}" if len(uniq) <= 12 else " (>12，仅显示数量)"))

    g = all_actions[:, 6]
    changes = int(np.count_nonzero(np.abs(np.diff(g)) > 1e-6))
    total = len(g)
    print(f"  dim 6 相邻帧发生变化的次数: {changes} / {total - 1} 帧  ({100.0 * changes / max(total - 1, 1):.3f}%)")
    print(f"  折算: 约 {changes / max(fps, 1):.2f} 秒的帧带有夹爪变化信号")
    if changes / max(total - 1, 1) < 0.01:
        print("  ⚠️  夹爪开合事件在整个数据集中占比 < 1%，flow matching 极易把这个突变平均掉。")
        print("      建议：在 GRASP / RELEASE 前后补保持帧，或对夹爪维单独加权。")

    # dim6 与 dim7 是否严格反号
    if np.allclose(all_actions[:, 6], -all_actions[:, 7], atol=1e-6):
        print("  ℹ️  dim 6 与 dim 7 严格反号（互为相反数），第 7 维不携带额外信息。")
    else:
        print("  ⚠️  dim 6 与 dim 7 并非严格反号，请确认采集端的夹爪约定是否中途变过。")

    # state 与 action 在夹爪维上的偏差（抓紧物体时 ctrl 与 qpos 必然分离）
    diff = np.abs(all_actions[:, 6] - all_qpos[:, 6])
    print(f"  |action[:,6] - state[:,6]| : mean={diff.mean():.5f}  max={diff.max():.5f}")
    print("      （抓紧螺丝刀时 ctrl 与 qpos 会分离，这是正常的；但若训练配置里对夹爪维")
    print("        启用了 delta_action，这个差值就是模型实际要学的量，务必核对 delta_action_mask 长度）")
    print("=" * 78)


def main(
    # 新任务的数据集目录
    data_dir: str = "datasets/screwdriver_cleanup",
    repo_id: str = "local/dummyx_screwdriver",
    # 💡 必须与采集器的真实录制频率一致。新采集器的 STEPS_PER_RECORD 是用硬编码
    #    SIM_TIMESTEP=0.001 算的，若场景 xml 里 <option timestep> 不是 0.001，
    #    真实频率就不是 50Hz，这里必须跟着改，否则 action horizon 对应的时长会翻倍/减半。
    fps: int = 50,
    # openpi 的 norm_stats 独立于 LeRobot 数据集目录，重转数据后不会自动失效
    purge_norm_stats: bool = False,
    openpi_assets_dir: str = "",
    push_to_hub: bool = False,
):
    # 动态获取根目录 dummyx/，并将传入的 data_dir 转换为绝对路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)

    print(f"⚙️  fps = {fps} Hz —— 请确认与场景 xml 的 <option timestep> × STEPS_PER_RECORD 一致")

    # 1. 查找所有 episode 文件夹（只取目录，排除同名文件）
    candidates = glob.glob(os.path.join(data_dir, "ep_*"))
    ep_dirs = [d for d in candidates if os.path.isdir(d)]
    skipped_non_dir = [d for d in candidates if not os.path.isdir(d)]
    for d in skipped_non_dir:
        print(f"⏭️  跳过非目录项: {os.path.basename(d)}")

    if not ep_dirs:
        print(f"❌ 错误：在 {data_dir} 中未找到任何数据文件夹！")
        return

    # 💡 自定义排序函数，按文件夹后的真实数字大小排序 (解决 ep_10 排在 ep_2 前面的问题)
    def get_ep_num(dir_path):
        try:
            return int(os.path.basename(dir_path).split("_")[1])
        except (ValueError, IndexError):
            return None

    valid = []
    for d in ep_dirs:
        if get_ep_num(d) is None:
            print(f"⏭️  跳过命名异常的目录（无法解析编号）: {os.path.basename(d)}")
        else:
            valid.append(d)
    ep_dirs = sorted(valid, key=get_ep_num)

    if not ep_dirs:
        print("❌ 错误：没有命名合法的 episode 目录！")
        return

    print(f"🔍 找到 {len(ep_dirs)} 条轨迹数据，准备转换为 LeRobot 格式...")

    # 2. 清理旧的同名 LeRobot 数据集（防止报错）
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"🧹 清理旧的数据集缓存: {output_path}")
        shutil.rmtree(output_path)

    # 2b. 可选：清理 openpi 侧的 norm_stats（它不在 HF_LEROBOT_HOME 下，重转数据不会自动失效）
    if purge_norm_stats:
        if not openpi_assets_dir:
            print("⚠️  指定了 --purge_norm_stats 但未提供 --openpi_assets_dir，跳过清理。")
        else:
            norm_path = Path(openpi_assets_dir).expanduser() / repo_id
            if norm_path.exists():
                print(f"🧹 清理旧的 norm_stats: {norm_path}")
                shutil.rmtree(norm_path)
            else:
                print(f"ℹ️  未找到 norm_stats 目录（可能路径不同）: {norm_path}")

    # 3. 严格按照 OpenPI 规范初始化 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dummyx",
        fps=fps,  # 💡 与采集器严格对齐
        features={
            "observation.images.cam_fixed": {
                "dtype": "image",
                "shape": (256, 256, 3),  # (Height, Width, Channel)
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),  # 6个大臂关节 + 2个夹爪关节(joint8/joint9，反号)
                "names": ["motors"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),  # 采集时下发的 ctrl 目标(同样是 6 关节 + 2 反号夹爪)
                "names": ["motors"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 4. 遍历并压入数据
    total_frames = 0
    written_eps = 0
    skipped_empty = []
    qpos_pool, action_pool = [], []
    instructions = set()

    for ep_dir in tqdm.tqdm(ep_dirs, desc="🔄 正在打包数据"):
        ep_name = os.path.basename(ep_dir)

        task_instruction, qpos, actions, num_frames = _load_episode(ep_dir)

        # 空 episode 直接跳过，避免 save_episode() 写出损坏分片
        if num_frames == 0:
            skipped_empty.append(ep_name)
            continue

        instructions.add(task_instruction)

        # 逐帧压入
        for i in range(num_frames):
            # 读取图片并从 BGR (OpenCV 默认) 转为 RGB
            img_f_path = os.path.join(ep_dir, "cam_fixed", f"{i:03d}.jpg")
            img_w_path = os.path.join(ep_dir, "cam_wrist", f"{i:03d}.jpg")

            raw_f = cv2.imread(img_f_path)
            raw_w = cv2.imread(img_w_path)
            if raw_f is None or raw_w is None:
                raise FileNotFoundError(
                    f"[{ep_name}] 图片缺失或损坏：{img_f_path if raw_f is None else img_w_path}\n"
                    f"（该 episode 的 npz 有 {num_frames} 帧）"
                )

            img_f = cv2.cvtColor(raw_f, cv2.COLOR_BGR2RGB)
            img_w = cv2.cvtColor(raw_w, cv2.COLOR_BGR2RGB)

            # 💡 旧版 API 写法：task 塞进 add_frame 的字典里，不作为 kwarg 传
            dataset.add_frame(
                {
                    "observation.images.cam_fixed": img_f,
                    "observation.images.cam_wrist": img_w,
                    "observation.state": qpos[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "task": task_instruction,
                }
            )

        qpos_pool.append(qpos.astype(np.float32))
        action_pool.append(actions.astype(np.float32))
        total_frames += num_frames
        written_eps += 1

        # 保存这一个 Episode（不需要参数）
        dataset.save_episode()

    print(f"\n📦 共写入 {written_eps} 条轨迹 / {total_frames} 帧。")
    if skipped_empty:
        print(f"⏭️  跳过 {len(skipped_empty)} 条空 episode: {skipped_empty}")
    print(f"🗣️  数据集中出现的不同指令共 {len(instructions)} 条:")
    for s in sorted(instructions):
        print(f"     - {s}")

    # 旧版环境中该调用已不需要手动触发，保持注释状态
    # dataset.consolidate()

    # 5. 统计报告
    if qpos_pool:
        _report_stats(np.concatenate(qpos_pool, axis=0), np.concatenate(action_pool, axis=0), fps)

    print(f"\n🎉 转换完成！数据集已保存在: {output_path}")
    print("\n" + "!" * 78)
    print("❗ 下一步必做：重新计算归一化统计量")
    print("   openpi 的 norm_stats 保存在独立的 assets 目录下，重转数据集不会让它失效。")
    print("   若沿用旧的 norm_stats（例如把 asset_id 指回 local/dummyx_grasp），")
    print("   新的夹爪约定 (±0.02 反号) 会被旧统计量 (0~0.04 同号) 错误反归一化，")
    print("   推理时夹爪会恒定在闭合附近、完全不动。")
    print(f"   请执行： uv run scripts/compute_norm_stats.py --config-name <你的训练配置>")
    print("!" * 78)

    # 6. 如果需要推送到 Hugging Face Hub (可选)
    if push_to_hub:
        print("🚀 正在推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["dummyx", "screwdriver", "pi0", "rlds"],
            private=True,
            push_videos=False,  # features 声明的是 image 而非 video
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)