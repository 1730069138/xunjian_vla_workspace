"""
将本地的 dummyx 仿真数据集 (多文件夹/JPG/NPZ 格式)
转换为 OpenPI 官方支持的 LeRobot (Parquet) 格式。

【本版说明】
- 导入方式、API 用法、数据处理逻辑与旧版 convert_dummyx_to_lerobot_v2.py 完全一致，
  仅把默认数据源换成新任务的 screwdriver_cleanup 数据集。
  => 适用于旧的虚拟环境 (lerobot.common.datasets 路径)。
- 夹爪两指的反号约定 (joint8 = -joint9, 例如 开=(-0.02, 0.02) / 合=(0.02, -0.02))
  原样透传，不做任何符号翻转、缩放或归一化，8 维向量按采集时的原始数值入库。
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


def main(
    # 新任务的数据集目录
    data_dir: str = "datasets/screwdriver_cleanup",
    repo_id: str = "local/dummyx_screwdriver",
    push_to_hub: bool = False
):
    # 动态获取根目录 dummyx/，并将传入的 data_dir 转换为绝对路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)

    # 1. 查找所有 episode 文件夹
    ep_dirs = glob.glob(os.path.join(data_dir, "ep_*"))
    if not ep_dirs:
        print(f"❌ 错误：在 {data_dir} 中未找到任何数据文件夹！")
        return

    # 💡 自定义排序函数，按文件夹后的真实数字大小排序 (解决 ep_10 排在 ep_2 前面的问题)
    def get_ep_num(dir_path):
        try:
            return int(os.path.basename(dir_path).split('_')[1])
        except ValueError:
            return -1

    ep_dirs = sorted(ep_dirs, key=get_ep_num)

    print(f"🔍 找到 {len(ep_dirs)} 条轨迹数据，准备转换为 LeRobot 格式...")

    # 2. 清理旧的同名 LeRobot 数据集（防止报错）
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"🧹 清理旧的数据集缓存: {output_path}")
        shutil.rmtree(output_path)

    # 3. 严格按照 OpenPI 规范初始化 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dummyx",
        fps=50,  # 💡 与采集器严格对齐，保持 50Hz 物理频率
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
    for ep_dir in tqdm.tqdm(ep_dirs, desc="🔄 正在打包数据"):
        # 读取指令
        with open(os.path.join(ep_dir, "instruction.txt"), "r", encoding="utf-8") as f:
            task_instruction = f.read().strip()

        # 读取物理状态
        joint_data = np.load(os.path.join(ep_dir, "joint_data.npz"))
        qpos = joint_data["qpos"]        # (N, 8)  夹爪两维为反号原值，不做处理
        actions = joint_data["actions"]  # (N, 8)  同上

        num_frames = qpos.shape[0]

        # 逐帧压入
        for i in range(num_frames):
            # 读取图片并从 BGR (OpenCV 默认) 转为 RGB
            img_f_path = os.path.join(ep_dir, "cam_fixed", f"{i:03d}.jpg")
            img_w_path = os.path.join(ep_dir, "cam_wrist", f"{i:03d}.jpg")

            raw_f = cv2.imread(img_f_path)
            raw_w = cv2.imread(img_w_path)
            if raw_f is None or raw_w is None:
                raise FileNotFoundError(
                    f"图片缺失或损坏：{img_f_path if raw_f is None else img_w_path}\n"
                    f"（该 episode 的 npz 有 {num_frames} 帧，请检查图片与状态帧数是否一致）"
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

        total_frames += num_frames

        # 保存这一个 Episode（不需要参数）
        dataset.save_episode()

    print(f"\n📦 共写入 {len(ep_dirs)} 条轨迹 / {total_frames} 帧。")

    # 旧版环境中该调用已不需要手动触发，保持注释状态
    # dataset.consolidate()

    print(f"\n🎉 转换完成！数据集已保存在: {output_path}")

    # 6. 如果需要推送到 Hugging Face Hub (可选)
    if push_to_hub:
        print("🚀 正在推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["dummyx", "screwdriver", "pi0", "rlds"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)