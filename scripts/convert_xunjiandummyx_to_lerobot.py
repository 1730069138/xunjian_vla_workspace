"""
将本地的 dummyx 仿真数据集 转换为 OpenPI 官方支持的 LeRobot 格式。
(具备双重兼容逻辑：自动适配新旧版 LeRobot 路径与元数据生成)
"""

import os
import glob
import cv2
import numpy as np
import shutil
from pathlib import Path
import tqdm
import tyro

# 👇 兼容导入：无论新版旧版 LeRobot 都能成功导包
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

HF_LEROBOT_HOME = Path(
    os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
).expanduser()
def main(
    data_dir: str = "datasets/screwdriver_cleanup", 
    repo_id: str = "local/dummyx_screwdriver", 
    push_to_hub: bool = False
):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
        
    ep_dirs = glob.glob(os.path.join(data_dir, "ep_*"))
    if not ep_dirs:
        print(f"❌ 错误：在 {data_dir} 中未找到任何数据文件夹！")
        return
        
    def get_ep_num(dir_path):
        try:
            return int(os.path.basename(dir_path).split('_')[1])
        except ValueError:
            return -1
            
    ep_dirs = sorted(ep_dirs, key=get_ep_num)
        
    print(f"🔍 找到 {len(ep_dirs)} 条轨迹数据，准备转换为 LeRobot 格式...")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"🧹 清理旧的数据集缓存: {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dummyx",
        fps=50, 
        features={
            "observation.images.cam_fixed": {
                "dtype": "image",
                "shape": (256, 256, 3), 
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["motors"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["motors"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for ep_dir in tqdm.tqdm(ep_dirs, desc="🔄 正在打包数据"):
        with open(os.path.join(ep_dir, "instruction.txt"), "r", encoding="utf-8") as f:
            task_instruction = f.read().strip()
            
        joint_data = np.load(os.path.join(ep_dir, "joint_data.npz"))
        qpos = joint_data["qpos"]       
        actions = joint_data["actions"] 
        
        num_frames = qpos.shape[0]
        
        for i in range(num_frames):
            img_f_path = os.path.join(ep_dir, "cam_fixed", f"{i:03d}.jpg")
            img_w_path = os.path.join(ep_dir, "cam_wrist", f"{i:03d}.jpg")
            
            img_f = cv2.cvtColor(cv2.imread(img_f_path), cv2.COLOR_BGR2RGB)
            img_w = cv2.cvtColor(cv2.imread(img_w_path), cv2.COLOR_BGR2RGB)
            
            dataset.add_frame(
                {
                    "observation.images.cam_fixed": img_f,
                    "observation.images.cam_wrist": img_w,
                    "observation.state": qpos[i],
                    "action": actions[i],
                    "task": task_instruction,
                }
            )
            
        dataset.save_episode()

    print("\n📦 正在整合数据集...")
    # 👇 容错合并：若版本支持 consolidate（能生成 OpenPI 需要的 tasks.jsonl）则执行，不支持则安全跳过
    try:
        dataset.consolidate()
    except AttributeError:
        pass
    
    print(f"\n🎉 转换完成！数据集已保存在: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)