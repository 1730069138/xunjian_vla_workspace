import mujoco
import mujoco.viewer
import cv2
import time
import numpy as np

def main():
    # 1. 确保路径正确（相对于脚本在 scripts/ 目录运行）
    model_path = "../scenes/xunjian_arm_scene.xml"
    
    # 加载模型和数据
    print("正在加载 MuJoCo 场景...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 2. 初始化双相机的离线渲染器
    # 统一使用 640x480 分辨率以兼顾清晰度与帧率
    cam_width, cam_height = 640, 480
    
    # 为全局相机和腕部相机分别创建一个渲染器
    renderer_global = mujoco.Renderer(model, height=cam_height, width=cam_width)
    renderer_wrist = mujoco.Renderer(model, height=cam_height, width=cam_width)

    print("启动成功！按下 'q' 键关闭双相机监控窗口。")

    # 3. 启动 MuJoCo 原生的被动可视化窗口 (主场景)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
            step_start = time.time()

            # 让物理引擎前进一步
            mujoco.mj_step(model, data)
            
            # 同步更新 MuJoCo 主窗口
            viewer.sync()

            # 4. 提取双相机画面并用 OpenCV 渲染拼接
            try:
                # --- 渲染全局相机 (Global Cam) ---
                renderer_global.update_scene(data, camera="global_cam")
                pixels_global = renderer_global.render()
                bgr_global = cv2.cvtColor(pixels_global, cv2.COLOR_RGB2BGR)

                # --- 渲染腕部相机 (Wrist Cam) ---
                renderer_wrist.update_scene(data, camera="d415_rgb")
                pixels_wrist = renderer_wrist.render()
                bgr_wrist = cv2.cvtColor(pixels_wrist, cv2.COLOR_RGB2BGR)

                # --- 水平拼接画面 ---
                # 将全局视角放在左边，腕部视角放在右边
                combined_img = np.hstack((bgr_global, bgr_wrist))

                # 实时显示合并后的 VLA 监控台画面
                cv2.imshow("VLA Dual Camera Dashboard (Global | Wrist)", combined_img)
                
            except Exception as e:
                print(f"相机渲染出错，请检查 XML 中相机的定义名称: {e}")
                break

            # 设置 OpenCV 窗口刷新及退出按键 (1ms 延迟保证刷新)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("收到退出信号，正在关闭...")
                break

            # 5. 控制仿真帧率与真实时间同步
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # 销毁所有 OpenCV 窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()