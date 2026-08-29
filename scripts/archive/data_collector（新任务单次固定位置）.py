import mujoco
import mujoco.viewer
import numpy as np
import time

def get_body_pose(data, body_name):
    """获取指定 body 的位置和旋转矩阵"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[body_id]
    mat = data.xmat[body_id].reshape(3, 3)
    return pos, mat

def get_site_pose(data, site_name):
    """获取指定 site 的位置和旋转矩阵"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    return pos, mat

def simple_ik_step(model, data, target_pos, step_size=0.01, damping=0.05):
    """改进版的雅可比伪逆运动学求解器"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    
    current_pos = data.site_xpos[site_id]
    err_pos = target_pos - current_pos
    
    max_err = 0.05 
    err_norm = np.linalg.norm(err_pos)
    if err_norm > max_err:
        err_pos = (err_pos / err_norm) * max_err
    
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    
    J = jacp
    J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
    
    dq = J_pinv @ err_pos * step_size
    
    for i in range(6):
        data.ctrl[i] += np.clip(dq[i], -0.05, 0.05)

def control_gripper(data, close=False):
    """控制夹爪闭合或张开"""
    if close:
        data.ctrl[6] = 0.02
        data.ctrl[7] = -0.02
    else:
        data.ctrl[6] = -0.02
        data.ctrl[7] = 0.02

# 1. 加载模型
model_path = "../scenes/xunjian_arm_scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 启动前对齐初始状态
mujoco.mj_step(model, data) 
for i in range(model.nu):
    data.ctrl[i] = data.qpos[i]
control_gripper(data, close=False)

# 记录机械臂前6个关节的初始完美姿态
initial_arm_qpos = np.copy(data.qpos[:6]) 

# 状态机变量
state = 0
wait_time = 0.0
saved_target_pos = np.zeros(3)  

# 启动可视化与控制循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 实时获取关键物体位置
        screw_pos, screw_mat = get_body_pose(data, "real_screwdriver")
        box_pos, _ = get_body_pose(data, "plasticbox")
        ee_pos, ee_mat = get_site_pose(data, "ee_site")

        # --- 状态机控制逻辑 ---
        
        if state == 0:
            target_pos = screw_pos + np.array([0, 0, 0.15])
            simple_ik_step(model, data, target_pos)
            
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                wait_time += 0.001
                if wait_time > 1.0:
                    state = 1
                    wait_time = 0.0
                    
        elif state == 1:
            target_pos = screw_pos + np.array([0, 0, 0.0])
            simple_ik_step(model, data, target_pos, step_size=0.005)
            
            dot_product = np.dot(ee_mat[:, 1], screw_mat[:, 1])
            if abs(dot_product) > 0.05:
                data.ctrl[5] -= 0.0005 * np.sign(dot_product)
            
            if abs(ee_pos[2] - screw_pos[2]) < 0.002:
                state = 2
                
        elif state == 2:
            control_gripper(data, close=True)
            wait_time += 0.001
            if wait_time > 1.0:
                state = 3
                wait_time = 0.0
                saved_target_pos = screw_pos + np.array([0, 0, 0.12])
                
        elif state == 3:
            simple_ik_step(model, data, saved_target_pos)
            if np.linalg.norm(ee_pos - saved_target_pos) < 0.02:
                state = 4
                
        elif state == 4:
            step_size_zeroing = 0.001
            if abs(data.ctrl[5]) > step_size_zeroing:
                data.ctrl[5] -= np.sign(data.ctrl[5]) * step_size_zeroing
            else:
                data.ctrl[5] = 0.0
            
            if abs(data.qpos[5]) < 0.05:
                state = 5
                saved_target_pos = box_pos + np.array([0, 0, 0.15])
                
        elif state == 5:
            simple_ik_step(model, data, saved_target_pos)
            if np.linalg.norm(ee_pos - saved_target_pos) < 0.03:
                state = 6
                
        elif state == 6:
            # 停止 IK，直接动第二三关节下探
            data.ctrl[1] -= 0.0005 
            data.ctrl[2] += 0.0005 
            
            wait_time += 0.001
            if wait_time > 1.5:
                state = 7
                wait_time = 0.0
                
        elif state == 7:
            # 释放并等待螺丝刀掉落
            control_gripper(data, close=False)
            wait_time += 0.001
            if wait_time > 1.0:  
                state = 8
                wait_time = 0.0
                
        elif state == 8:
            # 🌟 修复：完全对称反转 State 6 的关节动作，原路安全退回，避免 IK 造成的拉扯
            data.ctrl[1] += 0.0005 
            data.ctrl[2] -= 0.0005 
            
            wait_time += 0.001
            if wait_time > 1.5:  # 时间和下降的 1.5 秒严格一致
                state = 9
                wait_time = 0.0
                
        elif state == 9:
            # 平滑恢复到初始姿态
            step_size_home = 0.002  
            is_home = True
            
            # 遍历前6个关节，让它们匀速回到 initial_arm_qpos
            for i in range(6):
                diff = initial_arm_qpos[i] - data.ctrl[i]
                if abs(diff) > step_size_home:
                    data.ctrl[i] += np.sign(diff) * step_size_home
                    is_home = False
                else:
                    data.ctrl[i] = initial_arm_qpos[i]
            
            # 所有关节都复位后，进入待机
            if is_home:
                state = 10 
                
        elif state == 10:
            # 任务彻底完成，保持初始待机姿态
            pass 

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)