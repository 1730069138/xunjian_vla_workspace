"""
机械臂 VLA 数据集自动采集脚本 (基于 scenes/xunjian_arm_scene.xml + models/xunjian_arm.xml)。

流程: 自动循环 N 次 -> 设置自定义起手位姿 -> 停留 1 秒等待静止 -> 开启录制 
      -> 移动至全 0 关节位姿 -> 规划并物理试跑 -> 真实执行并以 50Hz 频率完整记录
      -> 以文件夹形式保存图片序列、npz 状态和自然语言指令 (含 10% 概率模拟失误与纠错)。

运行:
    conda activate xunjian_vla_workspace2
    pip install lerobot tqdm tyro mujoco opencv-python numpy
    python scripts/collect_vla_dataset.py --episodes 50 --out_dir ./datasets/screwdriver_cleanup
"""

import argparse
import time
import os
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_PATH = SCRIPT_DIR / ".." / "scenes" / "xunjian_arm_scene.xml"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ARM_ACTUATORS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
EE_SITE = "ee_site"

OBJECT_BODY = "real_screwdriver"
OBJECT_FREEJOINT = "fj_screwdriver"
STORAGE_BOX_BODY = "plasticbox"

# 随机初始摆放的机械臂可达范围 (会通过拒绝采样进一步限制在绿色区域内)
SPAWN_X_RANGE = (0.35, 0.44)
SPAWN_Y_RANGE = (-0.09, 0.09)
SPAWN_DROP_Z = 0.80

BASE_XY = np.array([0.13, 0.0])
RELEASE_TILT_DEG = 45.0
RELEASE_CLEARANCE = 0.028
HOVER_CLEARANCE = 0.16

GRIP_OPEN = (-0.02, 0.02)
GRIP_CLOSE = (0.02, -0.02)
GRIP_INIT = (0.0, 0.0)  # 用户指定的初始夹爪状态

LIFT_CLEARANCES = (0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01)
PREGRASP_HEIGHT = 0.08

# 全局动态安全限速 (弧度/秒) —— 已降低一半，确保运行极度平稳
MAX_JOINT_VELOCITY = 0.75

# 各阶段仿真的保底基础步数 (timestep=0.001s)
# 实际运行步数将根据 MAX_JOINT_VELOCITY 动态延长
STEPS_SETTLE = 1000      # 1.0s (用于在初始位姿停留一秒)
STEPS_PREGRASP = 400     # 0.4s
STEPS_DESCEND = 300      # 0.3s
STEPS_GRIP_HOLD = 300    # 0.3s
STEPS_LIFT = 300         # 0.3s
STEPS_TRANSPORT = 500    # 0.5s
STEPS_BOX_DESCEND = 300  # 0.3s
STEPS_RELEASE_HOLD = 300 # 0.3s
STEPS_RETREAT = 400      # 0.4s
STEPS_FINAL_SETTLE = 800 # 0.8s

Q_HOME = np.zeros(6)
# 用户指定的最原始 6 轴机械臂起手位姿
Q_INIT = np.array([0.0, 1.41, 1.3, 0.0, -0.7, 0.0])

IK_SEED_BANK = [
    np.array([0.0, -0.5, 0.5, 0.0, -1.0, 2.3]),
    np.zeros(6),
    np.array([0.0, -0.8, 0.0, 0.0, -1.5, 2.3]),
    np.array([0.0, -1.2, 0.8, 0.0, 0.5, 4.0]),
    np.array([0.3, -1.0, 1.0, 0.3, -1.0, 3.0]),
]

# VLA 数据集录制配置
CONTROL_HZ = 50  # 50Hz 录制频率
SIM_TIMESTEP = 0.001
STEPS_PER_RECORD = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)

LANGUAGE_INSTRUCTIONS = [
    "pick up the screwdriver and place it into the storage box.",
    "grasp the screwdriver and put it away.",
    "clear the screwdriver into the plastic box.",
    "clean up the workspace by placing the screwdriver into the box."
]

class VLARecorder:
    """用于采集和保存多模态 VLA 数据的录制器"""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.step_count = 0
        
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[1] = 0  
        self.vopt.geomgroup[2] = 0  
        
        for i in range(6):
            self.vopt.sitegroup[i] = 0
            
        self.renderer_global = mujoco.Renderer(model, height=256, width=256)
        self.renderer_wrist = mujoco.Renderer(model, height=256, width=256)
        
        self.obs_global = []
        self.obs_wrist = []
        self.obs_qpos = []
        self.actions = []
        
        self.arm_qpos_adr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.j8_qpos_adr = model.joint("joint8").qposadr[0]
        self.j9_qpos_adr = model.joint("joint9").qposadr[0]

    def reset(self):
        self.step_count = 0
        self.obs_global.clear()
        self.obs_wrist.clear()
        self.obs_qpos.clear()
        self.actions.clear()

    def record_step(self, ctrl_target):
        if self.step_count % STEPS_PER_RECORD == 0:
            self.renderer_global.update_scene(self.data, camera="global_cam", scene_option=self.vopt)
            self.obs_global.append(self.renderer_global.render().copy())
            
            self.renderer_wrist.update_scene(self.data, camera="d415_rgb", scene_option=self.vopt)
            self.obs_wrist.append(self.renderer_wrist.render().copy())
            
            arm_q = [self.data.qpos[adr] for adr in self.arm_qpos_adr]
            grip_q = [self.data.qpos[self.j8_qpos_adr], self.data.qpos[self.j9_qpos_adr]]
            self.obs_qpos.append(np.array(arm_q + grip_q, dtype=np.float32))
            
            self.actions.append(np.array(ctrl_target, dtype=np.float32))
            
        self.step_count += 1

    def save_episode(self, save_dir, ep_name):
        ep_dir = os.path.join(save_dir, ep_name)
        cam_f_dir = os.path.join(ep_dir, "cam_fixed")
        cam_w_dir = os.path.join(ep_dir, "cam_wrist")
        
        os.makedirs(cam_f_dir, exist_ok=True)
        os.makedirs(cam_w_dir, exist_ok=True)

        for i, (img_f, img_w) in enumerate(zip(self.obs_global, self.obs_wrist)):
            cv2.imwrite(os.path.join(cam_f_dir, f"{i:03d}.jpg"), cv2.cvtColor(img_f, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(cam_w_dir, f"{i:03d}.jpg"), cv2.cvtColor(img_w, cv2.COLOR_RGB2BGR))

        np.savez_compressed(
            os.path.join(ep_dir, "joint_data.npz"),
            qpos=np.array(self.obs_qpos, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32)
        )
        
        chosen_instruction = np.random.choice(LANGUAGE_INSTRUCTIONS)
        with open(os.path.join(ep_dir, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(chosen_instruction)


class ArmIK:
    """基于 mj_jacSite 的阻尼最小二乘逆运动学"""
    def __init__(self, model):
        self.model = model
        self.qpos_adr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dof_adr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.jnt_range = np.array([model.joint(j).range for j in ARM_JOINTS])
        self.site_id = model.site(EE_SITE).id
        self._scratch = mujoco.MjData(model)

    def _forward(self, d):
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def solve(self, base_qpos, q_seed, target_pos, target_mat,
              max_iter=400, tol=1e-4, damping=0.05, step=0.5):
        d = self._scratch
        d.qpos[:] = base_qpos
        d.qpos[self.qpos_adr] = q_seed

        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_err = np.zeros(3)
        rot_err = np.zeros(3)

        for _ in range(max_iter):
            self._forward(d)
            cur_pos = d.site_xpos[self.site_id]
            cur_mat = d.site_xmat[self.site_id].reshape(3, 3)
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())

            pos_err = target_pos - cur_pos
            rot_err = _quat_err_to_rotvec(_quat_mul(target_quat, _quat_conj(cur_quat)))
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < tol and np.linalg.norm(rot_err) < tol * 5:
                break

            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            new_q = np.clip(d.qpos[self.qpos_adr] + step * dq,
                             self.jnt_range[:, 0], self.jnt_range[:, 1])
            d.qpos[self.qpos_adr] = new_q

        return d.qpos[self.qpos_adr].copy(), np.linalg.norm(pos_err), np.linalg.norm(rot_err)

    def solve_multi_seed(self, base_qpos, seeds, target_pos, target_mat, tol=2e-3):
        """寻找距离当前位姿 (base_qpos) 最近的有效解，杜绝关节跳变飞转。"""
        arm_base_qpos = base_qpos[self.qpos_adr]
        best_sol = None
        best_dist = float('inf')
        fallback = None

        for seed in seeds:
            qsol, perr, rerr = self.solve(base_qpos, seed, target_pos, target_mat)
            is_valid = (perr < tol and rerr < tol * 10)
            
            if is_valid:
                dist = np.linalg.norm(qsol - arm_base_qpos)
                if dist < best_dist:
                    best_dist = dist
                    best_sol = (qsol, perr, rerr)
            else:
                score = perr + 0.05 * rerr
                if fallback is None or score < fallback[3]:
                    fallback = (qsol, perr, rerr, score)
                    
        if best_sol is not None:
            return best_sol[0], best_sol[1], best_sol[2]
        return fallback[0], fallback[1], fallback[2]

    def solve_above(self, base_qpos, q_seed, xy, base_z, target_mat,
                     clearances=LIFT_CLEARANCES, tol=2e-3, multi_seed=False):
        arm_base_qpos = base_qpos[self.qpos_adr]
        seeds = [q_seed, arm_base_qpos] + IK_SEED_BANK if multi_seed else [q_seed, arm_base_qpos]
        best = None
        for clearance in clearances:
            target = np.array([xy[0], xy[1], base_z + clearance])
            qsol, perr, rerr = self.solve_multi_seed(base_qpos, seeds, target, target_mat, tol=tol)
            if perr < tol and rerr < tol * 10:
                return qsol, clearance
            if best is None or perr < best[2]:
                best = (qsol, clearance, perr)
        return best[0], best[1]

def _quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
def _quat_err_to_rotvec(qe):
    w = np.clip(qe[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 1e-12))
    if s < 1e-8: return np.zeros(3)
    if angle > np.pi: angle -= 2 * np.pi
    return qe[1:] / s * angle

def mat_from_approach(x_hint, z_approach):
    x = x_hint / np.linalg.norm(x_hint)
    z = z_approach / np.linalg.norm(z_approach)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])

def mat_tilted_toward(tilt_deg, lean_dir):
    theta = np.radians(tilt_deg)
    yd = lean_dir / np.linalg.norm(lean_dir)
    z_approach = np.array([np.sin(theta) * yd[0], np.sin(theta) * yd[1], -np.cos(theta)])
    x_hint = np.array([yd[1], -yd[0], 0.0])
    return mat_from_approach(x_hint, z_approach)

def randomize_object_pose(model, data, rng):
    adr = model.joint(OBJECT_FREEJOINT).qposadr[0]
    
    GREEN_X_MIN, GREEN_X_MAX = 0.30, 0.60
    GREEN_Y_MIN, GREEN_Y_MAX = -0.125, 0.125
    L_HALF = 0.10 
    
    while True:
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
        yaw = rng.uniform(-np.pi, np.pi)
        
        p1_x = x + L_HALF * np.cos(yaw)
        p1_y = y + L_HALF * np.sin(yaw)
        p2_x = x - L_HALF * np.cos(yaw)
        p2_y = y - L_HALF * np.sin(yaw)
        
        if (GREEN_X_MIN <= p1_x <= GREEN_X_MAX and
            GREEN_X_MIN <= p2_x <= GREEN_X_MAX and
            GREEN_Y_MIN <= p1_y <= GREEN_Y_MAX and
            GREEN_Y_MIN <= p2_y <= GREEN_Y_MAX):
            break 
            
    data.qpos[adr:adr + 3] = [x, y, SPAWN_DROP_Z]
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    return x, y, yaw

def grasp_plan_for(ik, base_qpos, obj_xy, obj_z, sign_dir):
    R = mat_from_approach(sign_dir, np.array([0.0, 0.0, -1.0]))
    arm_base_qpos = base_qpos[ik.qpos_adr]
    
    # 强制将当前位姿 arm_base_qpos 加入最优选种子，防止大范围跨度
    seeds = [arm_base_qpos] + IK_SEED_BANK
    q_grasp, perr, _ = ik.solve_multi_seed(
        base_qpos, seeds, np.array([obj_xy[0], obj_xy[1], obj_z]), R, tol=1e-3)
        
    q_pre, _, _ = ik.solve_multi_seed(
        base_qpos, [q_grasp, arm_base_qpos] + IK_SEED_BANK, np.array([obj_xy[0], obj_xy[1], obj_z + PREGRASP_HEIGHT]), R, tol=2e-3)
        
    return dict(R=R, q_pre=q_pre, q_grasp=q_grasp, perr=perr, gripX=R[:, 0].copy(), q_grasp_target=q_grasp)


def set_ctrl(data, arm_act_ids, j8_id, j9_id, arm_q, grip):
    data.ctrl[arm_act_ids] = arm_q
    data.ctrl[j8_id] = grip[0]
    data.ctrl[j9_id] = grip[1]
    return list(arm_q) + list(grip)


def nlerp_mat(mat1, mat2, alpha):
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, mat1.flatten())
    mujoco.mju_mat2Quat(q1, mat2.flatten())
    if np.sum(q0 * q1) < 0.0: q1 = -q1
    qt = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(qt)
    if norm > 1e-6: qt /= norm
    else: qt = q0
    Rt = np.zeros(9)
    mujoco.mju_quat2Mat(Rt, qt)
    return Rt.reshape(3, 3)


def run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id,
                q_start, q_end, grip_target, n_steps, realtime=False, recorder=None):
    """带 VLA 状态录制的线性插值运动 (包含夹爪同步缓动，及最高速限制)"""
    # 动态限速：计算所需的最少步数，确保所有关节角速度均不越过安全阈值
    max_delta = np.max(np.abs(q_end - q_start))
    min_steps = int((max_delta / MAX_JOINT_VELOCITY) / model.opt.timestep)
    actual_steps = max(n_steps, min_steps)
    
    grip_start = np.array([data.ctrl[j8_id], data.ctrl[j9_id]])
    grip_end = np.array(grip_target)
    
    for i in range(1, actual_steps + 1):
        t0 = time.time()
        alpha = i / actual_steps
        alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)
        
        q = q_start + (q_end - q_start) * alpha_smooth
        grip_curr = grip_start + (grip_end - grip_start) * alpha_smooth
        
        ctrl_vec = set_ctrl(data, arm_act_ids, j8_id, j9_id, q, grip_curr)
        mujoco.mj_step(model, data)
        
        if recorder:
            recorder.record_step(ctrl_vec)
            
        if viewer is not None and viewer.is_running():
            viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)
    return True


def hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id, q_hold, grip_target, n_steps, realtime=False, recorder=None):
    return run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id,
                       q_hold, q_hold, grip_target, n_steps, realtime=realtime, recorder=recorder)


def run_parabola_motion(model, data, viewer, arm_act_ids, j8_id, j9_id, ik,
                        q_start, p_start, R_start, p_end, R_end, q_end, h_peak, grip_target, n_steps, realtime=False, recorder=None):
    """带 VLA 状态录制的抛物线轨迹运动 (包含动态限速与平滑夹爪)"""
    waypoint_steps = 30
    joint_waypoints = np.zeros((waypoint_steps + 1, 6))
    joint_waypoints[0] = q_start
    joint_waypoints[-1] = q_end
    
    q_curr_seed = q_start.copy()
    for i in range(1, waypoint_steps):
        alpha = i / waypoint_steps
        pt = p_start + (p_end - p_start) * alpha
        pt[2] += 4 * h_peak * alpha * (1.0 - alpha)
        Rt = nlerp_mat(R_start, R_end, alpha)
        q_sol, _, _ = ik.solve_multi_seed(data.qpos, [q_curr_seed, q_start] + IK_SEED_BANK, pt, Rt, tol=4e-3)
        joint_waypoints[i] = q_sol
        q_curr_seed = q_sol

    # 动态限速：评估整个路径最大关节行程的估算值
    max_accumulated_delta = np.max(np.sum(np.abs(np.diff(joint_waypoints, axis=0)), axis=0))
    min_steps = int((max_accumulated_delta / MAX_JOINT_VELOCITY) / model.opt.timestep)
    actual_steps = max(n_steps, min_steps)
        
    grip_start = np.array([data.ctrl[j8_id], data.ctrl[j9_id]])
    grip_end = np.array(grip_target)
        
    for i in range(1, actual_steps + 1):
        t0 = time.time()
        alpha_time = i / actual_steps
        alpha_smooth = 0.5 - 0.5 * np.cos(np.pi * alpha_time)
        
        idx_float = alpha_smooth * waypoint_steps
        idx_int = int(np.floor(idx_float))
        
        if idx_int >= waypoint_steps:
            q_target = joint_waypoints[-1]
        else:
            rem = idx_float - idx_int
            q_target = joint_waypoints[idx_int] * (1.0 - rem) + joint_waypoints[idx_int + 1] * rem
        
        grip_curr = grip_start + (grip_end - grip_start) * alpha_smooth
        
        ctrl_vec = set_ctrl(data, arm_act_ids, j8_id, j9_id, q_target, grip_curr)
        mujoco.mj_step(model, data)
        
        if recorder:
            recorder.record_step(ctrl_vec)
            
        if viewer is not None and viewer.is_running():
            viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)
            
    return True, joint_waypoints[-1]


def pick_and_place(model, data, viewer, arm_act_ids, j8_id, j9_id, ik, plan,
                   realtime=False, recorder=None, is_dry_run=False):
    """
    真实执行流程。包含 10% 概率触发抓取失误与纠错逻辑（模拟人类示范时的修正）。
    一旦传入 recorder，沿途的所有动作和观察结果都会被录制下来。
    """
    body_id = model.body(OBJECT_BODY).id
    box_bid = model.body(STORAGE_BOX_BODY).id
    R_grasp = plan["R"]
    q_cur = data.qpos[ik.qpos_adr].copy()
    box_xy0 = data.xpos[box_bid][:2].copy()

    intentional_miss = (np.random.rand() < 0.10) if not is_dry_run else False
    has_retried = False

    def move(q_from, q_to, grip_target, n):
        return run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id,
                          q_from, q_to, grip_target, n, realtime=realtime, recorder=recorder)

    def hold(q_hold, grip_target, n):
        return hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id,
                         q_hold, grip_target, n, realtime=realtime, recorder=recorder)

    while True:
        if not move(q_cur, plan["q_pre"], GRIP_OPEN, STEPS_PREGRASP): return None
        q_cur = plan["q_pre"]

        obj_now = data.xpos[body_id].copy()
        miss_offset = np.array([0.0, -0.06, 0.0]) if (intentional_miss and not has_retried) else np.zeros(3)
        target_pos = obj_now + miss_offset
        
        q_grasp, _, _ = ik.solve_multi_seed(
            data.qpos, [plan["q_grasp"], plan["q_pre"]],
            target_pos, R_grasp, tol=1e-3)
        if not move(q_cur, q_grasp, GRIP_OPEN, STEPS_DESCEND): return None
        q_cur = q_grasp

        if not hold(q_cur, GRIP_CLOSE, STEPS_GRIP_HOLD): return None

        if intentional_miss and not has_retried:
            if recorder:
                print("    ⚠️ 触发 10% 概率抓取偏移 (模拟纠错示范)，准备张开夹爪重试...")
            
            q_lift_miss, _ = ik.solve_above(data.qpos, q_cur, target_pos[:2], target_pos[2], R_grasp, multi_seed=True)
            if not move(q_cur, q_lift_miss, GRIP_CLOSE, STEPS_LIFT): return None
            q_cur = q_lift_miss
            
            if not hold(q_cur, GRIP_OPEN, STEPS_GRIP_HOLD): return None
            has_retried = True
            continue
            
        break

    grasp_xy = data.xpos[body_id][:2].copy()
    grasp_z = data.xpos[body_id][2]
    q_lift, lift_clearance = ik.solve_above(data.qpos, q_cur, grasp_xy, grasp_z, R_grasp, multi_seed=True)
    if not move(q_cur, q_lift, GRIP_CLOSE, STEPS_LIFT): return None
    q_cur = q_lift

    box_pos = data.xpos[box_bid].copy()
    box_xy = box_pos[:2].copy()
    box_wall_top = box_pos[2] + 0.06
    hover_z = box_wall_top + HOVER_CLEARANCE
    release_z = box_wall_top + RELEASE_CLEARANCE
    R_release = mat_tilted_toward(RELEASE_TILT_DEG, box_xy - BASE_XY)

    p_lift = np.array([grasp_xy[0], grasp_xy[1], grasp_z + lift_clearance])
    p_hover = np.array([box_xy[0], box_xy[1], hover_z])
    q_hover, _, _ = ik.solve_multi_seed(
        data.qpos, [q_cur, data.qpos[ik.qpos_adr]] + IK_SEED_BANK, p_hover, R_release, tol=6e-3)
        
    ok, q_cur = run_parabola_motion(
        model, data, viewer, arm_act_ids, j8_id, j9_id, ik,
        q_cur, p_lift, R_grasp, p_hover, R_release, q_hover, 
        h_peak=0.08, grip_target=GRIP_CLOSE, n_steps=STEPS_TRANSPORT, realtime=realtime, recorder=recorder)
    if not ok: return None

    q_release, _, _ = ik.solve_multi_seed(
        data.qpos, [q_cur, data.qpos[ik.qpos_adr]] + IK_SEED_BANK,
        np.array([box_xy[0], box_xy[1], release_z]), R_release, tol=6e-3)
    if not move(q_cur, q_release, GRIP_CLOSE, STEPS_BOX_DESCEND): return None
    q_cur = q_release
    if not hold(q_cur, GRIP_OPEN, STEPS_RELEASE_HOLD): return None

    q_retreat, _ = ik.solve_above(data.qpos, q_cur, box_xy, box_wall_top, R_release,
                                  clearances=(0.15, 0.12, 0.10, 0.08), multi_seed=True)
    if not move(q_cur, q_retreat, GRIP_OPEN, STEPS_RETREAT): return None
    if not move(q_retreat, Q_HOME, GRIP_OPEN, STEPS_RETREAT): return None
    if not hold(Q_HOME, GRIP_OPEN, STEPS_FINAL_SETTLE): return None

    final_pos = data.xpos[body_id].copy()
    box_moved = float(np.linalg.norm(data.xpos[box_bid][:2] - box_xy0))
    in_box = (abs(final_pos[0] - box_pos[0]) < 0.096 and
              abs(final_pos[1] - box_pos[1]) < 0.148 and
              final_pos[2] < box_wall_top)
    return dict(final_pos=final_pos, box_moved=box_moved, in_box=in_box)


def _hold_dry_run_ok(model, data, arm_act_ids, j8_id, j9_id, ik, plan):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    scratch = mujoco.MjData(model)
    mujoco.mj_setState(model, scratch, state, spec)
    mujoco.mj_forward(model, scratch)
    dry = pick_and_place(model, scratch, None, arm_act_ids, j8_id, j9_id, ik, plan,
                         realtime=False, recorder=None, is_dry_run=True)
    return dry is not None and dry["box_moved"] < 0.02 and dry["in_box"]


def choose_grasp_orientation(model, data, arm_act_ids, j8_id, j9_id, ik, grasp_xy, grasp_z, shaft_dir):
    candidates = [shaft_dir, -shaft_dir]
    forward_candidates = [d for d in candidates if d[0] >= -1e-5]
    forward_candidates.sort(key=lambda d: float(d[0]), reverse=True)

    for xdir in forward_candidates:
        plan = grasp_plan_for(ik, data.qpos, grasp_xy, grasp_z, xdir)
        if plan["perr"] >= 3e-3: continue
        
        if _hold_dry_run_ok(model, data, arm_act_ids, j8_id, j9_id, ik, plan):
            return plan

    current_q = data.qpos[ik.qpos_adr].copy()
    best_plan = None
    min_q_dist = float('inf')

    for xdir in candidates:
        plan = grasp_plan_for(ik, data.qpos, grasp_xy, grasp_z, xdir)
        if plan["perr"] >= 3e-3: continue
        
        q_dist = float(np.linalg.norm(plan["q_grasp_target"] - current_q))
        if q_dist < min_q_dist:
            min_q_dist = q_dist
            best_plan = plan

    if best_plan is not None:
        return best_plan
        
    return grasp_plan_for(ik, data.qpos, grasp_xy, grasp_z, candidates[0])


def collect_dataset(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    success_count = 0
    existing_eps = []
    for d in os.listdir(args.out_dir):
        if d.startswith("ep_") and os.path.isdir(os.path.join(args.out_dir, d)):
            try:
                num = int(d.split("_")[1])
                existing_eps.append(num)
            except ValueError:
                pass
    if existing_eps:
        success_count = max(existing_eps) + 1
        print(f"📂 检测到历史数据，本次将从 ep_{success_count} 开始续录。")
        
    start_count = success_count
    
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    arm_act_ids = np.array([model.actuator(a).id for a in ARM_ACTUATORS])
    j8_id = model.actuator("Joint8").id
    j9_id = model.actuator("Joint9").id
    ik = ArmIK(model)
    
    recorder = VLARecorder(model, data)
    
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    pbar = tqdm(total=args.episodes, desc="Collecting Episodes")
    
    rng = np.random.default_rng(args.seed)

    while (success_count - start_count) < args.episodes:
        mujoco.mj_resetData(model, data)
        
        # 赋予用户设定的机械臂起手位姿与夹爪状态
        data.qpos[ik.qpos_adr] = Q_INIT
        data.ctrl[arm_act_ids] = Q_INIT
        data.ctrl[j8_id] = GRIP_INIT[0]
        data.ctrl[j9_id] = GRIP_INIT[1]
        
        try:
            sponge_adr = model.joint("sponge_joint").qposadr[0]
            data.qpos[sponge_adr:sponge_adr+3] = [10.0, 10.0, -10.0]
        except KeyError:
            pass
        
        sx, sy, syaw = randomize_object_pose(model, data, rng)
        mujoco.mj_forward(model, data)
        
        # 1. 在初始位姿停留等待螺丝刀落定 (不录制，避免无意义的冗长静止画面)
        if not hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id, Q_INIT, GRIP_INIT, STEPS_SETTLE):
            break
            
        # ==================================
        # 提前开启正式录制：记录从 Q_INIT 开始的完整轨迹
        # ==================================
        recorder.reset()
        
        # 2. 移动到全 0 关节位姿 (Q_HOME) 进行规划准备，并同步录制这段运动
        if not run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id, Q_INIT, Q_HOME, GRIP_OPEN, 600, realtime=not args.headless, recorder=recorder):
            break
            
        # 短暂等待机械臂物理稳定，同样会被录制
        if not hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id, Q_HOME, GRIP_OPEN, 200, realtime=not args.headless, recorder=recorder):
            break
            
        body_id = model.body(OBJECT_BODY).id
        obj_pos = data.xpos[body_id].copy()
        obj_mat = data.xmat[body_id].reshape(3, 3).copy()

        shaft_dir = obj_mat @ np.array([0.0, 1.0, 0.0])
        shaft_dir[2] = 0.0
        if np.linalg.norm(shaft_dir) < 1e-3:
            shaft_dir = np.array([1.0, 0.0, 0.0])
        shaft_dir /= np.linalg.norm(shaft_dir)

        plan = choose_grasp_orientation(
            model, data, arm_act_ids, j8_id, j9_id, ik, obj_pos[:2], obj_pos[2], shaft_dir)
        
        # 真实执行轨迹并采集数据 (无缝接续刚刚开启的 recorder，is_dry_run=False，开放 10% 纠错录制)
        result = pick_and_place(model, data, viewer, arm_act_ids, j8_id, j9_id, ik, plan, 
                                realtime=not args.headless, recorder=recorder, is_dry_run=False)
                                
        if viewer is not None and not viewer.is_running():
            break

        if result is not None and result["in_box"] and result["box_moved"] < 0.02:
            ep_name = f"ep_{success_count}"
            recorder.save_episode(args.out_dir, ep_name)
            success_count += 1
            pbar.update(1)
            pbar.set_postfix({"Saved": ep_name})
        else:
            pass
            
    pbar.close()
    if viewer is not None:
        viewer.close()
    print(f"\n数据集采集完成! 共新收集 {success_count - start_count} 条有效轨迹，总计保存在: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动采集基于视觉的 VLA 数据集")
    parser.add_argument("--episodes", type=int, default=50, help="要收集的成功轨迹数量")
    parser.add_argument("--out_dir", type=str, default="../datasets/screwdriver_cleanup", help="保存数据文件夹的目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="是否无头模式运行(后台极速跑数据)")
    args = parser.parse_args()
    
    collect_dataset(args)