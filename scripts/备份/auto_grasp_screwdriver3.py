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

# 释放点朝"红色障碍区"方向偏移(m)：红区世界中心≈(0.45,0.225)，盒心(0.35,0.40)，
# 即朝 -y/+x 的盒内近侧。会自动夹在盒内壁安全范围内，不会撞墙。0 = 仍放盒中心。
RELEASE_TOWARD_OBSTACLE = 0.10
OBSTACLE_ZONE_XY = np.array([0.45, 0.225])   # 红色障碍区世界中心
BOX_INNER_MARGIN = 0.035                      # 距盒内壁至少留这么多，防止蹭到边

GRIP_OPEN = (-0.02, 0.02)
GRIP_CLOSE = (0.02, -0.02)
GRIP_INIT = GRIP_CLOSE  # 初始/默认状态：闭合(仅在接近并抓取螺丝刀时才张开)

LIFT_CLEARANCES = (0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01)
PREGRASP_HEIGHT = 0.08

# 抓取接近方向绕"杆身轴线"的候选倾斜角(rad)。0=正下方(优先)；正下方 IK 失败时
# (典型：螺丝刀轴向≈桌子长方向 世界Y，腕部卡限位)依次尝试倾斜，绕开限位。
# 绕杆身轴旋转不改变夹持质量(两指仍垂直于杆身)。
GRASP_TILT_CANDIDATES = (0.0,
                         np.radians(20), np.radians(-20),
                         np.radians(35), np.radians(-35),
                         np.radians(50), np.radians(-50))

# 全局动态安全限速 (弧度/秒) —— 提速：由 0.75 提高，运动更快但仍受平滑插值约束
MAX_JOINT_VELOCITY = 2.0

# 提速总开关：所有运动阶段步数 × SPEED_SCALE (0.5 = 快一倍)。太小会导致抖动/抓不稳，
# 建议 0.4~0.7；物体落定(STEPS_SETTLE)不缩放，那是物理沉降必须的时间。
SPEED_SCALE = 0.5

def _s(n):  # 按提速系数缩放步数，最少保底 80 步
    return max(80, int(n * SPEED_SCALE))

# 各阶段仿真的保底基础步数 (timestep=0.001s)
# 实际运行步数将根据 MAX_JOINT_VELOCITY 动态延长
STEPS_SETTLE = 1000          # 1.0s 物体落定(不缩放，需真实沉降时间)
STEPS_PREGRASP = _s(400)
STEPS_DESCEND = _s(300)
STEPS_GRIP_HOLD = _s(300)
STEPS_LIFT = _s(300)
STEPS_TRANSPORT = _s(500)
STEPS_BOX_DESCEND = _s(300)
STEPS_RELEASE_HOLD = _s(300)
STEPS_RETREAT = _s(400)
STEPS_FINAL_SETTLE = _s(800)

Q_HOME = np.zeros(6)
# 用户指定的最原始 6 轴机械臂起手位姿
Q_INIT = np.array([0.0, 1.41, 1.3, 0.0, -0.7, 0.0])

# —— 抓取偏航容差 ——
# 桌面长边沿世界 y；当螺丝刀轴向也≈±y 时，夹爪X(=±轴向)的 x 分量≈0，相机只能侧看，
# 朝前的候选不存在 → 规划失败。平行夹爪夹圆杆时偏航偏一些仍夹得稳，故允许绕竖直轴
# 在 ±GRASP_YAW_TOL_DEG 内偏转，用这点余量把相机转向前方，解决该退化情形。
GRASP_YAW_TOL_DEG = 30.0
GRASP_YAW_STEP_DEG = 10.0
# 认为"相机朝前"所需的最小 x 分量(0.35 ≈ 与 +x 夹角 70° 以内)
FORWARD_MIN_X = 0.35
# 朝向选择阶段的干跑验证：主流程已改为"先虚拟抓取整条轨迹再回放"，那一步即是验证，
# 这里再干跑就是重复计算 → 设 0 关闭(仍保留代码，需要时可调回 1)
DRYRUN_TOP_K = 0
# 抓到后若螺丝刀相对夹爪滑移超过该距离(m)，判定滑落 → 本条作废重采
SLIP_TOL = 0.05

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

def _rot_about_axis(axis, angle):
    """罗德里格斯公式：绕 axis 旋转 angle 的旋转矩阵"""
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def grasp_plan_for(ik, base_qpos, obj_xy, obj_z, sign_dir, tilt=0.0):
    # 接近方向绕"杆身轴线(sign_dir)"旋转 tilt：这是抓圆柱杆身的自由自由度 ——
    # 转过之后两指闭合方向仍垂直于杆身，夹持效果不变，但腕部姿态大幅改变，
    # 可绕开 螺丝刀轴向≈桌子长方向(世界Y) 时 joint5/joint6 卡限位导致的 IK 失败。
    z_app = _rot_about_axis(sign_dir, tilt) @ np.array([0.0, 0.0, -1.0])
    R = mat_from_approach(sign_dir, z_app)
    arm_base_qpos = base_qpos[ik.qpos_adr]
    
    # 强制将当前位姿 arm_base_qpos 加入最优选种子，防止大范围跨度
    seeds = [arm_base_qpos] + IK_SEED_BANK
    q_grasp, perr, _ = ik.solve_multi_seed(
        base_qpos, seeds, np.array([obj_xy[0], obj_xy[1], obj_z]), R, tol=1e-3)

    # 预抓取沿"接近轴反方向"退开(tilt=0 时即正上方，与原行为一致)
    pre_off = -z_app * PREGRASP_HEIGHT
    q_pre, _, _ = ik.solve_multi_seed(
        base_qpos, [q_grasp, arm_base_qpos] + IK_SEED_BANK,
        np.array([obj_xy[0] + pre_off[0], obj_xy[1] + pre_off[1], obj_z + pre_off[2]]),
        R, tol=2e-3)

    return dict(R=R, q_pre=q_pre, q_grasp=q_grasp, perr=perr,
                gripX=R[:, 0].copy(), q_grasp_target=q_grasp, tilt=tilt)


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
                q_start, q_end, grip_target, n_steps, realtime=False, recorder=None,
                vel_cap=None, ctrl_log=None):
    """带 VLA 状态录制的线性插值运动 (包含夹爪同步缓动，及最高速限制)"""
    # 动态限速：计算所需的最少步数，确保所有关节角速度均不越过安全阈值
    max_delta = np.max(np.abs(q_end - q_start))
    cap = MAX_JOINT_VELOCITY if vel_cap is None else vel_cap
    min_steps = int((max_delta / cap) / model.opt.timestep)
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
        
        if ctrl_log is not None:
            ctrl_log.append(list(ctrl_vec))
        if recorder:
            recorder.record_step(ctrl_vec)
            
        if viewer is not None and viewer.is_running():
            viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)
    return True


def hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id, q_hold, grip_target, n_steps, realtime=False, recorder=None, ctrl_log=None):
    return run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id,
                       q_hold, q_hold, grip_target, n_steps, realtime=realtime, recorder=recorder,
                       ctrl_log=ctrl_log)


def run_parabola_motion(model, data, viewer, arm_act_ids, j8_id, j9_id, ik,
                        q_start, p_start, R_start, p_end, R_end, q_end, h_peak, grip_target, n_steps, realtime=False, recorder=None, ctrl_log=None):
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
        
        if ctrl_log is not None:
            ctrl_log.append(list(ctrl_vec))
        
        if recorder:
            recorder.record_step(ctrl_vec)
            
        if viewer is not None and viewer.is_running():
            viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)
            
    return True, joint_waypoints[-1]


def replay_ctrl_log(model, data, viewer, arm_act_ids, j8_id, j9_id, ctrl_log,
                    realtime=False, recorder=None):
    """回放虚拟抓取时捕获的逐步 ctrl 序列。
    MuJoCo 是确定性的：同一初始状态 + 同一 ctrl 序列 → 完全相同的结果，
    所以这里不需要任何 IK/规划，录制过程绝对顺畅无停顿。"""
    for cv in ctrl_log:
        t0 = time.time()
        data.ctrl[arm_act_ids] = cv[:6]
        data.ctrl[j8_id] = cv[6]
        data.ctrl[j9_id] = cv[7]
        mujoco.mj_step(model, data)

        if recorder:
            recorder.record_step(cv)
        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()
        if realtime:
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)
    return True


def pick_and_place(model, data, viewer, arm_act_ids, j8_id, j9_id, ik, plan,
                   realtime=False, recorder=None, is_dry_run=False, ctrl_log=None):
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

    # 虚拟抓取(捕获轨迹)时按真实步长跑，保证之后回放的物理结果完全一致
    def move(q_from, q_to, grip_target, n):
        return run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id,
                          q_from, q_to, grip_target, n,
                          realtime=realtime, recorder=recorder, ctrl_log=ctrl_log)

    def hold(q_hold, grip_target, n):
        return hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id,
                         q_hold, grip_target, n,
                         realtime=realtime, recorder=recorder, ctrl_log=ctrl_log)

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
            print("    ⚠️ 触发 10% 概率抓取偏移 (模拟纠错示范)，准备张开夹爪重试...")
            
            q_lift_miss, _ = ik.solve_above(data.qpos, q_cur, target_pos[:2], target_pos[2], R_grasp, multi_seed=True)
            if not move(q_cur, q_lift_miss, GRIP_CLOSE, STEPS_LIFT): return None
            q_cur = q_lift_miss
            
            if not hold(q_cur, GRIP_OPEN, STEPS_GRIP_HOLD): return None
            has_retried = True
            continue
            
        break

    # 抓稳瞬间记录 螺丝刀相对夹爪 的偏移，作为后续判断"滑落"的基准
    grip_ref = data.xpos[body_id] - data.site_xpos[ik.site_id]

    def slipped(tag):
        """搬运途中螺丝刀相对夹爪偏移超过 SLIP_TOL 即判滑落"""
        d = float(np.linalg.norm((data.xpos[body_id] - data.site_xpos[ik.site_id]) - grip_ref))
        if d > SLIP_TOL:
            print(f"    ⚠️ 螺丝刀滑落({tag}, 偏移{d*1000:.0f}mm > {SLIP_TOL*1000:.0f}mm)，本条作废重采")
            return True
        return False

    grasp_xy = data.xpos[body_id][:2].copy()
    grasp_z = data.xpos[body_id][2]
    q_lift, lift_clearance = ik.solve_above(data.qpos, q_cur, grasp_xy, grasp_z, R_grasp, multi_seed=True)
    if not move(q_cur, q_lift, GRIP_CLOSE, STEPS_LIFT): return None
    if slipped("抬起后"): return None
    q_cur = q_lift

    box_pos = data.xpos[box_bid].copy()
    box_xy = box_pos[:2].copy()
    box_wall_top = box_pos[2] + 0.06
    hover_z = box_wall_top + HOVER_CLEARANCE
    release_z = box_wall_top + RELEASE_CLEARANCE
    R_release = mat_tilted_toward(RELEASE_TILT_DEG, box_xy - BASE_XY)

    # 释放点朝红色障碍区方向偏移，并夹在盒内壁安全范围内(盒内半尺寸 x 0.098 / y 0.148)
    to_obs = OBSTACLE_ZONE_XY - box_xy
    n = np.linalg.norm(to_obs)
    if n > 1e-6 and RELEASE_TOWARD_OBSTACLE > 0:
        drop_xy = box_xy + (to_obs / n) * RELEASE_TOWARD_OBSTACLE
        lim = np.array([0.098, 0.148]) - BOX_INNER_MARGIN     # 允许偏离盒心的最大量
        drop_xy = box_xy + np.clip(drop_xy - box_xy, -lim, lim)
    else:
        drop_xy = box_xy.copy()

    p_lift = np.array([grasp_xy[0], grasp_xy[1], grasp_z + lift_clearance])
    p_hover = np.array([drop_xy[0], drop_xy[1], hover_z])
    q_hover, _, _ = ik.solve_multi_seed(
        data.qpos, [q_cur, data.qpos[ik.qpos_adr]] + IK_SEED_BANK, p_hover, R_release, tol=6e-3)
        
    ok, q_cur = run_parabola_motion(
        model, data, viewer, arm_act_ids, j8_id, j9_id, ik,
        q_cur, p_lift, R_grasp, p_hover, R_release, q_hover, 
        h_peak=0.08, grip_target=GRIP_CLOSE, n_steps=STEPS_TRANSPORT, realtime=realtime, recorder=recorder,
        ctrl_log=ctrl_log)
    if not ok: return None
    if slipped("抛物线搬运后"): return None

    q_release, _, _ = ik.solve_multi_seed(
        data.qpos, [q_cur, data.qpos[ik.qpos_adr]] + IK_SEED_BANK,
        np.array([drop_xy[0], drop_xy[1], release_z]), R_release, tol=6e-3)
    if not move(q_cur, q_release, GRIP_CLOSE, STEPS_BOX_DESCEND): return None
    if slipped("入盒下降后"): return None
    q_cur = q_release
    if not hold(q_cur, GRIP_OPEN, STEPS_RELEASE_HOLD): return None

    q_retreat, _ = ik.solve_above(data.qpos, q_cur, box_xy, box_wall_top, R_release,
                                  clearances=(0.15, 0.12, 0.10, 0.08), multi_seed=True)
    if not move(q_cur, q_retreat, GRIP_CLOSE, STEPS_RETREAT): return None
    if not move(q_retreat, Q_HOME, GRIP_CLOSE, STEPS_RETREAT): return None
    # 最后回退到第一步的初始位姿 (Q_INIT + 初始夹爪状态=闭合) 再结束
    if not move(Q_HOME, Q_INIT, GRIP_INIT, STEPS_RETREAT): return None
    if not hold(Q_INIT, GRIP_INIT, STEPS_FINAL_SETTLE): return None

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


def _best_j6_equivalent(ik, plan, j6_cur):
    """joint6 限位 [0,4.76] 很宽：同一抓取姿态的 joint6 可 ±π(平行夹爪翻转,物理等价)
    或 ±2π 平移仍在限位内。这里挑离当前 joint6 最近的等价角，避免白转大半圈。
    返回 (最小转角, 优化后的 plan)。"""
    lo, hi = ik.jnt_range[5]
    q = plan["q_grasp_target"].copy()
    q_pre = plan["q_pre"].copy()
    j6 = float(q[5])

    best_j6, best_d = j6, abs(j6 - j6_cur)
    for k in (-2, -1, 1, 2):                # ±π(夹爪翻转,等价) 与 ±2π(整圈,等价)
        cand = j6 + k * np.pi
        if not (lo <= cand <= hi):
            continue
        d = abs(cand - j6_cur)
        if d < best_d:
            best_d, best_j6 = d, cand

    if best_j6 != j6:                       # 采用更近的等价角
        delta = best_j6 - j6
        q[5] = best_j6
        # 预抓取位姿同步平移，保持整段轨迹连贯(避免下探时又转回去)
        if lo <= q_pre[5] + delta <= hi:
            q_pre[5] = q_pre[5] + delta
        plan = dict(plan)
        plan["q_grasp_target"] = q
        plan["q_grasp"] = q
        plan["q_pre"] = q_pre
    return best_d, plan


def _yaw_rot(v, rad):
    """把水平方向 v 绕竖直轴旋转 rad 弧度"""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])


def choose_grasp_orientation(model, data, arm_act_ids, j8_id, j9_id, ik, grasp_xy, grasp_z, shaft_dir):
    # 候选 = {夹爪X = ±shaft_dir} × {绕竖直轴偏航 yaw} × {绕杆身轴倾斜 tilt}
    #   yaw : 让相机能朝前 —— 螺丝刀轴向≈桌子长方向(世界Y)时，夹爪X(=±轴向)的 x 分量≈0，
    #         相机只能侧看，朝前候选不存在而规划失败；平行夹爪夹圆杆允许偏航偏一些仍夹得稳。
    #   tilt: 绕开 joint5/joint6 卡限位导致的 IK 失败(夹持质量不变)。
    # 优先级：① 相机朝前 ② 偏航小 ③ 倾斜小 ④ joint6 转角小 ⑤ 总关节距离近；再做干跑验证。
    current_q = data.qpos[ik.qpos_adr].copy()
    j6_cur = float(current_q[5])

    yaws = [0.0]
    step = np.radians(GRASP_YAW_STEP_DEG)
    tol = np.radians(GRASP_YAW_TOL_DEG)
    y = step
    while y <= tol + 1e-9:
        yaws += [y, -y]
        y += step

    def collect(tilts):
        out = []
        for tilt in tilts:
            for base in (shaft_dir, -shaft_dir):
                for yaw in yaws:
                    xdir = _yaw_rot(base, yaw)
                    n = np.linalg.norm(xdir)
                    if n < 1e-9:
                        continue
                    plan = grasp_plan_for(ik, data.qpos, grasp_xy, grasp_z, xdir / n, tilt=tilt)
                    if plan["perr"] >= 3e-3:
                        continue                            # 不可达候选丢弃
                    dj6, plan = _best_j6_equivalent(ik, plan, j6_cur)
                    out.append(dict(
                        plan=plan, dj6=dj6, yaw=abs(yaw), tilt=abs(float(tilt)),
                        fwd_x=float(plan["gripX"][0]),
                        qdist=float(np.linalg.norm(plan["q_grasp_target"] - current_q))))
        return out

    # 先只试正下方抓取(tilt=0)；有"相机朝前"的可达候选就不必再展开倾斜
    cands = collect((0.0,))
    if not any(c["fwd_x"] >= FORWARD_MIN_X for c in cands):
        cands += collect([t for t in GRASP_TILT_CANDIDATES if abs(t) > 1e-9])

    if not cands:                                           # 全部不可达 → 保底
        return grasp_plan_for(ik, data.qpos, grasp_xy, grasp_z, shaft_dir)

    cands.sort(key=lambda c: (c["fwd_x"] < FORWARD_MIN_X, c["yaw"], c["tilt"],
                              c["dj6"], c["qdist"]))

    def _log(c, note):
        print(f"[规划] yaw={np.degrees(c['yaw']):.0f}° tilt={np.degrees(c['tilt']):.0f}° "
              f"相机前向x={c['fwd_x']:+.2f} Δjoint6={np.degrees(c['dj6']):.1f}° ({note})")

    # 干跑验证代价高，只验排序靠前的若干候选
    for c in cands[:DRYRUN_TOP_K]:
        if _hold_dry_run_ok(model, data, arm_act_ids, j8_id, j9_id, ik, c["plan"]):
            _log(c, "干跑通过")
            return c["plan"]

    _log(cands[0], "未过干跑,取最优候选")
    return cands[0]["plan"]


def _scan_existing_eps(out_dir):
    """扫描已有的 ep_N 文件夹，返回已存在的序号集合"""
    found = set()
    for d in os.listdir(out_dir):
        if d.startswith("ep_") and os.path.isdir(os.path.join(out_dir, d)):
            try:
                found.add(int(d.split("_")[1]))
            except ValueError:
                pass
    return found


def _next_slot(existing, target):
    """返回下一个应填充的序号：优先补 [0,target) 中被删掉的空缺，其次往后顺延"""
    for i in range(target):
        if i not in existing:
            return i
    n = target
    while n in existing:
        n += 1
    return n


def collect_dataset(args):
    os.makedirs(args.out_dir, exist_ok=True)

    target = args.target          # 目标总数(文件夹里达到这么多条就结束)
    existing = _scan_existing_eps(args.out_dir)
    start_have = len(existing)

    if existing:
        gaps = sorted(i for i in range(target) if i not in existing)
        print(f"📂 已有 {start_have} 条数据集。"
              + (f" 待补空缺序号: {gaps[:12]}{' …' if len(gaps) > 12 else ''}"
                 if gaps else " 无空缺。"))
    if start_have >= target:
        print(f"✅ 已有 {start_have} 条，已达目标 {target} 条，无需采集。")
        return

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    
    # ---------------------------------------------------------
    # 新增修改：动态对称镜像全局相机的 X 坐标 (从前向后看)
    # 目标中心X=0.45，原相机X=-0.1，对称镜像后X=1.0
    cam_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "global_cam_body")
    if cam_body_id != -1:
        model.body_pos[cam_body_id][0] = 1.0
    # ---------------------------------------------------------

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

    pbar = tqdm(total=target, initial=start_have, desc="数据集总数")

    rng = np.random.default_rng(args.seed)

    while len(existing) < target:
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
        if not run_motion(model, data, viewer, arm_act_ids, j8_id, j9_id, Q_INIT, Q_HOME, GRIP_CLOSE, 600, realtime=not args.headless, recorder=recorder):
            break
            
        # 短暂等待机械臂物理稳定，同样会被录制
        if not hold_pose(model, data, viewer, arm_act_ids, j8_id, j9_id, Q_HOME, GRIP_CLOSE, 200, realtime=not args.headless, recorder=recorder):
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

        # ==================================
        # 先"虚拟抓取"一遍：在状态副本上跑完整流程并捕获逐步 ctrl 序列。
        # 不渲染/不录制/不实时，纯算；成功了才把这条轨迹原样回放录制，
        # 于是录制过程中 0 次 IK、0 规划，完全顺畅无停顿。
        # ==================================
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.zeros(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)

        scratch = mujoco.MjData(model)
        mujoco.mj_setState(model, scratch, state, spec)
        mujoco.mj_forward(model, scratch)

        ctrl_log = []
        dry = pick_and_place(model, scratch, None, arm_act_ids, j8_id, j9_id, ik, plan,
                             realtime=False, recorder=None, is_dry_run=False, ctrl_log=ctrl_log)

        if dry is None or not dry["in_box"] or dry["box_moved"] >= 0.02:
            continue                      # 虚拟抓取失败(含滑落) → 换个摆放重来，不录制

        # 虚拟成功 → 恢复到同一初始状态，原样回放并录制(确定性保证结果一致)
        mujoco.mj_setState(model, data, state, spec)
        mujoco.mj_forward(model, data)
        if not replay_ctrl_log(model, data, viewer, arm_act_ids, j8_id, j9_id, ctrl_log,
                               realtime=not args.headless, recorder=recorder):
            break
        result = dry

        if viewer is not None and not viewer.is_running():
            break

        if result is not None and result["in_box"] and result["box_moved"] < 0.02:
            # 重新扫盘(采集期间可能有人手动删了数据)，取下一个待填充序号：优先补空缺
            existing = _scan_existing_eps(args.out_dir)
            slot = _next_slot(existing, target)
            ep_name = f"ep_{slot}"
            recorder.save_episode(args.out_dir, ep_name)
            existing.add(slot)
            pbar.n = len(existing)
            pbar.refresh()
            pbar.set_postfix({"Saved": ep_name})
        else:
            pass

    pbar.close()
    if viewer is not None:
        viewer.close()
    have = len(_scan_existing_eps(args.out_dir))
    print(f"\n数据集采集完成! 本次新增 {have - start_have} 条，"
          f"当前共 {have} 条 (目标 {target})，保存在: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动采集基于视觉的 VLA 数据集")
    parser.add_argument("--target", type=int, default=150,
                        help="目标数据集总数：文件夹里达到这么多条即结束(会先补被删掉的空缺序号)")
    parser.add_argument("--out_dir", type=str, default="../datasets/screwdriver_cleanup", help="保存数据文件夹的目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="是否无头模式运行(后台极速跑数据)")
    args = parser.parse_args()
    
    collect_dataset(args)