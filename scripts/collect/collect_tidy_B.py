#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_tidy_B.py —— 螺丝刀入快递盒 · 自动化采集
执行内核搬自 data_collector_v2.py：速度级(resolved-rate)伺服 + 状态机。

为什么换掉之前那套"先解 IK 再关节插值"：
  那套在 IK 解不出来时是**硬失败**，而这台臂工作面就在肩高附近、姿态余量很小，
  一堆位置都会解不出来。速度级伺服没有这个概念 —— 它每步只算一次
  q_dot = damped_pinv(J) @ twist 然后积分，够不到就慢慢挪，配合死锁超时兜底。

原样搬过来的：
  get_site_jacobian_6d / damped_pinv / get_orientation_error /
  compute_6d_twist / compute_3d_velocity
  状态机 HOVER → DESCEND → GRASP → MOVE_ARC → RELEASE → RETURN_ARC →
        RETURN_JOINT → WAIT_HOME，以及 RECOVER_OPEN 脱手重试
  虚拟兔子(carrot)抛物线跟踪、phase_steps>4000 死锁作废
  边跑边录（不做虚拟-回放两趟）、ep_N 目录、joint_data.npz + instruction.txt

针对 tidy_B / xunjian_arm 必须改的地方
================================================================================
1. 夹爪编码。原脚本 ctrl[6]=ctrl[7]=gripper_target（两维**同号**，0.04=开 0.0=合）。
   xunjian_arm 是 joint8 = -joint9，±0.02，**正值=闭合**。照抄会让两指互相打架，
   不报错、不产生 NaN，只是永远夹不住。这正是 robot_spec 开头写的那个历史 bug。
2. 索引。原脚本用 data.qpos[:6] / data.qpos[:8] / data.ctrl[:6] 硬编码下标。
   这里全部按名字解析（ARM_JOINTS / ARM_ACTUATORS / joint8 / joint9）。
3. tcp_site -> ee_site；rear_cam/wrist_cam -> robot_spec.CAM_FIXED/CAM_WRIST。
4. 目标点 = 两夹爪夹取点连线中点（[C1]），不是物体 body 原点，
   也不再用 z_offset=-0.015 那种经验下潜量。夹取点从碰撞胶囊 sd_handle 反算。
5. HOVER/DESCEND 改用 3D 速度（只约束位置），不用 6D twist。
   你要求过"不强制垂直下抓，末端小球到抓取点就行"。原脚本这两段是 6D twist，
   会把腕部硬拧成竖直；这台臂指尖朝下时 TCP 高度上限只有 ~0.789，硬拧会一直较劲。
   compute_3d_velocity 和 J_3d 是原脚本搬运段本来就在用的，这里复用。
6. [C2] 转移走廊。原脚本 MOVE_ARC 的 H_peak=0.18 且不设上限，在这个场景会冲到
   ~0.96 以上，高过手电筒顶 0.958。夹持期每个虚拟兔子的 z 都硬夹进
   (0.805, 0.943)，并在成功后复核实际 TCP 高度，越界整条作废。
================================================================================
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common import robot_spec as rspec

SCENE_PATH = PROJECT_ROOT / "scenes" / "tidy_B_record_preview.xml"

ARM_JOINTS = rspec.ARM_JOINTS
ARM_ACTUATORS = rspec.ARM_ACTUATORS
EE_SITE = rspec.EE_SITE

OBJECT_BODY = "screwdriver"
OBJECT_FREEJOINT = "fj_screwdriver"
STORAGE_BOX_BODY = "delivery_box"
HANDLE_GEOM = "sd_handle"

# ---------- 夹爪（±0.02，正值=闭合，joint8 = -joint9）----------
GRIP_OPEN = rspec.GRIP_OPEN
GRIP_CLOSE = rspec.GRIP_CLOSE

# ---------- 高度常量（由 XML 推出）----------
Z_TABLE_TOP = 0.732
Z_BOX_WALL_TOP = 0.790            # delivery_box: 0.734 + 0.031 + 0.025
Z_BOX_INNER_FLOOR = 0.740
Z_FLASHLIGHT_TOP = 0.958          # 0.730 + 0.228，采集时不存在，部署时放回来
CORRIDOR_MARGIN = 0.015
Z_CORRIDOR_LO = Z_BOX_WALL_TOP + CORRIDOR_MARGIN     # 0.805
Z_CORRIDOR_HI = Z_FLASHLIGHT_TOP - CORRIDOR_MARGIN   # 0.943

SAFE_Z = 0.845                    # 抓取前的悬停高度（此时未夹持，不受 C2 约束）
H_PEAK = 0.06                     # 搬运抛物线拱高（原脚本 0.18，这里会顶穿手电筒）
RET_H_PEAK = 0.05
RELEASE_YAW_TOL_DEG = 30.0   # 装箱安全锥 43°，留裕度取 30°
PARALLEL_TOL_DEG = 8.0       # ALIGN 阶段"与盒子长边平行"的收敛阈值（不是成功判据）
FIT_TOL = 0.005              # 成功判据容差：最高点允许高过盒壁顶多少（AABB 偏保守）
FIT_EDGE_MARGIN = 0.010      # 规划约束：物体包络距内壁至少留这么多
Z_PLACE_CLEAR = 0.012        # 松手时物体最低点高出盒内底多少（贴着放，不是扔）
BOX_INNER_HALF = np.array([0.145, 0.085])

# ---------- 生成区（tidy_B 绿区 x∈[0.30,0.60], y∈[-0.35,-0.05]）----------
SPAWN_X_RANGE = (0.38, 0.50)
SPAWN_Y_RANGE = (-0.26, -0.14)
SPAWN_DROP_Z = 0.80
# 螺丝刀绕世界 z 轴的偏航随机幅度。放开之后抓取几何每条都不同，
# 采集分布变宽；但要留意释放前的偏航对齐锥是 ±35°，两者要能对上。
YAW_JITTER = np.radians(30.0)
GREEN_X_MIN, GREEN_X_MAX = 0.31, 0.59
GREEN_Y_MIN, GREEN_Y_MAX = -0.34, -0.06

FPS = rspec.CONTROL_HZ                       # 50
STEPS_PER_RECORD = rspec.STEPS_PER_RECORD    # 20
Q_INIT = rspec.Q_INIT
Q_ZERO = np.zeros(6)      # 关节全 0：每条 episode 的起手与收尾位姿
LANGUAGE_INSTRUCTIONS = rspec.LANGUAGE_INSTRUCTIONS

MISS_PROB = 0.10                  # 故意抓偏，制造纠错示范
SLIP_TRANSLATION_TOL = 0.035      # 物体相对夹持中心的平移变化上限（m）
SLIP_ROTATION_TOL_DEG = 30.0      # 物体相对夹持中心的旋转变化上限（deg）
TASK_GRIPPER_KP = 400.0           # 仅本任务提高夹持力；机械臂其余关节保持 XML 参数
PRELIFT_DISTANCE = 0.050          # 正式搬运前慢速试提 5cm
PRELIFT_CLEARANCE = 0.012         # 物体最低点至少离桌 12mm 才算真正抓起
PRELIFT_SPEED = 0.08              # 试提速度，经过 SPEED_SCALE 后实际更低
PRELIFT_SETTLE_TIME = 0.20        # 离桌且双指接触后稳定多久再标定抓取参考
PRELIFT_TIMEOUT = 3.0
MAX_GRASP_RETRIES = 1             # 故意抓偏允许纠错一次，再失败则整条重采
JAW_LEVEL_K = 20.0                # 抓取前夹爪闭合轴调平增益（实测比 4.0 收敛快）
JAW_LEVEL_RHO = 0.05              # 调平伺服阻尼（实测比 0.10 收敛快）
JAW_AXIS_TOL_DEG = 4.0            # 闭合轴与目标水平方向的夹角容差
JAW_HEIGHT_TOL = 0.002            # 左右夹爪 body 原点高度差容差（m）
PREALIGN_CLEARANCE = 0.045         # 在手柄中心上方 45mm 边下降边调平
# ---------- IK 预对齐：先解出"夹爪水平"的关节位形，再关节空间走过去 ----------
# 速度级伺服跨不过运动学分支：夹爪水平的解几乎全在 joint6≈70~150° 一侧，
# 而"就近选符号"会把 joint6 推到行程下限 0 顶死，残留十几度高低差压不下来。
# 离线 IK 两侧符号都试，取限位余量最大的解，再用关节空间运动过去。
IK_PREALIGN_CLEARANCE = 0.10      # IK 目标点取在手柄中心上方 10cm（悬停高度）
IK_MAX_ITERS = 300
IK_DLS_RHO = 0.05
IK_POS_TOL = 0.002
IK_AXIS_TOL_DEG = 1.0
IK_RANDOM_STARTS = 24
IK_APPROACH_MAX_TILT_DEG = 40.0   # 接近方向偏离竖直下方的上限，太斜没法直下抓
IK_GOOD_MARGIN = np.radians(28.0)  # 限位余量到这个程度就不再继续搜
IK_MOVE_TOL = 0.05                # 关节空间到位判据（rad，L2）
IK_MOVE_TIMEOUT = 4.0
IK_OBJECT_DISTURB_TOL = 0.010     # 关节空间转移途中碰动物体的容忍上限（m）
PREALIGN_SETTLE_TIME = 0.15
PREALIGN_TIMEOUT = 3.0
# 位置到位后把线速度压到很小，让 6 个自由度几乎全部让给调平任务。
# 远处接近奇异时，位置任务和姿态任务抢自由度是残余高低差压不下去的主因。
PREALIGN_FREEZE_DIST = 0.02       # 到这个距离内视为位置已到位
PREALIGN_FREEZE_GAIN = 0.15       # 冻结时保留的位置增益，仅用于防漂
PREALIGN_STALL_WINDOW = 1.0       # 调平误差停滞检测窗口（s）
# 停滞用"归一化残差"（轴误差/角度容差 与 高度差/高度容差 取大者）衡量，
# 窗口内至少要改善这么多倍容差才算仍在收敛。阈值定太粗会把正常的末段
# 慢收敛误判成不可达，从而把大量本可成功的 episode 推去走备用路径。
PREALIGN_STALL_IMPROVE_RATIO = 0.05
# 下降段迟迟无法同时满足到位+水平的判定时限。必须小于 DEADLOCK_STEPS 对应
# 的时长（timestep=0.001s 时约 5.7s），否则永远先撞上阶段死锁、走不到补救分支。
DESCEND_ALIGN_TIMEOUT = 4.0
# ---------- 调平不可达时的备用路径：先把螺丝刀推到更易调平的区域 ----------
MAX_REPOSE_RETRIES = 1            # 预调整最多触发一次，仍不可达才判失败
REPOSE_ANCHOR_XY = np.array([0.42, -0.18])   # 可达性较好的工作区中心
REPOSE_PUSH_DISTANCE = 0.06       # 单次推动的最大位移
REPOSE_MIN_PUSH = 0.015           # 小于这个位移不值得推
REPOSE_PUSH_CLEARANCE = 0.020     # 推之前落在物体包络之外的余量
REPOSE_PUSH_Z_OFFSET = 0.022      # 推动高度（桌面之上），贴着杆身推
REPOSE_GREEN_MARGIN = 0.030       # 推动终点距绿区边界的安全余量
REPOSE_SPEED = 0.18
# 备用路径共 5 个子阶段，所有子阶段超时之和必须明显小于 DEADLOCK_STEPS
# 对应的时长（timestep=0.001s 时约 5.7s），否则会先撞上阶段死锁判定。
REPOSE_STAGE_TIMEOUT = 0.9
REPOSE_TIMEOUT = 4.5
LIFT_SPEED = 0.15                 # 验证后的继续抬升速度
MOVE_SPEED = 0.30                 # 搬运速度，降低启停冲击
RELEASE_YAW_TOL = np.radians(35.0)
W_ORIENT = 2.5
# 抓到之后锁死腕部三关节，只用 joint1/2/3 搬运。
# 代价：3 个自由度正好被 3 个位置约束占满，冗余度为 0 ——
# 保持水平、偏航对齐全部失效，螺丝刀朝向由前三关节的位置解唯一决定。
# 抓完之后是否锁死腕部。锁死则零空间为空，偏航完全不可控 —— 螺丝刀入盒的
# 朝向只由抓取朝向和 joint1 转角决定。要求"长边与盒子长边平行"必须解锁。
LOCK_WRIST_AFTER_GRASP = False
SPEED_SCALE = 0.70           # 全局速度系数。慢下来夹持更稳，但每条 episode 更长。
# 抓取时在夹取点中点之下再压这么多。手柄半径 0.0175、底面贴着桌面 0.732，
# 夹取点中点在 0.7495 —— 往下压 8mm 后指垫中心到 0.7415，仍高于手柄底面，
# 不会顶到桌子。压得越深指垫包住手柄的弧长越多，抗滑力矩越大。
GRASP_DEPTH = 0.008
DEADLOCK_STEPS = int(4000 / SPEED_SCALE)   # 慢下来之后超时阈值要同步放宽


# ==============================================================================
# 逆运动学底层（原样搬自 data_collector_v2）
# ==============================================================================
def get_site_jacobian_6d(model, data, site_id, dof_adr):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack((jacp, jacr))[:, dof_adr]


def damped_pinv(J, rho=0.10):
    return J.T @ np.linalg.inv(J @ J.T + rho ** 2 * np.eye(J.shape[0]))


def get_orientation_error(target_mat, current_mat):
    err_mat = target_mat @ current_mat.T
    angle = np.arccos(np.clip((np.trace(err_mat) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([err_mat[2, 1] - err_mat[1, 2],
                     err_mat[0, 2] - err_mat[2, 0],
                     err_mat[1, 0] - err_mat[0, 1]])
    n = np.linalg.norm(axis)
    return np.zeros(3) if n < 1e-6 else (axis / n) * angle


def compute_6d_twist(cur_pos, tgt_pos, cur_mat, tgt_mat, speed_lin):
    v = tgt_pos - cur_pos
    dl = np.linalg.norm(v)
    if dl > 1e-4:
        v = (v / dl) * min(dl * 10.0, speed_lin)
    w_err = get_orientation_error(tgt_mat, cur_mat)
    da = np.linalg.norm(w_err)
    w = (w_err / da) * min(da * 10.0, speed_lin * 4.0) if da > 1e-4 else np.zeros(3)
    return np.concatenate([v, w])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def jac_at_offset(J6, d_world):
    """把 site 处的雅可比搬到"相对 site 偏移 d_world 的刚连点"上。
    v_p = v_s + omega x d = (Jp - skew(d) Jr) qdot
    不做这一步，伺服送到目标点的是 ee_site 小球，而不是实际接触手柄的指端面；
    夹爪一斜，这段偏移就会投影到手柄轴向，导致接触位置偏到细颈上。
    """
    return J6[:3, :] - skew(d_world) @ J6[3:, :]


def control_point_kinematics(data, H, J6):
    """真实接触端面中心的位置、姿态和位置雅可比。

    ee_site 只是 link7 上的标记点；真正与物体交互的是沿工具 z 轴偏移后的两指
    接触端面中心。偏移随末端姿态旋转，因此位置和雅可比必须每个控制步重新计算。
    """
    R = data.site_xmat[H.site_ee].reshape(3, 3)
    d_world = R @ H.tcp_offset
    pos = data.site_xpos[H.site_ee] + d_world
    return pos, R, jac_at_offset(J6, d_world)


def relative_object_pose(data, H, tool_pos, tool_mat):
    """物体在真实夹持中心坐标系下的相对位姿。"""
    obj_pos = data.xpos[H.body_obj]
    obj_mat = data.xmat[H.body_obj].reshape(3, 3)
    return tool_mat.T @ (obj_pos - tool_pos), tool_mat.T @ obj_mat


def grasp_pose_error(reference, current):
    """返回抓取相对位姿的平移误差（m）和旋转误差（deg）。"""
    ref_pos, ref_mat = reference
    cur_pos, cur_mat = current
    trans = float(np.linalg.norm(cur_pos - ref_pos))
    err_mat = cur_mat @ ref_mat.T
    angle = np.arccos(np.clip((np.trace(err_mat) - 1.0) * 0.5, -1.0, 1.0))
    return trans, float(np.degrees(angle))


def jaw_axis_sign_near(model, data, tool_mat):
    """离当前闭合轴更近的那个基准符号。"""
    base = grasp_tool_frame(model, data)[:, 1]
    return -1.0 if float(np.dot(tool_mat[:, 1], base)) < 0.0 else 1.0


def horizontal_jaw_target(model, data, tool_mat, sign=None):
    """与手柄垂直且与桌面平行的夹爪闭合轴。

    夹爪两指对称，闭合轴 +a 和 -a 是等价的抓取姿态，但对应的 joint6 解相差 π。
    joint6 行程是 [0, 4.76] rad 的单边范围：就近的那个符号常常要求它往下限
    0 以外转，于是卡死在 0，残留十几度高低差怎么压都压不下去 —— 这正是
    "远处夹爪不平行"的真实成因，不是奇异也不是增益问题。
    sign 显式给定时用给定符号（调平停滞后翻转到另一个等价解）；
    sign=None 时按就近选取，并且不随伺服过程中的姿态变化来回翻，避免振荡。
    """
    base = grasp_tool_frame(model, data)[:, 1]
    if sign is None:
        sign = jaw_axis_sign_near(model, data, tool_mat)
    return base * sign


def jaw_alignment_error(data, H, tool_mat, target_axis):
    """返回闭合轴夹角（deg）和左右夹爪高度差（m）。"""
    current = tool_mat[:, 1]
    axis_angle = np.degrees(np.arccos(np.clip(np.dot(current, target_axis), -1.0, 1.0)))
    z8 = float(data.xpos[H.finger_bodies[0]][2])
    z9 = float(data.xpos[H.finger_bodies[1]][2])
    return float(axis_angle), abs(z8 - z9)


def pos_jaw_axis_qdot(J6, Jp, v_lin, current_axis, target_axis,
                      k_axis=JAW_LEVEL_K, rho=JAW_LEVEL_RHO):
    """主任务 = 3D位置 + 夹爪闭合轴方向；保留绕闭合轴自转自由度。

    堆叠雅可比是 [Jp; k*J_axis]，k 同时决定姿态任务在阻尼最小二乘里的相对权重。
    k 和 rho 是实测选出来的：同一批 spawn 上量"调平到 4°/2mm 需要多久"，
    k=20/rho=0.05 明显快于原来的 k=4/rho=0.10；按最小奇异值自适应加大阻尼的做法
    实测反而完全不收敛，已弃用。
    """
    current_axis = current_axis / max(np.linalg.norm(current_axis), 1e-9)
    target_axis = target_axis / max(np.linalg.norm(target_axis), 1e-9)
    projector = np.eye(3) - np.outer(current_axis, current_axis)
    J_axis = projector @ J6[3:, :]
    w_des = k_axis * np.cross(current_axis, target_axis)
    J = np.vstack([Jp, k_axis * J_axis])
    e = np.concatenate([v_lin, w_des])
    return damped_pinv(J, rho) @ e


def solve_level_jaw_ik(model, H, tgt_pos, base_frame, q_seed, rng):
    """离线求解"控制点到 tgt_pos 且夹爪闭合轴水平"的关节位形。

    约束是 5 维（位置 3 + 闭合轴方向 2），绕闭合轴的自转留作冗余 —— 与在线
    伺服的主任务一致。把姿态约成完整 6 维（额外要求严格垂直下抓）实测只有
    5/12 个 spawn 有解，放开这一维后是 11/12。

    两侧闭合轴符号都试（joint6 相差约 π），取限位余量最大的解，因为顶在
    joint6=0 的行程端点正是"夹爪不平行"的直接成因。
    返回 (q, sign)；无解返回 (None, None)。
    """
    jr = np.array([model.jnt_range[model.joint(j).id] for j in ARM_JOINTS])
    d = mujoco.MjData(model)
    starts = [np.asarray(q_seed, dtype=float), np.array(Q_INIT, dtype=float),
              np.zeros(6)]
    starts += [rng.uniform(jr[:, 0], jr[:, 1]) for _ in range(IK_RANDOM_STARTS)]
    min_down = np.cos(np.radians(IK_APPROACH_MAX_TILT_DEG))
    axis_tol = np.sin(np.radians(IK_AXIS_TOL_DEG))
    best = None
    for sign in (1.0, -1.0):
        t_axis = base_frame[:, 1] * sign
        for q0 in starts:
            q = np.clip(np.asarray(q0, dtype=float), jr[:, 0], jr[:, 1])
            converged = False
            for _ in range(IK_MAX_ITERS):
                d.qpos[H.arm_qadr] = q
                mujoco.mj_kinematics(model, d)
                mujoco.mj_comPos(model, d)
                J6 = get_site_jacobian_6d(model, d, H.site_ee, H.arm_dof)
                R = d.site_xmat[H.site_ee].reshape(3, 3)
                d_world = R @ H.tcp_offset
                e_p = tgt_pos - (d.site_xpos[H.site_ee] + d_world)
                a = R[:, 1]
                e_w = np.cross(a, t_axis)
                if np.linalg.norm(e_p) < IK_POS_TOL and np.linalg.norm(e_w) < axis_tol:
                    converged = True
                    break
                J = np.vstack([jac_at_offset(J6, d_world),
                               (np.eye(3) - np.outer(a, a)) @ J6[3:, :]])
                e = np.concatenate([np.clip(e_p, -0.05, 0.05),
                                    np.clip(e_w, -0.5, 0.5)])
                q = np.clip(q + damped_pinv(J, IK_DLS_RHO) @ e, jr[:, 0], jr[:, 1])
            if not converged:
                continue
            down = -float(d.site_xmat[H.site_ee].reshape(3, 3)[2, 2])
            if down < min_down:            # 太斜，直下抓会撞桌面或蹭到物体
                continue
            margin = float(np.min(np.minimum(q - jr[:, 0], jr[:, 1] - q)))
            score = margin + 0.5 * down
            if best is None or score > best[0]:
                best = (score, q.copy(), sign, margin)
        if best is not None and best[3] >= IK_GOOD_MARGIN:
            break                          # 余量已经够大，不必再搜另一侧
    return (None, None) if best is None else (best[1], best[2])


def plan_repose_push(model, data, H):
    """规划一次把螺丝刀推向可达中心的直线推动。

    调平不可达的根因是物体落在工作空间边缘：那里冗余度被位置约束吃光，
    闭合轴水平这个方向解不出来。与其丢掉整条 episode，不如闭爪当作实心块，
    贴着桌面把杆身往 REPOSE_ANCHOR_XY 推一段，再重走抓取流程。

    返回 (推动起点 xy, 推动终点 xy)；若物体已接近锚点或可推距离过短则返回 None。
    """
    c = object_center_xy(model, data, H.body_obj)
    d = REPOSE_ANCHOR_XY - c
    n = float(np.linalg.norm(d))
    if n < REPOSE_MIN_PUSH:
        return None
    u = d / n
    end_c = c + u * min(REPOSE_PUSH_DISTANCE, n)
    lo = np.array([GREEN_X_MIN, GREEN_Y_MIN]) + REPOSE_GREEN_MARGIN
    hi = np.array([GREEN_X_MAX, GREEN_Y_MAX]) - REPOSE_GREEN_MARGIN
    end_c = np.clip(end_c, lo, hi)
    travel = float(np.linalg.norm(end_c - c))
    if travel < REPOSE_MIN_PUSH:
        return None
    # 起点必须落在物体水平包络之外，否则一放下去就压在杆身上。
    obj_lo, obj_hi = object_extent_xy(model, data, H.body_obj)
    half = 0.5 * (obj_hi - obj_lo)
    reach = float(abs(half[0] * u[0]) + abs(half[1] * u[1]))
    start = c - u * (reach + REPOSE_PUSH_CLEARANCE)
    # 终点要在接触点之后再多走 travel，否则工具只走到物体后缘就停下，
    # 物体根本没被推动。接触发生在离起点 REPOSE_PUSH_CLEARANCE 处。
    return start, start + u * (REPOSE_PUSH_CLEARANCE + travel)


def bilateral_grasp_contacts(model, data, H):
    """返回左右手指当前是否分别与被抓物体发生接触。"""
    touched = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = int(model.geom_bodyid[contact.geom1])
        b2 = int(model.geom_bodyid[contact.geom2])
        if b1 == H.body_obj and b2 in H.finger_bodies:
            touched.add(b2)
        elif b2 == H.body_obj and b1 in H.finger_bodies:
            touched.add(b1)
    return tuple(body_id in touched for body_id in H.finger_bodies)


def arm3_qdot(J3, v_lin, rho=0.10):
    """只用 joint1/2/3 求解的位置伺服。J3 是 3xN 的位置雅可比，取前三列。

    3 关节对 3 个位置约束是恰定问题，没有冗余度可用来管姿态。
    腕部保持抓取瞬间的角度不动。
    """
    Ja = J3[:, :3]
    qd = np.zeros(6)
    qd[:3] = Ja.T @ np.linalg.inv(Ja @ Ja.T + rho ** 2 * np.eye(3)) @ v_lin
    return qd


def compute_3d_velocity(cur_pos, tgt_pos, speed_lin):
    """所有笛卡尔速度都过这里，统一乘 SPEED_SCALE。
    降速的地方散在十几处，逐个改数字迟早漏掉一处，也没法一键调回去。"""
    v = tgt_pos - cur_pos
    dl = np.linalg.norm(v)
    lim = speed_lin * SPEED_SCALE
    return (v / dl) * min(dl * 10.0 * SPEED_SCALE, lim) if dl > 1e-4 else np.zeros(3)


# ==============================================================================
# 姿态控制：主任务永远是 3D 位置，姿态放进零空间
#
# 速度级伺服的代价是 6 个自由度里只有 3 个被位置占住，剩下 3 个由伪逆的最小范数
# 解随便定 —— 物体朝向变成"算出来的副产品"。视频里螺丝刀被吊成斜 60°、刀尖蹭桌面
# 就是这么来的。放进零空间既能管住姿态，又不会像硬约束那样造成 IK 失败/死锁。
# ==============================================================================
K_LEVEL = 2.5            # 把螺丝刀长轴压回水平的增益
K_YAW = 3.0              # 偏航保持/对齐增益。搬运全程就按住，别等到释放前才拧
K_TOOL = 2.0             # 抓取前对齐夹爪姿态的增益


def pos_yaw_qdot(J6, v_lin, yaw_err_rad, Jp=None, k_yaw=2.5, rho=0.10):
    """主任务 = 3 个位置 + 1 个偏航。6 自由度里仍留 2 个冗余。

    偏航放零空间按不住：从抓取点摆到盒子，底座要转 74°，要保持物体朝向不变
    腕部就得反向补 74°，零空间那点权限做不到，结果一路漂到 +57° ——
    而盒子上方可达偏航是 -90~0 和 +60~+90 两个孤岛，中间 +15~+45 是空洞，
    一旦漂进 +60 那个岛就再也拧不回来了。所以必须进主任务。
    """
    Jp = J6[:3, :] if Jp is None else Jp
    Jyaw = J6[5:6, :]                    # 绕世界 z 的角速度行
    J = np.vstack([Jp, k_yaw * Jyaw])
    e = np.concatenate([v_lin, [k_yaw * yaw_err_rad]])
    return damped_pinv(J, rho) @ e


def yaw_error(h_now, h_target):
    """两个水平方向之间的偏航误差（弧度，带符号，取最短路）。"""
    a = np.arctan2(h_now[1], h_now[0])
    b = np.arctan2(h_target[1], h_target[0])
    return float((b - a + np.pi) % (2 * np.pi) - np.pi)


def nullspace_qdot(J3, Jr, w_des, rho=0.10):
    """把期望角速度 w_des 投影到位置任务的零空间里。

    位置永远优先满足；姿态只用剩下的冗余自由度去争取，争不到就让步。
    """
    Jp_inv = damped_pinv(J3, rho)
    N = np.eye(J3.shape[1]) - Jp_inv @ J3
    dq_sec = damped_pinv(Jr, rho) @ w_des
    return N @ dq_sec


def level_and_yaw_twist(h, target_dir=None):
    """给定物体当前长轴 h（世界系），返回把它摆平、并可选拧到 target_dir 的角速度。"""
    h = h / max(np.linalg.norm(h), 1e-9)
    hl = np.array([h[0], h[1], 0.0])
    n = np.linalg.norm(hl)
    if n < 1e-6:
        return np.zeros(3)
    hl = hl / n
    w = K_LEVEL * np.cross(h, hl)                 # 压回水平
    if target_dir is not None:
        t = np.array([target_dir[0], target_dir[1], 0.0])
        t = t / max(np.linalg.norm(t), 1e-9)
        w = w + K_YAW * np.cross(hl, t)           # 拧到目标偏航
    return w


def box_long_axis(model, data, box_body_id, ref_dir=None):
    """快递盒长边在世界系的水平方向（单位向量）。

    从盒子 body 的姿态读，不硬编码世界 x —— 盒子在 XML 里挪个角度，
    这里自动跟着走。BOX_INNER_HALF=(0.145,0.085)，x 半宽更大，
    所以 body 局部 +x 就是长边。
    ref_dir 非空时挑 ±axis 中与它同向的那个：螺丝刀在夹爪里翻不了 180°，
    要求它翻转等于要求一个解不出来的目标。
    """
    R = data.xmat[box_body_id].reshape(3, 3)
    a = np.array([R[0, 0], R[1, 0], 0.0])
    n = np.linalg.norm(a)
    a = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else a / n
    if ref_dir is not None and np.dot(a, np.asarray(ref_dir)[:3]) < 0:
        a = -a
    return a


def fit_shift_xy(model, data, H):
    """把物体当前的水平包络整体塞进盒内腔所需的平移量 (dx,dy)，以及能否塞下。

    规划约束的核心：光把"包络中心"对准盒心不够 —— 偏航一歪，长的那一头就会
    先顶出去。这里直接算包络的四条边离内腔边界还差多少，需要往哪边挪多少。
    返回 (shift, fits)；fits=False 表示当前偏航下无论怎么挪都装不下，
    必须继续拧偏航。
    """
    lo, hi = object_extent_xy(model, data, H.body_obj)
    box = data.xpos[H.body_box][:2]
    inner_lo = box - BOX_INNER_HALF + FIT_EDGE_MARGIN
    inner_hi = box + BOX_INNER_HALF - FIT_EDGE_MARGIN
    half = 0.5 * (hi - lo)
    room = (inner_hi - inner_lo) * 0.5
    fits = bool((half <= room).all())
    ctr = 0.5 * (lo + hi)
    # 包络中心该落在哪：既要居中，又不能让任何一边越界
    tgt = np.clip(ctr, inner_lo + half, inner_hi - half) if fits else 0.5 * (inner_lo + inner_hi)
    return tgt - ctr, fits


def object_top_z(model, data, body_id):
    """物体最高点的世界高度。"""
    hi = -1e9
    for g in range(model.body_geomadr[body_id],
                   model.body_geomadr[body_id] + model.body_geomnum[body_id]):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        c, hs = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = data.geom_xpos[g] + R @ (c + np.array([sx, sy, sz]) * hs)
                    hi = max(hi, float(w[2]))
    return hi


def dropped_into_box(model, data, H):
    """螺丝刀是否已经落进盒子范围（水平上）。

    掉进盒子之后再张开夹爪回去重抓，等于让机械臂伸进盒腔里捞 —— 指尖会撞盒壁，
    而且这种"从盒里往外捞"的轨迹根本不是我们要教给策略的动作，
    混进数据集是纯污染。所以这种情况直接作废整条重来。
    """
    ext = object_extent_xy(model, data, H.body_obj)
    ctr = 0.5 * (ext[0] + ext[1])
    box_xy = data.xpos[H.body_box][:2]
    return bool((np.abs(ctr - box_xy) < BOX_INNER_HALF + 0.03).all())


def object_extent_xy(model, data, body_id):
    """物体在水平面上的包络角点 (min_xy, max_xy)。用来判"整根都在盒里"。"""
    lo = np.array([1e9, 1e9]); hi = np.array([-1e9, -1e9])
    for g in range(model.body_geomadr[body_id],
                   model.body_geomadr[body_id] + model.body_geomnum[body_id]):
        c = model.geom_aabb[g][:3]; hs = model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3); o = data.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = (o + R @ (c + np.array([sx, sy, sz]) * hs))[:2]
                    lo = np.minimum(lo, w); hi = np.maximum(hi, w)
    return lo, hi


def object_center_xy(model, data, body_id):
    """物体水平包络的中心。

    螺丝刀的手柄在一头、刀杆伸出去 0.21m，所以"把手柄放到盒心"会让刀杆
    整根挂在盒外。要对准盒心的是整根的包络中心，不是抓持点。
    """
    lo, hi = object_extent_xy(model, data, body_id)
    return 0.5 * (lo + hi)


def object_bottom_z(model, data, body_id):
    """被搬物体的最低点世界 z。

    [C2] 要保证"高于盒子最高点"的是**物体**，不是 TCP。物体斜挂在夹爪下方时
    刀尖能比 TCP 低十几厘米 —— 只夹 TCP 的话走廊约束等于没做。
    """
    lo = 1e9
    for g in range(model.body_geomadr[body_id],
                   model.body_geomadr[body_id] + model.body_geomnum[body_id]):
        c = model.geom_aabb[g][:3]
        hs = model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3)
        o = data.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = o + R @ (c + np.array([sx, sy, sz]) * hs)
                    lo = min(lo, float(w[2]))
    return lo


# ==============================================================================
# [C1] 抓取几何：夹取点与杆身方向，从碰撞胶囊 sd_handle 现场反算
# ==============================================================================
def grasp_geometry(model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, HANDLE_GEOM)
    if gid < 0:
        raise RuntimeError(f"场景里找不到 geom '{HANDLE_GEOM}'")
    c = data.geom_xpos[gid].copy()
    R = data.geom_xmat[gid].reshape(3, 3)
    h = R[:, 2] / np.linalg.norm(R[:, 2])
    j = np.cross([0.0, 0.0, 1.0], h)
    n = np.linalg.norm(j)
    j = np.array([0.0, 1.0, 0.0]) if n < 1e-6 else j / n
    r = float(model.geom_size[gid][0])
    return c + r * j, c - r * j, h


def grasp_tool_frame(model, data):
    """抓取时期望的工具姿态：手指朝下、两指连线 ⟂ 手柄。
    只作为零空间里的软目标，不做硬约束 —— 你要求过不强制垂直下抓，
    而且这台臂指尖朝下时 TCP 高度上限只有 ~0.789，硬拧会一直较劲。"""
    _, _, h = grasp_geometry(model, data)
    a = np.array([0.0, 0.0, -1.0])
    c = np.cross(a, h)
    n = np.linalg.norm(c)
    c = np.array([0.0, 1.0, 0.0]) if n < 1e-6 else c / n
    return np.column_stack([np.cross(c, a), c, a])   # [x, y, z] = [.., close, approach]


def jaw_mid_now(model, data):
    """[C1] 两夹取点连线中点 —— IK/伺服的目标点。"""
    ja, jb, _ = grasp_geometry(model, data)
    return 0.5 * (ja + jb)


def object_axis_now(model, data):
    """螺丝刀长轴在世界系的方向（单位向量）。"""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, HANDLE_GEOM)
    R = data.geom_xmat[gid].reshape(3, 3)
    return R[:, 2] / np.linalg.norm(R[:, 2])


def object_lowest_z(model, data, H):
    """被搬物体的最低点世界高度。

    [C2] 走廊约束的对象必须是**物体**，不是 TCP。物体斜挂在夹爪下方时刀尖能比
    TCP 低十几厘米 —— 只夹 TCP 的话刀尖照样贴着桌面走，走廊等于没约束。
    """
    lo = 1e9
    b = H.body_obj
    for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        Rg = data.geom_xmat[g].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = data.geom_xpos[g] + Rg @ (c + np.array([sx, sy, sz]) * h)
                    lo = min(lo, float(w[2]))
    return lo


def desired_axis(a_cur, align_yaw):
    """目标长轴方向：先压回水平；align_yaw 时再把偏航收进装箱安全锥。

    盒内腔 y 向半宽 0.085，螺丝刀半长 0.123 -> |0.123*sin(phi)| < 0.085 即
    |phi| < 43 度。留裕度取 35 度。不收的话刀杆会整根横跨盒壁挂在外面。
    """
    t = np.array([a_cur[0], a_cur[1], 0.0])
    n = np.linalg.norm(t)
    if n < 1e-9:
        return None
    t = t / n
    if not align_yaw:
        return t
    phi = np.arctan2(t[1], t[0])
    psi = ((phi + np.pi / 2) % np.pi) - np.pi / 2
    psi_t = float(np.clip(psi, -RELEASE_YAW_TOL, RELEASE_YAW_TOL))
    dd = psi_t - psi
    c, sn = np.cos(dd), np.sin(dd)
    return np.array([c * t[0] - sn * t[1], sn * t[0] + c * t[1], 0.0])


def shaft_yaw_now(model, data):
    """杆身水平偏航。原脚本用 place_obj 时记下的 yaw，这里现场从胶囊轴读 ——
    tidy_B 里螺丝刀长轴是 local X 不是 local Y，任何基于固定轴的推算都是错的。"""
    _, _, h = grasp_geometry(model, data)
    return float(np.arctan2(h[1], h[0]))


# ==============================================================================
# 场景加载与复位
# ==============================================================================
class Handles:
    def __init__(self, model):
        M = mujoco.mjtObj
        self.site_ee = model.site(EE_SITE).id
        self.body_obj = model.body(OBJECT_BODY).id
        self.body_box = model.body(STORAGE_BOX_BODY).id
        self.obj_qadr = model.joint(OBJECT_FREEJOINT).qposadr[0]
        self.arm_qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.grip_qadr = np.array([model.joint(j).qposadr[0] for j in rspec.GRIP_JOINTS])
        self.arm_act = np.array([model.actuator(a).id for a in ARM_ACTUATORS])
        self.j8 = model.actuator("Joint8").id
        self.j9 = model.actuator("Joint9").id
        self.finger_bodies = (model.body("link8").id, model.body("link9").id)
        self.cam_fixed = mujoco.mj_name2id(model, M.mjOBJ_CAMERA, rspec.CAM_FIXED)
        self.cam_wrist = mujoco.mj_name2id(model, M.mjOBJ_CAMERA, rspec.CAM_WRIST)
        self.tcp_offset = self._pad_offset(model)

    def _pad_offset(self, model):
        """ee_site -> 两指接触端面中心（工具系）。

        ee_site 挂在 link7 的 (0,0,0.08)，而 link8/link9 的 body 原点在
        (0,±0.023831,0.016)。手指碰撞网格沿工具 z 轴跨过 ee_site，但实际接触
        桌面物体的是朝向物体的局部 +z 端面，不是整根手指 AABB 的几何中心。
        取两指碰撞几何在工具系下的最大 z，作为闭合轴中点处的接触端面。
        """
        d = mujoco.MjData(model)
        d.qpos[self.arm_qadr] = Q_INIT
        mujoco.mj_forward(model, d)
        R = d.site_xmat[self.site_ee].reshape(3, 3)
        p = d.site_xpos[self.site_ee]
        zs = []
        for bn in ("link8", "link9"):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if b < 0:
                continue
            for g in range(model.body_geomadr[b],
                           model.body_geomadr[b] + model.body_geomnum[b]):
                if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                    continue
                c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
                Rg = d.geom_xmat[g].reshape(3, 3)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            w = d.geom_xpos[g] + Rg @ (c + np.array([sx, sy, sz]) * h)
                            zs.append(float((R.T @ (w - p))[2]))
        if not zs:
            return np.zeros(3)
        return np.array([0.0, 0.0, max(zs)])


def configure_task_gripper(model, verbose=True):
    """只提高本采集任务的夹爪位置伺服增益，不修改共享机器人 XML。"""
    configured = []
    for name in ("Joint8", "Joint9"):
        aid = model.actuator(name).id
        old_kp = float(model.actuator_gainprm[aid, 0])
        old_kv = -float(model.actuator_biasprm[aid, 2])
        if old_kp <= 0:
            raise RuntimeError(f"夹爪执行器 {name} 不是有效的位置伺服，kp={old_kp}")
        scale = np.sqrt(TASK_GRIPPER_KP / old_kp)
        model.actuator_gainprm[aid, 0] = TASK_GRIPPER_KP
        model.actuator_biasprm[aid, 1] = -TASK_GRIPPER_KP
        model.actuator_biasprm[aid, 2] = -old_kv * scale
        configured.append((name, old_kp, -model.actuator_biasprm[aid, 2]))
    if verbose:
        detail = ", ".join(
            f"{name}: kp {old_kp:g}->{TASK_GRIPPER_KP:g}, kv={kv:.3f}"
            for name, old_kp, kv in configured)
        print(f"🤏 任务夹爪增力 | {detail}")


def load_scene_tidyB(verbose=True):
    """rspec.load_scene() 走的是旧场景路径，且 assert_contract 会去查
    real_screwdriver / plasticbox / dynamic_pillar —— tidy_B 里都不存在。
    这里自己走等价流程，但相机镜像仍调 rspec 的那一个，保证与部署端逐字一致。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    configure_task_gripper(model, verbose=verbose)
    rspec.apply_camera_overrides(model, verbose=verbose)   # 必须在建 MjData 之前
    data = mujoco.MjData(model)

    bad = []
    if abs(model.opt.timestep - rspec.SIM_TIMESTEP) > 1e-12:
        bad.append(f"timestep {model.opt.timestep} != {rspec.SIM_TIMESTEP}")
    for nm in list(ARM_JOINTS) + list(rspec.GRIP_JOINTS):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) < 0:
            bad.append(f"缺关节 {nm}")
    for nm in list(ARM_ACTUATORS) + list(rspec.GRIP_ACTUATORS):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm) < 0:
            bad.append(f"缺执行器 {nm}")
    for nm, ot in [(EE_SITE, mujoco.mjtObj.mjOBJ_SITE),
                   (OBJECT_BODY, mujoco.mjtObj.mjOBJ_BODY),
                   (STORAGE_BOX_BODY, mujoco.mjtObj.mjOBJ_BODY),
                   (HANDLE_GEOM, mujoco.mjtObj.mjOBJ_GEOM),
                   (rspec.CAM_FIXED, mujoco.mjtObj.mjOBJ_CAMERA),
                   (rspec.CAM_WRIST, mujoco.mjtObj.mjOBJ_CAMERA)]:
        if mujoco.mj_name2id(model, ot, nm) < 0:
            bad.append(f"场景里找不到 {nm}")
    if bad:
        raise RuntimeError("❌ 契约自检未通过：\n  - " + "\n  - ".join(bad))
    if verbose:
        print(f"✅ 契约自检通过 | {FPS}Hz × {STEPS_PER_RECORD} 子步 | "
              f"夹爪 ±{rspec.GRIP_LIMIT} 反号")
        print(f"   [C2] 走廊 ({Z_BOX_WALL_TOP}, {Z_FLASHLIGHT_TOP}) "
              f"夹紧 ({Z_CORRIDOR_LO:.3f}, {Z_CORRIDOR_HI:.3f})")
    return model, data


def reset_scene(model, data, H, rng):
    """复位到起手状态并在绿区随机摆放螺丝刀（位置 + 绕世界 z 轴 ±30° 偏航）。"""
    mujoco.mj_resetData(model, data)
    data.qpos[H.arm_qadr] = Q_INIT
    data.qpos[H.grip_qadr] = GRIP_OPEN
    data.ctrl[H.arm_act] = Q_INIT
    data.ctrl[H.j8], data.ctrl[H.j9] = GRIP_OPEN

    q_base = np.array(model.body(OBJECT_BODY).quat, dtype=float)
    L_HALF = 0.13
    sampled = None
    for _ in range(200):
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
        # 绕世界 z 轴的偏航随机。左乘 qz 才是"绕世界 z 转"，右乘是绕物体自身
        # 局部轴转 —— 螺丝刀的 XML 初始 quat 不是单位阵，两者结果不同。
        dyaw = rng.uniform(-YAW_JITTER, YAW_JITTER)
        qz = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, qz, q_base)
        data.qpos[H.obj_qadr + 3:H.obj_qadr + 7] = qq
        data.qpos[H.obj_qadr:H.obj_qadr + 3] = [x, y, SPAWN_DROP_Z]
        mujoco.mj_forward(model, data)
        yaw = shaft_yaw_now(model, data)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        p1, p2 = np.array([x, y]) + L_HALF * u, np.array([x, y]) - L_HALF * u
        if (GREEN_X_MIN <= p1[0] <= GREEN_X_MAX and GREEN_X_MIN <= p2[0] <= GREEN_X_MAX
                and GREEN_Y_MIN <= p1[1] <= GREEN_Y_MAX
                and GREEN_Y_MIN <= p2[1] <= GREEN_Y_MAX):
            sampled = {
                "position": [float(x), float(y), float(SPAWN_DROP_Z)],
                "quaternion": qq.tolist(),
                "yaw_jitter_deg": float(np.degrees(dyaw)),
            }
            break
    if sampled is None:
        raise RuntimeError("拒绝采样 200 次仍未找到绿区内的合法螺丝刀位姿")
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(300):                # 落定
        mujoco.mj_step(model, data)
    sampled["settled_position"] = data.xpos[H.body_obj].tolist()
    sampled["settled_quaternion"] = data.qpos[H.obj_qadr + 3:H.obj_qadr + 7].tolist()
    return sampled


def get_drop_point(model, data, H, rng):
    """盒内投放点。原脚本用 ±0.03 抖动，这里按 delivery_box 的内腔限幅。"""
    p = data.xpos[H.body_box].copy()
    # 物体中心的可放范围要分轴算：螺丝刀半长 0.123，释放时偏航被收进 ±35°，
    # 所以 x 向半包络 0.123*cos35=0.101、y 向 0.123*sin35=0.071。
    # 原来统一减 0.10，y 方向算出负数（0.085-0.10=-0.015），rng.uniform 直接抛错。
    half = 0.123 * np.array([np.cos(RELEASE_YAW_TOL), np.sin(RELEASE_YAW_TOL)])
    lim = np.maximum(BOX_INNER_HALF - half - 0.005, 0.0)
    p[0] += rng.uniform(-lim[0], lim[0]) if lim[0] > 0 else 0.0
    p[1] += rng.uniform(-lim[1], lim[1]) if lim[1] > 0 else 0.0
    p[2] = Z_BOX_WALL_TOP + 0.028       # 略高于盒沿释放，避免指尖蹭壁
    return p


def episode_is_complete(ep_dir):
    """识别原子写盘的新 episode，并兼容修改前已完整写出的旧 episode。"""
    ep_dir = Path(ep_dir)
    fixed = ep_dir / "cam_fixed"
    wrist = ep_dir / "cam_wrist"
    core_complete = (
        (ep_dir / "joint_data.npz").is_file()
        and (ep_dir / "instruction.txt").is_file()
        and fixed.is_dir() and any(fixed.glob("*.jpg"))
        and wrist.is_dir() and any(wrist.glob("*.jpg"))
    )
    marker = ep_dir / "complete.json"
    if marker.is_file():
        try:
            marked = bool(json.loads(marker.read_text(encoding="utf-8")).get("complete"))
            return marked and core_complete
        except (OSError, ValueError, TypeError):
            return False
    # 历史数据没有 complete.json；只有核心文件和两路非空图像都存在才视为完整。
    return core_complete


def write_episode_atomic(out_dir, slot, imgs_f, imgs_w, ep_qpos, ep_act,
                         instruction, metadata):
    """先写同盘临时目录，全部成功后再原子发布为 ep_N。"""
    lengths = {len(imgs_f), len(imgs_w), len(ep_qpos), len(ep_act)}
    if len(lengths) != 1 or not ep_qpos:
        raise ValueError(
            "episode 各数据流长度必须相同且非空："
            f"fixed={len(imgs_f)} wrist={len(imgs_w)} "
            f"qpos={len(ep_qpos)} actions={len(ep_act)}")
    out_dir = Path(out_dir)
    ep_dir = out_dir / f"ep_{slot}"
    if ep_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有 episode 目录：{ep_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f".ep_{slot}.tmp-", dir=out_dir))
    try:
        cf, cw = tmp_dir / "cam_fixed", tmp_dir / "cam_wrist"
        cf.mkdir()
        cw.mkdir()
        for i, (a, b) in enumerate(zip(imgs_f, imgs_w)):
            ok_f = cv2.imwrite(str(cf / f"{i:03d}.jpg"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
            ok_w = cv2.imwrite(str(cw / f"{i:03d}.jpg"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
            if not ok_f or not ok_w:
                raise OSError(f"第 {i} 帧 JPEG 写入失败")
        np.savez_compressed(
            tmp_dir / "joint_data.npz",
            qpos=np.asarray(ep_qpos, dtype=np.float32),
            actions=np.asarray(ep_act, dtype=np.float32),
        )
        (tmp_dir / "instruction.txt").write_text(instruction, encoding="utf-8")
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (tmp_dir / "complete.json").write_text(
            json.dumps({"complete": True, "episode": slot}) + "\n", encoding="utf-8")
        os.replace(tmp_dir, ep_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return ep_dir


# ==============================================================================
# 采集主循环
# ==============================================================================
def collect(args):
    model, data = load_scene_tidyB()
    H = Handles(model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = rspec.make_renderer(model)
    vopt = rspec.make_scene_option()        # 两端必须逐字一致，别自己写一份

    done, occupied = set(), set()
    for ep_dir in out_dir.iterdir():
        if not ep_dir.is_dir() or not ep_dir.name.startswith("ep_"):
            continue
        suffix = ep_dir.name.removeprefix("ep_")
        if not suffix.isdigit():
            continue
        slot = int(suffix)
        occupied.add(slot)
        if episode_is_complete(ep_dir):
            done.add(slot)
    start_have = len(done)
    if start_have >= args.target:
        print(f"✅ 已有 {start_have} 条，达到目标 {args.target}。")
        return

    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed 必须是非负整数")
    global_seed = (int(args.seed) if args.seed is not None else
                   int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]))
    print(f"🌱 全局采样 seed: {global_seed}")
    viewer = mujoco.viewer.launch_passive(model, data) if not args.headless else None
    attempts = 0
    attempts_by_slot = {}
    reasons = {}

    while len(done) < args.target:
        if viewer is not None and not viewer.is_running():
            break
        slot = 0
        while slot in occupied:
            slot += 1
        attempt_index = attempts_by_slot.get(slot, 0)
        attempts_by_slot[slot] = attempt_index + 1
        seed_seq = np.random.SeedSequence(global_seed, spawn_key=(slot, attempt_index))
        attempt_seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(attempt_seed)
        attempts += 1
        reset_info = reset_scene(model, data, H, rng)

        # home 位姿在 GO_ZERO 到位之后才记录（见下），这样收臂也回到全 0，
        # 一条 episode 的首尾状态一致 —— 对 VLA 来说首尾不一致会让策略学到
        # "同一个观测对应两种动作"。
        home_point = None
        home_qpos = None

        phase, phase_steps, wait_steps, step_counter = "GO_ZERO", 0, 0, 0
        success, fail_why = False, None
        imgs_f, imgs_w, ep_qpos, ep_act = [], [], [], []
        intentional_miss = rng.random() < MISS_PROB
        has_retried = False
        grasp_retries = 0
        drop_point = get_drop_point(model, data, H, rng)
        grip_ref = None
        hold_dir = np.array([1.0, 0.0, 0.0])
        carry_z, carry_bot = [], []
        max_slip_translation = 0.0
        max_slip_rotation_deg = 0.0
        max_gripper_force_n = 0.0
        bilateral_contact_seen = False
        grasp_verified = False
        verify_stable_steps = 0
        prelift_start = None
        prelift_obj_z = None
        prealign_stable_steps = 0
        gripper_aligned = False
        pregrasp_axis_error_deg = None
        pregrasp_jaw_height_diff_m = None
        prealign_best_axis_err = None
        prealign_stall_steps = 0
        prealign_unreachable = False
        prealign_last_axis_err_deg = None
        prealign_last_jaw_height_diff_m = None
        jaw_axis_sign = None      # 由 IK 解确定；停滞时可翻到等价的另一侧
        jaw_axis_flipped = False
        ik_target_q = None
        ik_move_obj_xy = None
        ik_solved = False
        repose_count = 0
        repose_stage = None
        repose_stage_steps = 0
        repose_plan = None
        phase_step_counts = {}
        t0 = time.time()

        while True:
            phase_steps += 1
            phase_step_counts[phase] = phase_step_counts.get(phase, 0) + 1
            if phase_steps > DEADLOCK_STEPS:
                fail_why = f"[{phase}] 阶段死锁"
                break

            jaw_mid = jaw_mid_now(model, data)
            J6 = get_site_jacobian_6d(model, data, H.site_ee, H.arm_dof)
            tool_pos, tool_mat, J3 = control_point_kinematics(data, H, J6)
            obj_bot = object_bottom_z(model, data, H.body_obj)
            max_gripper_force_n = max(
                max_gripper_force_n,
                float(np.max(np.abs(data.actuator_force[[H.j8, H.j9]]))),
            )

            if data.xpos[H.body_obj][2] < Z_TABLE_TOP - 0.05:
                fail_why = "螺丝刀掉到桌下"
                break

            miss = np.array([0.0, 0.06, 0.0]) if (intentional_miss and not has_retried) \
                else np.zeros(3)
            hover_point = np.array([jaw_mid[0], jaw_mid[1], SAFE_Z]) + miss
            prealign_point = jaw_mid + miss + np.array([0.0, 0.0, PREALIGN_CLEARANCE])
            # 往下压 GRASP_DEPTH：指垫包住手柄更多，抗滑。miss 是 5% 概率的
            # 故意抓偏（纠错示范），不受这个影响。
            grasp_point = jaw_mid + miss - np.array([0.0, 0.0, GRASP_DEPTH])

            q_dot = np.zeros(6)
            grip = GRIP_OPEN

            # ---- HOVER：移到抓取点正上方。只约束位置（不强制垂直下抓）----
            # ---- GO_ZERO：先从起手位姿 Q_INIT 走到关节全 0，再去抓 ----
            # 全 0 时 ee_site 在 (0.554, -0.006, 1.110)，机械臂近乎直立。
            # 此时空载未夹持，不受 [C2] 走廊约束；而且这一段是录进数据集的，
            # 策略会学到"先立直再下探"的起手动作。
            if phase == "GO_ZERO":
                grip = GRIP_OPEN
                q_err = Q_ZERO - data.qpos[H.arm_qadr]
                dq = float(np.linalg.norm(q_err))
                if dq < 0.06 or (phase_steps > int(800/SPEED_SCALE) and dq < 0.15):
                    home_qpos = data.qpos[H.arm_qadr].copy()
                    home_point = tool_pos.copy()
                    phase, phase_steps = "HOVER", 0
                else:
                    q_dot = (q_err / dq) * max(min(dq / 0.5 * 2.5, 2.5), 0.6) * SPEED_SCALE

            elif phase == "HOVER":
                dist = np.linalg.norm(hover_point - tool_pos)
                if dist < 0.02 or (phase_steps > int(400/SPEED_SCALE) and dist < 0.05):
                    phase, phase_steps = "IK_PREALIGN", 0
                    ik_target_q = None
                else:
                    q_dot = damped_pinv(J3) @ compute_3d_velocity(tool_pos, hover_point, 0.45)
                    w = K_TOOL * get_orientation_error(grasp_tool_frame(model, data), tool_mat)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], w)

            # ---- IK_PREALIGN：离线解出夹爪水平的关节位形，再关节空间走过去 ----
            # 目标点取在手柄中心上方 IK_PREALIGN_CLEARANCE（≈悬停高度），
            # 这段转移离物体还远，不会蹭到螺丝刀；余下的几厘米交给
            # PREALIGN_GRIPPER 带着轴约束伺服下去 —— 那时已经在正确的分支上。
            elif phase == "IK_PREALIGN":
                grip = GRIP_OPEN
                if ik_target_q is None:
                    ik_q, ik_sign = solve_level_jaw_ik(
                        model, H,
                        jaw_mid + miss + np.array([0.0, 0.0, IK_PREALIGN_CLEARANCE]),
                        grasp_tool_frame(model, data),
                        data.qpos[H.arm_qadr], rng)
                    if ik_q is None:
                        # 这个位形下不存在夹爪水平的关节解 -> 走预调整备用路径
                        prealign_unreachable = True
                        if repose_count >= MAX_REPOSE_RETRIES:
                            fail_why = "无夹爪水平的关节解，且预调整已用尽"
                            break
                        phase, phase_steps = "REPOSE_OBJECT", 0
                        repose_stage, repose_stage_steps, repose_plan = "LIFT_OUT", 0, None
                        continue
                    ik_target_q, jaw_axis_sign, ik_solved = ik_q, ik_sign, True
                    ik_move_obj_xy = object_center_xy(model, data, H.body_obj)
                # 转移途中把螺丝刀碰跑了就作废：抓取点已经不是解算时那个了。
                if (ik_move_obj_xy is not None
                        and np.linalg.norm(object_center_xy(model, data, H.body_obj)
                                           - ik_move_obj_xy) > IK_OBJECT_DISTURB_TOL):
                    fail_why = "IK 预对齐转移途中碰动了螺丝刀"
                    break
                q_err = ik_target_q - data.qpos[H.arm_qadr]
                dq = float(np.linalg.norm(q_err))
                elapsed = phase_steps * model.opt.timestep
                if dq < IK_MOVE_TOL or (elapsed >= IK_MOVE_TIMEOUT
                                        and dq < 2 * IK_MOVE_TOL):
                    phase, phase_steps = "PREALIGN_GRIPPER", 0
                    prealign_stable_steps = 0
                    prealign_best_axis_err = None
                    prealign_stall_steps = 0
                elif elapsed >= 1.5 * IK_MOVE_TIMEOUT:
                    fail_why = f"IK 预对齐关节到位失败 (残差 {dq:.3f} rad)"
                    break
                else:
                    q_dot = (q_err / dq) * max(min(dq / 0.5 * 2.5, 2.5), 0.6) * SPEED_SCALE

            # ---- PREALIGN_GRIPPER：位置保持在悬停点，先把左右夹爪调到等高 ----
            elif phase == "PREALIGN_GRIPPER":
                if jaw_axis_sign is None:
                    jaw_axis_sign = jaw_axis_sign_near(model, data, tool_mat)
                target_axis = horizontal_jaw_target(
                    model, data, tool_mat, jaw_axis_sign)
                axis_err_deg, jaw_height_diff = jaw_alignment_error(
                    data, H, tool_mat, target_axis)
                dist = np.linalg.norm(prealign_point - tool_pos)
                v = compute_3d_velocity(tool_pos, prealign_point, 0.12)
                # 位置到位后冻结位置任务：只留很小的保位增益防漂，
                # 剩下的自由度全部交给闭合轴调平。
                frozen = dist <= PREALIGN_FREEZE_DIST
                if frozen:
                    v = v * PREALIGN_FREEZE_GAIN
                q_dot = pos_jaw_axis_qdot(
                    J6, J3, v, tool_mat[:, 1], target_axis)

                aligned_now = (
                    axis_err_deg <= JAW_AXIS_TOL_DEG
                    and jaw_height_diff <= JAW_HEIGHT_TOL
                    and dist <= PREALIGN_FREEZE_DIST
                )
                prealign_stable_steps = prealign_stable_steps + 1 if aligned_now else 0
                # 收敛停滞检测：位置已到位、残差仍超容差，却连续这么久压不下去，
                # 说明这个位形上"水平"根本解不出来，不必空等满超时。
                # 残差取两项对各自容差的归一化最大值，避免只盯轴误差时
                # 高度差还在慢慢收敛却被判死。
                residual = max(axis_err_deg / max(JAW_AXIS_TOL_DEG, 1e-9),
                               jaw_height_diff / max(JAW_HEIGHT_TOL, 1e-9))
                if not frozen or residual <= 1.0:
                    prealign_stall_steps = 0
                    prealign_best_axis_err = residual
                elif (prealign_best_axis_err is None
                        or residual < prealign_best_axis_err - PREALIGN_STALL_IMPROVE_RATIO):
                    prealign_best_axis_err = residual
                    prealign_stall_steps = 0
                else:
                    prealign_stall_steps += 1
                stalled = prealign_stall_steps >= max(
                    1, int(PREALIGN_STALL_WINDOW / model.opt.timestep))

                if prealign_stable_steps >= max(
                        1, int(PREALIGN_SETTLE_TIME / model.opt.timestep)):
                    gripper_aligned = True
                    phase, phase_steps = "DESCEND", 0
                elif stalled or phase_steps * model.opt.timestep >= PREALIGN_TIMEOUT:
                    # 第一优先补救：翻到等价的另一侧闭合轴解（joint6 差 π，
                    # 那一侧离行程端点远得多）。只在**真停滞**时翻：残差还在
                    # 稳步下降却因超时翻过去，会白白从 180° 重新收敛一遍。
                    if stalled and not jaw_axis_flipped:
                        jaw_axis_sign = -jaw_axis_sign
                        jaw_axis_flipped = True
                        prealign_stall_steps = 0
                        prealign_best_axis_err = None
                        prealign_stable_steps = 0
                        phase_steps = 0
                        continue
                    prealign_unreachable = True
                    prealign_last_axis_err_deg = axis_err_deg
                    prealign_last_jaw_height_diff_m = jaw_height_diff
                    if repose_count >= MAX_REPOSE_RETRIES:
                        fail_why = (
                            f"夹爪调平失败 (轴误差 {axis_err_deg:.1f}°, "
                            f"高度差 {jaw_height_diff*1000:.1f}mm, "
                            f"已预调整 {repose_count} 次)")
                        break
                    phase, phase_steps = "REPOSE_OBJECT", 0
                    repose_stage, repose_stage_steps, repose_plan = "LIFT_OUT", 0, None

            # ---- REPOSE_OBJECT：调平不可达时的备用路径 ----
            # 闭爪当实心块，贴桌把螺丝刀往可达中心推一段，然后回 HOVER 重走抓取。
            # 这段动作会完整录进 episode，metadata 里用 repose_used 标记以便事后筛选。
            elif phase == "REPOSE_OBJECT":
                grip = GRIP_CLOSE
                repose_stage_steps += 1
                if repose_plan is None:
                    repose_plan = plan_repose_push(model, data, H)
                    if repose_plan is None:
                        fail_why = (
                            "夹爪调平失败且物体已在可达中心附近，无可用预调整方向")
                        break
                if phase_steps * model.opt.timestep >= REPOSE_TIMEOUT:
                    fail_why = "预调整螺丝刀超时"
                    break
                push_start, push_end = repose_plan
                push_z = Z_TABLE_TOP + REPOSE_PUSH_Z_OFFSET
                stage_timeout = repose_stage_steps * model.opt.timestep >= REPOSE_STAGE_TIMEOUT

                if repose_stage == "LIFT_OUT":
                    tgt = np.array([tool_pos[0], tool_pos[1], SAFE_Z])
                    if tool_pos[2] >= SAFE_Z - 0.01 or stage_timeout:
                        repose_stage, repose_stage_steps = "APPROACH", 0
                elif repose_stage == "APPROACH":
                    tgt = np.array([push_start[0], push_start[1], SAFE_Z])
                    if np.linalg.norm(tgt[:2] - tool_pos[:2]) < 0.012 or stage_timeout:
                        repose_stage, repose_stage_steps = "DOWN", 0
                elif repose_stage == "DOWN":
                    tgt = np.array([push_start[0], push_start[1], push_z])
                    if abs(tool_pos[2] - push_z) < 0.006 or stage_timeout:
                        repose_stage, repose_stage_steps = "PUSH", 0
                elif repose_stage == "PUSH":
                    tgt = np.array([push_end[0], push_end[1], push_z])
                    if np.linalg.norm(tgt[:2] - tool_pos[:2]) < 0.008 or stage_timeout:
                        repose_stage, repose_stage_steps = "RETREAT", 0
                else:                                   # RETREAT
                    tgt = np.array([tool_pos[0], tool_pos[1], SAFE_Z])
                    if tool_pos[2] >= SAFE_Z - 0.01 or stage_timeout:
                        repose_count += 1
                        prealign_stable_steps = 0
                        prealign_best_axis_err = None
                        prealign_stall_steps = 0
                        jaw_axis_sign, jaw_axis_flipped = None, False
                        ik_target_q, ik_move_obj_xy = None, None
                        gripper_aligned = False
                        repose_stage, repose_plan = None, None
                        phase, phase_steps, wait_steps = "HOVER", 0, 0

                if phase == "REPOSE_OBJECT":
                    v = compute_3d_velocity(tool_pos, tgt, REPOSE_SPEED)
                    q_dot = damped_pinv(J3) @ v
                    w = K_TOOL * get_orientation_error(
                        grasp_tool_frame(model, data), tool_mat)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], w)

            # ---- DESCEND：保持闭合轴水平，下到夹取点中点 ----
            elif phase == "DESCEND":
                dist = np.linalg.norm(grasp_point - tool_pos)
                target_axis = horizontal_jaw_target(
                    model, data, tool_mat, jaw_axis_sign)
                axis_err_deg, jaw_height_diff = jaw_alignment_error(
                    data, H, tool_mat, target_axis)
                aligned_now = (
                    axis_err_deg <= JAW_AXIS_TOL_DEG
                    and jaw_height_diff <= JAW_HEIGHT_TOL
                )
                # 旧兜底允许 15mm 误差，实测会横向偏 12mm、垂直浅 9mm 就闭合，
                # 只夹到手柄边缘。兜底只能略放宽，不能大于 GRASP_DEPTH 本身。
                arrived = dist < 0.004 or (
                    phase_steps > int(1000/SPEED_SCALE) and dist < 0.006)
                if arrived and aligned_now:
                    pregrasp_axis_error_deg = axis_err_deg
                    pregrasp_jaw_height_diff_m = jaw_height_diff
                    phase, phase_steps = "GRASP", 0
                elif phase_steps * model.opt.timestep >= DESCEND_ALIGN_TIMEOUT:
                    # 下降段迟迟不能同时满足到位与水平：先试等价的另一侧解，
                    # 再退到预调整备用路径，而不是硬等到 DEADLOCK_STEPS 丢整条。
                    if not jaw_axis_flipped:
                        jaw_axis_sign = -jaw_axis_sign
                        jaw_axis_flipped = True
                        phase_steps = 0
                        continue
                    prealign_unreachable = True
                    prealign_last_axis_err_deg = axis_err_deg
                    prealign_last_jaw_height_diff_m = jaw_height_diff
                    if repose_count >= MAX_REPOSE_RETRIES:
                        fail_why = (
                            f"下降段无法保持夹爪水平 (轴误差 {axis_err_deg:.1f}°, "
                            f"高度差 {jaw_height_diff*1000:.1f}mm, "
                            f"已预调整 {repose_count} 次)")
                        break
                    phase, phase_steps = "REPOSE_OBJECT", 0
                    repose_stage, repose_stage_steps, repose_plan = "LIFT_OUT", 0, None
                else:
                    v = compute_3d_velocity(tool_pos, grasp_point, 0.15)
                    q_dot = pos_jaw_axis_qdot(
                        J6, J3, v, tool_mat[:, 1], target_axis)

            elif phase == "GRASP":
                grip = GRIP_CLOSE
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.5 * FPS):
                    phase, phase_steps, wait_steps = "VERIFY_GRASP", 0, 0
                    prelift_start = tool_pos.copy()
                    prelift_obj_z = float(data.xpos[H.body_obj][2])
                    verify_stable_steps = 0
                    # 记下抓取瞬间的螺丝刀朝向。搬运全程按住这个偏航 ——
                    # 放任不管的话它会被底座旋转带到 +57°，而盒子上方可达偏航是
                    # -90~0 和 +60~+90 两个孤岛，+15~+45 是不可达空洞：
                    # 一旦漂进 +60 那个岛，就再也拧不回 0 了（穿不过空洞）。
                    _, _, h_g = grasp_geometry(model, data)
                    # 目标偏航直接设成盒子长边方向：搬运全程边走边拧，
                    # 到盒子上方时已经基本平行，ALIGN 只需微调。
                    # 等到 ALIGN 才一次性扭几十度，既慢又容易撞上不可达区。
                    hold_dir = box_long_axis(model, data, H.body_box,
                                             ref_dir=np.array([h_g[0], h_g[1], 0.0]))

            # ---- VERIFY_GRASP：慢速试提，离桌且双指接触稳定后才标定抓取参考 ----
            # 旧流程在物体仍贴桌时记录 grip_ref。试提时手柄在两指间自然就位产生的
            # 位移会被误判为滑移，随后 RECOVER_OPEN 主动张爪，看起来就像自己掉落。
            elif phase == "VERIFY_GRASP":
                grip = GRIP_CLOSE
                left_contact, right_contact = bilateral_grasp_contacts(model, data, H)
                bilateral = left_contact and right_contact
                bilateral_contact_seen = bilateral_contact_seen or bilateral
                # 用物体 body 的相对上升量判断离桌；object_bottom_z() 基于保守 AABB，
                # 初始值可能低于真实桌面，不能拿绝对高度做这一步的判据。
                cleared = (float(data.xpos[H.body_obj][2]) >=
                           prelift_obj_z + PRELIFT_CLEARANCE)

                if bilateral and cleared:
                    verify_stable_steps += 1
                    if verify_stable_steps >= max(
                            1, int(PRELIFT_SETTLE_TIME / model.opt.timestep)):
                        grip_ref = relative_object_pose(data, H, tool_pos, tool_mat)
                        grasp_verified = True
                        _, _, h_g = grasp_geometry(model, data)
                        hold_dir = box_long_axis(
                            model, data, H.body_box,
                            ref_dir=np.array([h_g[0], h_g[1], 0.0]))
                        phase, phase_steps = "LIFT", 0
                else:
                    verify_stable_steps = 0
                    target = prelift_start + np.array([0.0, 0.0, PRELIFT_DISTANCE])
                    v = compute_3d_velocity(tool_pos, target, PRELIFT_SPEED)
                    _, _, h_now = grasp_geometry(model, data)
                    q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, hold_dir), Jp=J3)
                    q_dot = q_dot + nullspace_qdot(
                        J3, J6[3:], level_and_yaw_twist(h_now))

                if phase == "VERIFY_GRASP" and (
                        phase_steps * model.opt.timestep >= PRELIFT_TIMEOUT):
                    if grasp_retries >= MAX_GRASP_RETRIES:
                        fail_why = "预抬验证失败：物体未离桌或未保持双指接触"
                        break
                    phase, phase_steps, wait_steps = "RECOVER_OPEN", 0, 0
                    continue

            # ---- LIFT：先原地垂直抬到走廊里，再开始横move ----
            # 抛物线的起点是抓取点(z≈0.75)，若直接开始横move，伺服要边走边爬，
            # 而虚拟兔子只前瞻 0.10，整段都在追 —— 物体中段实测只有 0.73~0.77，
            # 低于盒顶 0.79，[C2] 必然判不过。先抬够再走。
            elif phase == "LIFT":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    slip_t, slip_r = grasp_pose_error(
                        grip_ref, relative_object_pose(data, H, tool_pos, tool_mat))
                    max_slip_translation = max(max_slip_translation, slip_t)
                    max_slip_rotation_deg = max(max_slip_rotation_deg, slip_r)
                    if slip_t > SLIP_TRANSLATION_TOL or slip_r > SLIP_ROTATION_TOL_DEG:
                        if grasp_retries >= MAX_GRASP_RETRIES:
                            fail_why = "抬升阶段重复滑移，超过重抓上限"
                            break
                        phase, phase_steps, wait_steps = "RECOVER_OPEN", 0, 0
                        continue
                need = Z_CORRIDOR_LO + 0.012 - obj_bot      # 物体底还差多少
                if need <= 0 or phase_steps > int(1200/SPEED_SCALE):
                    phase, phase_steps = "MOVE_ARC", 0
                    arc_start = tool_pos.copy()
                    arc_end = drop_point.copy()
                    arc_end[2] = max(arc_end[2], Z_CORRIDOR_LO + (tool_pos[2] - obj_bot))
                    vec_xy = arc_end[:2] - arc_start[:2]
                    L_sq = float(np.dot(vec_xy, vec_xy)) or 1e-6
                else:
                    up = np.array([tool_pos[0], tool_pos[1],
                                   min(tool_pos[2] + need, Z_CORRIDOR_HI)])
                    v = compute_3d_velocity(tool_pos, up, LIFT_SPEED)
                    _, _, h_now = grasp_geometry(model, data)
                    if LOCK_WRIST_AFTER_GRASP:
                        q_dot = arm3_qdot(J3, v)      # 腕部锁死，姿态不可控
                    else:
                        q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, hold_dir), Jp=J3)
                        q_dot = q_dot + nullspace_qdot(J3, J6[3:],
                                                       level_and_yaw_twist(h_now))

            # ---- MOVE_ARC：虚拟兔子沿抛物线搬运。[C2] 每个兔子的 z 都夹进走廊 ----
            elif phase == "MOVE_ARC":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    slip_t, slip_r = grasp_pose_error(
                        grip_ref, relative_object_pose(data, H, tool_pos, tool_mat))
                    max_slip_translation = max(max_slip_translation, slip_t)
                    max_slip_rotation_deg = max(max_slip_rotation_deg, slip_r)
                    if slip_t > SLIP_TRANSLATION_TOL or slip_r > SLIP_ROTATION_TOL_DEG:
                        if dropped_into_box(model, data, H):
                            fail_why = "螺丝刀已掉进盒子，不重抓，整条作废"
                            break
                        if grasp_retries >= MAX_GRASP_RETRIES:
                            fail_why = "搬运阶段重复滑移，超过重抓上限"
                            break
                        phase, phase_steps, wait_steps = "RECOVER_OPEN", 0, 0
                        continue
                w = tool_pos[:2] - arc_start[:2]
                p = float(np.clip(np.dot(w, vec_xy) / L_sq if L_sq > 1e-6 else 1.0, 0.0, 1.0))
                if p >= 0.97 and np.linalg.norm(arc_end - tool_pos) < 0.035:
                    phase, phase_steps = "ALIGN", 0
                else:
                    # 让**物体包络中心**落到投放点，而不是让夹持中心落到投放点
                    lead = drop_point[:2] - (object_center_xy(model, data, H.body_obj)
                                             - tool_pos[:2])
                    vec_xy = lead - arc_start[:2]
                    L_sq = float(np.dot(vec_xy, vec_xy)) or 1e-6
                    arc_end[:2] = lead
                    pt = float(np.clip(p + 0.10, 0.0, 1.0))
                    txy = arc_start[:2] + vec_xy * pt
                    tz = arc_start[2] + (arc_end[2] - arc_start[2]) * pt \
                        + H_PEAK * np.sin(pt * np.pi)
                    # [C2] 约束的是**物体**最低点，不是夹持中心。物体挂在夹爪下方，
                    # sag = 夹持中心到物体最低点的落差，每步实测（姿态会变，不是常数）。
                    sag = tool_pos[2] - obj_bot
                    tz = float(np.clip(tz,
                                       Z_CORRIDOR_LO + sag,      # 物体底 > 盒顶+裕度
                                       Z_CORRIDOR_HI))
                    rabbit = np.array([txy[0], txy[1], tz])
                    _, _, h_now = grasp_geometry(model, data)
                    # 主任务：位置 + 偏航保持
                    v = compute_3d_velocity(tool_pos, rabbit, MOVE_SPEED)
                    if LOCK_WRIST_AFTER_GRASP:
                        q_dot = arm3_qdot(J3, v)
                    else:
                        q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, hold_dir), Jp=J3)
                        # 零空间：把螺丝刀压回水平
                        q_dot = q_dot + nullspace_qdot(
                            J3, J6[3:], level_and_yaw_twist(h_now))
                    # [C2] 只在真正横越的一段统计：p<0.2 是起抬、p>0.9 是入箱下降，
                    # 这两段必然穿过盒顶高度，算进去等于永远判不过。
                    if 0.2 <= p <= 0.9:
                        carry_z.append(float(tool_pos[2]))
                        carry_bot.append(obj_bot)

            # ---- ALIGN：原地把偏航拧进装箱安全锥，再松手 ----
            # 盒内腔 y 向半宽 0.085，螺丝刀半长 0.123 -> |偏航| 必须 < 43° 才装得下。
            # 视频里没有这一步，结果刀杆整根横跨盒壁挂在外面。
            elif phase == "ALIGN":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    slip_t, slip_r = grasp_pose_error(
                        grip_ref, relative_object_pose(data, H, tool_pos, tool_mat))
                    max_slip_translation = max(max_slip_translation, slip_t)
                    max_slip_rotation_deg = max(max_slip_rotation_deg, slip_r)
                    if slip_t > SLIP_TRANSLATION_TOL or slip_r > SLIP_ROTATION_TOL_DEG:
                        # ALIGN 已经在盒子正上方了，脱手基本必然落进盒里
                        if dropped_into_box(model, data, H):
                            fail_why = "螺丝刀已掉进盒子，不重抓，整条作废"
                            break
                        if grasp_retries >= MAX_GRASP_RETRIES:
                            fail_why = "对齐阶段重复滑移，超过重抓上限"
                            break
                        phase, phase_steps, wait_steps = "RECOVER_OPEN", 0, 0
                        continue
                _, _, h_now = grasp_geometry(model, data)
                tgt = box_long_axis(model, data, H.body_box, ref_dir=h_now)
                yaw_err = np.degrees(abs(yaw_error(h_now, tgt)))
                # 判据从"装得进安全锥"收紧到"与盒子长边平行"。
                # 前者只保证不架在盒壁上，后者才是你要的对齐。
                shift, fits = fit_shift_xy(model, data, H)
                aligned = yaw_err < PARALLEL_TOL_DEG
                placed = fits and np.linalg.norm(shift) < 0.006
                # 必须同时满足：偏航平行 + 整根包络装得进内腔。
                # 只判偏航不够 —— 偏航对了但整体偏出去，长的一头照样挂在盒外。
                if LOCK_WRIST_AFTER_GRASP or (aligned and placed) or phase_steps > int(1500/SPEED_SCALE):
                    phase, phase_steps = "LOWER_IN", 0
                else:
                    end = tool_pos.copy()
                    end[:2] = tool_pos[:2] + shift      # 按实测越界量平移
                    v = compute_3d_velocity(tool_pos, end, 0.10)
                    q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, tgt), Jp=J3)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], level_and_yaw_twist(h_now))

            # ---- LOWER_IN：下放到贴近盒底再松手 ----
            # 原来在盒沿上方 0.028 就松手，物体要自由落 5cm 才到盒底，落下去弹一下
            # 就翘起来架在沿上。改成先放到离盒底 12mm 再张爪，基本是"放"而不是"扔"。
            elif phase == "LOWER_IN":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    slip_t, slip_r = grasp_pose_error(
                        grip_ref, relative_object_pose(data, H, tool_pos, tool_mat))
                    max_slip_translation = max(max_slip_translation, slip_t)
                    max_slip_rotation_deg = max(max_slip_rotation_deg, slip_r)
                    if slip_t > SLIP_TRANSLATION_TOL or slip_r > SLIP_ROTATION_TOL_DEG:
                        fail_why = "盒内下放阶段夹持相对位姿失稳"
                        break
                need = obj_bot - (Z_BOX_INNER_FLOOR + Z_PLACE_CLEAR)
                if need <= 0 or phase_steps > int(900/SPEED_SCALE):
                    phase, phase_steps, wait_steps = "RELEASE", 0, 0
                else:
                    shift, _ = fit_shift_xy(model, data, H)
                    end = np.array([tool_pos[0] + shift[0], tool_pos[1] + shift[1],
                                    tool_pos[2] - need])
                    _, _, h_now2 = grasp_geometry(model, data)
                    v = compute_3d_velocity(tool_pos, end, 0.10)
                    q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now2, tgt), Jp=J3)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:],
                                                   level_and_yaw_twist(h_now2))
            elif phase == "RECOVER_OPEN":
                grip = GRIP_OPEN
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.3 * FPS):
                    has_retried, grip_ref = True, None
                    grasp_retries += 1
                    grasp_verified = False
                    verify_stable_steps = 0
                    prelift_start = None
                    prelift_obj_z = None
                    prealign_stable_steps = 0
                    prealign_best_axis_err = None
                    prealign_stall_steps = 0
                    jaw_axis_sign, jaw_axis_flipped = None, False
                    ik_target_q, ik_move_obj_xy = None, None
                    gripper_aligned = False
                    pregrasp_axis_error_deg = None
                    pregrasp_jaw_height_diff_m = None
                    phase, phase_steps, wait_steps = "HOVER", 0, 0

            elif phase == "RELEASE":
                grip = GRIP_OPEN
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.4 * FPS):
                    phase, phase_steps, wait_steps = "RETURN_ARC", 0, 0
                    ret_start = tool_pos.copy()
                    ret_end = home_point.copy()
                    ret_vec = ret_end[:2] - ret_start[:2]
                    ret_L = float(np.dot(ret_vec, ret_vec))

            elif phase == "RETURN_ARC":
                grip = GRIP_OPEN
                w = tool_pos[:2] - ret_start[:2]
                p = float(np.clip(np.dot(w, ret_vec) / ret_L if ret_L > 1e-6 else 1.0, 0.0, 1.0))
                if p >= 0.97 and np.linalg.norm(ret_end - tool_pos) < 0.05:
                    phase, phase_steps = "RETURN_JOINT", 0
                else:
                    pt = float(np.clip(p + 0.10, 0.0, 1.0))
                    txy = ret_start[:2] + ret_vec * pt
                    tz = ret_start[2] + (ret_end[2] - ret_start[2]) * pt \
                        + RET_H_PEAK * np.sin(pt * np.pi)
                    rabbit = np.array([txy[0], txy[1], tz])
                    q_dot = damped_pinv(J3) @ compute_3d_velocity(tool_pos, rabbit, 0.55)

            elif phase == "RETURN_JOINT":
                grip = GRIP_OPEN
                q_err = home_qpos - data.qpos[H.arm_qadr]
                dq = float(np.linalg.norm(q_err))
                # 收紧到位判据：原来 (>300步 and dq<0.25) 太松，实测只走到范数
                # 0.217 就提前跳走，等于没真正回到全 0，中间这一站形同虚设。
                if dq < 0.05 or (phase_steps > int(700/SPEED_SCALE) and dq < 0.10):
                    phase, phase_steps, wait_steps = "GO_INIT", 0, 0
                else:
                    q_dot = (q_err / dq) * max(min(dq / 0.5 * 2.5, 2.5), 0.6) * SPEED_SCALE

            # ---- GO_INIT：从全 0 再回到起手位姿 Q_INIT，然后收工 ----
            # 一条 episode 的关节序列因此是 Q_INIT -> 全0 -> 抓放 -> 全0 -> Q_INIT，
            # 首尾完全闭合。下一条 episode 从 reset 出来正好也是 Q_INIT，
            # 所以数据集里不存在"同一观测对应两种动作"的歧义。
            elif phase == "GO_INIT":
                grip = GRIP_OPEN
                q_err = Q_INIT - data.qpos[H.arm_qadr]
                dq = float(np.linalg.norm(q_err))
                if dq < 0.08 or (phase_steps > int(600/SPEED_SCALE) and dq < 0.20):
                    phase, phase_steps, wait_steps = "WAIT_HOME", 0, 0
                else:
                    q_dot = (q_err / dq) * max(min(dq / 0.5 * 2.5, 2.5), 0.6) * SPEED_SCALE

            elif phase == "WAIT_HOME":
                grip = GRIP_OPEN
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.8 * FPS):
                    success = True
                    break

            q_dot = np.clip(q_dot, -6.0, 6.0)

            # ---- 录制 ----
            if step_counter % STEPS_PER_RECORD == 0:
                renderer.update_scene(data, camera=rspec.CAM_FIXED, scene_option=vopt)
                imgs_f.append(renderer.render().copy())
                renderer.update_scene(data, camera=rspec.CAM_WRIST, scene_option=vopt)
                imgs_w.append(renderer.render().copy())
                cur = np.concatenate([data.qpos[H.arm_qadr], data.qpos[H.grip_qadr]])
                act = np.zeros(8, dtype=np.float32)
                act[:6] = data.qpos[H.arm_qadr] + q_dot * (1.0 / FPS)
                # 夹爪严格反号。原脚本 action[6]=action[7]=target 是两维同号的
                # 旧约定，照抄会让两指互相打架 —— 不报错，只是永远夹不住。
                act[6], act[7] = grip
                ep_qpos.append(cur.astype(np.float32))
                ep_act.append(act)

            # ---- 积分下发 ----
            data.ctrl[H.arm_act] = data.ctrl[H.arm_act] + q_dot * model.opt.timestep
            err = data.ctrl[H.arm_act] - data.qpos[H.arm_qadr]
            data.ctrl[H.arm_act] = data.qpos[H.arm_qadr] + np.clip(err, -0.40, 0.40)
            data.ctrl[H.j8], data.ctrl[H.j9] = grip

            mujoco.mj_step(model, data)
            if viewer is not None and step_counter % STEPS_PER_RECORD == 0:
                if not viewer.is_running():
                    fail_why = "viewer 关闭"
                    break
                viewer.sync()
            step_counter += 1

        # ---- 成功判定 ----
        if success:
            # 判据用手柄中心 + 整体水平包络，不用 body 原点 ——
            # 螺丝刀的 body 原点在手柄和刀尖之间，离实际抓持点 4~9cm，判起来偏。
            # 判据只看两件事：**躺平** + **落到盒底**（没斜挂在盒沿上）。
            # 不再要求整根包络都在盒内 —— 刀杆稍微探出盒沿一点，只要它是平躺在
            # 盒底而不是架在边上，就算完成任务。
            box_xy = data.xpos[H.body_box][:2]
            gm = jaw_mid_now(model, data)
            # 判据只有一条：**整根螺丝刀的最高点不高过盒壁顶**。
            # 只要有一头翘起来（斜挂在盒沿、竖插、半个身子在外），最高点必然超标，
            # 所以不用再单独判倾角 —— 倾斜本身就会被这一条抓住。实测：
            #   平躺盒底 0.790 / 斜15° 0.834 / 斜挂30° 0.902 / 竖插 0.986
            # 容差 5mm：geom_aabb 是保守包围盒，平躺时算出来正好卡在盒壁顶上。
            obj_top_fin = object_top_z(model, data, H.body_obj)
            in_xy = (np.abs(gm[:2] - box_xy) < BOX_INNER_HALF).all()

            if not in_xy:
                success, fail_why = False, (
                    f"手柄不在盒内 ({gm[0]:.3f},{gm[1]:.3f})")
            elif obj_top_fin > Z_BOX_WALL_TOP + FIT_TOL:
                success, fail_why = False, (
                    f"没全进盒子 (最高点 {obj_top_fin:.3f} > 盒壁顶 "
                    f"{Z_BOX_WALL_TOP}+{FIT_TOL:.3f})")
            elif not carry_z:
                success, fail_why = False, "没采到搬运段"
            elif max(carry_z) > Z_FLASHLIGHT_TOP:
                success, fail_why = False, f"[C2] 夹持中心最高 {max(carry_z):.4f} 越过手电筒顶"
            elif min(carry_bot) < Z_BOX_WALL_TOP:
                success, fail_why = False, (f"[C2] 物体最低 {min(carry_bot):.4f} 低于盒顶 "
                                            f"{Z_BOX_WALL_TOP}")

        if not success:
            reasons[fail_why] = reasons.get(fail_why, 0) + 1
            print(f"  ↻ 第 {attempts} 次尝试失败: {fail_why}")
            if attempts % 15 == 0:
                top = sorted(reasons.items(), key=lambda x: -x[1])[:3]
                print("  ⚠ 失败统计: " + "; ".join(f"{k} ×{v}" for k, v in top))
            continue

        # ---- 写盘（保持旧格式，并增加元数据与完成标记）----
        instruction = str(rng.choice(LANGUAGE_INSTRUCTIONS))
        phase_durations = {
            name: round(steps * model.opt.timestep, 6)
            for name, steps in phase_step_counts.items()
        }
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "episode": slot,
            "attempt_index": attempt_index,
            "global_attempt": attempts,
            "requested_seed": args.seed,
            "global_seed": global_seed,
            "attempt_seed": attempt_seed,
            "initial_object": reset_info,
            "drop_point": drop_point.tolist(),
            "intentional_miss": bool(intentional_miss),
            "recovered": bool(has_retried),
            "grasp_retries": grasp_retries,
            "grasp_verified": bool(grasp_verified),
            "bilateral_contact_seen": bool(bilateral_contact_seen),
            "gripper_aligned_before_grasp": bool(gripper_aligned),
            "pregrasp_jaw_axis_error_deg": pregrasp_axis_error_deg,
            "pregrasp_jaw_height_diff_m": pregrasp_jaw_height_diff_m,
            "ik_prealign_solved": bool(ik_solved),
            "jaw_axis_sign": None if jaw_axis_sign is None else float(jaw_axis_sign),
            "jaw_axis_flipped": bool(jaw_axis_flipped),
            "prealign_unreachable": bool(prealign_unreachable),
            "prealign_last_axis_error_deg": prealign_last_axis_err_deg,
            "prealign_last_jaw_height_diff_m": prealign_last_jaw_height_diff_m,
            "repose_used": bool(repose_count),
            "repose_count": repose_count,
            "instruction": instruction,
            "frames": len(ep_qpos),
            "simulation_steps": step_counter,
            "simulation_duration_s": round(step_counter * model.opt.timestep, 6),
            "wall_duration_s": round(time.time() - t0, 6),
            "phase_durations_s": phase_durations,
            "tool_control": {
                "site": EE_SITE,
                "offset_local": H.tcp_offset.tolist(),
            },
            "quality": {
                "max_grasp_translation_slip_m": max_slip_translation,
                "max_grasp_rotation_slip_deg": max_slip_rotation_deg,
                "grasp_translation_tolerance_m": SLIP_TRANSLATION_TOL,
                "grasp_rotation_tolerance_deg": SLIP_ROTATION_TOL_DEG,
                "max_gripper_actuator_force_n": max_gripper_force_n,
                "task_gripper_kp": TASK_GRIPPER_KP,
                "prelift_clearance_m": PRELIFT_CLEARANCE,
                "carry_control_point_z_min": float(min(carry_z)),
                "carry_control_point_z_max": float(max(carry_z)),
                "carry_object_bottom_z_min": float(min(carry_bot)),
                "carry_object_bottom_z_max": float(max(carry_bot)),
                "final_object_top_z": float(obj_top_fin),
            },
            "final_object": {
                "position": data.xpos[H.body_obj].tolist(),
                "quaternion": data.qpos[H.obj_qadr + 3:H.obj_qadr + 7].tolist(),
            },
        }
        ep_dir = write_episode_atomic(
            out_dir, slot, imgs_f, imgs_w, ep_qpos, ep_act, instruction, metadata)
        done.add(slot)
        occupied.add(slot)
        print(f"📁 ep_{slot}  {len(ep_qpos)} 帧  物体底 {min(carry_bot):.3f}~{max(carry_bot):.3f}  "
              f"{time.time()-t0:.1f}s  [{len(done)}/{args.target}]")

    renderer.close()
    if viewer is not None:
        viewer.close()
    print(f"\n完成：新增 {len(done)-start_have} 条，共 {len(done)} 条，"
          f"尝试 {attempts} 次 -> {out_dir.resolve()}")


def cmd_inspect():
    model, data = load_scene_tidyB()
    H = Handles(model)
    data.qpos[H.arm_qadr] = Q_INIT
    mujoco.mj_forward(model, data)
    ja, jb, h = grasp_geometry(model, data)
    print("-" * 68)
    print("[C1] 抓取几何（来源：碰撞胶囊 sd_handle）")
    print(f"  jaw_a = {np.round(ja,4)}")
    print(f"  jaw_b = {np.round(jb,4)}")
    print(f"  中点  = {np.round(0.5*(ja+jb),4)}   <- 伺服目标点")
    print(f"  杆身偏航 = {np.degrees(shaft_yaw_now(model, data)):.1f}°")
    print("-" * 68)
    print("[C2] 走廊")
    print(f"  盒顶 {Z_BOX_WALL_TOP}  手电筒顶 {Z_FLASHLIGHT_TOP}")
    print(f"  搬运期虚拟兔子 z 夹紧到 ({Z_CORRIDOR_LO:.3f}, {Z_CORRIDOR_HI:.3f})")
    print(f"  抛物线拱高 {H_PEAK}（原脚本 0.18，在此场景会顶穿手电筒）")
    print("-" * 68)
    print(f"夹爪 开={GRIP_OPEN} 合={GRIP_CLOSE}（正值=闭合，joint8=-joint9）")
    print("-" * 68)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="tidy_B 螺丝刀入盒 VLA 采集（速度级伺服）")
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--out_dir", type=str, default="datasets/screwdriver_tidyB")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        cmd_inspect()
    else:
        collect(a)
