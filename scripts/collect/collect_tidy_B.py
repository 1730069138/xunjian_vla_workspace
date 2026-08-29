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
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))
from common import robot_spec as rspec

SCENE_PATH = SCRIPT_DIR.parent / "scenes" / "tidy_B_record_preview.xml"

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
SLIP_TOL = 0.06
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
    不做这一步，伺服送到目标点的是 ee_site 小球，而真正夹东西的指垫在它后面
    ~33mm —— 夹爪一斜，这 33mm 就沿手柄轴滑过去，夹到细颈上。
    """
    return J6[:3, :] - skew(d_world) @ J6[3:, :]


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


def pos_yaw_qdot(J6, v_lin, yaw_err_rad, k_yaw=2.5, rho=0.10):
    """主任务 = 3 个位置 + 1 个偏航。6 自由度里仍留 2 个冗余。

    偏航放零空间按不住：从抓取点摆到盒子，底座要转 74°，要保持物体朝向不变
    腕部就得反向补 74°，零空间那点权限做不到，结果一路漂到 +57° ——
    而盒子上方可达偏航是 -90~0 和 +60~+90 两个孤岛，中间 +15~+45 是空洞，
    一旦漂进 +60 那个岛就再也拧不回来了。所以必须进主任务。
    """
    Jp = J6[:3, :]
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
        self.cam_fixed = mujoco.mj_name2id(model, M.mjOBJ_CAMERA, rspec.CAM_FIXED)
        self.cam_wrist = mujoco.mj_name2id(model, M.mjOBJ_CAMERA, rspec.CAM_WRIST)
        self.tcp_offset = self._pad_offset(model)

    def _pad_offset(self, model):
        """ee_site -> 两指夹持面中心（工具系）。

        ee_site 挂在 link7 的 (0,0,0.08)，而 link8/link9 的 body 原点在
        (0,±0.023831,0.016) —— ee_site 在指根上方 64mm，落在指尖处甚至指尖之外，
        不是夹持面中心。从指垫碰撞 geom 的 AABB 反算真实中心。
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
        return np.array([0.0, 0.0, 0.5 * (min(zs) + max(zs))])


def load_scene_tidyB(verbose=True):
    """rspec.load_scene() 走的是旧场景路径，且 assert_contract 会去查
    real_screwdriver / plasticbox / dynamic_pillar —— tidy_B 里都不存在。
    这里自己走等价流程，但相机镜像仍调 rspec 的那一个，保证与部署端逐字一致。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
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
            break
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(300):                # 落定
        mujoco.mj_step(model, data)


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


# ==============================================================================
# 采集主循环
# ==============================================================================
def collect(args):
    model, data = load_scene_tidyB()
    H = Handles(model)
    os.makedirs(args.out_dir, exist_ok=True)

    renderer = rspec.make_renderer(model)
    vopt = rspec.make_scene_option()        # 两端必须逐字一致，别自己写一份

    done = set()
    for d in os.listdir(args.out_dir):
        if d.startswith("ep_") and os.path.isdir(os.path.join(args.out_dir, d)):
            try:
                done.add(int(d.split("_")[1]))
            except ValueError:
                pass
    start_have = len(done)
    if start_have >= args.target:
        print(f"✅ 已有 {start_have} 条，达到目标 {args.target}。")
        return

    rng = np.random.default_rng(args.seed)
    viewer = mujoco.viewer.launch_passive(model, data) if not args.headless else None
    attempts = 0
    reasons = {}

    while len(done) < args.target:
        if viewer is not None and not viewer.is_running():
            break
        attempts += 1
        reset_scene(model, data, H, rng)

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
        drop_point = get_drop_point(model, data, H, rng)
        grip_ref = None
        hold_dir = np.array([1.0, 0.0, 0.0])
        carry_z, carry_bot = [], []
        t0 = time.time()

        while True:
            phase_steps += 1
            if phase_steps > DEADLOCK_STEPS:
                fail_why = f"[{phase}] 阶段死锁"
                break

            jaw_mid = jaw_mid_now(model, data)
            tcp_pos = data.site_xpos[H.site_ee].copy()
            obj_bot = object_bottom_z(model, data, H.body_obj)

            if data.xpos[H.body_obj][2] < Z_TABLE_TOP - 0.05:
                fail_why = "螺丝刀掉到桌下"
                break

            miss = np.array([0.0, 0.06, 0.0]) if (intentional_miss and not has_retried) \
                else np.zeros(3)
            hover_point = np.array([jaw_mid[0], jaw_mid[1], SAFE_Z]) + miss
            # 往下压 GRASP_DEPTH：指垫包住手柄更多，抗滑。miss 是 5% 概率的
            # 故意抓偏（纠错示范），不受这个影响。
            grasp_point = jaw_mid + miss - np.array([0.0, 0.0, GRASP_DEPTH])

            q_dot = np.zeros(6)
            grip = GRIP_OPEN
            J6 = get_site_jacobian_6d(model, data, H.site_ee, H.arm_dof)
            J3 = J6[:3, :]

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
                    home_point = tcp_pos.copy()
                    phase, phase_steps = "HOVER", 0
                else:
                    q_dot = (q_err / dq) * max(min(dq / 0.5 * 2.5, 2.5), 0.6) * SPEED_SCALE

            elif phase == "HOVER":
                dist = np.linalg.norm(hover_point - tcp_pos)
                if dist < 0.02 or (phase_steps > int(400/SPEED_SCALE) and dist < 0.05):
                    phase, phase_steps = "DESCEND", 0
                else:
                    q_dot = damped_pinv(J3) @ compute_3d_velocity(tcp_pos, hover_point, 0.45)
                    Rt = data.site_xmat[H.site_ee].reshape(3, 3)
                    w = K_TOOL * get_orientation_error(grasp_tool_frame(model, data), Rt)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], w)

            # ---- DESCEND：末端小球下到夹取点中点 ----
            elif phase == "DESCEND":
                dist = np.linalg.norm(grasp_point - tcp_pos)
                if dist < 0.006 or (phase_steps > int(500/SPEED_SCALE) and dist < 0.015):
                    phase, phase_steps = "GRASP", 0
                else:
                    q_dot = damped_pinv(J3) @ compute_3d_velocity(tcp_pos, grasp_point, 0.15)
                    Rt = data.site_xmat[H.site_ee].reshape(3, 3)
                    w = K_TOOL * get_orientation_error(grasp_tool_frame(model, data), Rt)
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], w)

            elif phase == "GRASP":
                grip = GRIP_CLOSE
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.5 * FPS):
                    phase, phase_steps, wait_steps = "LIFT", 0, 0
                    # arc_start / arc_end / vec_xy / L_sq 移到 LIFT 结束时再算：
                    # 抬升之后 TCP 高度才是抛物线真正的起点。
                    grip_ref = jaw_mid - tcp_pos
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

            # ---- LIFT：先原地垂直抬到走廊里，再开始横move ----
            # 抛物线的起点是抓取点(z≈0.75)，若直接开始横move，伺服要边走边爬，
            # 而虚拟兔子只前瞻 0.10，整段都在追 —— 物体中段实测只有 0.73~0.77，
            # 低于盒顶 0.79，[C2] 必然判不过。先抬够再走。
            elif phase == "LIFT":
                grip = GRIP_CLOSE
                need = Z_CORRIDOR_LO + 0.012 - obj_bot      # 物体底还差多少
                if need <= 0 or phase_steps > int(1200/SPEED_SCALE):
                    phase, phase_steps = "MOVE_ARC", 0
                    arc_start = tcp_pos.copy()
                    arc_end = drop_point.copy()
                    arc_end[2] = max(arc_end[2], Z_CORRIDOR_LO + (tcp_pos[2] - obj_bot))
                    vec_xy = arc_end[:2] - arc_start[:2]
                    L_sq = float(np.dot(vec_xy, vec_xy)) or 1e-6
                else:
                    up = np.array([tcp_pos[0], tcp_pos[1],
                                   min(tcp_pos[2] + need, Z_CORRIDOR_HI)])
                    v = compute_3d_velocity(tcp_pos, up, 0.35)
                    _, _, h_now = grasp_geometry(model, data)
                    if LOCK_WRIST_AFTER_GRASP:
                        q_dot = arm3_qdot(J3, v)      # 腕部锁死，姿态不可控
                    else:
                        q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, hold_dir))
                        q_dot = q_dot + nullspace_qdot(J3, J6[3:],
                                                       level_and_yaw_twist(h_now))

            # ---- MOVE_ARC：虚拟兔子沿抛物线搬运。[C2] 每个兔子的 z 都夹进走廊 ----
            elif phase == "MOVE_ARC":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    slip = np.linalg.norm((jaw_mid - tcp_pos) - grip_ref)
                    if slip > SLIP_TOL:
                        if dropped_into_box(model, data, H):
                            fail_why = "螺丝刀已掉进盒子，不重抓，整条作废"
                            break
                        phase, phase_steps, wait_steps = "RECOVER_OPEN", 0, 0
                        continue
                w = tcp_pos[:2] - arc_start[:2]
                p = float(np.clip(np.dot(w, vec_xy) / L_sq if L_sq > 1e-6 else 1.0, 0.0, 1.0))
                if p >= 0.97 and np.linalg.norm(arc_end - tcp_pos) < 0.035:
                    phase, phase_steps = "ALIGN", 0
                else:
                    # 让**物体包络中心**落到投放点，而不是让 TCP 落到投放点
                    lead = drop_point[:2] - (object_center_xy(model, data, H.body_obj)
                                             - tcp_pos[:2])
                    vec_xy = lead - arc_start[:2]
                    L_sq = float(np.dot(vec_xy, vec_xy)) or 1e-6
                    arc_end[:2] = lead
                    pt = float(np.clip(p + 0.10, 0.0, 1.0))
                    txy = arc_start[:2] + vec_xy * pt
                    tz = arc_start[2] + (arc_end[2] - arc_start[2]) * pt \
                        + H_PEAK * np.sin(pt * np.pi)
                    # [C2] 约束的是**物体**最低点，不是 TCP。物体挂在夹爪下方，
                    # sag = TCP 到物体最低点的落差，每步实测（姿态会变，不是常数）。
                    sag = tcp_pos[2] - obj_bot
                    tz = float(np.clip(tz,
                                       Z_CORRIDOR_LO + sag,      # 物体底 > 盒顶+裕度
                                       Z_CORRIDOR_HI))
                    rabbit = np.array([txy[0], txy[1], tz])
                    _, _, h_now = grasp_geometry(model, data)
                    # 主任务：位置 + 偏航保持
                    v = compute_3d_velocity(tcp_pos, rabbit, 0.45)
                    if LOCK_WRIST_AFTER_GRASP:
                        q_dot = arm3_qdot(J3, v)
                    else:
                        q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, hold_dir))
                        # 零空间：把螺丝刀压回水平
                        q_dot = q_dot + nullspace_qdot(
                            J3, J6[3:], level_and_yaw_twist(h_now))
                    # [C2] 只在真正横越的一段统计：p<0.2 是起抬、p>0.9 是入箱下降，
                    # 这两段必然穿过盒顶高度，算进去等于永远判不过。
                    if 0.2 <= p <= 0.9:
                        carry_z.append(float(tcp_pos[2]))
                        carry_bot.append(obj_bot)

            # ---- ALIGN：原地把偏航拧进装箱安全锥，再松手 ----
            # 盒内腔 y 向半宽 0.085，螺丝刀半长 0.123 -> |偏航| 必须 < 43° 才装得下。
            # 视频里没有这一步，结果刀杆整根横跨盒壁挂在外面。
            elif phase == "ALIGN":
                grip = GRIP_CLOSE
                if grip_ref is not None:
                    if np.linalg.norm((jaw_mid - tcp_pos) - grip_ref) > SLIP_TOL:
                        # ALIGN 已经在盒子正上方了，脱手基本必然落进盒里
                        if dropped_into_box(model, data, H):
                            fail_why = "螺丝刀已掉进盒子，不重抓，整条作废"
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
                    end = tcp_pos.copy()
                    end[:2] = tcp_pos[:2] + shift      # 按实测越界量平移
                    v = compute_3d_velocity(tcp_pos, end, 0.10)
                    q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now, tgt))
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:], level_and_yaw_twist(h_now))

            # ---- LOWER_IN：下放到贴近盒底再松手 ----
            # 原来在盒沿上方 0.028 就松手，物体要自由落 5cm 才到盒底，落下去弹一下
            # 就翘起来架在沿上。改成先放到离盒底 12mm 再张爪，基本是"放"而不是"扔"。
            elif phase == "LOWER_IN":
                grip = GRIP_CLOSE
                need = obj_bot - (Z_BOX_INNER_FLOOR + Z_PLACE_CLEAR)
                if need <= 0 or phase_steps > int(900/SPEED_SCALE):
                    phase, phase_steps, wait_steps = "RELEASE", 0, 0
                else:
                    shift, _ = fit_shift_xy(model, data, H)
                    end = np.array([tcp_pos[0] + shift[0], tcp_pos[1] + shift[1],
                                    tcp_pos[2] - need])
                    _, _, h_now2 = grasp_geometry(model, data)
                    v = compute_3d_velocity(tcp_pos, end, 0.10)
                    q_dot = pos_yaw_qdot(J6, v, yaw_error(h_now2, tgt))
                    q_dot = q_dot + nullspace_qdot(J3, J6[3:],
                                                   level_and_yaw_twist(h_now2))
            elif phase == "RECOVER_OPEN":
                grip = GRIP_OPEN
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.3 * FPS):
                    has_retried, grip_ref = True, None
                    phase, phase_steps, wait_steps = "HOVER", 0, 0

            elif phase == "RELEASE":
                grip = GRIP_OPEN
                if step_counter % STEPS_PER_RECORD == 0:
                    wait_steps += 1
                if wait_steps >= int(0.4 * FPS):
                    phase, phase_steps, wait_steps = "RETURN_ARC", 0, 0
                    ret_start = tcp_pos.copy()
                    ret_end = home_point.copy()
                    ret_vec = ret_end[:2] - ret_start[:2]
                    ret_L = float(np.dot(ret_vec, ret_vec))

            elif phase == "RETURN_ARC":
                grip = GRIP_OPEN
                w = tcp_pos[:2] - ret_start[:2]
                p = float(np.clip(np.dot(w, ret_vec) / ret_L if ret_L > 1e-6 else 1.0, 0.0, 1.0))
                if p >= 0.97 and np.linalg.norm(ret_end - tcp_pos) < 0.05:
                    phase, phase_steps = "RETURN_JOINT", 0
                else:
                    pt = float(np.clip(p + 0.10, 0.0, 1.0))
                    txy = ret_start[:2] + ret_vec * pt
                    tz = ret_start[2] + (ret_end[2] - ret_start[2]) * pt \
                        + RET_H_PEAK * np.sin(pt * np.pi)
                    rabbit = np.array([txy[0], txy[1], tz])
                    q_dot = damped_pinv(J3) @ compute_3d_velocity(tcp_pos, rabbit, 0.55)

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
                success, fail_why = False, f"[C2] TCP 最高 {max(carry_z):.4f} 越过手电筒顶"
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

        # ---- 写盘（格式与既有数据集一致）----
        slot = next((i for i in range(args.target) if i not in done),
                    max(done) + 1 if done else 0)
        ep_dir = os.path.join(args.out_dir, f"ep_{slot}")
        cf, cw = os.path.join(ep_dir, "cam_fixed"), os.path.join(ep_dir, "cam_wrist")
        os.makedirs(cf, exist_ok=True)
        os.makedirs(cw, exist_ok=True)
        for i, (a, b) in enumerate(zip(imgs_f, imgs_w)):
            cv2.imwrite(os.path.join(cf, f"{i:03d}.jpg"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(cw, f"{i:03d}.jpg"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        np.savez_compressed(os.path.join(ep_dir, "joint_data.npz"),
                            qpos=np.array(ep_qpos, dtype=np.float32),
                            actions=np.array(ep_act, dtype=np.float32))
        with open(os.path.join(ep_dir, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(str(rng.choice(LANGUAGE_INSTRUCTIONS)))
        done.add(slot)
        print(f"📁 ep_{slot}  {len(ep_qpos)} 帧  物体底 {min(carry_bot):.3f}~{max(carry_bot):.3f}  "
              f"{time.time()-t0:.1f}s  [{len(done)}/{args.target}]")

    renderer.close()
    if viewer is not None:
        viewer.close()
    print(f"\n完成：新增 {len(done)-start_have} 条，共 {len(done)} 条，"
          f"尝试 {attempts} 次 -> {os.path.abspath(args.out_dir)}")


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
