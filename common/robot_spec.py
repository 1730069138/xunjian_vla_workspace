"""
dummyx / common / robot_spec.py

采集端与部署端之间那份**唯一的契约**。

历史教训：夹爪编码、相机位姿、控制频率、起手状态这四件事，原本各自散落在
auto_grasp_screwdriver2.py 和 deploy_screwdriver_client2.py 里重复定义。
2026-08 夹爪约定从「(0.04=开, 0.0=合)，两维同号」改成「±0.02，joint8 = -joint9」时，
只有采集端跟着改了，部署端的阈值判决 `0.04 if a[6] > 0.02 else 0.0` 留在原地 ——
新行程的上界恰好等于旧阈值，判断永远为假，夹爪 ctrl 被钉死在 0.0，全程不动。
同一时期部署端也漏掉了 global_cam_body 的镜像，视觉输入直接 OOD。

两个 bug 都不抛异常、不改变数组形状、不产生 NaN，而且全在推理链路最末端 ——
数据集、norm_stats、delta mask、训练 loss 逐个检查都是对的。

结论：凡是"两端必须一致"的量，都只在本文件定义一次，两端 import。
改约定时改这里，另一端自动跟上；真的不一致时，assert_contract() 会当场报错。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # 允许在没装 mujoco 的环境里只 import 常量
    mujoco = None


# ==============================================================================
# 路径
# ==============================================================================
# 本文件在 dummyx/common/ 下，项目根目录即上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "scenes" / "xunjian_arm_scene.xml"
DATASET_DIR = PROJECT_ROOT / "datasets" / "screwdriver_cleanup"


# ==============================================================================
# 时间尺度
# ==============================================================================
CONTROL_HZ = 50                                        # 录制 / 下发动作的频率
SIM_TIMESTEP = 0.001                                   # 必须与场景 xml 的 <option timestep> 一致
STEPS_PER_RECORD = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)   # = 20

# 部署端每个策略动作要跑的物理子步数。与采集端 STEPS_PER_RECORD 是同一个量：
# 采集时"每 20 个 mj_step 记录一帧"，部署时就必须"每个动作跑 20 个 mj_step"。
# 曾经部署端写死 range(50)，等于把每段动作拉长 2.5 倍。
SUBSTEPS_PER_ACTION = STEPS_PER_RECORD


# ==============================================================================
# 位姿
# ==============================================================================
Q_HOME = np.zeros(6)
Q_INIT = np.array([0.0, 1.41, 1.3, 0.0, -0.7, 0.0])    # 起手位姿，录制从这里开始


# ==============================================================================
# 夹爪约定
#   joint8 与 joint9 严格反号，行程 ±GRIP_LIMIT。
#   正值 = 闭合，负值 = 张开。默认状态是闭合。
# ==============================================================================
GRIP_LIMIT = 0.02
GRIP_OPEN = (-GRIP_LIMIT, +GRIP_LIMIT)
GRIP_CLOSE = (+GRIP_LIMIT, -GRIP_LIMIT)
GRIP_INIT = GRIP_CLOSE


def encode_gripper(closed: bool) -> tuple[float, float]:
    """离散意图 -> (joint8, joint9) ctrl 目标。采集端规划时用。"""
    return GRIP_CLOSE if closed else GRIP_OPEN


def decode_gripper(action_vec) -> tuple[float, float]:
    """模型输出 -> (joint8, joint9) ctrl 目标。部署端用。

    模型回归的就是 joint8 的绝对目标值（DeltaActions 的 mask 让后两维保持绝对），
    所以这里**不做阈值判决**，只裁剪行程。阈值化会把连续开合压成二值，而且历史上
    那个阈值正好落在行程端点上，实际永远不成立。
    """
    g = float(np.clip(np.asarray(action_vec).reshape(-1)[6], -GRIP_LIMIT, GRIP_LIMIT))
    return g, -g


def is_gripper_closed(g_cmd: float) -> bool:
    """正值为闭合。判断"是否已夹住 / 是否已释放"时统一走这里。"""
    return float(g_cmd) > 0.0


# ==============================================================================
# 实体名（全部按名字解析，绝不硬编码下标）
# ==============================================================================
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ARM_ACTUATORS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
GRIP_JOINTS = ["joint8", "joint9"]
GRIP_ACTUATORS = ["Joint8", "Joint9"]

EE_SITE = "ee_site"
OBJECT_BODY = "real_screwdriver"
OBJECT_FREEJOINT = "fj_screwdriver"
STORAGE_BOX_BODY = "plasticbox"
OBSTACLE_BODY = "dynamic_pillar"
OBSTACLE_JOINT = "pillar_joint"
GLOBAL_CAM_BODY = "global_cam_body"

STATE_DIM = 8      # 6 臂关节 + 2 夹爪
ACTION_DIM = 8


# ==============================================================================
# 相机
# ==============================================================================
CAM_FIXED = "global_cam"     # -> observation.images.cam_fixed -> observation/image
CAM_WRIST = "d415_rgb"       # -> observation.images.cam_wrist -> observation/wrist_image
IMG_H = IMG_W = 256

# 采集时对全局相机做的运行时改动：目标中心 x≈0.45，原相机 x=-0.1，对称镜像后 x=1.0。
# 这是"从前向后看"的视角。部署端必须复现，否则策略看到的画面与训练时不是同一个视角。
GLOBAL_CAM_MIRROR_X = 1.0


def apply_camera_overrides(model, verbose: bool = True) -> None:
    """把采集时对相机的运行时改动应用到 model 上。

    必须在 MjData 创建之前、模型加载之后立刻调用，采集端和部署端都要调。
    """
    cam_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GLOBAL_CAM_BODY)
    if cam_body_id == -1:
        raise RuntimeError(
            f"场景里找不到 body '{GLOBAL_CAM_BODY}'，无法对齐全局相机视角。"
            f"训练与推理的视觉输入将不一致。"
        )
    model.body_pos[cam_body_id][0] = GLOBAL_CAM_MIRROR_X
    if verbose:
        print(f"📷 global_cam_body 已对齐: body_pos = {model.body_pos[cam_body_id]}")


def make_scene_option():
    """采集时的画面屏蔽选项：隐藏坐标系、桌面色块、所有 site 标记。

    只作用于相机 renderer，不影响 viewer。两端必须一致，否则图像里会多出/少掉东西。
    """
    vopt = mujoco.MjvOption()
    vopt.geomgroup[1] = 0    # 坐标系
    vopt.geomgroup[2] = 0    # 桌面红/绿/蓝区域
    for i in range(6):
        vopt.sitegroup[i] = 0
    return vopt


def make_renderer(model):
    return mujoco.Renderer(model, height=IMG_H, width=IMG_W)


# ==============================================================================
# 语言指令
# ==============================================================================
LANGUAGE_INSTRUCTIONS = [
    "pick up the screwdriver and place it into the storage box.",
    "grasp the screwdriver and put it away.",
    "clear the screwdriver into the plastic box.",
    "clean up the workspace by placing the screwdriver into the box.",
]

# 部署时默认发这一条。训练用的是 prompt_from_task=True，即数据集 instruction.txt 的原文，
# 所以这里必须**逐字**命中上面列表里的某一条。
DEFAULT_PROMPT = LANGUAGE_INSTRUCTIONS[0]


def check_prompt(prompt: str) -> str:
    """校验 prompt 在训练分布内，不在就当场报错（而不是安静地喂 OOD 输入）。"""
    if prompt not in LANGUAGE_INSTRUCTIONS:
        raise ValueError(
            f"prompt 不在训练指令集中，语言条件将 OOD:\n"
            f"  收到: {prompt!r}\n"
            f"  可用: " + "\n         ".join(repr(s) for s in LANGUAGE_INSTRUCTIONS)
        )
    return prompt


# ==============================================================================
# 索引解析：全部按名字查，一次解析，到处复用
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class RobotIndices:
    arm_qpos_adr: np.ndarray     # (6,) 臂关节在 qpos 里的地址
    grip_qpos_adr: np.ndarray    # (2,) joint8 / joint9 在 qpos 里的地址
    arm_act_ids: np.ndarray      # (6,) 臂执行器 id
    j8_id: int
    j9_id: int
    ee_site_id: int
    object_body_id: int
    box_body_id: int
    obstacle_body_id: int

    @property
    def state_qpos_adr(self) -> np.ndarray:
        """8 维 observation.state 对应的 qpos 地址，顺序与训练数据严格一致。"""
        return np.concatenate([self.arm_qpos_adr, self.grip_qpos_adr])

    @property
    def is_contiguous(self) -> bool:
        """臂+夹爪是否恰好占据 qpos[0:8]。为真时 qpos[:8] 才等价于 state。"""
        return np.array_equal(self.state_qpos_adr, np.arange(STATE_DIM))


def resolve_indices(model) -> RobotIndices:
    def jid(name):
        return model.joint(name).qposadr[0]

    def aid(name):
        return model.actuator(name).id

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    return RobotIndices(
        arm_qpos_adr=np.array([jid(j) for j in ARM_JOINTS]),
        grip_qpos_adr=np.array([jid(j) for j in GRIP_JOINTS]),
        arm_act_ids=np.array([aid(a) for a in ARM_ACTUATORS]),
        j8_id=aid("Joint8"),
        j9_id=aid("Joint9"),
        ee_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE),
        object_body_id=bid(OBJECT_BODY),
        box_body_id=bid(STORAGE_BOX_BODY),
        obstacle_body_id=bid(OBSTACLE_BODY),
    )


# ==============================================================================
# 读写机器人状态 / 指令：两端共用，保证语义完全一致
# ==============================================================================
def get_state(data, idx: RobotIndices) -> np.ndarray:
    """读出 8 维 observation.state（6 臂关节 qpos + joint8/joint9 qpos）。

    与采集端 VLARecorder.record_step 里记录的 obs_qpos 完全同构。
    """
    return np.array([data.qpos[a] for a in idx.state_qpos_adr], dtype=np.float32)


def set_ctrl(data, idx: RobotIndices, arm_q, grip) -> np.ndarray:
    """下发 8 维 ctrl，并返回同样的 8 维向量（即训练数据里的 action）。"""
    data.ctrl[idx.arm_act_ids] = arm_q
    data.ctrl[idx.j8_id] = grip[0]
    data.ctrl[idx.j9_id] = grip[1]
    return np.array([*np.asarray(arm_q).reshape(-1)[:6], grip[0], grip[1]], dtype=np.float32)


def reset_to_init(data, idx: RobotIndices) -> None:
    """把机械臂和夹爪复位到采集时的起手状态（含默认闭合的夹爪）。"""
    data.qpos[idx.arm_qpos_adr] = Q_INIT
    data.qpos[idx.grip_qpos_adr] = GRIP_INIT
    data.ctrl[idx.arm_act_ids] = Q_INIT
    data.ctrl[idx.j8_id] = GRIP_INIT[0]
    data.ctrl[idx.j9_id] = GRIP_INIT[1]


# ==============================================================================
# 契约自检
# ==============================================================================
def assert_contract(model, idx: RobotIndices | None = None, verbose: bool = True) -> RobotIndices:
    """在采集端和部署端启动时各调一次。任何一条对不上就当场抛错。

    宁可启动即失败，也不要跑完 50 个 episode 才发现数据是废的。
    """
    problems = []
    idx = idx or resolve_indices(model)

    # 1. 物理步长必须与硬编码的 SIM_TIMESTEP 一致，否则 CONTROL_HZ 是假的
    if abs(model.opt.timestep - SIM_TIMESTEP) > 1e-12:
        problems.append(
            f"场景 timestep = {model.opt.timestep}，但 robot_spec.SIM_TIMESTEP = {SIM_TIMESTEP}。"
            f"实际录制频率是 {1.0 / (model.opt.timestep * STEPS_PER_RECORD):.1f}Hz 而非 {CONTROL_HZ}Hz。"
        )

    # 2. 夹爪反号约定
    if not np.allclose(GRIP_OPEN, [-x for x in GRIP_CLOSE]):
        problems.append("GRIP_OPEN 与 GRIP_CLOSE 不是严格反号。")

    # 3. 夹爪关节行程必须覆盖 ±GRIP_LIMIT（恰好相等是允许的，容差 1e-6）
    for name, adr in zip(GRIP_JOINTS, idx.grip_qpos_adr):
        jnt = model.joint(name)
        if jnt.limited:
            lo, hi = float(jnt.range[0]), float(jnt.range[1])
            if lo > -GRIP_LIMIT + 1e-6 or hi < GRIP_LIMIT - 1e-6:
                problems.append(
                    f"关节 {name} 行程 [{lo:.4f}, {hi:.4f}] 覆盖不了 ±{GRIP_LIMIT}，指令会被 clip。"
                )

    # 4. 关键实体都能找到
    for label, val in [
        ("ee_site", idx.ee_site_id),
        (OBJECT_BODY, idx.object_body_id),
        (STORAGE_BOX_BODY, idx.box_body_id),
    ]:
        if val == -1:
            problems.append(f"场景里找不到 {label}。")

    # 5. qpos[:8] 是否等价于 state（部署端旧代码依赖这个假设）
    if not idx.is_contiguous:
        problems.append(
            f"臂+夹爪在 qpos 里的地址是 {idx.state_qpos_adr.tolist()}，不是 0..7。"
            f"任何 data.qpos[:8] 的写法都会错位，必须改用 robot_spec.get_state()。"
        )

    if problems:
        raise RuntimeError(
            "❌ 机器人契约自检未通过：\n  - " + "\n  - ".join(problems)
        )

    if verbose:
        print(
            f"✅ 契约自检通过 | {CONTROL_HZ}Hz × {STEPS_PER_RECORD} 子步 | "
            f"夹爪 ±{GRIP_LIMIT} 反号 | state 地址 {idx.state_qpos_adr.tolist()}"
        )
    return idx


def load_scene(verbose: bool = True):
    """标准加载流程：读 xml -> 应用相机改动 -> 建 data -> 契约自检。

    采集端和部署端都走这一个入口，就不可能再出现"一端做了镜像另一端没做"。
    """
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    apply_camera_overrides(model, verbose=verbose)
    data = mujoco.MjData(model)
    idx = assert_contract(model, verbose=verbose)
    return model, data, idx