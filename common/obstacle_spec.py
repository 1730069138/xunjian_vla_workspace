"""障碍物（手电筒）在采集 / 部署两个阶段的开关 —— 建议并入 common/robot_spec.py。

背景：采集示教数据时桌面上不放障碍物，障碍只在推理/部署阶段引入。
两个阶段共用同一份 scenes/xunjian_arm_scene_tidy_v10.xml，靠这里的
set_obstacle() 切换，而不是维护两份场景文件 —— 两份 XML 迟早会漂移，
而漂移在这个项目里从来不报错，只会让策略学出奇怪的东西。

关键点：不能只把障碍物"渲染关掉"。手电筒立在 x=0.45，正好压在
螺丝刀 (0.45, -0.20) 到快递盒 (0.45, +0.20) 的转移航道上。只关渲染的话，
示教轨迹仍会撞上一个看不见的圆柱，采出来的数据莫名其妙地卡顿，
而且没有任何异常抛出。所以必须同时断掉碰撞、挪走、并关掉外观。

用法：
    from common.robot_spec import ObstacleSwitch
    obs = ObstacleSwitch(model)          # 加载模型后建一次，快照原始参数
    obs.set(model, data, enabled=False)  # 采集
    obs.set(model, data, enabled=True)   # 部署，位姿从 qpos0 读回
"""

import numpy as np
import mujoco

OBSTACLE_BODY = "flashlight_obstacle"

# 停放点：桌面下方 5 m，远到任何相机和任何距离场都够不着
PARK_XYZ = (0.0, 0.0, -5.0)


class ObstacleSwitch:
    """在模型加载后构造一次，快照障碍物 geom 的原始碰撞/外观参数。

    快照是必须的：关掉的时候要把 contype/conaffinity/alpha 改掉，
    再打开时得原样还回去。硬编码"原始值"是下一个静默 bug 的温床。
    """

    def __init__(self, model: mujoco.MjModel):
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBSTACLE_BODY)
        if self.bid < 0:
            raise KeyError(
                f"场景里找不到 body '{OBSTACLE_BODY}'。"
                "如果用的是 tidy_B_record_preview.xml，那份文件本来就没有障碍物，"
                "不要在它上面调 ObstacleSwitch。"
            )
        g0 = model.body_geomadr[self.bid]
        self.gids = list(range(g0, g0 + model.body_geomnum[self.bid]))

        self.contype = model.geom_contype[self.gids].copy()
        self.conaffinity = model.geom_conaffinity[self.gids].copy()
        self.rgba = model.geom_rgba[self.gids].copy()

        jadr = model.body_jntadr[self.bid]
        if jadr < 0 or model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"'{OBSTACLE_BODY}' 必须带 freejoint 才能停放")
        self.qadr = model.jnt_qposadr[jadr]
        self.vadr = model.jnt_dofadr[jadr]

        # 摆放位姿的唯一来源是 XML 编译出的 qpos0，不在 Python 里抄一遍坐标。
        # 以后改了 XML 里的障碍物位置，这里自动跟上。
        self.home_qpos = model.qpos0[self.qadr:self.qadr + 7].copy()

    def set(self, model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> None:
        if enabled:
            model.geom_contype[self.gids] = self.contype
            model.geom_conaffinity[self.gids] = self.conaffinity
            model.geom_rgba[self.gids] = self.rgba
            data.qpos[self.qadr:self.qadr + 7] = self.home_qpos
        else:
            model.geom_contype[self.gids] = 0
            model.geom_conaffinity[self.gids] = 0
            model.geom_rgba[self.gids, 3] = 0.0
            # 停放后它会一直自由落体。这没关系：contype 已置 0，不产生任何
            # 接触，也不在任何相机视野里，只是 qvel 单调增大而已。
            # 顺带记一笔踩过的坑：想用重力补偿把它钉住是行不通的 ——
            # model.body_gravcomp[bid] = 1.0 在运行时改毫无效果，因为
            # model.ngravcomp 是编译期统计的，XML 里没写 gravcomp 就是 0，
            # 整个重力补偿计算会被直接跳过。实测验证过。
            # 而在 XML 里写 gravcomp="1" 又会让它摆上桌面时失重、撞了也不倒，
            # 破坏 R[2,2] < 0.9 的倾倒判据。所以就让它落。
            data.qpos[self.qadr:self.qadr + 3] = PARK_XYZ
            data.qpos[self.qadr + 3:self.qadr + 7] = (1.0, 0.0, 0.0, 0.0)

        data.qvel[self.vadr:self.vadr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def assert_state(self, model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> None:
        """每个 episode 开录前调一次。宁可当场断言失败，
        也不要采完 150 条才发现障碍物混进了训练集。"""
        z = float(data.qpos[self.qadr + 2])
        on_table = z > 0.5
        assert on_table == enabled, (
            f"障碍物状态不符：期望 enabled={enabled}，实际 z={z:.3f}"
        )
        alpha = float(np.max(model.geom_rgba[self.gids, 3]))
        assert (alpha > 0.0) == enabled, (
            f"障碍物外观不符：期望 enabled={enabled}，实际 max alpha={alpha:.3f}"
        )
