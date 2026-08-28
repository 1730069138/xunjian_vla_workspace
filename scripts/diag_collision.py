#!/usr/bin/env python3
"""
碰撞体诊断：定位"下抓时被顶住抓不深"的元凶。

用法（在 xunjian_vla_workspace 根目录下）：
    python scripts/diag_collision.py                # 只列静态碰撞体清单
    python scripts/diag_collision.py --probe        # 额外做末端下探扫描，打印首次接触

把 SCENE 改成你的实际路径，或直接用 common.robot_spec.SCENE_PATH。
"""

import argparse
import numpy as np
import mujoco

SCENE = "scenes/xunjian_arm_scene.xml"
EE_SITE = "ee_site"


def name_of(model, objtype, i):
    n = mujoco.mj_id2name(model, objtype, i)
    return n if n else f"<unnamed_{i}>"


def can_collide(model, a, b):
    """MuJoCo 的 contype/conaffinity 位掩码规则。"""
    return bool(
        (model.geom_contype[a] & model.geom_conaffinity[b])
        or (model.geom_contype[b] & model.geom_conaffinity[a])
    )


def dump_geoms(model):
    print(f"\n{'='*100}")
    print("全部 geom 清单（★ = 参与碰撞）")
    print(f"{'='*100}")
    hdr = f"{'id':>3} {'name':<30} {'body':<22} {'type':>4} {'ct':>3} {'ca':>3} {'grp':>3} {'rbound':>8}"
    print(hdr)
    print("-" * 100)
    for i in range(model.ngeom):
        gname = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
        ct, ca = model.geom_contype[i], model.geom_conaffinity[i]
        star = "★" if (ct and ca) or ct or ca else " "
        print(
            f"{star}{i:>3} {gname:<30} {bname:<22} {model.geom_type[i]:>4} "
            f"{ct:>3} {ca:>3} {model.geom_group[i]:>3} {model.geom_rbound[i]:>8.4f}"
        )


def dump_ee_vs_screwdriver(model):
    """末端相关 geom 与螺丝刀碰撞体的配对关系 + 包围盒尺寸对比。"""
    screw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    screw_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == screw_bid]

    print(f"\n{'='*100}")
    print("螺丝刀 body 上的 geom —— 注意视觉 mesh 与真实碰撞体的差别")
    print(f"{'='*100}")
    for i in screw_geoms:
        aabb = model.geom_aabb[i]  # [cx cy cz  hx hy hz]
        print(
            f"  {name_of(model, mujoco.mjtObj.mjOBJ_GEOM, i):<30} "
            f"ct={model.geom_contype[i]} ca={model.geom_conaffinity[i]} "
            f"rbound={model.geom_rbound[i]:.4f}  "
            f"half_extent=({aabb[3]:.4f}, {aabb[4]:.4f}, {aabb[5]:.4f})"
        )
    collidable_screw = [i for i in screw_geoms if model.geom_contype[i] or model.geom_conaffinity[i]]

    # 末端相关：ee_site 所在 body 及其所有祖先 body 上的 geom
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if sid == -1:
        print(f"\n[!] 找不到 site '{EE_SITE}' —— 末端小球可能是 geom 而不是 site，见上面清单。")
        return
    ee_bid = model.site_bodyid[sid]
    print(f"\nee_site 挂在 body: {name_of(model, mujoco.mjtObj.mjOBJ_BODY, ee_bid)}  "
          f"(site 本身不参与任何碰撞)")

    ee_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == ee_bid]
    if not ee_geoms:
        print("  该 body 上没有 geom。")
    print(f"\n{'='*100}")
    print("末端 body 的 geom 是否会与螺丝刀碰撞体成对")
    print(f"{'='*100}")
    for g in ee_geoms:
        hits = [s for s in collidable_screw if can_collide(model, g, s)]
        tag = "会碰撞 →" if hits else "不碰撞  "
        print(f"  {tag} {name_of(model, mujoco.mjtObj.mjOBJ_GEOM, g):<30} "
              f"size={model.geom_size[g]}  rbound={model.geom_rbound[g]:.4f}")


def probe(model, data):
    """把手臂钉在初始位姿跑几步，打印当前所有接触。"""
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    print(f"\n{'='*100}")
    print(f"沉降 200 步后的接触对（ncon={data.ncon}）")
    print(f"{'='*100}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = name_of(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"  {g1:<30} <-> {g2:<30} dist={c.dist:+.5f}  pos_z={c.pos[2]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--probe", action="store_true", help="额外跑 200 步并打印接触对")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    dump_geoms(model)
    dump_ee_vs_screwdriver(model)
    if args.probe:
        probe(model, data)


if __name__ == "__main__":
    main()
