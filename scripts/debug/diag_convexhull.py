#!/usr/bin/env python3
"""
凸包膨胀诊断：MuJoCo 的 mesh 碰撞用的是凸包，不是你看到的网格。
夹爪手掌那种带凹槽的形状，凸包会把槽填平，形成一块看不见的实心挡板。

用法（在 xunjian_vla_workspace 根目录）：
    python scripts/debug/diag_convexhull.py
    python scripts/debug/diag_convexhull.py --bodies link7 link8 link9 --site ee_site

判读：
    hull/mesh 体积比 ≈ 1.0  -> 网格本身就是凸的，碰撞形状 = 你看到的形状
    hull/mesh 体积比 > 1.3  -> 凸包填掉了腔体，存在"看不见的碰撞体"
"""

import argparse
import numpy as np
import mujoco

try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SCENE = "scenes/xunjian_arm_scene.xml"


def mesh_arrays(model, mesh_id):
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = np.array(model.mesh_vert[va: va + vn], dtype=float).reshape(-1, 3)
    faces = np.array(model.mesh_face[fa: fa + fn], dtype=int).reshape(-1, 3)
    return verts, faces


def mesh_volume(verts, faces):
    """闭合三角网格的有符号体积（散度定理）。"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--bodies", nargs="*", default=["link7", "link8", "link9"])
    ap.add_argument("--site", default="ee_site")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)

    if not HAVE_SCIPY:
        print("[!] 没装 scipy，凸包体积算不了（pip install scipy）。仍会输出包围盒信息。\n")

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    site_body = model.site_bodyid[sid] if sid != -1 else -1
    site_pos_local = np.array(model.site_pos[sid], dtype=float) if sid != -1 else None

    for bname in args.bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid == -1:
            print(f"[!] 找不到 body '{bname}'，跳过。")
            continue

        print("=" * 92)
        print(f"body: {bname}")
        print("=" * 92)

        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            collidable = bool(model.geom_contype[g] or model.geom_conaffinity[g])
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}"

            mesh_id = model.geom_dataid[g]
            verts, faces = mesh_arrays(model, mesh_id)

            # mesh 局部坐标 -> body 坐标
            R = quat_to_mat(model.geom_quat[g])
            p = np.array(model.geom_pos[g], dtype=float)
            vb = verts @ R.T + p

            v_mesh = mesh_volume(verts, faces)
            lo, hi = vb.min(axis=0), vb.max(axis=0)

            tag = "★碰撞" if collidable else " 视觉"
            print(f"\n  {tag}  {gname}   (mesh id {mesh_id}, {len(verts)} 顶点 / {len(faces)} 面)")
            print(f"        body 系包围盒  x[{lo[0]:+.4f}, {hi[0]:+.4f}]  "
                  f"y[{lo[1]:+.4f}, {hi[1]:+.4f}]  z[{lo[2]:+.4f}, {hi[2]:+.4f}]")

            if HAVE_SCIPY and len(verts) >= 4:
                try:
                    hull = ConvexHull(verts)
                    ratio = hull.volume / v_mesh if v_mesh > 1e-12 else float("nan")
                    verdict = ("✅ 凸包≈网格，无隐形碰撞体"
                               if ratio < 1.15 else
                               "⚠️  凸包比网格大不少，腔体被填平"
                               if ratio < 1.6 else
                               "🚨 凸包远大于网格 —— 开口/凹槽被完全填死")
                    print(f"        网格体积 {v_mesh*1e6:9.2f} cm³   "
                          f"凸包体积 {hull.volume*1e6:9.2f} cm³   "
                          f"比值 {ratio:5.2f}   {verdict}")
                except Exception as e:  # 退化网格等
                    print(f"        [凸包计算失败: {e}]")

            if collidable and sid != -1 and site_body == bid:
                d = lo - site_pos_local
                print(f"        相对 {args.site}：凸包最低点在 "
                      f"x{d[0]:+.4f} y{d[1]:+.4f} z{d[2]:+.4f} 处")
                print(f"        （若某个方向上 TCP 位于凸包内部，该方向就永远压不下去）")
                inside = np.all(site_pos_local >= lo) and np.all(site_pos_local <= hi)
                if inside:
                    print(f"        🚨 {args.site} 落在该碰撞体的包围盒内部！")

    print("\n" + "=" * 92)
    print("下一步：在 viewer 里开 mjVIS_CONVEXHULL 目视确认")
    print("=" * 92)
    print("    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True")


if __name__ == "__main__":
    main()
