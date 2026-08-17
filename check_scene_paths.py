#!/usr/bin/env python3
"""按 MuJoCo 的规则解析场景里每一条资产路径，报告哪些文件缺失。

在 MuJoCo 报一个含糊的 "Error: could not open file" 之前先跑这个，
可以一次看到全部缺失项，而不是修一个报一个。

用法（工作区根目录下）：
    python3 check_scene_paths.py scenes/xunjian_arm_scene_tidy.xml
"""
import os
import re
import sys
import xml.etree.ElementTree as ET


def load_with_includes(path, seen=None):
    """把 <include> 递归展开，返回 (根元素, 每个元素来自哪个文件) 。"""
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        return None, {}
    seen.add(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        line, col = e.position
        print(f"XML 格式错误: {path}")
        print(f"  第 {line} 行第 {col} 列: {e}")
        try:
            src = open(path, encoding="utf-8").read().split("\n")[line - 1]
            print(f"  内容: {src.strip()[:100]}")
            if "--" in src and src.strip().startswith(("-", "<!--")):
                print("  提示: XML 注释里不允许出现连续两个减号 '--'（包括用减号"
                      "画的分隔线）。MuJoCo 的 TinyXML 会放行，但标准解析器不会。")
        except (OSError, IndexError):
            pass
        return None, {}
    origin = {id(root): path}
    for parent in list(root.iter()):
        for child in list(parent):
            origin[id(child)] = path
            if child.tag == "include":
                inc = os.path.normpath(
                    os.path.join(os.path.dirname(path), child.get("file")))
                if not os.path.exists(inc):
                    print(f"  [缺失] include -> {inc}")
                    continue
                sub, sub_origin = load_with_includes(inc, seen)
                if sub is None:
                    continue
                origin.update(sub_origin)
                idx = list(parent).index(child)
                parent.remove(child)
                for off, node in enumerate(list(sub)):
                    parent.insert(idx + off, node)
    return root, origin


def main(scene):
    scene = os.path.abspath(scene)
    base = os.path.dirname(scene)          # MuJoCo 以主模型文件所在目录为基准
    root, _ = load_with_includes(scene)
    if root is None:
        sys.exit("无法解析场景文件")

    comp = root.find("compiler")
    assetdir = comp.get("assetdir") if comp is not None else None
    meshdir = (comp.get("meshdir") if comp is not None else None) or assetdir or ""
    texdir = (comp.get("texturedir") if comp is not None else None) or assetdir or ""
    print(f"主模型目录 : {base}")
    print(f"meshdir    : {meshdir!r}")
    print(f"texturedir : {texdir!r}\n")

    rows, missing = [], 0
    for tag, d in (("mesh", meshdir), ("texture", texdir), ("hfield", meshdir)):
        for el in root.iter(tag):
            f = el.get("file")
            if not f:
                continue                    # builtin 贴图没有 file
            resolved = os.path.normpath(os.path.join(base, d, f))
            ok = os.path.exists(resolved)
            missing += (not ok)
            rows.append((("OK " if ok else "缺失"), tag,
                         el.get("name") or "-", f, resolved))

    w = max(len(r[3]) for r in rows) if rows else 10
    for status, tag, name, f, resolved in rows:
        print(f"[{status}] {tag:<7} {name:<22} {f:<{w}}  ->  {resolved}")

    print()
    if missing:
        print(f"共 {missing} 个资产缺失。")
        sys.exit(1)
    print(f"全部 {len(rows)} 个资产路径解析成功。")

    try:
        import mujoco
        mujoco.MjModel.from_xml_path(scene)
        print("MuJoCo 编译通过。")
    except ImportError:
        print("（未安装 mujoco，跳过编译检查）")
    except Exception as e:
        print(f"MuJoCo 编译失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
