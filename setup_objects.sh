#!/usr/bin/env bash
# =============================================================================
# 在 xunjian_vla_workspace 下布置桌面归位任务所需的三个物体资产。
#
# 用法（在工作区根目录 ~/xunjian_vla_workspace 下运行）：
#     bash setup_objects.sh
#
# 前置条件：multimeter 的两张贴图 mm_panel.png / mm_case.png 已经在当前目录，
# 或者用 make_mm_texture.py 现场生成。
# =============================================================================
set -euo pipefail

WS="$(pwd)"
if [[ ! -d "$WS/scenes" || ! -d "$WS/models" ]]; then
  echo "错误：请在 xunjian_vla_workspace 根目录下运行（需要能看到 scenes/ 和 models/）" >&2
  exit 1
fi

OBJ="$WS/assets/objects"
mkdir -p "$OBJ/screwdriver" "$OBJ/flashlight" "$OBJ/multimeter"

# ---------- 1. 从 mujoco_scanned_objects 稀疏拉取两个 GSO 模型 ----------
TMP="$(mktemp -d)"
echo "[1/3] 拉取 GSO 模型到 $TMP ..."
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/kevinzakka/mujoco_scanned_objects.git "$TMP/mso"
cd "$TMP/mso"
git sparse-checkout init --cone
git sparse-checkout set \
    models/Craftsman_Grip_Screwdriver_Phillips_Cushion \
    models/HeavyDuty_Flashlight
git checkout

# 只取视觉网格和贴图。32 个 model_collision_*.obj 不需要：
# 场景里的碰撞体用 primitive（胶囊/圆柱/长方体）近似。
cp models/Craftsman_Grip_Screwdriver_Phillips_Cushion/model.obj   "$OBJ/screwdriver/model.obj"
cp models/Craftsman_Grip_Screwdriver_Phillips_Cushion/texture.png "$OBJ/screwdriver/texture.png"
cp models/HeavyDuty_Flashlight/model.obj                          "$OBJ/flashlight/model.obj"
cp models/HeavyDuty_Flashlight/texture.png                        "$OBJ/flashlight/texture.png"
cd "$WS"
rm -rf "$TMP"

# ---------- 2. 万用表贴图 ----------
echo "[2/3] 布置万用表贴图 ..."
if [[ -f "$WS/mm_panel.png" && -f "$WS/mm_case.png" ]]; then
  cp "$WS/mm_panel.png" "$WS/mm_case.png" "$OBJ/multimeter/"
elif [[ -f "$WS/make_mm_texture.py" ]]; then
  python3 - <<PY
import sys, importlib.util
spec = importlib.util.spec_from_file_location("mk", "$WS/make_mm_texture.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
mk.make_panel().save("$OBJ/multimeter/mm_panel.png")
mk.make_case().save("$OBJ/multimeter/mm_case.png")
PY
else
  echo "  跳过：找不到 mm_panel.png / mm_case.png，也找不到 make_mm_texture.py" >&2
fi

# ---------- 3. 结果 ----------
echo "[3/3] 完成。assets/objects/ 目录内容："
find "$OBJ" -type f | sed "s|$WS/||" | sort
