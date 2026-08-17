"""
开环回放自检 (open-loop replay test)

把**训练集里的真实帧**喂给正在运行的 policy server，把返回的动作块和 npz 里的
真值 action 逐维对比。不启动 MuJoCo、不跑物理，只测「模型 + 推理链路」这一段。

它能立刻抓到什么：
  - 夹爪维输出恒定 / 不跨零   -> 夹爪解码或归一化有问题
  - 臂关节 MAE 明显偏大        -> delta/absolute 语义、norm_stats、state 语义对不上
  - 返回维度 < 8               -> LiberoOutputs 把 8 维截成了 7 维
  - 全维度都很差               -> asset_id / checkpoint 用错了数据集

它抓不到什么（这些只能在闭环里暴露）：
  - 相机镜像、vopt 等**渲染侧**的不一致 —— 因为本测试直接读采集时存下来的图片，
    绕过了渲染。渲染一致性由 robot_spec.load_scene() 统一入口来保证。

用法:
    python scripts/test_policy_replay.py                       # 默认取 ep_000
    python scripts/test_policy_replay.py --ep ep_007 --starts 0 40 120
    python scripts/test_policy_replay.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import robot_spec as rspec  # noqa: E402

from openpi_client import websocket_client_policy  # noqa: E402


def load_episode(ep_dir: Path):
    npz = np.load(ep_dir / "joint_data.npz")
    qpos, actions = npz["qpos"], npz["actions"]
    if qpos.shape[0] != actions.shape[0]:
        raise ValueError(f"{ep_dir.name}: qpos {qpos.shape[0]} 帧 != actions {actions.shape[0]} 帧")
    instruction = (ep_dir / "instruction.txt").read_text(encoding="utf-8").strip()
    return qpos, actions, instruction


def load_frame(ep_dir: Path, i: int):
    """读回采集时存的两张图。存的时候做过 RGB->BGR，这里要转回来。"""
    out = []
    for cam in ("cam_fixed", "cam_wrist"):
        p = ep_dir / cam / f"{i:03d}.jpg"
        raw = cv2.imread(str(p))
        if raw is None:
            raise FileNotFoundError(f"读不到 {p}")
        out.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    return out


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    h = min(len(pred), len(gt))
    pred, gt = pred[:h], gt[:h]
    d = min(pred.shape[1], gt.shape[1])
    err = np.abs(pred[:, :d] - gt[:, :d])
    return {
        "horizon": h,
        "pred_dim": pred.shape[1],
        "mae": err.mean(axis=0),
        "arm_mae": err[:, :6].mean(),
        "grip_pred": pred[:, 6] if pred.shape[1] > 6 else None,
        "grip_gt": gt[:, 6],
    }


def report(tag: str, r: dict) -> list[str]:
    warns = []
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    print(f"\n─── {tag} ───")
    print(f"  返回动作块: horizon={r['horizon']}  dim={r['pred_dim']}")
    if r["pred_dim"] < rspec.ACTION_DIM:
        warns.append(
            f"{tag}: server 只返回 {r['pred_dim']} 维（应为 {rspec.ACTION_DIM}）。"
            f"多半是 LiberoOutputs 的 actions[:, :7] 截断 —— 换成自定义 DummyxOutputs。"
        )
    print(f"  逐维 MAE : {r['mae']}")
    print(f"  臂关节 MAE: {r['arm_mae']:.5f} rad")

    if r["arm_mae"] > 0.10:
        warns.append(f"{tag}: 臂关节 MAE {r['arm_mae']:.4f} rad 偏大，检查 norm_stats / delta 语义。")

    gp, gg = r["grip_pred"], r["grip_gt"]
    if gp is None:
        warns.append(f"{tag}: 返回里没有夹爪维。")
        return warns

    span_p, span_g = gp.max() - gp.min(), gg.max() - gg.min()
    print(f"  夹爪 预测 : min={gp.min():+.4f} max={gp.max():+.4f} span={span_p:.4f}")
    print(f"  夹爪 真值 : min={gg.min():+.4f} max={gg.max():+.4f} span={span_g:.4f}")

    if abs(gp).max() > rspec.GRIP_LIMIT * 1.5:
        warns.append(
            f"{tag}: 夹爪预测超出行程 ±{rspec.GRIP_LIMIT} 太多（max |g|={abs(gp).max():.4f}），"
            f"可能用了旧约定(0~0.04)的 norm_stats。"
        )
    if span_g > 1e-3 and span_p < 1e-3:
        warns.append(
            f"{tag}: 真值段内夹爪在动(span={span_g:.4f})，预测却几乎不动(span={span_p:.4f})。"
            f"策略在夹爪维塌到了均值。"
        )
    if span_g > 1e-3 and span_p > 1e-3:
        c = np.corrcoef(gp, gg)[0, 1]
        print(f"  夹爪 相关 : {c:+.3f}")
        if c < -0.3:
            warns.append(f"{tag}: 夹爪预测与真值**负相关**({c:+.3f}) —— 极性反了，检查开/合符号。")
        elif c < 0.3:
            warns.append(f"{tag}: 夹爪预测与真值几乎不相关({c:+.3f})。")
    return warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(rspec.DATASET_DIR))
    ap.add_argument("--ep", default=None, help="episode 目录名，默认取排序后的第一条")
    ap.add_argument("--starts", type=int, nargs="*", default=None,
                    help="从哪些帧开始推理，默认在整条轨迹上均匀取 4 个点")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.ep:
        ep_dir = data_dir / args.ep
    else:
        eps = sorted([d for d in data_dir.glob("ep_*") if d.is_dir()],
                     key=lambda p: int(p.name.split("_")[1]))
        if not eps:
            sys.exit(f"❌ {data_dir} 下没有 ep_* 目录")
        ep_dir = eps[0]

    qpos, actions, instruction = load_episode(ep_dir)
    n = len(qpos)
    print("=" * 78)
    print(f"🔁 开环回放自检  {ep_dir.name}  共 {n} 帧")
    print(f"   指令: {instruction!r}")
    rspec.check_prompt(instruction)   # 训练集自己的指令一定在集合内，不在就是数据脏了
    print("=" * 78)

    starts = args.starts if args.starts else [int(n * f) for f in (0.0, 0.25, 0.5, 0.75)]
    starts = [s for s in starts if 0 <= s < n]

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    all_warns = []
    for s in starts:
        img_f, img_w = load_frame(ep_dir, s)
        result = policy.infer({
            "observation/image": img_f,
            "observation/wrist_image": img_w,
            "observation/state": qpos[s].astype(np.float32),
            "prompt": instruction,
        })
        pred = np.asarray(result["actions"])
        all_warns += report(f"frame {s:>4d}/{n}", evaluate(pred, actions[s:]))

    print("\n" + "=" * 78)
    if all_warns:
        print(f"⚠️  发现 {len(all_warns)} 个问题：")
        for w in all_warns:
            print(f"   - {w}")
        print("=" * 78)
        sys.exit(1)
    print("✅ 全部通过：动作维度、臂关节精度、夹爪极性与幅度均正常。")
    print("=" * 78)


if __name__ == "__main__":
    main()