#!/usr/bin/env python3
"""Evaluate M2 Grasp PPO checkpoint (lift / grasp_success)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="M2 Grasp PPO eval")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--hover-offset", type=float, default=0.06)
    p.add_argument("--success-dist", type=float, default=0.05)
    p.add_argument("--lift-height", type=float, default=0.04)
    p.add_argument("--delta-q-scale", type=float, default=None)
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-latch", action="store_true")
    args = p.parse_args()
    if args.delta_q_scale is None:
        args.delta_q_scale = 0.035 if args.no_latch else 0.05

    import numpy as np
    from stable_baselines3 import PPO

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    env = PickPlaceMujocoEnv(
        phase="grasp",
        hover_offset=args.hover_offset,
        success_dist=args.success_dist,
        lift_height=args.lift_height,
        delta_q_scale=args.delta_q_scale,
        max_episode_steps=args.max_episode_steps,
        randomize_box=False,
        hold_gripper_open=False,
        early_terminate_success=True,
        grasp_latch=not args.no_latch,
    )
    model = PPO.load(args.checkpoint, device="cpu")

    successes = 0
    lifts = []
    returns = []
    final_dists = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        last_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        ok = bool(last_info.get("grasp_success") or last_info.get("success"))
        successes += int(ok)
        box_z = float(last_info["object_pose"][2])
        lifts.append(box_z)
        final_dists.append(
            float(
                np.linalg.norm(
                    np.asarray(last_info["ee_pose"][:3])
                    - np.asarray(last_info["object_pose"][:3])
                )
            )
        )
        returns.append(ep_ret)
        if ep < 10 or ok or ep % 20 == 0:
            print(
                f"  ep={ep:03d} grasp={ok} box_z={box_z:.3f} "
                f"ee_box={final_dists[-1]:.3f} ret={ep_ret:.2f}"
            )

    env.close()
    summary = {
        "phase": "grasp",
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(args.episodes, 1),
        "mean_box_z": float(np.mean(lifts)),
        "mean_ee_box_dist": float(np.mean(final_dists)),
        "mean_return": float(np.mean(returns)),
        "hover_offset": args.hover_offset,
        "lift_height": args.lift_height,
        "exit_criterion": "lift success >= 70% over >=100 eval episodes",
        "exit_met": (successes / max(args.episodes, 1)) >= 0.70,
    }
    print(
        f"[eval_grasp] {successes}/{args.episodes} "
        f"rate={summary['success_rate']:.2%} exit_met={summary['exit_met']}"
    )
    out = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).parent / "eval_summary.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[eval_grasp] wrote {out}")
    return 0 if summary["exit_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
