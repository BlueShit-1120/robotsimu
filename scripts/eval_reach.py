#!/usr/bin/env python3
"""Evaluate M1 Reach PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="M1 Reach PPO eval")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--hover-offset", type=float, default=0.30)
    p.add_argument("--success-dist", type=float, default=0.05)
    p.add_argument("--delta-q-scale", type=float, default=0.06)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    import numpy as np
    from stable_baselines3 import PPO

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    env = PickPlaceMujocoEnv(
        phase="reach",
        hover_offset=args.hover_offset,
        success_dist=args.success_dist,
        delta_q_scale=args.delta_q_scale,
        max_episode_steps=args.max_episode_steps,
        randomize_box=False,
        hold_gripper_open=True,
        early_terminate_success=True,
    )
    model = PPO.load(args.checkpoint, device="cpu")

    successes = 0
    final_dists = []
    returns = []
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
        ok = bool(last_info.get("reach_success") or last_info.get("success"))
        successes += int(ok)
        final_dists.append(float(last_info.get("reach_dist", np.nan)))
        returns.append(ep_ret)
        print(
            f"  ep={ep:02d} success={ok} dist={final_dists[-1]:.4f} ret={ep_ret:.2f}"
        )

    env.close()
    summary = {
        "phase": "reach",
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(args.episodes, 1),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_return": float(np.mean(returns)),
        "hover_offset": args.hover_offset,
        "success_dist": args.success_dist,
    }
    print(
        f"[eval_reach] {successes}/{args.episodes} "
        f"rate={summary['success_rate']:.2f} mean_dist={summary['mean_final_dist']:.4f}"
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[eval_reach] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
