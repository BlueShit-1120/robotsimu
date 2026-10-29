#!/usr/bin/env python3
"""Evaluate a Reach PPO checkpoint (or random policy smoke).

Usage:
  python scripts/eval_reach.py --checkpoint runs/reach_*/ppo_reach.zip
  python scripts/eval_reach.py --random --episodes 5   # no checkpoint smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Reach eval / smoke")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random", action="store_true", help="Random policy (no ckpt)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--success-dist", type=float, default=0.03)
    parser.add_argument("--hover-offset", type=float, default=0.08)
    parser.add_argument("--randomize-box", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv
    from robot_simu_rl.tasks.pick_place import ReachConfig

    model = None
    if not args.random:
        if not args.checkpoint:
            print("[eval_reach] need --checkpoint or --random", file=sys.stderr)
            return 2
        from stable_baselines3 import PPO

        model = PPO.load(args.checkpoint, device=args.device)
        print(f"[eval_reach] loaded {args.checkpoint}")

    env = PickPlaceMujocoEnv(
        phase="reach",
        randomize_box=args.randomize_box,
        hold_gripper_open=True,
        max_episode_steps=args.max_episode_steps,
        reach_cfg=ReachConfig(
            hover_offset_m=args.hover_offset,
            success_dist_m=args.success_dist,
        ),
        terminate_on_reach_success=True,
    )

    successes = 0
    dists: list[float] = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        last_info = info
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
        dist = float(last_info["reach_dist"])
        ok = dist < args.success_dist or bool(last_info.get("reach_success"))
        if terminated and not truncated and dist < args.success_dist * 1.5:
            ok = True
        successes += int(ok)
        dists.append(dist)
        print(
            f"  ep={ep} dist={dist:.4f} success={ok} "
            f"steps={last_info.get('steps')} r_end={reward:.3f}"
        )

    env.close()
    rate = successes / max(args.episodes, 1)
    mean_d = sum(dists) / max(len(dists), 1)
    summary = {
        "episodes": args.episodes,
        "success_rate": rate,
        "mean_final_dist_m": mean_d,
        "final_dists_m": dists,
        "checkpoint": args.checkpoint,
        "random": args.random,
    }
    print(json.dumps(summary, indent=2))
    print(f"[eval_reach] success_rate={rate:.2f} mean_dist={mean_d:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
