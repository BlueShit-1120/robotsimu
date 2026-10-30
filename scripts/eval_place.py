#!/usr/bin/env python3
"""Evaluate M3 Place PPO checkpoint (full pick-place success)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="M3 Place PPO eval")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--hover-offset", type=float, default=0.06)
    p.add_argument("--lift-height", type=float, default=0.03)
    p.add_argument("--place-tol-xy", type=float, default=0.12)
    p.add_argument("--place-tol-z", type=float, default=0.04)
    p.add_argument("--place-stable-steps", type=int, default=3)
    p.add_argument("--delta-q-scale", type=float, default=0.05)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-latch", action="store_true")
    p.add_argument("--full-pick", action="store_true")
    args = p.parse_args()

    import numpy as np
    from stable_baselines3 import PPO

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    env = PickPlaceMujocoEnv(
        phase="place",
        hover_offset=args.hover_offset,
        lift_height=args.lift_height,
        place_tol_xy=args.place_tol_xy,
        place_tol_z=args.place_tol_z,
        place_stable_steps=args.place_stable_steps,
        delta_q_scale=args.delta_q_scale,
        max_episode_steps=args.max_episode_steps,
        randomize_box=False,
        hold_gripper_open=False,
        early_terminate_success=True,
        grasp_latch=not args.no_latch,
        place_start_holding=not args.full_pick,
    )
    model = PPO.load(args.checkpoint, device="cpu")

    successes = 0
    grasps = 0
    place_dists = []
    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        last_info = info
        saw_grasp = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_ret += float(reward)
            if last_info.get("grasp_success") or last_info.get("latched"):
                saw_grasp = True
            done = terminated or truncated
        ok = bool(last_info.get("place_success") or last_info.get("success"))
        successes += int(ok)
        grasps += int(saw_grasp or last_info.get("grasp_success", False))
        place_dists.append(float(last_info.get("place_dist_xy", np.nan)))
        returns.append(ep_ret)
        if ep < 10 or ok or ep % 20 == 0:
            print(
                f"  ep={ep:03d} place={ok} grasp/latch={saw_grasp} "
                f"xy={place_dists[-1]:.3f} box_z={float(last_info['object_pose'][2]):.3f} "
                f"ret={ep_ret:.2f}"
            )

    env.close()
    rate = successes / max(args.episodes, 1)
    summary = {
        "phase": "place",
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": rate,
        "grasp_or_latch_rate": grasps / max(args.episodes, 1),
        "mean_place_dist_xy": float(np.nanmean(place_dists)),
        "mean_return": float(np.mean(returns)),
        "place_tol_xy": args.place_tol_xy,
        "place_tol_z": args.place_tol_z,
        "lift_height": args.lift_height,
        "exit_criterion": "full pick-place >= 50% over >=100 eval episodes",
        "exit_met": rate >= 0.50,
    }
    print(
        f"[eval_place] {successes}/{args.episodes} "
        f"rate={rate:.2%} exit_met={summary['exit_met']}"
    )
    out = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).parent / "eval_summary.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[eval_place] wrote {out}")
    return 0 if summary["exit_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
