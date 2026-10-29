#!/usr/bin/env python3
"""M0 smoke: load pick-place scene and step a random / zero policy without crash.

Usage (from repo root, env rs-mujoco or pip install -e .):
  python scripts/smoke_mujoco.py
  python scripts/smoke_mujoco.py --steps 200 --seed 0
  python scripts/smoke_mujoco.py --policy zero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="M0 MuJoCo pick-place smoke test")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("random", "zero", "open_close"),
        default="random",
        help="Action policy for the smoke roll-out",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional MJCF path (default: assets/mjcf/pick_place_scene.xml)",
    )
    args = parser.parse_args()

    import numpy as np
    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    env = PickPlaceMujocoEnv(model_path=args.model, randomize_box=True)
    obs, info = env.reset(seed=args.seed)
    print(
        f"[smoke] model={env.model_path.name} nq={env.model.nq} nu={env.model.nu} "
        f"obs_dim={obs.shape} action_dim={env.action_space.shape}"
    )
    print(f"[smoke] box0={info['object_pose'][:3]} ee0={info['ee_pose'][:3]}")

    grip_phase = 0.0
    for t in range(args.steps):
        if args.policy == "zero":
            action = np.zeros(8, dtype=np.float32)
            action[7] = 0.0
        elif args.policy == "open_close":
            action = np.zeros(8, dtype=np.float32)
            grip_phase += 0.02
            action[7] = 0.5 * (1.0 + np.sin(grip_phase))
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        if t % 25 == 0 or terminated or truncated:
            print(
                f"  t={t:4d} r={reward:7.3f} box_z={info['object_pose'][2]:.3f} "
                f"term={terminated} trunc={truncated}"
            )
        if terminated or truncated:
            obs, info = env.reset()
            print("  (reset)")

    env.close()
    print("[smoke] OK — completed without crash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
