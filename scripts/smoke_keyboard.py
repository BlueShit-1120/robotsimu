#!/usr/bin/env python3
"""Optional keyboard teleop smoke (requires a display + mujoco viewer).

Keys (when viewer focused — printed on start):
  random policy fallback if no display: same as smoke_mujoco.py --policy open_close

This script primarily validates viewer launch; headless CI should use smoke_mujoco.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import numpy as np
    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    env = PickPlaceMujocoEnv(render_mode="human", randomize_box=True)
    obs, info = env.reset(seed=args.seed)
    print("[keyboard] Launching passive viewer if display available…")
    print("[keyboard] No key bindings wired yet — cycling open/close gripper + small arm noise.")
    print("[keyboard] For headless smoke use: python scripts/smoke_mujoco.py")

    phase = 0.0
    try:
        for t in range(args.steps):
            phase += 0.03
            action = np.zeros(8, dtype=np.float32)
            action[:7] = 0.05 * np.sin(phase + np.arange(7) * 0.4)
            action[7] = 0.5 * (1.0 + np.sin(phase * 0.5))
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()
    print("[keyboard] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
