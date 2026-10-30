#!/usr/bin/env python3
"""Train M1 Reach PPO (SB3) on PickPlaceMujocoEnv(phase='reach')."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="M1 Reach PPO train")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--out", type=str, default="runs/reach_m1")
    p.add_argument("--hover-offset", type=float, default=0.30)
    p.add_argument("--success-dist", type=float, default=0.05)
    p.add_argument("--delta-q-scale", type=float, default=0.06)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--n-envs", type=int, default=1)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _make():
        env = PickPlaceMujocoEnv(
            phase="reach",
            hover_offset=args.hover_offset,
            success_dist=args.success_dist,
            delta_q_scale=args.delta_q_scale,
            max_episode_steps=args.max_episode_steps,
            randomize_box=False,
            hold_gripper_open=True,
        )
        return Monitor(env)

    venv = DummyVecEnv([_make for _ in range(args.n_envs)])

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device=args.device,
            seed=args.seed,
            tensorboard_log=str(out / "tb"),
        )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0
    ckpt = out / "ppo_reach.zip"
    model.save(str(ckpt))
    meta = {
        "phase": "reach",
        "timesteps": args.timesteps,
        "elapsed_s": elapsed,
        "fps": args.timesteps / max(elapsed, 1e-6),
        "hover_offset": args.hover_offset,
        "success_dist": args.success_dist,
        "delta_q_scale": args.delta_q_scale,
        "checkpoint": str(ckpt),
        "seed": args.seed,
        "device": args.device,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_reach] saved {ckpt} elapsed={elapsed:.1f}s fps={meta['fps']:.0f}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
