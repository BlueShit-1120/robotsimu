#!/usr/bin/env python3
"""Train M2 Grasp PPO (SB3) on PickPlaceMujocoEnv(phase='grasp').

Curriculum C1: approach → close gripper → lift. Place reward off.
Optional --resume from a Reach checkpoint (policy transfer; same obs/act dims).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="M2 Grasp PPO train")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--out", type=str, default="runs/grasp_m2")
    p.add_argument("--hover-offset", type=float, default=0.06)
    p.add_argument("--success-dist", type=float, default=0.05)
    p.add_argument("--lift-height", type=float, default=0.03)
    p.add_argument(
        "--delta-q-scale",
        type=float,
        default=None,
        help="Arm Δq scale (default 0.05 latch / 0.035 no-latch)",
    )
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="SB3 zip (reach or grasp)")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument(
        "--no-latch",
        action="store_true",
        help="Disable kinematic grasp latch (hard contact-only mode)",
    )
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.02)
    args = p.parse_args()
    if args.delta_q_scale is None:
        args.delta_q_scale = 0.035 if args.no_latch else 0.05

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _make():
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
        return Monitor(env)

    venv = DummyVecEnv([_make for _ in range(args.n_envs)])

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device)
        # Allow continued learning with ent_coef for exploration on grasp
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
    else:
        policy_kwargs = dict(log_std_init=-1.0)
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            device=args.device,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(out / "tb"),
        )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0
    ckpt = out / "ppo_grasp.zip"
    model.save(str(ckpt))
    meta = {
        "phase": "grasp",
        "timesteps": args.timesteps,
        "elapsed_s": elapsed,
        "fps": args.timesteps / max(elapsed, 1e-6),
        "hover_offset": args.hover_offset,
        "success_dist": args.success_dist,
        "lift_height": args.lift_height,
        "delta_q_scale": args.delta_q_scale,
        "checkpoint": str(ckpt),
        "resume": args.resume,
        "grasp_latch": not args.no_latch,
        "seed": args.seed,
        "device": args.device,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_grasp] saved {ckpt} elapsed={elapsed:.1f}s fps={meta['fps']:.0f}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
