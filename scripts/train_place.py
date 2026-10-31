#!/usr/bin/env python3
"""Train M3/M5 Place PPO (SB3) on PickPlaceMujocoEnv(phase='place').

Curriculum C2–C3: grasp/lift → carry to goal → release & settle.
Optional --resume from a Grasp checkpoint (same obs/act dims).
Optional --bootstrap-grasp: freeze a grasp policy until lift (M5 full-pick).
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
    p = argparse.ArgumentParser(description="M3/M5 Place PPO train")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--out", type=str, default="runs/place_m3")
    p.add_argument("--hover-offset", type=float, default=0.06)
    p.add_argument("--lift-height", type=float, default=0.03)
    p.add_argument("--place-tol-xy", type=float, default=0.12)
    p.add_argument("--place-tol-z", type=float, default=0.04)
    p.add_argument("--place-stable-steps", type=int, default=3)
    p.add_argument(
        "--delta-q-scale",
        type=float,
        default=None,
        help="Arm Δq scale (default 0.05 latch / 0.03 no-latch full-pick)",
    )
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None, help="SB3 zip (grasp or place)")
    p.add_argument(
        "--bootstrap-grasp",
        type=str,
        default=None,
        help="Frozen grasp SB3 zip; actions overridden until lift (M5)",
    )
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--no-latch", action="store_true")
    p.add_argument(
        "--full-pick",
        action="store_true",
        help="Start from open grasp (harder); default starts already holding",
    )
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument(
        "--carry-assist",
        action="store_true",
        help="Train-only: keep grip closed after lift until near place goal",
    )
    p.add_argument(
        "--contact-hold",
        action="store_true",
        help="Acquire kinematic hold only after dual pad↔box contact (M5)",
    )
    args = p.parse_args()
    if args.delta_q_scale is None:
        args.delta_q_scale = 0.03 if (args.no_latch and args.full_pick) else 0.05
    if args.max_episode_steps is None:
        args.max_episode_steps = 250 if args.full_pick else 200

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from robot_simu_rl.backends.mujoco import (
        GraspBootstrapWrapper,
        PickPlaceMujocoEnv,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grasp_model = None
    if args.bootstrap_grasp:
        grasp_model = PPO.load(args.bootstrap_grasp, device=args.device)

    def _make():
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
            contact_hold=bool(args.contact_hold),
            place_start_holding=not args.full_pick,
            carry_assist=bool(args.carry_assist),
        )
        if grasp_model is not None:
            env = GraspBootstrapWrapper(env, grasp_model, deterministic=True)
        return Monitor(env)

    venv = DummyVecEnv([_make for _ in range(args.n_envs)])

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device)
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: float(args.learning_rate)
    else:
        policy_kwargs = dict(log_std_init=-0.5) if args.no_latch else {}
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
            policy_kwargs=policy_kwargs or None,
            tensorboard_log=str(out / "tb"),
        )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0
    ckpt = out / "ppo_place.zip"
    model.save(str(ckpt))
    meta = {
        "phase": "place",
        "timesteps": args.timesteps,
        "elapsed_s": elapsed,
        "fps": args.timesteps / max(elapsed, 1e-6),
        "hover_offset": args.hover_offset,
        "lift_height": args.lift_height,
        "place_tol_xy": args.place_tol_xy,
        "place_tol_z": args.place_tol_z,
        "place_stable_steps": args.place_stable_steps,
        "delta_q_scale": args.delta_q_scale,
        "checkpoint": str(ckpt),
        "resume": args.resume,
        "bootstrap_grasp": args.bootstrap_grasp,
        "grasp_latch": not args.no_latch,
        "place_start_holding": not args.full_pick,
        "full_pick": bool(args.full_pick),
        "carry_assist": bool(args.carry_assist),
        "contact_hold": bool(args.contact_hold),
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "device": args.device,
        "label": "M5 full-pick no-latch",
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train_place] saved {ckpt} elapsed={elapsed:.1f}s fps={meta['fps']:.0f}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
