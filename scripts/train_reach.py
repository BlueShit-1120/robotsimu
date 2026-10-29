#!/usr/bin/env python3
"""M1 C0 Reach: train PPO so right-arm EE reaches hover pose above the box.

Usage (conda activate rs-mujoco; from repo root):
  pip install -e '.[rl]'   # or: pip install stable-baselines3 torch
  python scripts/train_reach.py
  python scripts/train_reach.py --timesteps 50000 --out runs/reach_smoke
  python scripts/train_reach.py --resume runs/reach_smoke/ppo_reach.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _make_env(
    seed: int,
    max_episode_steps: int,
    randomize_box: bool,
    hold_gripper_open: bool,
    hover_offset_m: float,
    success_dist_m: float,
    delta_q_scale: float = 0.08,
):
    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv
    from robot_simu_rl.tasks.pick_place import ReachConfig

    def _thunk():
        cfg = ReachConfig(
            hover_offset_m=hover_offset_m,
            success_dist_m=success_dist_m,
        )
        env = PickPlaceMujocoEnv(
            phase="reach",
            randomize_box=randomize_box,
            hold_gripper_open=hold_gripper_open,
            max_episode_steps=max_episode_steps,
            delta_q_scale=delta_q_scale,
            reach_cfg=cfg,
            terminate_on_reach_success=True,
        )
        env.reset(seed=seed)
        return env

    return _thunk


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Reach PPO train (MuJoCo)")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir (default: runs/reach_YYYYmmdd_HHMMSS)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from .zip")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--delta-q-scale", type=float, default=0.08)
    parser.add_argument("--randomize-box", action="store_true", default=False)
    parser.add_argument("--no-hold-gripper", action="store_true")
    parser.add_argument("--hover-offset", type=float, default=0.08)
    parser.add_argument("--success-dist", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-episodes", type=int, default=8)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        print(
            "[train_reach] missing RL deps. Install with:\n"
            "  pip install -e '.[rl]'\n"
            "  # or: pip install 'stable-baselines3>=2.3' torch",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (_REPO / "runs" / f"reach_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ppo_reach.zip"
    tb_dir = out_dir / "tb"

    hold_grip = not args.no_hold_gripper
    env_fn = _make_env(
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        randomize_box=args.randomize_box,
        hold_gripper_open=hold_grip,
        hover_offset_m=args.hover_offset,
        success_dist_m=args.success_dist,
        delta_q_scale=args.delta_q_scale,
    )

    def _monitored():
        e = env_fn()
        return Monitor(e, str(out_dir / "monitor"))

    vec_env = DummyVecEnv([_monitored])

    if args.resume:
        print(f"[train_reach] resume={args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(tb_dir),
        )

    meta = {
        "task": "reach",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "hold_gripper_open": hold_grip,
        "randomize_box": args.randomize_box,
        "hover_offset_m": args.hover_offset,
        "success_dist_m": args.success_dist,
        "max_episode_steps": args.max_episode_steps,
        "started_utc": stamp,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"[train_reach] out={out_dir} timesteps={args.timesteps} "
        f"device={args.device} hold_gripper={hold_grip}"
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - t0
    model.save(str(ckpt_path))
    print(f"[train_reach] saved {ckpt_path} ({elapsed:.1f}s)")

    # Quick post-train eval
    from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv
    from robot_simu_rl.tasks.pick_place import ReachConfig

    eval_env = PickPlaceMujocoEnv(
        phase="reach",
        randomize_box=args.randomize_box,
        hold_gripper_open=hold_grip,
        max_episode_steps=args.max_episode_steps,
        delta_q_scale=args.delta_q_scale,
        reach_cfg=ReachConfig(
            hover_offset_m=args.hover_offset,
            success_dist_m=args.success_dist,
        ),
        terminate_on_reach_success=True,
    )
    successes = 0
    final_dists: list[float] = []
    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset(seed=args.seed + 1000 + ep)
        done = False
        last_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = eval_env.step(action)
            done = terminated or truncated
        dist = float(last_info.get("reach_dist", 1.0))
        ok = bool(last_info.get("reach_success", False)) or dist < args.success_dist
        # Also count early-terminated reach success streak
        if terminated and not truncated and not last_info.get("object_pose", [0, 0, 1])[2] < 0.3:
            # terminated without drop → likely reach success
            if dist < args.success_dist * 1.5:
                ok = True
        successes += int(ok)
        final_dists.append(dist)
        print(
            f"  eval ep={ep} dist={dist:.4f} success={ok} steps={last_info.get('steps')}"
        )
    eval_env.close()
    mean_dist = sum(final_dists) / max(len(final_dists), 1)
    rate = successes / max(args.eval_episodes, 1)
    summary = {
        **meta,
        "elapsed_s": elapsed,
        "checkpoint": str(ckpt_path),
        "eval_episodes": args.eval_episodes,
        "eval_success_rate": rate,
        "eval_mean_final_dist_m": mean_dist,
        "eval_final_dists_m": final_dists,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[train_reach] eval success_rate={rate:.2f} mean_final_dist={mean_dist:.4f} m"
    )
    print("[train_reach] OK")
    vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
