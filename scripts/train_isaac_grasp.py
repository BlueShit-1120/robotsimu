#!/usr/bin/env python3
"""Isaac Lab Grasp PPO train (SB3) on ManagerBased ReachGraspEnv.

Defaults lean on M12 control fixes (latched Δq, reward dt weight, stiff arm) and
MuJoCo M4 no-latch contact knobs (hover=0.06, lift=0.03, delta_q_scale=0.03).

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/train_isaac_grasp.py \\
        --headless --timesteps 100000 --num-envs 4 --out runs/isaac_grasp_m13 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab Grasp PPO train (SB3).")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", type=str, default="runs/isaac_grasp_m13")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--hover-offset", type=float, default=0.06)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--success-dist", type=float, default=0.05)
    # M4 no-latch contact used 0.03; Isaac has no kinematic latch.
    parser.add_argument("--delta-q-scale", type=float, default=0.03)
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    # Keep total PPO rollout ≈2048 like MuJoCo / M12 (n_steps * num_envs).
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--force-urdf", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    meta: dict = {"ok": False, "phase": "grasp"}
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper
        from stable_baselines3 import PPO

        from robot_simu_rl.backends.isaac_lab.action_space import configure_sb3_action_space
        from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD
        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        prefer_usd = not bool(args.force_urdf)
        device = args.device if getattr(args, "device", None) else ("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg = make_reach_grasp_env_cfg(
            phase="grasp",
            num_envs=args.num_envs,
            prefer_usd=prefer_usd,
            delta_q_scale=args.delta_q_scale,
            hover_offset=args.hover_offset,
            success_dist=args.success_dist,
            lift_height=args.lift_height,
            enable_contact=True,
            early_terminate_success=True,
            max_episode_steps=args.max_episode_steps,
        )
        cfg.sim.device = device
        cfg.seed = args.seed

        env = ManagerBasedRLEnv(cfg=cfg)
        configure_sb3_action_space(env)
        vec_env = Sb3VecEnvWrapper(env)

        policy_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        if args.resume:
            model = PPO.load(args.resume, env=vec_env, device=policy_device)
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=10,
                learning_rate=args.learning_rate,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=args.ent_coef,
                device=policy_device,
                seed=args.seed,
                tensorboard_log=str(out / "tb"),
                policy_kwargs={"log_std_init": -1.0},
            )

        t0 = time.perf_counter()
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        elapsed = time.perf_counter() - t0

        ckpt = out / "ppo_isaac_grasp.zip"
        model.save(str(ckpt))

        meta.update(
            {
                "ok": True,
                "timesteps": args.timesteps,
                "elapsed_s": round(elapsed, 3),
                "fps": round(args.timesteps / max(elapsed, 1e-6), 1),
                "num_envs": args.num_envs,
                "hover_offset": args.hover_offset,
                "lift_height": args.lift_height,
                "success_dist": args.success_dist,
                "delta_q_scale": args.delta_q_scale,
                "max_episode_steps": args.max_episode_steps,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "seed": args.seed,
                "sim_device": device,
                "policy_device": policy_device,
                "prefer_usd": prefer_usd,
                "enable_contact": True,
                "proxy_urdf": str(PROXY_URDF),
                "robot_usd": str(ROBOT_USD),
                "usd_exists": ROBOT_USD.is_file(),
                "checkpoint": str(ckpt),
                "cuda": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "obs_dim": int(vec_env.observation_space.shape[-1]),
                "act_dim": int(vec_env.action_space.shape[-1]),
            }
        )
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[train_isaac_grasp] saved {ckpt} elapsed={elapsed:.1f}s fps={meta['fps']}")

        vec_env.close()
    except Exception as exc:  # noqa: BLE001
        meta["ok"] = False
        meta["error"] = f"{type(exc).__name__}: {exc}"
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("[train_isaac_grasp] FAIL", json.dumps(meta, indent=2))
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1) from exc
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0 if meta.get("ok") else 1)


if __name__ == "__main__":
    main()
