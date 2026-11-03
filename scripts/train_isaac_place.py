#!/usr/bin/env python3
"""Isaac Lab Place PPO short train (SB3) — M3 C3 release curriculum.

Defaults: place_start_holding (kinematic hold-assist), place_tol_xy=0.12,
place_tol_z=0.04, place_stable_steps=2, M12 latched Δq + reward-dt.

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/train_isaac_place.py \\
        --headless --timesteps 50000 --num-envs 4 --out runs/isaac_place_m16
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
    parser = argparse.ArgumentParser(description="Isaac Lab Place PPO train (SB3).")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out", type=str, default="runs/isaac_place_m16")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--delta-q-scale", type=float, default=0.03)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--place-tol-xy", type=float, default=0.12)
    parser.add_argument("--place-tol-z", type=float, default=0.04)
    parser.add_argument("--place-stable-steps", type=int, default=2)
    parser.add_argument("--no-place-start-holding", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
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

    meta: dict = {"ok": False, "phase": "place"}
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
            phase="place",
            num_envs=args.num_envs,
            prefer_usd=prefer_usd,
            delta_q_scale=args.delta_q_scale,
            lift_height=args.lift_height,
            place_tol_xy=args.place_tol_xy,
            place_tol_z=args.place_tol_z,
            place_stable_steps=args.place_stable_steps,
            place_start_holding=not bool(args.no_place_start_holding),
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
                learning_rate=args.learning_rate,
                ent_coef=args.ent_coef,
                seed=args.seed,
                device=policy_device,
            )

        t0 = time.time()
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        elapsed = time.time() - t0
        ckpt = out / "ppo_isaac_place.zip"
        model.save(str(ckpt))

        obs_space = getattr(env, "single_observation_space", env.observation_space)["policy"]
        act_space = getattr(env, "single_action_space", env.action_space)
        meta.update(
            {
                "ok": True,
                "timesteps": args.timesteps,
                "elapsed_s": round(elapsed, 3),
                "fps": round(args.timesteps / max(elapsed, 1e-6), 1),
                "num_envs": args.num_envs,
                "delta_q_scale": args.delta_q_scale,
                "lift_height": args.lift_height,
                "place_tol_xy": args.place_tol_xy,
                "place_tol_z": args.place_tol_z,
                "place_stable_steps": args.place_stable_steps,
                "place_start_holding": not bool(args.no_place_start_holding),
                "max_episode_steps": args.max_episode_steps,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "ent_coef": args.ent_coef,
                "seed": args.seed,
                "sim_device": device,
                "policy_device": policy_device,
                "prefer_usd": prefer_usd,
                "proxy_urdf": str(PROXY_URDF),
                "robot_usd": str(ROBOT_USD),
                "usd_exists": ROBOT_USD.is_file(),
                "checkpoint": str(ckpt),
                "cuda": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "obs_dim": int(obs_space.shape[-1]),
                "act_dim": int(act_space.shape[-1]),
            }
        )
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[train_isaac_place] saved {ckpt} elapsed={elapsed:.1f}s fps={meta['fps']}")
        vec_env.close()
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("[train_isaac_place] FAIL", json.dumps(meta, indent=2))
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
