#!/usr/bin/env python3
"""Evaluate Isaac Lab Place PPO checkpoint (SB3).

Reports place success (released near goal, stable steps), mean place_xy / box_z.

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/eval_isaac_place.py \\
        --headless --checkpoint runs/isaac_place_m16/ppo_isaac_place.zip --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _as_np(x):
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab Place PPO eval (SB3).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--delta-q-scale", type=float, default=0.03)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--place-tol-xy", type=float, default=0.12)
    parser.add_argument("--place-tol-z", type=float, default=0.04)
    parser.add_argument("--place-stable-steps", type=int, default=2)
    parser.add_argument("--no-place-start-holding", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-urdf", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    out = Path(args.out) if args.out else ckpt.parent / "eval_summary.json"
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    summary: dict = {"ok": False, "phase": "place"}
    try:
        import numpy as np
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
        vec_env = Sb3VecEnvWrapper(env, fast_variant=True)
        policy_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        model = PPO.load(str(ckpt), env=vec_env, device=policy_device)

        obs = vec_env.reset()
        successes: list[bool] = []
        place_xys: list[float] = []
        box_zs: list[float] = []
        releaseds: list[bool] = []
        returns: list[float] = []
        ep_ret = np.zeros(args.num_envs, dtype=np.float64)
        ep_success = np.zeros(args.num_envs, dtype=bool)
        ep_last = {
            "place_xy": np.full(args.num_envs, np.nan),
            "box_z": np.full(args.num_envs, np.nan),
            "released": np.zeros(args.num_envs, dtype=bool),
        }

        max_steps = args.episodes * args.max_episode_steps * 4 + args.max_episode_steps
        steps = 0
        while len(successes) < args.episodes and steps < max_steps:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = vec_env.step(actions)
            steps += 1
            ep_ret += rewards.astype(np.float64)

            unwrapped = vec_env.unwrapped
            log = unwrapped.extras.get("log", {}) if hasattr(unwrapped, "extras") else {}
            gsucc = _as_np(log.get("place_success"))
            pxy = _as_np(log.get("place_dist_xy"))
            box_z = _as_np(log.get("box_z"))
            released = _as_np(log.get("released"))

            if gsucc is not None:
                ep_success |= gsucc.astype(bool).reshape(-1)
            if pxy is not None:
                ep_last["place_xy"] = pxy.astype(np.float64).reshape(-1)
            if box_z is not None:
                ep_last["box_z"] = box_z.astype(np.float64).reshape(-1)
            if released is not None:
                ep_last["released"] = released.astype(bool).reshape(-1)

            for i in np.flatnonzero(dones):
                if len(successes) >= args.episodes:
                    break
                ok = bool(ep_success[i])
                successes.append(ok)
                place_xys.append(float(ep_last["place_xy"][i]) if np.isfinite(ep_last["place_xy"][i]) else float("nan"))
                box_zs.append(float(ep_last["box_z"][i]) if np.isfinite(ep_last["box_z"][i]) else float("nan"))
                releaseds.append(bool(ep_last["released"][i]))
                returns.append(float(ep_ret[i]))
                print(
                    f"  ep={len(successes) - 1:02d} success={ok} "
                    f"place_xy={place_xys[-1]:.3f} box_z={box_zs[-1]:.3f} "
                    f"released={releaseds[-1]} ret={ep_ret[i]:.2f}"
                )
                ep_ret[i] = 0.0
                ep_success[i] = False

        n = len(successes)
        rate = float(sum(successes) / max(n, 1))
        summary.update(
            {
                "ok": n >= args.episodes,
                "checkpoint": str(ckpt),
                "episodes": n,
                "successes": int(sum(successes)),
                "success_rate": rate,
                "mean_place_xy": float(np.nanmean(place_xys)) if place_xys else float("nan"),
                "mean_box_z": float(np.nanmean(box_zs)) if box_zs else float("nan"),
                "release_rate": float(sum(releaseds) / max(n, 1)),
                "mean_return": float(np.mean(returns)) if returns else float("nan"),
                "place_tol_xy": args.place_tol_xy,
                "place_tol_z": args.place_tol_z,
                "place_stable_steps": args.place_stable_steps,
                "place_start_holding": not bool(args.no_place_start_holding),
                "delta_q_scale": args.delta_q_scale,
                "max_episode_steps": args.max_episode_steps,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "sim_device": device,
                "policy_device": policy_device,
                "prefer_usd": prefer_usd,
                "proxy_urdf": str(PROXY_URDF),
                "robot_usd": str(ROBOT_USD),
                "usd_exists": ROBOT_USD.is_file(),
                "place_xys": [None if not np.isfinite(z) else round(float(z), 4) for z in place_xys],
                "box_zs": [None if not np.isfinite(z) else round(float(z), 4) for z in box_zs],
                "eval_steps": steps,
                "exit_criterion": "place success rate (M3 aimed >=50%)",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[eval_isaac_place] {summary['successes']}/{n} rate={rate:.2f} "
            f"mean_place_xy={summary['mean_place_xy']:.3f} "
            f"mean_box_z={summary['mean_box_z']:.3f} release={summary['release_rate']:.2f}"
        )
        print(f"[eval_isaac_place] wrote {out}")
        vec_env.close()
    except Exception as exc:  # noqa: BLE001
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("[eval_isaac_place] FAIL", json.dumps(summary, indent=2))
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
        raise SystemExit(0 if summary.get("ok") else 1)


if __name__ == "__main__":
    main()
