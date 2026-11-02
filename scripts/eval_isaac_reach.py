#!/usr/bin/env python3
"""Evaluate Isaac Lab Reach PPO checkpoint (SB3).

Reports success rate and mean end-effector hover distance (success_dist ~5 cm).

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/eval_isaac_reach.py \\
        --headless --checkpoint runs/isaac_reach_long/ppo_isaac_reach.zip \\
        --episodes 10 --num-envs 1 --out runs/isaac_reach_long/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _hover_dists(env, hover_offset: float):
    """Compute EE→hover-target distances for all envs (world frame)."""
    import torch

    robot = env.scene["robot"]
    obj = env.scene["workpiece"]
    if not hasattr(env, "_eval_ee_id"):
        ee_ids, _ = robot.find_bodies(["right_arm_ee_link"], preserve_order=True)
        env._eval_ee_id = ee_ids[0]
    ee = robot.data.body_pos_w[:, env._eval_ee_id]
    box = obj.data.root_pos_w
    target = box.clone()
    target[:, 2] = box[:, 2] + hover_offset
    return torch.norm(ee - target, dim=-1)


def _as_np(x):
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab Reach PPO eval (SB3).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1, help="Prefer 1 for episode accounting.")
    parser.add_argument("--hover-offset", type=float, default=0.30)
    parser.add_argument("--success-dist", type=float, default=0.05)
    parser.add_argument("--delta-q-scale", type=float, default=0.05)
    parser.add_argument("--max-episode-steps", type=int, default=200)
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

    summary: dict = {"ok": False, "phase": "reach"}
    try:
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper
        from stable_baselines3 import PPO

        from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD
        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        prefer_usd = not bool(args.force_urdf)
        device = args.device if getattr(args, "device", None) else ("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg = make_reach_grasp_env_cfg(
            phase="reach",
            num_envs=args.num_envs,
            prefer_usd=prefer_usd,
            delta_q_scale=args.delta_q_scale,
            hover_offset=args.hover_offset,
            success_dist=args.success_dist,
            enable_contact=False,
            # Keep early success so short successful episodes still count.
            early_terminate_success=True,
            max_episode_steps=args.max_episode_steps,
        )
        cfg.sim.device = device
        cfg.seed = args.seed

        env = ManagerBasedRLEnv(cfg=cfg)
        vec_env = Sb3VecEnvWrapper(env, fast_variant=True)
        policy_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        model = PPO.load(str(ckpt), env=vec_env, device=policy_device)

        obs = vec_env.reset()
        successes: list[bool] = []
        final_dists: list[float] = []
        best_dists: list[float] = []
        returns: list[float] = []
        ep_ret = np.zeros(args.num_envs, dtype=np.float64)
        best_dist = np.full(args.num_envs, np.inf, dtype=np.float64)
        # Pre-reset distance from the latest physics step (extras log or lagged scene read).
        term_dist = np.full(args.num_envs, np.nan, dtype=np.float64)

        max_steps = args.episodes * args.max_episode_steps * 4 + args.max_episode_steps
        steps = 0
        while len(successes) < args.episodes and steps < max_steps:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = vec_env.step(actions)
            steps += 1
            ep_ret += rewards.astype(np.float64)

            unwrapped = vec_env.unwrapped
            log = unwrapped.extras.get("log", {}) if hasattr(unwrapped, "extras") else {}
            log_dist = _as_np(log.get("reach_dist")) if isinstance(log, dict) else None
            # Scene read AFTER step is post-reset for done envs; still useful for active envs.
            scene_dist = _hover_dists(unwrapped, args.hover_offset).detach().cpu().numpy()

            if log_dist is not None:
                # Reward log is computed before auto-reset → valid terminal distance.
                term_dist = log_dist.astype(np.float64).reshape(-1)
                best_dist = np.minimum(best_dist, term_dist)
            else:
                active = ~np.asarray(dones, dtype=bool)
                best_dist[active] = np.minimum(best_dist[active], scene_dist[active])
                term_dist[active] = scene_dist[active]

            done_ids = np.flatnonzero(dones)
            for i in done_ids:
                if len(successes) >= args.episodes:
                    break
                dist_val = float(term_dist[i]) if np.isfinite(term_dist[i]) else float(scene_dist[i])
                best_val = float(best_dist[i]) if np.isfinite(best_dist[i]) else dist_val
                ok = bool(dist_val < args.success_dist or best_val < args.success_dist)
                successes.append(ok)
                final_dists.append(dist_val)
                best_dists.append(best_val)
                returns.append(float(ep_ret[i]))
                print(
                    f"  ep={len(successes) - 1:02d} env={int(i)} success={ok} "
                    f"dist={dist_val:.4f} best={best_val:.4f} ret={ep_ret[i]:.2f}"
                )
                ep_ret[i] = 0.0
                best_dist[i] = np.inf
                term_dist[i] = np.nan

        n = len(successes)
        summary.update(
            {
                "ok": n >= args.episodes,
                "checkpoint": str(ckpt),
                "episodes": n,
                "episodes_requested": args.episodes,
                "successes": int(sum(successes)),
                "success_rate": float(sum(successes) / max(n, 1)),
                "mean_final_dist": float(np.mean(final_dists)) if final_dists else float("nan"),
                "mean_best_dist": float(np.mean(best_dists)) if best_dists else float("nan"),
                "mean_return": float(np.mean(returns)) if returns else float("nan"),
                "hover_offset": args.hover_offset,
                "success_dist": args.success_dist,
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
                "final_dists": [round(float(d), 6) for d in final_dists],
                "best_dists": [round(float(d), 6) for d in best_dists],
                "eval_steps": steps,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[eval_isaac_reach] {summary['successes']}/{n} "
            f"rate={summary['success_rate']:.2f} mean_dist={summary['mean_final_dist']:.4f} "
            f"mean_best={summary['mean_best_dist']:.4f}"
        )
        print(f"[eval_isaac_reach] wrote {out}")
        vec_env.close()
    except Exception as exc:  # noqa: BLE001
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("[eval_isaac_reach] FAIL", json.dumps(summary, indent=2))
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
