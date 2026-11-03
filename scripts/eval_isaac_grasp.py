#!/usr/bin/env python3
"""Evaluate Isaac Lab Grasp PPO checkpoint (SB3).

Reports lift/contact grasp success rate, mean box height, tip–box distance.

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/eval_isaac_grasp.py \\
        --headless --checkpoint runs/isaac_grasp_m13/ppo_isaac_grasp.zip \\
        --episodes 20 --num-envs 1 --out runs/isaac_grasp_m13/eval_summary.json
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


def _scene_metrics(env, lift_height: float):
    """Fallback metrics from scene if extras log missing (active envs only)."""
    import torch

    from robot_simu_rl.common.joints import GRIPPER_JOINT_LIMIT, URDF_RIGHT_GRIPPER_JOINT

    robot = env.scene["robot"]
    obj = env.scene["workpiece"]
    if not hasattr(env, "_eval_tip_ids"):
        tip_l, _ = robot.find_bodies(["right_robotiq_85_left_finger_tip_link"], preserve_order=True)
        tip_r, _ = robot.find_bodies(["right_robotiq_85_right_finger_tip_link"], preserve_order=True)
        grip, _ = robot.find_joints([URDF_RIGHT_GRIPPER_JOINT], preserve_order=True)
        env._eval_tip_ids = (tip_l[0], tip_r[0])
        env._eval_grip_id = grip[0]
        if not hasattr(env, "_box_z0"):
            env._box_z0 = torch.full((env.num_envs,), 0.75, device=env.device)
    tip_mid = 0.5 * (
        robot.data.body_pos_w[:, env._eval_tip_ids[0]] + robot.data.body_pos_w[:, env._eval_tip_ids[1]]
    )
    box = obj.data.root_pos_w
    dist = torch.norm(tip_mid - box, dim=-1)
    lift = box[:, 2] - env._box_z0
    grip_lo, grip_hi = GRIPPER_JOINT_LIMIT
    grip_q = robot.data.joint_pos[:, env._eval_grip_id]
    grip_norm = (grip_q - grip_lo) / (grip_hi - grip_lo + 1e-9)
    holding = (dist < 0.05) & (grip_norm > 0.35)
    success = holding & (lift > lift_height)
    return {
        "grasp_dist": dist,
        "box_lift": lift,
        "box_z": box[:, 2],
        "holding": holding.float(),
        "grasp_success": success.float(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab Grasp PPO eval (SB3).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--hover-offset", type=float, default=0.06)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--success-dist", type=float, default=0.05)
    parser.add_argument("--delta-q-scale", type=float, default=0.03)
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

    summary: dict = {"ok": False, "phase": "grasp"}
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
        vec_env = Sb3VecEnvWrapper(env, fast_variant=True)
        policy_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        model = PPO.load(str(ckpt), env=vec_env, device=policy_device)

        obs = vec_env.reset()
        successes: list[bool] = []
        box_zs: list[float] = []
        lifts: list[float] = []
        tip_dists: list[float] = []
        holdings: list[bool] = []
        pinches: list[bool] = []
        returns: list[float] = []
        ep_ret = np.zeros(args.num_envs, dtype=np.float64)
        # Track best lift / success within episode (pre-reset).
        ep_success = np.zeros(args.num_envs, dtype=bool)
        ep_best_lift = np.full(args.num_envs, -np.inf, dtype=np.float64)
        ep_last = {
            "box_z": np.full(args.num_envs, np.nan),
            "lift": np.full(args.num_envs, np.nan),
            "dist": np.full(args.num_envs, np.nan),
            "holding": np.zeros(args.num_envs, dtype=bool),
            "pinch": np.zeros(args.num_envs, dtype=bool),
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
            if not isinstance(log, dict) or log.get("grasp_success") is None:
                log = _scene_metrics(unwrapped, args.lift_height)

            gsucc = _as_np(log.get("grasp_success"))
            lift = _as_np(log.get("box_lift"))
            dist = _as_np(log.get("grasp_dist"))
            holding = _as_np(log.get("holding"))
            pinch = _as_np(log.get("pinch"))
            box_z = _as_np(log.get("box_z"))
            if box_z is None and lift is not None:
                # Reconstruct from lift + table approx if log lacked box_z.
                z0 = getattr(unwrapped, "_box_z0", None)
                if z0 is not None:
                    box_z = _as_np(z0) + lift

            if gsucc is not None:
                gsucc = gsucc.astype(bool).reshape(-1)
                ep_success |= gsucc
            if lift is not None:
                lift = lift.astype(np.float64).reshape(-1)
                ep_best_lift = np.maximum(ep_best_lift, lift)
                ep_last["lift"] = lift
            if dist is not None:
                ep_last["dist"] = dist.astype(np.float64).reshape(-1)
            if holding is not None:
                ep_last["holding"] = holding.astype(bool).reshape(-1)
            if pinch is not None:
                ep_last["pinch"] = pinch.astype(bool).reshape(-1)
            if box_z is not None:
                ep_last["box_z"] = box_z.astype(np.float64).reshape(-1)

            done_ids = np.flatnonzero(dones)
            for i in done_ids:
                if len(successes) >= args.episodes:
                    break
                ok = bool(ep_success[i] or (ep_best_lift[i] > args.lift_height and ep_last["holding"][i]))
                successes.append(ok)
                box_zs.append(float(ep_last["box_z"][i]) if np.isfinite(ep_last["box_z"][i]) else float("nan"))
                lifts.append(float(ep_best_lift[i]) if np.isfinite(ep_best_lift[i]) else float("nan"))
                tip_dists.append(float(ep_last["dist"][i]) if np.isfinite(ep_last["dist"][i]) else float("nan"))
                holdings.append(bool(ep_last["holding"][i]))
                pinches.append(bool(ep_last["pinch"][i]))
                returns.append(float(ep_ret[i]))
                print(
                    f"  ep={len(successes) - 1:02d} env={int(i)} success={ok} "
                    f"box_z={box_zs[-1]:.3f} lift={lifts[-1]:.3f} "
                    f"tip_dist={tip_dists[-1]:.3f} hold={holdings[-1]} pinch={pinches[-1]} "
                    f"ret={ep_ret[i]:.2f}"
                )
                ep_ret[i] = 0.0
                ep_success[i] = False
                ep_best_lift[i] = -np.inf

        n = len(successes)
        rate = float(sum(successes) / max(n, 1))
        summary.update(
            {
                "ok": n >= args.episodes,
                "checkpoint": str(ckpt),
                "episodes": n,
                "episodes_requested": args.episodes,
                "successes": int(sum(successes)),
                "success_rate": rate,
                "mean_box_z": float(np.nanmean(box_zs)) if box_zs else float("nan"),
                "mean_lift": float(np.nanmean(lifts)) if lifts else float("nan"),
                "mean_tip_box_dist": float(np.nanmean(tip_dists)) if tip_dists else float("nan"),
                "holding_rate": float(sum(holdings) / max(n, 1)),
                "pinch_rate": float(sum(pinches) / max(n, 1)),
                "mean_return": float(np.mean(returns)) if returns else float("nan"),
                "hover_offset": args.hover_offset,
                "lift_height": args.lift_height,
                "success_dist": args.success_dist,
                "delta_q_scale": args.delta_q_scale,
                "max_episode_steps": args.max_episode_steps,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "sim_device": device,
                "policy_device": policy_device,
                "prefer_usd": prefer_usd,
                "enable_contact": True,
                "proxy_urdf": str(PROXY_URDF),
                "robot_usd": str(ROBOT_USD),
                "usd_exists": ROBOT_USD.is_file(),
                "box_zs": [None if not np.isfinite(z) else round(float(z), 4) for z in box_zs],
                "lifts": [None if not np.isfinite(z) else round(float(z), 4) for z in lifts],
                "tip_dists": [None if not np.isfinite(z) else round(float(z), 4) for z in tip_dists],
                "eval_steps": steps,
                "exit_criterion": "lift/grasp success rate (report; M4 aimed >=70% over >=100 ep)",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[eval_isaac_grasp] {summary['successes']}/{n} "
            f"rate={rate:.2f} mean_box_z={summary['mean_box_z']:.3f} "
            f"mean_lift={summary['mean_lift']:.3f} mean_tip={summary['mean_tip_box_dist']:.3f}"
        )
        print(f"[eval_isaac_grasp] wrote {out}")
        vec_env.close()
    except Exception as exc:  # noqa: BLE001
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("[eval_isaac_grasp] FAIL", json.dumps(summary, indent=2))
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
