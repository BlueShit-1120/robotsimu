#!/usr/bin/env python3
"""Headless smoke: ManagerBased Reach/Grasp env — reset + step (+ optional random policy).

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/smoke_isaac_reach_grasp.py \\
        --headless --phase reach --steps 30
    ./isaaclab.sh -p /path/to/robotsimu/scripts/smoke_isaac_reach_grasp.py \\
        --headless --phase grasp --steps 40 --random-policy
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
    parser = argparse.ArgumentParser(description="Isaac Lab Reach/Grasp ManagerBased smoke.")
    parser.add_argument("--phase", choices=("reach", "grasp"), default="reach")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--delta-q-scale", type=float, default=0.05)
    parser.add_argument("--hover-offset", type=float, default=0.30)
    parser.add_argument("--lift-height", type=float, default=0.03)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--force-urdf", action="store_true")
    parser.add_argument("--no-contact", action="store_true", help="Disable fingertip ContactSensors.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "assets/isaac/usd/SMOKE_REACH_GRASP_OK.json",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    result: dict = {"ok": False, "phase": args.phase}
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD
        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        prefer_usd = not bool(args.force_urdf)
        result["proxy_urdf"] = str(PROXY_URDF)
        result["robot_usd"] = str(ROBOT_USD)
        result["usd_exists"] = ROBOT_USD.is_file()
        result["prefer_usd"] = prefer_usd
        result["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)

        cfg = make_reach_grasp_env_cfg(
            phase=args.phase,
            num_envs=args.num_envs,
            prefer_usd=prefer_usd,
            delta_q_scale=args.delta_q_scale,
            hover_offset=args.hover_offset,
            lift_height=args.lift_height,
            enable_contact=not args.no_contact,
            early_terminate_success=False,
            max_episode_steps=max(args.steps + 5, 50),
        )
        # Device from AppLauncher
        if hasattr(args, "device") and args.device:
            cfg.sim.device = args.device

        env = ManagerBasedRLEnv(cfg=cfg)
        result["num_envs"] = int(env.num_envs)
        # Prefer single_* spaces (unbatched) when available.
        obs_space = getattr(env, "single_observation_space", env.observation_space)["policy"]
        act_space = getattr(env, "single_action_space", env.action_space)
        result["obs_dim"] = int(obs_space.shape[-1])
        result["act_dim"] = int(act_space.shape[-1])
        result["action_space_low"] = act_space.low.tolist() if hasattr(act_space, "low") else None
        result["action_space_high"] = act_space.high.tolist() if hasattr(act_space, "high") else None

        obs, info = env.reset()
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        result["obs_shape"] = list(policy_obs.shape)
        result["obs_finite_reset"] = bool(torch.isfinite(policy_obs).all().item())

        robot = env.scene["robot"]
        result["num_joints"] = int(robot.num_joints)
        result["right_arm_present"] = all(
            any(n == j for j in robot.joint_names) for n in [f"right_arm_joint{i}" for i in range(1, 8)]
        )
        result["right_grip_present"] = any(
            "right_robotiq_85_left_knuckle" in n for n in robot.joint_names
        )
        result["contact_sensors"] = [
            k for k in env.scene.keys() if "contact" in k
        ]

        t0 = time.perf_counter()
        last_rew = None
        last_log: dict = {}
        for i in range(args.steps):
            if args.random_policy:
                # Arm Δq ~ U(-1,1); gripper U(0,1) (reach holds open inside action term).
                act = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
                act[:, :7] = 2.0 * torch.rand(env.num_envs, 7, device=env.device) - 1.0
                act[:, 7] = torch.rand(env.num_envs, device=env.device)
            else:
                act = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
            obs, rew, terminated, truncated, extras = env.step(act)
            last_rew = rew.detach().cpu()
            last_log = {k: v.detach().cpu() for k, v in extras.get("log", {}).items() if torch.is_tensor(v)}
            if i == 0:
                result["reward_step0"] = float(last_rew.mean().item())
                box0 = env.scene["workpiece"].data.root_pos_w[0].detach().cpu().tolist()
                result["box_pos_step0"] = box0

        result["reward_final_mean"] = float(last_rew.mean().item()) if last_rew is not None else None
        result["log_final"] = {
            k: (float(v.mean().item()) if v.numel() else None) for k, v in last_log.items()
        }
        result["box_pos_final"] = env.scene["workpiece"].data.root_pos_w[0].detach().cpu().tolist()
        result["robot_root_final"] = robot.data.root_pos_w[0].detach().cpu().tolist()
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        result["obs_finite_final"] = bool(torch.isfinite(policy_obs).all().item())
        result["steps"] = args.steps
        result["random_policy"] = bool(args.random_policy)
        result["elapsed_s"] = round(time.perf_counter() - t0, 3)

        # Soft checks
        ok = (
            result["obs_dim"] == 31
            and result["act_dim"] == 8
            and result["right_arm_present"]
            and result["right_grip_present"]
            and result["obs_finite_reset"]
            and result["obs_finite_final"]
        )
        result["ok"] = bool(ok)
        result["checks"] = {
            "obs_dim_31": result["obs_dim"] == 31,
            "act_dim_8": result["act_dim"] == 8,
            "arm": result["right_arm_present"],
            "grip": result["right_grip_present"],
            "finite_obs": result["obs_finite_reset"] and result["obs_finite_final"],
        }

        print("SMOKE_REACH_GRASP", json.dumps(result, indent=2, default=str))
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

        env.close()
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print("SMOKE_REACH_GRASP_FAIL", json.dumps(result, indent=2, default=str))
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
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
        raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
