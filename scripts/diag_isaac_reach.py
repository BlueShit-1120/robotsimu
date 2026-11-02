#!/usr/bin/env python3
"""Headless diagnostic: hold drift + const-Δq tracking + reward scale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--delta-q-scale", type=float, default=0.06)
    parser.add_argument("--hover-offset", type=float, default=0.30)
    parser.add_argument("--force-urdf", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    out: dict = {"ok": False}
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        from robot_simu_rl.backends.isaac_lab.reach_grasp_env_cfg import make_reach_grasp_env_cfg

        cfg = make_reach_grasp_env_cfg(
            phase="reach",
            num_envs=1,
            prefer_usd=not args.force_urdf,
            delta_q_scale=args.delta_q_scale,
            hover_offset=args.hover_offset,
            success_dist=0.05,
            enable_contact=False,
            early_terminate_success=False,
            max_episode_steps=400,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.sim.device = device
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()

        robot = env.scene["robot"]
        obj = env.scene["workpiece"]
        ee_ids, _ = robot.find_bodies(["right_arm_ee_link"], preserve_order=True)
        ee_id = ee_ids[0]
        arm_ids, _ = robot.find_joints(
            [f"right_arm_joint{i}" for i in range(1, 8)], preserve_order=True
        )

        def snapshot(tag: str) -> dict:
            ee = robot.data.body_pos_w[0, ee_id] - env.scene.env_origins[0]
            box = obj.data.root_pos_w[0] - env.scene.env_origins[0]
            target = box.clone()
            target[2] = box[2] + args.hover_offset
            dist = torch.norm(ee - target).item()
            q = robot.data.joint_pos[0, arm_ids].detach().cpu()
            return {
                "tag": tag,
                "ee": [round(float(x), 4) for x in ee.tolist()],
                "target": [round(float(x), 4) for x in target.tolist()],
                "dist": round(dist, 4),
                "q": [round(float(x), 4) for x in q.tolist()],
            }

        out["reset"] = snapshot("reset")
        zero = torch.zeros(1, 8, device=env.device)
        for _ in range(args.steps):
            env.step(zero)
        out["after_hold"] = snapshot("hold")
        out["hold_dq_max"] = round(
            max(abs(out["after_hold"]["q"][i] - out["reset"]["q"][i]) for i in range(7)),
            6,
        )

        env.reset()
        out["reset2"] = snapshot("reset2")
        action = torch.zeros(1, 8, device=env.device)
        action[0, 1] = 1.0
        rews = []
        for _ in range(args.steps):
            _o, rew, *_ = env.step(action)
            rews.append(float(rew[0].item()))
        out["after_j2"] = snapshot("j2")
        out["reward_mean"] = round(sum(rews) / max(len(rews), 1), 6)
        out["dq_joint2"] = round(out["after_j2"]["q"][1] - out["reset2"]["q"][1], 4)
        expected = args.delta_q_scale * args.steps
        out["dq_joint2_expected"] = round(expected, 4)
        out["tracking_ratio"] = round(out["dq_joint2"] / max(expected, 1e-6), 4)

        # Random policy best dist over 200 steps (same budget as MuJoCo probe).
        env.reset()
        best = out["reset"]["dist"]
        for _ in range(200):
            a = torch.empty(1, 8, device=env.device).uniform_(-1.0, 1.0)
            a[0, 7] = 0.0
            env.step(a)
            ee = robot.data.body_pos_w[0, ee_id] - env.scene.env_origins[0]
            box = obj.data.root_pos_w[0] - env.scene.env_origins[0]
            target = box.clone()
            target[2] = box[2] + args.hover_offset
            best = min(best, float(torch.norm(ee - target).item()))
        out["random_best_dist"] = round(best, 4)
        out["ok"] = True
        print(json.dumps(out, indent=2))
        env.close()
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(out, indent=2))
        raise SystemExit(1) from exc
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0 if out.get("ok") else 1)


if __name__ == "__main__":
    main()
