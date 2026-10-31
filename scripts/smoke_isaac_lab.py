#!/usr/bin/env python3
"""Headless smoke: spawn frozen-base SPR + table + box; reset + N physics steps.

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/smoke_isaac_lab.py --headless --steps 50

Falls back to on-the-fly URDF import when USD is missing.
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
    parser = argparse.ArgumentParser(description="Isaac Lab pick-place scene smoke.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--prefer-usd", action="store_true", default=True)
    parser.add_argument("--force-urdf", action="store_true", help="Skip USD even if present.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "assets/isaac/usd/SMOKE_OK.json",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    result: dict = {"ok": False}
    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD
        from robot_simu_rl.backends.isaac_lab.scene_cfg import build_pick_place_scene_cfg

        prefer_usd = bool(args.prefer_usd) and not bool(args.force_urdf)
        result["proxy_urdf"] = str(PROXY_URDF)
        result["robot_usd"] = str(ROBOT_USD)
        result["usd_exists"] = ROBOT_USD.is_file()
        result["prefer_usd"] = prefer_usd
        result["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)

        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
        sim = SimulationContext(sim_cfg)
        scene_cfg = build_pick_place_scene_cfg(
            num_envs=args.num_envs, prefer_usd=prefer_usd
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        robot = scene["robot"]
        box = scene["workpiece"]
        result["num_joints"] = int(robot.num_joints)
        result["joint_names_sample"] = list(robot.joint_names[:12])
        result["right_arm_present"] = all(
            any(n == j or n in j for j in robot.joint_names)
            for n in [f"right_arm_joint{i}" for i in range(1, 8)]
        )
        result["right_grip_present"] = any(
            "right_robotiq_85_left_knuckle" in n for n in robot.joint_names
        )

        t0 = time.perf_counter()
        for i in range(args.steps):
            # Hold default joint targets (implicit PD).
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            if i == 0:
                result["box_pos_step0"] = box.data.root_pos_w[0].detach().cpu().tolist()
        result["box_pos_final"] = box.data.root_pos_w[0].detach().cpu().tolist()
        result["robot_root_final"] = robot.data.root_pos_w[0].detach().cpu().tolist()
        result["steps"] = args.steps
        result["elapsed_s"] = round(time.perf_counter() - t0, 3)
        result["ok"] = True
        print("SMOKE_OK", json.dumps(result, indent=2))
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — surface full failure for docs
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        print("SMOKE_FAIL", json.dumps(result, indent=2))
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(1) from exc
    finally:
        # Kit close can hang headless; hard-exit after writing the marker.
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
