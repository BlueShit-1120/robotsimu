#!/usr/bin/env python3
"""Convert proxy URDF → USD via Isaac Lab UrdfConverter (headless-friendly).

Usage (conda env_isaacsim)::

    export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/robotsimu/scripts/convert_robot_usd.py --headless

Or::

    python /path/to/robotsimu/scripts/convert_robot_usd.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robot_simu_rl.backends.isaac_lab.paths import PROXY_URDF, ROBOT_USD, USD_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DVT1 proxy URDF to USD.")
    parser.add_argument("--urdf", type=Path, default=PROXY_URDF)
    parser.add_argument("--out", type=Path, default=ROBOT_USD)
    parser.add_argument("--joint-stiffness", type=float, default=80.0)
    parser.add_argument("--joint-damping", type=float, default=4.0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if not args.urdf.is_file():
        raise SystemExit(
            f"URDF not found: {args.urdf}. Run scripts/make_proxy_urdf.py first."
        )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.utils.dict import print_dict

        out = args.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = UrdfConverterCfg(
            asset_path=str(args.urdf.resolve()),
            usd_dir=str(out.parent),
            usd_file_name=out.name,
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args.joint_stiffness,
                    damping=args.joint_damping,
                ),
                target_type="position",
            ),
        )
        print("UrdfConverterCfg:")
        print_dict(cfg.to_dict(), nesting=0)
        converter = UrdfConverter(cfg)
        print(f"Generated USD: {converter.usd_path}")
        # Touch marker for CI/smoke without opening GUI.
        marker = USD_DIR / "CONVERT_OK.txt"
        marker.write_text(f"{converter.usd_path}\n", encoding="utf-8")
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(0)


if __name__ == "__main__":
    main()
