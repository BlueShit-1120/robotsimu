#!/usr/bin/env python3
"""Verify vendored meshes + M0 MJCF load."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    import mujoco
    from robot_simu_rl.common.paths import PICK_PLACE_SCENE, ROBOT_M0
    from robot_simu_rl.common.joints import MJCF_RIGHT_ARM_JOINTS, MJCF_RIGHT_GRIPPER_JOINT

    meshes = _REPO / "assets/DVT1/meshes"
    n_vis = len(list((meshes / "visual").glob("*.STL"))) + len(
        list((meshes / "visual").glob("*.stl"))
    )
    n_col = len(list((meshes / "collision").glob("*.STL"))) + len(
        list((meshes / "collision").glob("*.stl"))
    )
    print(f"meshes visual={n_vis} collision={n_col}")

    for path in (ROBOT_M0, PICK_PLACE_SCENE):
        m = mujoco.MjModel.from_xml_path(str(path))
        print(f"load OK {path.name}: nq={m.nq} nu={m.nu} njnt={m.njnt}")
        for name in (*MJCF_RIGHT_ARM_JOINTS, MJCF_RIGHT_GRIPPER_JOINT, "workpiece_free"):
            if path == ROBOT_M0 and name == "workpiece_free":
                continue
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, name
        # floating_base must be gone
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        assert fb < 0, "floating_base should be removed (frozen base)"
    print("check_assets OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
