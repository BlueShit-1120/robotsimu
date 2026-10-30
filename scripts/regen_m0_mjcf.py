#!/usr/bin/env python3
"""Regenerate assets/mjcf/robot_assemble_m0.xml from in-repo DVT1 MJCF.

Replaces missing Robotiq/FT meshes with box/cylinder proxies, freezes floating
base, drops external scene include / grasp welds.

M4 contact-grasp: right fingertip pads become sticky spheres (contype=2), other
right-gripper collisions disabled, noslip/elliptic friction strengthened.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "assets/DVT1/mjcf/robot_assemble.xml"
OUT = REPO / "assets/mjcf/robot_assemble_m0.xml"

PROXY = {
    "ft300_mounting_plate_vis": ("cylinder", "0.035 0.004", "0.6 0.6 0.65 1"),
    "ft300_mounting_plate_col": ("cylinder", "0.035 0.004", None),
    "ft300_sensor_vis": ("cylinder", "0.025 0.02", "0.45 0.45 0.5 1"),
    "ft300_sensor_col": ("cylinder", "0.025 0.02", None),
    "robotiq_base_vis": ("box", "0.043 0.035 0.03", "0.2 0.2 0.22 1"),
    "robotiq_left_knuckle_vis": ("box", "0.012 0.012 0.015", "0.25 0.25 0.28 1"),
    "robotiq_right_knuckle_vis": ("box", "0.012 0.012 0.015", "0.25 0.25 0.28 1"),
    "robotiq_left_finger_vis": ("box", "0.01 0.012 0.025", "0.3 0.3 0.32 1"),
    "robotiq_right_finger_vis": ("box", "0.01 0.012 0.025", "0.3 0.3 0.32 1"),
    "robotiq_left_finger_tip_vis": ("box", "0.012 0.014 0.02", "0.15 0.15 0.18 1"),
    "robotiq_right_finger_tip_vis": ("box", "0.012 0.014 0.02", "0.15 0.15 0.18 1"),
    "robotiq_left_inner_knuckle_vis": ("box", "0.008 0.008 0.03", "0.35 0.35 0.38 1"),
    "robotiq_right_inner_knuckle_vis": ("box", "0.008 0.008 0.03", "0.35 0.35 0.38 1"),
}


def replace_geom(match: re.Match) -> str:
    full = match.group(0)
    mesh = match.group(1)
    if mesh not in PROXY:
        return full
    gtype, size, rgba = PROXY[mesh]
    name_m = re.search(r'name="([^"]+)"', full)
    class_m = re.search(r'class="([^"]+)"', full)
    mass_m = re.search(r'mass="([^"]+)"', full)
    name = name_m.group(1) if name_m else "proxy"
    cls = class_m.group(1) if class_m else "visual"
    # M4: right fingertip pads → sticky spheres that skip table contacts.
    if "right_gripper" in name and "fingertip" in name:
        if cls == "visual":
            return (
                f'<geom name="{name}" type="sphere" size="0.016" class="visual" '
                f'rgba="0.15 0.15 0.18 1" />'
            )
        mass = mass_m.group(1) if mass_m else "0.0427"
        return (
            f'<geom name="{name}" type="sphere" size="0.018" class="collision" '
            f'friction="3.5 0.3 0.01" solref="0.045 1" solimp="0.9 0.95 0.001" '
            f'condim="6" contype="2" conaffinity="0" mass="{mass}" />'
        )
    parts = [f'<geom name="{name}" type="{gtype}" size="{size}" class="{cls}"']
    if rgba and cls == "visual":
        parts.append(f' rgba="{rgba}"')
    if "fingertip" in name and cls == "collision":
        parts.append(' friction="1.0 0.005 0.0001"')
    if mass_m:
        parts.append(f' mass="{mass_m.group(1)}"')
    if "inner_knuckle" in name and cls == "collision":
        parts.append(' contype="0" conaffinity="0"')
    # M4: disable non-pad right gripper collisions (knuckles shove the box).
    if (
        cls == "collision"
        and name.startswith("right_gripper")
        and "fingertip" not in name
    ):
        parts.append(' contype="0" conaffinity="0"')
    parts.append(" />")
    return "".join(parts)


def main() -> int:
    if not SRC.is_file():
        print(f"missing source MJCF: {SRC}", file=sys.stderr)
        return 1
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(
        r'[ \t]*<mesh\s+name="(?:ft300_|robotiq_)[^"]+"\s+file="[^"]+"\s*/>\s*\n?',
        "",
        text,
    )
    text = text.replace('<freejoint name="floating_base" />', "")
    text = re.sub(
        r'[ \t]*<include\s+file="[^"]*scene_containers\.xml"\s*/>\s*\n?', "", text
    )
    text = re.sub(r'[ \t]*<weld\s+name="grasp_weld_[^"]+"[^/]*/>\s*\n?', "", text)
    text = text.replace(
        'noslip_iterations="20"',
        'noslip_iterations="20" impratio="20" cone="elliptic"',
    )
    if "noslip_iterations" not in text:
        text = text.replace(
            "<compiler",
            '<option noslip_iterations="20" noslip_tolerance="1e-8" '
            'impratio="20" cone="elliptic" />\n  <compiler',
            1,
        )
    text = re.sub(r'<geom\b[^>]*\bmesh="([^"]+)"[^/]*/>', replace_geom, text)
    text = text.replace('<mujoco model="spr_robot">', '<mujoco model="spr_robot_m0_proxy">')
    text = text.replace('meshdir="../meshes"', 'meshdir="../DVT1/meshes"')
    text = text.replace(
        '<position name="right_gripper" joint="right_gripper_left_finger_joint" kp="50" />',
        '<position name="right_gripper" joint="right_gripper_left_finger_joint" '
        'kp="70" ctrlrange="0 0.8" ctrllimited="true" '
        'forcerange="-80 80" forcelimited="true" />',
    )
    text = text.replace(
        '<position name="left_gripper" joint="left_gripper_left_finger_joint" kp="50" />',
        '<position name="left_gripper" joint="left_gripper_left_finger_joint" '
        'kp="50" ctrlrange="0 0.8" ctrllimited="true" />',
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
