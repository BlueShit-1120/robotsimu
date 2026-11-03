#!/usr/bin/env python3
"""Build an Isaac-ready proxy URDF from DVT1 robot_assemble.urdf.

Missing Robotiq / FT meshes are replaced with box / cylinder / sphere proxies
(aligned with MuJoCo M0/M4 sizes). Body STL paths are rewritten relative to
assets/isaac/. Does not modify the MuJoCo MJCF path.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_URDF = REPO / "assets/DVT1/urdf/robot_assemble.urdf"
OUT_URDF = REPO / "assets/isaac/robot_assemble_proxy.urdf"

# MJCF half-extents → URDF full box size (2×). Spheres use radius.
# M4: right fingertip pads are sticky spheres (r=0.018); other right-gripper
# collision geoms are stripped below so only tips contact the workpiece.
PROXY_BY_SUBSTR: list[tuple[str, dict]] = [
    ("robotiq_ft300-G-062", {"kind": "cylinder", "radius": 0.035, "length": 0.008}),
    ("robotiq_ft300", {"kind": "cylinder", "radius": 0.025, "length": 0.04}),
    ("robotiq_base", {"kind": "box", "size": (0.086, 0.07, 0.06)}),
    ("left_knuckle", {"kind": "box", "size": (0.024, 0.024, 0.03)}),
    ("right_knuckle", {"kind": "box", "size": (0.024, 0.024, 0.03)}),
    ("left_inner_knuckle", {"kind": "box", "size": (0.016, 0.016, 0.06)}),
    ("right_inner_knuckle", {"kind": "box", "size": (0.016, 0.016, 0.06)}),
    ("left_finger_tip", {"kind": "sphere", "radius": 0.018}),
    ("right_finger_tip", {"kind": "sphere", "radius": 0.018}),
    ("left_finger", {"kind": "box", "size": (0.02, 0.024, 0.05)}),
    ("right_finger", {"kind": "box", "size": (0.02, 0.024, 0.05)}),
]

# MuJoCo M4 disables collision on right gripper base/knuckle/finger (keep tips).
_RIGHT_GRIPPER_NO_COLLISION_SUBSTR = (
    "right_robotiq_85_base_link",
    "right_robotiq_85_left_knuckle_link",
    "right_robotiq_85_right_knuckle_link",
    "right_robotiq_85_left_finger_link",
    "right_robotiq_85_right_finger_link",
    "right_robotiq_85_left_inner_knuckle_link",
    "right_robotiq_85_right_inner_knuckle_link",
)


def _proxy_for(filename: str) -> dict | None:
    base = Path(filename).name.lower()
    for key, spec in PROXY_BY_SUBSTR:
        if key.lower() in base:
            return spec
    return None


def _replace_mesh_geom(geom: ET.Element, spec: dict) -> None:
    mesh = geom.find("mesh")
    if mesh is not None:
        geom.remove(mesh)
    if spec["kind"] == "box":
        sx, sy, sz = spec["size"]
        ET.SubElement(geom, "box", {"size": f"{sx} {sy} {sz}"})
    elif spec["kind"] == "cylinder":
        ET.SubElement(
            geom,
            "cylinder",
            {"radius": str(spec["radius"]), "length": str(spec["length"])},
        )
    elif spec["kind"] == "sphere":
        ET.SubElement(geom, "sphere", {"radius": str(spec["radius"])})
    else:
        raise ValueError(spec)


def os_path_rel(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(str(target), str(start))


def build(src: Path = SRC_URDF, out: Path = OUT_URDF) -> Path:
    if not src.is_file():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(src)
    root = tree.getroot()
    urdf_dir = src.parent
    out_dir = out.parent

    n_proxy = 0
    n_keep = 0
    # Collect first — mutating during root.iter() skips siblings in ElementTree.
    targets: list[tuple[ET.Element, ET.Element]] = []
    for section in root.iter():
        if section.tag not in ("visual", "collision"):
            continue
        geometry = section.find("geometry")
        if geometry is None:
            continue
        mesh = geometry.find("mesh")
        if mesh is None:
            continue
        targets.append((geometry, mesh))

    for geometry, mesh in targets:
        filename = mesh.get("filename", "")
        if not filename:
            continue
        abs_mesh = (urdf_dir / filename).resolve()
        if abs_mesh.is_file():
            mesh.set("filename", os_path_rel(abs_mesh, out_dir))
            n_keep += 1
            continue
        spec = _proxy_for(filename)
        if spec is None:
            geometry.remove(mesh)
            ET.SubElement(geometry, "box", {"size": "0.02 0.02 0.02"})
            n_proxy += 1
            continue
        _replace_mesh_geom(geometry, spec)
        n_proxy += 1

    # Strip non-tip right-gripper collision (M4: tips only contact the box).
    n_strip = 0
    for link in root.findall("link"):
        name = link.get("name", "")
        if name not in _RIGHT_GRIPPER_NO_COLLISION_SUBSTR:
            continue
        for col in list(link.findall("collision")):
            link.remove(col)
            n_strip += 1

    # Prefer absolute-ish comment for humans; ElementTree strips comments on write.
    tree.write(out, encoding="utf-8", xml_declaration=True)
    # Prepend a short header comment after xml decl.
    text = out.read_text(encoding="utf-8")
    text = re.sub(
        r"(<\?xml[^?]+\?>\s*)",
        rf"\1<!-- Autogenerated by scripts/make_proxy_urdf.py from {src.name}. "
        f"proxy={n_proxy} keep={n_keep} strip_right_grip_col={n_strip}. Do not edit by hand. -->\n",
        text,
        count=1,
    )
    out.write_text(text, encoding="utf-8")
    print(
        f"Wrote {out} (proxy geoms={n_proxy}, kept meshes={n_keep}, "
        f"stripped right-grip collisions={n_strip})"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=SRC_URDF)
    p.add_argument("--out", type=Path, default=OUT_URDF)
    args = p.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
