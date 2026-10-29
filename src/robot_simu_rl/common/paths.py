"""Resolve asset paths relative to the repo root."""

from __future__ import annotations

from pathlib import Path

# src/robot_simu_rl/common/paths.py → repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS = REPO_ROOT / "assets"
MJCF_DIR = ASSETS / "mjcf"
PICK_PLACE_SCENE = MJCF_DIR / "pick_place_scene.xml"
ROBOT_M0 = MJCF_DIR / "robot_assemble_m0.xml"
