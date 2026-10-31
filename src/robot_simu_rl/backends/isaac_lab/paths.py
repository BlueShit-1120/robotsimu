"""Isaac Lab asset paths (repo-local)."""

from __future__ import annotations

from pathlib import Path

from robot_simu_rl.common.paths import REPO_ROOT

ISAAC_ASSETS = REPO_ROOT / "assets" / "isaac"
PROXY_URDF = ISAAC_ASSETS / "robot_assemble_proxy.urdf"
USD_DIR = ISAAC_ASSETS / "usd"
ROBOT_USD = USD_DIR / "robot_assemble_proxy.usd"
