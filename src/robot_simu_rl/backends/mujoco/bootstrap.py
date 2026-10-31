"""Grasp→place bootstrap: freeze a grasp policy until lift, then hand off."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class GraspBootstrapWrapper(gym.Wrapper):
    """Override actions with a frozen grasp policy until the box is lifted.

    After `_lifted_once` (or contact hold above lift_height), the wrapped
    policy's action is used so PPO can learn carry + release. Eval can use
    the same wrapper so full-pick success does not require one net to retain
    grasp when fine-tuning place.
    """

    def __init__(
        self,
        env: gym.Env,
        grasp_model: Any,
        deterministic: bool = True,
    ) -> None:
        super().__init__(env)
        self.grasp_model = grasp_model
        self.deterministic = bool(deterministic)
        self._obs: Optional[np.ndarray] = None
        self._phase = "grasp"  # grasp | place

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = obs
        self._phase = "grasp"
        return obs, info

    def _should_bootstrap(self) -> bool:
        base = self.env.unwrapped
        if getattr(base, "_lifted_once", False):
            return False
        # Contact-hold engaged: pads pinched — hand off so place policy can
        # lift/carry (M4 post-pinch actions often open and drop).
        if getattr(base, "_latched", False) and getattr(base, "contact_hold", False):
            base._lifted_once = True
            base._reached_once = True
            return False
        # Hand off on grasp-success or clear lift even if contact flickers
        # (M4 proxy pads often report holding=False the step lift clears).
        if base._grasp_success():
            base._lifted_once = True
            return False
        box_z = float(base.data.xpos[base._box_body_id][2])
        if box_z > base._box_z0 + base.lift_height:
            base._lifted_once = True
            return False
        return True

    def step(self, action):
        if self._should_bootstrap() and self._obs is not None:
            action, _ = self.grasp_model.predict(
                self._obs, deterministic=self.deterministic
            )
            self._phase = "grasp"
        else:
            self._phase = "place"
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        info = dict(info)
        info["bootstrap_phase"] = self._phase
        return obs, reward, terminated, truncated, info
