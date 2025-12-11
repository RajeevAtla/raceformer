"""
racecar_gym wrapper to emit CMHT-style observations.

Scaffold for matching simulator outputs to the multimodal pretraining format and
handling missing modalities (e.g., radar/IR).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import racecar_gym  # type: ignore
except ImportError:  # pragma: no cover - simulator installed separately
    racecar_gym = None


class GymStep(NamedTuple):
    obs: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class RacecarGymWrapper:
    """
    Wrap the racecar_gym environment to align observation keys with CMHT.

    Missing modalities (radar/IR) should be masked with zeros or learnable tokens
    inside the model; this wrapper only ensures consistent keys and shapes.
    """

    def __init__(self, env_id: str = "Racecar-v0", seed: int = 0):
        if racecar_gym is None:
            raise RuntimeError("racecar_gym is not installed; install simulator before use.")
        self.env = racecar_gym.make(env_id)  # type: ignore[attr-defined]
        self.env.seed(seed)
        # Cache action bounds for scaling in PPO.
        if hasattr(self.env, "action_space"):
            space = self.env.action_space
            self.action_low = np.asarray(getattr(space, "low", None))
            self.action_high = np.asarray(getattr(space, "high", None))
        else:  # pragma: no cover - fallback when space absent
            self.action_low = None
            self.action_high = None

    def reset(self) -> dict[str, Any]:
        obs = self.env.reset()
        return self._format_obs(obs)

    def step(self, action: Any) -> GymStep:
        obs, reward, done, info = self.env.step(action)
        return GymStep(self._format_obs(obs), reward, done, info)

    def _format_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """
        Map simulator observations to CMHT-style dict.

        Expected keys: lidar, rgb, gps, imu; radar and ir are absent in sim and set to zeros.
        """
        lidar = raw_obs.get("lidar")
        rgb = raw_obs.get("rgb")
        gps = raw_obs.get("gps")
        imu = raw_obs.get("imu")
        radar = jnp.zeros(1)  # placeholder mask for missing radar
        ir = jnp.zeros(1)  # placeholder mask for missing IR
        return {"lidar": lidar, "rgb": rgb, "radar": radar, "ir": ir, "gps": gps, "imu": imu}


def jax_step(env: RacecarGymWrapper) -> Callable[[Any, Any], GymStep]:
    """
    Return a JAX-friendly step function.

    Host-callback may be required because racecar_gym uses PyBullet on CPU.
    """

    def _step(state: Any, action: Any) -> GymStep:
        return jax.pure_callback(env.step, GymStep, action)  # type: ignore[arg-type]

    return _step
