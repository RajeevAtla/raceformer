"""
PPO finetuning scaffold using racecar_gym observations.

Loads pretrained weights, optionally freezes encoders, masks missing modalities,
and optimizes policy/value heads.
"""

from typing import Any

import jax
import jax.numpy as jnp

from src.data.gym_wrapper import RacecarGymWrapper, jax_step
from src.models.encoders import ModalityViT
from src.models.fusion import MultimodalFusion
from src.models.heads import PolicyHead, ValueHead


def run_ppo(config: dict[str, Any]) -> None:
    """Run PPO loop scaffold."""
    env = RacecarGymWrapper(config.get("env_id", "Racecar-v0"), seed=config.get("seed", 0))
    step_fn = jax_step(env)
    model_cfg = config["model"]

    encoder = ModalityViT(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        patch_size=model_cfg.get("patch_size", 16),
    )
    fusion = MultimodalFusion(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg.get("fusion_depth", model_cfg["depth"]),
        num_heads=model_cfg["num_heads"],
    )
    policy = PolicyHead(action_dim=config["action_dim"])
    value = ValueHead()

    del encoder, fusion, policy, value  # TODO: remove when implementing RL forward pass
    _ = step_fn
    # TODO: implement PPO rollout, masking for missing modalities, optimizer, and checkpointing.


def mask_missing_modalities(obs: dict[str, Any]) -> dict[str, Any]:
    """Mask radar/IR with zeros or learned tokens before model forward."""
    masked = dict(obs)
    masked["radar"] = jnp.zeros_like(jnp.asarray(obs["radar"]))
    masked["ir"] = jnp.zeros_like(jnp.asarray(obs["ir"]))
    return masked


def main() -> None:
    example_config = {
        "env_id": "Racecar-v0",
        "seed": 0,
        "model": {"embed_dim": 256, "depth": 6, "num_heads": 8},
        "action_dim": 2,
    }
    run_ppo(example_config)


if __name__ == "__main__":
    main()
