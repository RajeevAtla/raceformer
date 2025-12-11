"""
Masked multimodal pretraining loop scaffold.

Implements data loading, masking, forward pass, reconstruction loss, and checkpointing.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from src.data.cmht_loader import CMHTDataSource
from src.models.encoders import ModalityViT
from src.models.fusion import MultimodalFusion
from src.models.heads import PretrainHead


def run_pretraining(config: dict[str, Any]) -> None:
    """
    Execute masked multimodal modeling.

    This scaffold wires together the data source, model, masking, loss, and checkpoints.
    """
    data_path = Path(config["data"]["path"])
    data = CMHTDataSource(data_path)
    model_cfg = config["model"]
    training_cfg = config["training"]

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
    head = PretrainHead(embed_dim=model_cfg["embed_dim"], out_dim=model_cfg.get("out_dim", 0))

    del encoder, fusion, head  # TODO: remove when implementing forward pass
    del data
    _ = training_cfg
    # TODO: implement masking, forward pass, loss, optimizer, and orbax checkpointing.


def _mask_tokens(tokens: jnp.ndarray, mask_ratio: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Randomly mask a ratio of tokens; returns masked tokens and mask indices."""
    rng = jax.random.PRNGKey(0)  # TODO: use config/step-based rngs.
    mask = jax.random.bernoulli(rng, mask_ratio, tokens.shape[:-1])
    masked = jnp.where(mask[..., None], 0, tokens)
    return masked, mask


def main() -> None:
    example_config = {
        "model": {"embed_dim": 256, "depth": 6, "num_heads": 8},
        "training": {"batch_size": 64, "learning_rate": 3e-4, "steps": 1000, "mask_ratio": 0.5},
        "data": {"path": "/path/to/cmht"},
    }
    run_pretraining(example_config)


if __name__ == "__main__":
    main()
