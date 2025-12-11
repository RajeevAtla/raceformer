"""
Multimodal fusion module using Flax NNX.

Concatenate modality tokens and process with transformer blocks; extend with
cross-attention if needed.
"""

from flax import nnx
import jax.numpy as jnp


class MultimodalFusion(nnx.Module):
    """Placeholder fusion stack."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int):
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        # TODO: initialize transformer blocks.

    def __call__(self, tokens: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        _ = train
        # TODO: apply transformer layers for fusion.
        raise NotImplementedError
