"""
Vision Transformer encoders for each modality using Flax NNX.

This is a scaffold; implement patch embedding, positional encodings, and transformer
blocks tailored to each sensor modality.
"""

from typing import Any

from flax import nnx
import jax.numpy as jnp


class PatchEmbed(nnx.Module):
    """Placeholder for modality-specific patch embedding."""

    def __init__(self, embed_dim: int, patch_size: int):
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class ModalityViT(nnx.Module):
    """
    Transformer encoder per modality.

    Expected output: sequence of latent tokens Z_modality.
    """

    def __init__(self, embed_dim: int, depth: int, num_heads: int, patch_size: int):
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch_size=patch_size)
        # TODO: add positional embeddings and transformer encoder stack.

    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        _ = train
        tokens = self.patch_embed(x)
        # TODO: apply positional embeddings and transformer blocks.
        raise NotImplementedError
