"""
Vision Transformer encoder for each modality using Flax NNX.

Implements patch embedding, learnable positional embeddings, and a stack of
Transformer blocks that output modality-specific token sequences.
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx


def _num_patches(height: int, width: int, patch: int) -> int:
    return (height // patch) * (width // patch)


class PatchEmbed(nnx.Module):
    """Patchify an image-like tensor and linearly project to tokens."""

    def __init__(self, embed_dim: int, patch_size: int, in_channels: int, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nnx.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=embed_dim,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        ph = pw = self.patch_size
        if h % ph != 0 or w % pw != 0:
            raise ValueError(f"Input {(h, w)} not divisible by patch size {ph}")
        x = x.reshape(b, h // ph, ph, w // pw, pw, c)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        patches = x.reshape(b, -1, ph * pw * c)
        return self.proj(patches)


class MLPBlock(nnx.Module):
    """Feed-forward block with GELU activation."""

    def __init__(self, dim: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nnx.Module):
    """Self-attention + MLP with pre-LayerNorm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(dim, use_bias=True, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(dim, use_bias=True, rngs=rngs)
        self.mlp = MLPBlock(dim=dim, mlp_dim=int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln1(x), decode=False)
        x = x + self.mlp(self.ln2(x))
        return x


class ModalityViT(nnx.Module):
    """
    Transformer encoder per modality.

    Expected input: (batch, H, W, C) tensor.
    Output: sequence of latent tokens Z_modality.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        in_channels: int = 3,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, patch_size=patch_size, in_channels=in_channels, rngs=rngs)
        # Positional embedding initialized lazily based on input spatial dims.
        self.positional = nnx.data(None)
        self.blocks = nnx.List([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(depth)
        ])
        self.norm = nnx.LayerNorm(embed_dim, use_bias=True, rngs=rngs)

    def _init_positional(self, num_tokens: int) -> None:
        if self.positional is None:
            self.positional = nnx.Param(jnp.zeros((1, num_tokens, self.embed_dim)))

    def __call__(self, x: jnp.ndarray, *, train: bool = True, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
        _ = train
        tokens = self.patch_embed(x)
        b, n, _ = tokens.shape
        self._init_positional(n)
        pos = self.positional[...] if isinstance(self.positional, nnx.Param) else self.positional  # type: ignore[arg-type]
        tokens = tokens + pos
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens
