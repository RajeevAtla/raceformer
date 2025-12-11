"""
Multimodal fusion module using Flax NNX.

Concatenates modality token sequences and processes them with a transformer stack.
"""

import jax.numpy as jnp
from flax import nnx


class FusionMLP(nnx.Module):
    """Feed-forward block with GELU."""

    def __init__(self, dim: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class FusionBlock(nnx.Module):
    """Self-attention + MLP with pre-LayerNorm for fused tokens."""

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
        self.mlp = FusionMLP(dim=dim, mlp_dim=int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln1(x), decode=False)
        x = x + self.mlp(self.ln2(x))
        return x


class MultimodalFusion(nnx.Module):
    """Transformer stack over concatenated modality tokens."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int, *, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.blocks = nnx.List([
            FusionBlock(dim=embed_dim, num_heads=num_heads, rngs=rngs)
            for _ in range(depth)
        ])
        self.norm = nnx.LayerNorm(embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, tokens: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        _ = train
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens
