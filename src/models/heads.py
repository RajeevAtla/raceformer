"""
Heads for pretraining reconstruction and RL policy/value outputs.
"""

import jax.numpy as jnp
from flax import nnx


class PretrainHead(nnx.Module):
    """MAE-style reconstruction or contrastive projection."""

    def __init__(self, embed_dim: int, out_dim: int, *, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.embed_dim = embed_dim
        self.out_dim = out_dim or embed_dim
        self.proj1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.proj2 = nnx.Linear(embed_dim, self.out_dim, rngs=rngs)

    def __call__(self, tokens: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        _ = train
        x = nnx.gelu(self.proj1(tokens))
        x = self.proj2(x)
        return x


class PolicyHead(nnx.Module):
    """Continuous action head: Dense -> Tanh."""

    def __init__(self, embed_dim: int, action_dim: int, *, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.action_dim = action_dim
        self.policy = nnx.Linear(in_features=embed_dim, out_features=action_dim, rngs=rngs)

    def __call__(self, fused_tokens: jnp.ndarray) -> jnp.ndarray:
        pooled = fused_tokens.mean(axis=1)  # simple token pooling
        logits = self.policy(pooled)
        return jnp.tanh(logits)


class ValueHead(nnx.Module):
    """Scalar critic head."""

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.value = nnx.Linear(in_features=embed_dim, out_features=1, rngs=rngs)

    def __call__(self, fused_tokens: jnp.ndarray) -> jnp.ndarray:
        pooled = fused_tokens.mean(axis=1)
        return self.value(pooled)
