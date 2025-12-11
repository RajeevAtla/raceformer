"""
Heads for pretraining reconstruction and RL policy/value outputs.
"""

from flax import nnx
import jax.numpy as jnp


class PretrainHead(nnx.Module):
    """MAE-style reconstruction or contrastive projection head scaffold."""

    def __init__(self, embed_dim: int, out_dim: int):
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        # TODO: define decoder or projection layers.

    def __call__(self, tokens: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        _ = train
        raise NotImplementedError


class PolicyHead(nnx.Module):
    """Continuous action head: Dense -> Tanh."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        # TODO: define dense layer.

    def __call__(self, fused_tokens: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class ValueHead(nnx.Module):
    """Scalar critic head."""

    def __init__(self):
        # TODO: define dense layer.
        ...

    def __call__(self, fused_tokens: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
