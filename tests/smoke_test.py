"""
Lightweight smoke tests for scaffolded components.

These tests do not rely on real CMHT data or racecar_gym; they ensure shapes and
basic forward passes do not crash on dummy data.
"""

import jax.numpy as jnp
from flax import nnx

from src.models.encoders import ModalityViT
from src.models.fusion import MultimodalFusion
from src.models.heads import PretrainHead, PolicyHead, ValueHead
from src.training.pretrainer import _mask_tokens
from src.data.cmht_loader import CMHTDataSource
from src.training.rl_agent import PPOModel, _dummy_batch


def test_modality_vit_forward():
    rngs = nnx.Rngs(0)
    model = ModalityViT(embed_dim=32, depth=2, num_heads=4, patch_size=8, in_channels=3, rngs=rngs)
    x = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)
    tokens = model(x, train=False, rngs=rngs)
    assert tokens.shape[0] == 2


def test_fusion_and_heads_forward():
    rngs = nnx.Rngs(1)
    fusion = MultimodalFusion(embed_dim=32, depth=2, num_heads=4, rngs=rngs)
    head = PretrainHead(embed_dim=32, out_dim=32, rngs=rngs)
    tokens = jnp.zeros((2, 16, 32))
    fused = fusion(tokens, train=False)
    preds = head(fused, train=False)
    assert preds.shape == fused.shape


def test_mask_tokens():
    tokens = jnp.ones((2, 4, 8))
    masked, mask = _mask_tokens(tokens, 0.5, rng=nnx.Rngs(2).random())
    assert masked.shape == tokens.shape
    assert mask.shape == tokens.shape[:2]


def test_ppo_model_forward_dummy():
    rngs = nnx.Rngs(3)
    model = PPOModel(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=8,
        action_dim=2,
        modalities=("lidar", "rgb"),
        rngs=rngs,
    )
    batch = _dummy_batch(batch_size=2, action_dim=2)
    mean, value, log_std = model({"lidar": batch["lidar"], "rgb": batch["rgb"]}, rngs=rngs)
    assert mean.shape == (2, 2)
    assert value.shape == (2,)
    assert log_std.shape == (2, 2)


def test_cmht_loader_iterates():
    loader = CMHTDataSource("/nonexistent/path")
    it = iter(loader)
    sample = next(it)
    assert "lidar" in sample and "rgb" in sample
