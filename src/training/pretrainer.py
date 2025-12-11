"""
Masked multimodal pretraining loop.

Implements data loading, masking, forward pass, reconstruction loss, and checkpointing scaffolds.
"""

from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import optax
try:
    import orbax.checkpoint as ocp
except ImportError:  # pragma: no cover - optional dependency
    ocp = None
from flax import nnx

from src.data.cmht_loader import CMHTDataSource
from src.models.encoders import ModalityViT
from src.models.fusion import MultimodalFusion
from src.models.heads import PretrainHead


class MultimodalPretrainModel(nnx.Module):
    """Encoders + fusion + reconstruction head."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        modalities: Iterable[str],
        *,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.modalities = tuple(modalities)
        self.encoders = {
            name: ModalityViT(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=patch_size,
                in_channels=3 if name == "rgb" else 1,
                rngs=rngs,
            )
            for name in self.modalities
        }
        self.fusion = MultimodalFusion(embed_dim=embed_dim, depth=depth, num_heads=num_heads, rngs=rngs)
        self.head = PretrainHead(embed_dim=embed_dim, out_dim=embed_dim, rngs=rngs)

    def __call__(self, batch: dict[str, jnp.ndarray], mask_ratio: float, *, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray]:
        tokens = []
        for name in self.modalities:
            if name not in batch:
                continue
            tokens.append(self.encoders[name](batch[name], train=True, rngs=rngs))
        fused = jnp.concatenate(tokens, axis=1)
        masked, mask = _mask_tokens(fused, mask_ratio, rng=rngs.random())
        fused_out = self.fusion(masked, train=True)
        preds = self.head(fused_out, train=True)
        return preds, mask


def reconstruction_loss(preds: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute MSE on masked tokens only."""
    mask = mask[..., None]
    mse = (preds - targets) ** 2
    masked_mse = jnp.where(mask, mse, 0.0)
    denom = jnp.maximum(mask.sum(), 1.0)
    return masked_mse.sum() / denom


def run_pretraining(config: dict[str, Any]) -> None:
    """
    Execute masked multimodal modeling.

    Wires together the data source, model, masking, loss, optimizer, and checkpointing scaffold.
    """
    data_path = Path(config["data"]["path"])
    modalities: Iterable[str] = config["data"].get(
        "include_modalities", ("lidar", "rgb", "radar", "ir")
    )
    data_source = CMHTDataSource(
        data_path,
        include_modalities=tuple(modalities),
        image_size=config["data"].get("image_size", 224),
    )
    model_cfg = config["model"]
    training_cfg = config["training"]

    rngs = nnx.Rngs(config["training"].get("seed", 0))
    model = MultimodalPretrainModel(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        patch_size=model_cfg.get("patch_size", 16),
        modalities=modalities,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(training_cfg["learning_rate"]))
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "runs/example/ckpt"))
    ckpt_mgr = _init_checkpoint_manager(checkpoint_dir) if ocp is not None else None
    if ckpt_mgr:
        restored = _restore_checkpoint(ckpt_mgr, model, optimizer)
        if restored:
            model, optimizer = restored

    def loss_fn(model: MultimodalPretrainModel, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs) -> jnp.ndarray:
        preds, mask = model(batch, training_cfg["mask_ratio"], rngs=rngs)
        targets = jnp.concatenate(
            [model.encoders[m](batch[m], train=False, rngs=rngs) for m in model.modalities if m in batch],
            axis=1,
        )
        return reconstruction_loss(preds, targets, mask)

    @jax.jit
    def train_step(model: MultimodalPretrainModel, opt: nnx.Optimizer, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs):
        value_and_grad = nnx.value_and_grad(loss_fn)
        loss, grads = value_and_grad(model, batch, rngs)
        opt = opt.apply_gradient(grads)
        return model, opt, loss

    # Placeholder loop; replace with real Grain loader iteration.
    batch_iter = _batch_iterator(data_source, training_cfg["batch_size"])
    for step in range(training_cfg["steps"]):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = _batch_iterator(data_source, training_cfg["batch_size"])
            batch = next(batch_iter)
        step_rngs = rngs.split()
        model, optimizer, loss = train_step(model, optimizer, batch, step_rngs)
        if step % training_cfg.get("log_every", 100) == 0:
            print(f"step {step}: loss {loss}")
        if ckpt_mgr and step % training_cfg.get("checkpoint_every", 1000) == 0:
            _save_checkpoint(ckpt_mgr, step, model, optimizer)


def _mask_tokens(tokens: jnp.ndarray, mask_ratio: float, *, rng: jax.random.KeyArray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Randomly mask a ratio of tokens; returns masked tokens and mask indices."""
    mask = jax.random.bernoulli(rng, mask_ratio, tokens.shape[:-1])
    masked = jnp.where(mask[..., None], 0, tokens)
    return masked, mask


def _dummy_batch(batch_size: int) -> dict[str, jnp.ndarray]:
    """Generate a small dummy batch for smoke-testing the training loop."""
    h, w = 32, 32
    return {
        "rgb": jnp.zeros((batch_size, h, w, 3), dtype=jnp.float32),
        "lidar": jnp.zeros((batch_size, h, w, 1), dtype=jnp.float32),
        "radar": jnp.zeros((batch_size, h, w, 1), dtype=jnp.float32),
        "ir": jnp.zeros((batch_size, h, w, 1), dtype=jnp.float32),
    }


def _batch_iterator(data_source: CMHTDataSource, batch_size: int):
    """
    Collate samples from CMHTDataSource into batches of JAX arrays.

    Falls back to dummy batches if the datasource cannot be iterated.
    """
    try:
        buffer = []
        for sample in data_source:
            buffer.append(sample)
            if len(buffer) == batch_size:
                yield _collate(buffer)
                buffer = []
        if buffer:
            yield _collate(buffer)
    except Exception as exc:  # pragma: no cover - allows running without dataset
        print(f"CMHTDataSource failed ({exc}); using dummy batches.")
        while True:
            yield _dummy_batch(batch_size)


def _collate(samples: list[dict[str, Any]]) -> dict[str, jnp.ndarray]:
    """Stack list of sample dicts into batch dict of arrays."""
    batch = {}
    for key in samples[0]:
        batch[key] = jnp.asarray([s[key] for s in samples])
    return batch


def _init_checkpoint_manager(checkpoint_dir: Path):
    """Initialize an Orbax checkpoint manager if available."""
    if ocp is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_names=("train_state",),
        options=options,
        checkpointers={"train_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler())},
    )


def _save_checkpoint(ckpt_mgr, step: int, model: MultimodalPretrainModel, optimizer: nnx.Optimizer) -> None:
    """Save model and optimizer state with Orbax."""
    if ckpt_mgr is None or ocp is None:
        return
    to_save = {
        "model": nnx.state(model),
        "optimizer": optimizer.state,
    }
    try:
        ckpt_mgr.save(step, args=to_save)
    except Exception as exc:  # pragma: no cover - avoid hard failure in scaffolding
        print(f"Checkpoint save failed at step {step}: {exc}")


def _restore_checkpoint(ckpt_mgr, model: MultimodalPretrainModel, optimizer: nnx.Optimizer):
    """Restore latest checkpoint if present; returns (model, optimizer) or None."""
    if ckpt_mgr is None or ocp is None:
        return None
    latest = ckpt_mgr.latest_step()
    if latest is None:
        return None
    try:
        restored = ckpt_mgr.restore(latest, args=ocp.args.Composite(
            model=nnx.state(model),
            optimizer=optimizer.state,
        ))
        model = nnx.merge(model, restored["model"])
        optimizer = optimizer.replace_state(restored["optimizer"])
        print(f"Restored checkpoint from step {latest}")
        return model, optimizer
    except Exception as exc:  # pragma: no cover
        print(f"Checkpoint restore failed: {exc}")
        return None


def main() -> None:
    example_config = {
        "model": {"embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 8},
        "training": {"batch_size": 2, "learning_rate": 3e-4, "steps": 2, "mask_ratio": 0.5, "seed": 0},
        "data": {"path": "/path/to/cmht"},
    }
    run_pretraining(example_config)


if __name__ == "__main__":
    main()
