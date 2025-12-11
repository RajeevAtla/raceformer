"""
PPO finetuning scaffold using racecar_gym observations.

Loads pretrained weights, optionally freezes encoders, masks missing modalities,
and optimizes policy/value heads.
"""

from typing import Any, Iterable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from src.data.gym_wrapper import RacecarGymWrapper, jax_step
from src.models.encoders import ModalityViT
from src.models.fusion import MultimodalFusion
from src.models.heads import PolicyHead, ValueHead


class PPOModel(nnx.Module):
    """Encoders + fusion + policy/value heads."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        action_dim: int,
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
        self.policy = PolicyHead(embed_dim=embed_dim, action_dim=action_dim, rngs=rngs)
        self.value = ValueHead(embed_dim=embed_dim, rngs=rngs)

    def __call__(self, obs: dict[str, jnp.ndarray], *, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray]:
        tokens = []
        for name in self.modalities:
            if name not in obs:
                continue
            tokens.append(self.encoders[name](obs[name], train=True, rngs=rngs))
        fused = jnp.concatenate(tokens, axis=1)
        fused = self.fusion(fused, train=True)
        actions = self.policy(fused)
        values = self.value(fused)
        return actions, values.squeeze(-1)


def ppo_loss(
    model: PPOModel,
    batch: dict[str, jnp.ndarray],
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    *,
    rngs: nnx.Rngs,
) -> jnp.ndarray:
    actions_pred, values_pred = model(batch, rngs=rngs)
    # Placeholder policy loss: MSE between predicted actions and taken actions.
    policy_loss = jnp.mean((actions_pred - batch["actions"]) ** 2 * advantages[..., None])
    value_loss = jnp.mean((values_pred - returns) ** 2)
    return policy_loss + value_loss


def run_ppo(config: dict[str, Any]) -> None:
    """Run PPO loop scaffold."""
    model_cfg = config["model"]
    modalities = ("lidar", "rgb")  # simulator lacks radar/ir; gps/imu handled outside token flow.
    rngs = nnx.Rngs(config.get("seed", 0))
    model = PPOModel(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        patch_size=model_cfg.get("patch_size", 16),
        action_dim=config["action_dim"],
        modalities=modalities,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adam(config.get("learning_rate", 3e-4)))

    try:
        env = RacecarGymWrapper(config.get("env_id", "Racecar-v0"), seed=config.get("seed", 0))
        step_fn = jax_step(env)
    except Exception as exc:  # pragma: no cover - allows smoke testing without simulator
        print(f"racecar_gym unavailable ({exc}); using dummy rollout.")
        env = None
        step_fn = None

    def loss_fn(model: PPOModel, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs) -> jnp.ndarray:
        adv = batch["advantages"]
        ret = batch["returns"]
        return ppo_loss(model, batch, adv, ret, rngs=rngs)

    @jax.jit
    def train_step(model: PPOModel, opt: nnx.Optimizer, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs):
        value_and_grad = nnx.value_and_grad(loss_fn)
        loss, grads = value_and_grad(model, batch, rngs)
        opt = opt.apply_gradient(grads)
        return model, opt, loss

    for step in range(config.get("steps", 2)):
        batch = _rollout_batch(env, step_fn, config.get("batch_size", 2), config["action_dim"]) if env else _dummy_batch(config.get("batch_size", 2), config["action_dim"])
        model, optimizer, loss = train_step(model, optimizer, batch, rngs)
        if step % config.get("log_every", 1) == 0:
            print(f"ppo step {step}: loss {loss}")
    # TODO: add checkpointing and proper PPO advantages/returns from trajectories.


def _rollout_batch(env: RacecarGymWrapper, step_fn, batch_size: int, action_dim: int) -> dict[str, jnp.ndarray]:
    """
    Collect a small rollout batch from the environment.

    Placeholder: uses random actions; replace with policy sampling and GAE/returns.
    """
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    values = []
    for _ in range(batch_size):
        obs_masked = mask_missing_modalities(obs)
        action = jnp.zeros((action_dim,))  # replace with policy sampling
        observations.append(obs_masked)
        actions.append(action)
        values.append(0.0)
        step_out = env.step(action) if step_fn is None else step_fn(None, action)
        rewards.append(step_out.reward)
        obs = step_out.obs
    advantages = jnp.ones((batch_size,))
    returns = jnp.asarray(rewards)
    return {
        "lidar": jnp.asarray([o["lidar"] for o in observations]),
        "rgb": jnp.asarray([o["rgb"] for o in observations]),
        "actions": jnp.asarray(actions),
        "advantages": advantages,
        "returns": returns,
    }


def _dummy_batch(batch_size: int, action_dim: int) -> dict[str, jnp.ndarray]:
    """Create a dummy batch for offline smoke testing."""
    h, w = 32, 32
    return {
        "lidar": jnp.zeros((batch_size, h, w, 1), dtype=jnp.float32),
        "rgb": jnp.zeros((batch_size, h, w, 3), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, action_dim), dtype=jnp.float32),
        "advantages": jnp.ones((batch_size,), dtype=jnp.float32),
        "returns": jnp.zeros((batch_size,), dtype=jnp.float32),
    }


def mask_missing_modalities(obs: dict[str, Any]) -> dict[str, Any]:
    """Mask radar/IR with zeros or learned tokens before model forward."""
    masked = dict(obs)
    masked["radar"] = jnp.zeros_like(jnp.asarray(obs.get("radar", 0.0)))
    masked["ir"] = jnp.zeros_like(jnp.asarray(obs.get("ir", 0.0)))
    return masked


def main() -> None:
    example_config = {
        "env_id": "Racecar-v0",
        "seed": 0,
        "model": {"embed_dim": 64, "depth": 2, "num_heads": 4, "patch_size": 8},
        "action_dim": 2,
        "batch_size": 2,
        "steps": 2,
    }
    run_ppo(example_config)


if __name__ == "__main__":
    main()
