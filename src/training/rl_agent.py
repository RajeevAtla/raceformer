"""
PPO finetuning scaffold using racecar_gym observations.

Loads pretrained weights, optionally freezes encoders, masks missing modalities,
and optimizes policy/value heads.
"""

from typing import Any, Iterable

import jax
import jax.numpy as jnp
import optax
try:
    import orbax.checkpoint as ocp
except ImportError:  # pragma: no cover - optional
    ocp = None
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
        self.log_std = nnx.Param(jnp.zeros((action_dim,)))
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

    def __call__(self, obs: dict[str, jnp.ndarray], *, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        tokens = []
        for name in self.modalities:
            if name not in obs:
                continue
            tokens.append(self.encoders[name](obs[name], train=True, rngs=rngs))
        fused = jnp.concatenate(tokens, axis=1)
        fused = self.fusion(fused, train=True)
        action_mean = self.policy(fused)
        values = self.value(fused)
        log_std = jnp.broadcast_to(self.log_std.value, action_mean.shape)
        return action_mean, values.squeeze(-1), log_std


def ppo_loss(
    model: PPOModel,
    batch: dict[str, jnp.ndarray],
    clip_eps: float,
    *,
    rngs: nnx.Rngs,
) -> jnp.ndarray:
    action_mean, value_pred, log_std = model(batch, rngs=rngs)
    log_prob = _gaussian_logprob(batch["actions"], action_mean, log_std)
    ratio = jnp.exp(log_prob - batch["log_prob"])
    adv = batch["advantages"]
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((value_pred - batch["returns"]) ** 2)
    entropy = jnp.mean(0.5 * (1.0 + jnp.log(2 * jnp.pi)) + log_std)
    return policy_loss + 0.5 * value_loss - 0.01 * entropy


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
    checkpoint_dir = config.get("checkpoint_dir", "runs/ppo/ckpt")
    ckpt_mgr = _init_checkpoint_manager(checkpoint_dir) if ocp is not None else None
    if ckpt_mgr:
        restored = _restore_checkpoint(ckpt_mgr, model, optimizer)
        if restored:
            model, optimizer = restored

    try:
        env = RacecarGymWrapper(config.get("env_id", "Racecar-v0"), seed=config.get("seed", 0))
        step_fn = jax_step(env)
    except Exception as exc:  # pragma: no cover - allows smoke testing without simulator
        print(f"racecar_gym unavailable ({exc}); using dummy rollout.")
        env = None
        step_fn = None

    def loss_fn(model: PPOModel, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs) -> jnp.ndarray:
        return ppo_loss(model, batch, config.get("clip_eps", 0.2), rngs=rngs)

    @jax.jit
    def train_step(model: PPOModel, opt: nnx.Optimizer, batch: dict[str, jnp.ndarray], rngs: nnx.Rngs):
        value_and_grad = nnx.value_and_grad(loss_fn)
        loss, grads = value_and_grad(model, batch, rngs)
        opt = opt.apply_gradient(grads)
        return model, opt, loss

    gamma = config.get("gamma", 0.99)
    gae_lambda = config.get("gae_lambda", 0.95)
    update_epochs = config.get("update_epochs", 1)
    rollout_rng = rngs
    for step in range(config.get("steps", 2)):
        batch, rollout_rng = _rollout_batch(
            env,
            model,
            step_fn,
            config.get("batch_size", 2),
            config["action_dim"],
            rollout_rng,
            gamma,
            gae_lambda,
        ) if env else (_dummy_batch(config.get("batch_size", 2), config["action_dim"]), rollout_rng)
        loss = 0.0
        for _ in range(update_epochs):
            model, optimizer, loss = train_step(model, optimizer, batch, rngs)
        if step % config.get("log_every", 1) == 0:
            print(f"ppo step {step}: loss {loss}")
        if ckpt_mgr and step % config.get("checkpoint_every", 1000) == 0:
            _save_checkpoint(ckpt_mgr, step, model, optimizer)
    # TODO: add proper PPO advantages/returns from trajectories.


def _rollout_batch(
    env: RacecarGymWrapper,
    model: PPOModel,
    step_fn,
    batch_size: int,
    action_dim: int,
    rngs: nnx.Rngs,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, jnp.ndarray], nnx.Rngs]:
    """
    Collect a rollout batch using the current policy.

    Uses a simple on-policy sampler with fixed-horizon returns.
    """
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    rng = rngs
    for _ in range(batch_size):
        obs_masked = mask_missing_modalities(obs)
        action_mean, value, log_std = model(obs_masked, rngs=rng)
        rng, sample_key = rng.split()
        action = _sample_gaussian(action_mean, log_std, sample_key)
        lp = _gaussian_logprob(action, action_mean, log_std)
        observations.append(obs_masked)
        actions.append(action)
        values.append(value)
        log_probs.append(lp)
        step_out = env.step(action) if step_fn is None else step_fn(None, action)
        rewards.append(step_out.reward)
        obs = step_out.obs
    values = jnp.asarray(values)
    rewards = jnp.asarray(rewards)
    returns, advantages = _compute_returns_advantages(rewards, values, gamma=gamma, gae_lambda=gae_lambda)
    batch = {
        "lidar": jnp.asarray([o["lidar"] for o in observations]),
        "rgb": jnp.asarray([o["rgb"] for o in observations]),
        "actions": jnp.asarray(actions),
        "log_prob": jnp.asarray(log_probs),
        "advantages": advantages,
        "returns": returns,
    }
    return batch, rng


def _dummy_batch(batch_size: int, action_dim: int) -> dict[str, jnp.ndarray]:
    """Create a dummy batch for offline smoke testing."""
    h, w = 32, 32
    return {
        "lidar": jnp.zeros((batch_size, h, w, 1), dtype=jnp.float32),
        "rgb": jnp.zeros((batch_size, h, w, 3), dtype=jnp.float32),
        "actions": jnp.zeros((batch_size, action_dim), dtype=jnp.float32),
        "log_prob": jnp.zeros((batch_size,), dtype=jnp.float32),
        "advantages": jnp.ones((batch_size,), dtype=jnp.float32),
        "returns": jnp.zeros((batch_size,), dtype=jnp.float32),
    }


def mask_missing_modalities(obs: dict[str, Any]) -> dict[str, Any]:
    """Mask radar/IR with zeros or learned tokens before model forward."""
    masked = dict(obs)
    masked["radar"] = jnp.zeros_like(jnp.asarray(obs.get("radar", 0.0)))
    masked["ir"] = jnp.zeros_like(jnp.asarray(obs.get("ir", 0.0)))
    return masked


def _sample_gaussian(mean: jnp.ndarray, log_std: jnp.ndarray, rng: jax.random.KeyArray) -> jnp.ndarray:
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    return mean + noise * std


def _gaussian_logprob(actions: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    var = jnp.exp(2.0 * log_std)
    log_prob = -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return log_prob.sum(axis=-1)


def _compute_returns_advantages(rewards: jnp.ndarray, values: jnp.ndarray, gamma: float, gae_lambda: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE(lambda) and returns for a single trajectory."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        next_value = values[t]
    advantages = jnp.asarray(advantages)
    returns = advantages + values
    return returns, advantages


def _init_checkpoint_manager(checkpoint_dir: str):
    """Initialize an Orbax checkpoint manager if available."""
    if ocp is None:
        return None
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    return ocp.CheckpointManager(
        path,
        item_names=("train_state",),
        options=options,
        checkpointers={"train_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler())},
    )


def _save_checkpoint(ckpt_mgr, step: int, model: PPOModel, optimizer: nnx.Optimizer) -> None:
    if ckpt_mgr is None or ocp is None:
        return
    to_save = {
        "model": nnx.state(model),
        "optimizer": optimizer.state,
    }
    try:
        ckpt_mgr.save(step, args=to_save)
    except Exception as exc:  # pragma: no cover
        print(f"PPO checkpoint save failed at step {step}: {exc}")


def _restore_checkpoint(ckpt_mgr, model: PPOModel, optimizer: nnx.Optimizer):
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
        print(f"Restored PPO checkpoint from step {latest}")
        return model, optimizer
    except Exception as exc:  # pragma: no cover
        print(f"PPO checkpoint restore failed: {exc}")
        return None


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
