"""Hybrid Soft Actor-Critic with twin quantile critics and two temperatures."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from env.gamblers_taxi import DRIVE, NUM_ACTIONS


@dataclass(frozen=True)
class AgentConfig:
    width: int = 128
    quantiles: int = 24
    learning_rate: float = 0.00025
    temperature_rate: float = 0.00007
    target_rate: float = 0.008
    risk_weight: float = 0.
    upside_weight: float = 0.
    tail_mass: float = 0.20
    demo_weight: float = 0.35
    gamma: float = 0.997


class Transition(NamedTuple):
    features: jax.Array
    mask: jax.Array
    control: jax.Array
    choice: jax.Array
    reward: jax.Array
    done: jax.Array
    next_features: jax.Array
    next_mask: jax.Array


class Learner(NamedTuple):
    actor: tuple
    critics: tuple
    targets: tuple
    log_temperatures: jax.Array
    actor_opt: tuple
    critic_opt: tuple
    temperature_opt: tuple


def _init_mlp(key, sizes, output_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    weights = []
    for i, (layer_key, inputs, outputs) in enumerate(zip(keys, sizes, sizes[1:])):
        scale = output_scale if i == len(keys) - 1 else 1.4
        w = jax.random.normal(layer_key, (inputs, outputs)) * (scale / inputs**0.5)
        weights.append((w, jnp.zeros((outputs,))))
    return tuple(weights)


def _mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def _encode(features, mask):
    return jnp.concatenate((features, mask.astype(features.dtype)), axis=-1)


def actor_outputs(actor, features, mask):
    out = _mlp(actor, _encode(features, mask))
    logits = jnp.where(mask, out[..., 4:], -1e9)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    return out[..., :2], jnp.clip(out[..., 2:4], -4.5, 0.5), log_prob


def sample(actor, key, features, mask, deterministic=False):
    key_control, key_choice = jax.random.split(key)
    mean, log_std, log_disc = actor_outputs(actor, features, mask)
    if deterministic:
        raw = mean
        choice = jnp.argmax(log_disc, axis=-1).astype(jnp.int32)
    else:
        raw = mean + jnp.exp(log_std)*jax.random.normal(key_control, mean.shape)
        choice = jax.random.categorical(key_choice, log_disc, axis=-1).astype(jnp.int32)
    return jnp.where((choice == DRIVE)[..., None], jnp.tanh(raw), 0.), choice


def _sample_all(actor, key, features, mask):
    mean, log_std, log_disc = actor_outputs(actor, features, mask)
    noise = jax.random.normal(key, mean.shape)
    raw = mean + jnp.exp(log_std) * noise
    log_normal = -0.5 * (noise**2 + jnp.log(2*jnp.pi)) - log_std
    log_jacobian = 2.*(jnp.log(2.) - raw - jax.nn.softplus(-2.*raw))
    log_control = (log_normal - log_jacobian).sum(axis=-1)
    return jnp.tanh(raw), log_control, jnp.exp(log_disc), log_disc


def _critics(critics, features, mask, control):
    # All branches use the parked control of zero except DRIVE.
    encoded = _encode(features, mask)
    shape = (*encoded.shape[:-1], NUM_ACTIONS)
    control = jnp.zeros((*shape, 2), dtype=control.dtype).at[..., DRIVE, :].set(control)
    encoded = jnp.broadcast_to(encoded[..., None, :], (*shape, encoded.shape[-1]))
    actions = jnp.broadcast_to(jnp.eye(NUM_ACTIONS), (*shape, NUM_ACTIONS))
    inputs = jnp.concatenate((encoded, control, actions), axis=-1)
    return jnp.stack([_mlp(critic, inputs) for critic in critics])


def _taken(critics, batch):
    actions = jax.nn.one_hot(batch.choice, NUM_ACTIONS)
    control = jnp.where((batch.choice == DRIVE)[..., None], batch.control, 0.)
    inputs = jnp.concatenate((_encode(batch.features, batch.mask), control, actions), axis=-1)
    return jnp.stack([_mlp(critic, inputs) for critic in critics])


def _soft_terms(log_c, log_d, temperatures):
    alpha_c, alpha_d = temperatures
    return alpha_d*log_d + alpha_c*jax.nn.one_hot(DRIVE, NUM_ACTIONS)*log_c[..., None]


def _distribution_loss(pred, target, prob, config):
    tau = (jnp.arange(config.quantiles) + 0.5)/config.quantiles
    error = target[None, :, None, :, :] - pred[..., None, None]
    absolute = jnp.abs(error)
    huber = jnp.where(absolute < 1., 0.5*error**2, absolute - 0.5)
    weights = jnp.abs(tau[None, None, :, None, None] - (error < 0).astype(error.dtype))
    return jnp.mean(jnp.sum(jnp.mean(huber*weights, axis=-1)*prob[None, :, None, :], axis=-1))


def _risk_score(values, config):
    mean = values.mean(axis=-1)
    mass = jnp.clip(config.tail_mass*config.quantiles - jnp.arange(config.quantiles), 0., 1.)
    tail = (values * mass).sum(axis=-1) / mass.sum()
    upside = (values * mass[::-1]).sum(axis=-1) / mass.sum()
    return ((1. - config.risk_weight - config.upside_weight)*mean
            + config.risk_weight*tail + config.upside_weight*upside)


def imitation_loss(actor, demo):
    mean, log_std, log_disc = actor_outputs(actor, demo.features, demo.mask)
    active = demo.choice == DRIVE
    actual = jnp.take_along_axis(log_disc, demo.choice[..., None], axis=-1)[..., 0]
    continuous = jnp.mean((jnp.tanh(mean) - demo.control)**2, axis=-1)
    regularizer = 0.006*jnp.mean((log_std + 1.7)**2, axis=-1)
    return -actual.mean() + jnp.sum(active*(continuous + regularizer))/jnp.maximum(active.sum(), 1)


class HybridDSAC:
    def __init__(self, config):
        self.config = config
        self.opt = optax.chain(optax.clip_by_global_norm(5.), optax.adam(config.learning_rate))
        self.temp_opt = optax.adam(config.temperature_rate)

    def init(self, key, feature_count):
        c = self.config
        key_a, key_q1, key_q2 = jax.random.split(key, 3)
        encoded = feature_count + NUM_ACTIONS
        actor = _init_mlp(key_a, (encoded, c.width, c.width, 4 + NUM_ACTIONS), 0.18)
        w, b = actor[-1]
        actor = (*actor[:-1], (w, b.at[2:4].set(-1.4)))
        critics = tuple(
            _init_mlp(k, (encoded + 2 + NUM_ACTIONS, c.width, c.width, c.quantiles), 0.24)
            for k in (key_q1, key_q2)
        )
        log_t = jnp.log(jnp.array((0.006, 0.012)))
        return Learner(actor, critics, critics, log_t, self.opt.init(actor), self.opt.init(critics),
                       self.temp_opt.init(log_t))

    def imitate(self, learner, demo):
        loss, grad = jax.value_and_grad(imitation_loss)(learner.actor, demo)
        updates, state = self.opt.update(grad, learner.actor_opt)
        return learner._replace(actor=optax.apply_updates(learner.actor, updates), actor_opt=state), loss

    def update(self, learner, key, batch, demo):
        c = self.config
        target_key, actor_key = jax.random.split(key)
        temperatures = jnp.exp(learner.log_temperatures)
        next_control, next_log_c, next_prob, next_log_d = _sample_all(
            learner.actor, target_key, batch.next_features, batch.next_mask
        )
        candidates = _critics(learner.targets, batch.next_features, batch.next_mask, next_control)
        # Pick a whole distribution by its lower mean. This preserves quantile structure.
        target = jnp.where((candidates[0].mean(axis=-1) <= candidates[1].mean(axis=-1))[..., None],
                           candidates[0], candidates[1])
        entropy = _soft_terms(next_log_c, next_log_d, temperatures)
        target = batch.reward[:, None, None] + c.gamma*(~batch.done[:, None, None])*(
            target - entropy[..., None]
        )
        target, next_prob = jax.lax.stop_gradient((target, next_prob))

        def critic_loss(params):
            return _distribution_loss(_taken(params, batch), target, next_prob, c)

        loss_q, grad = jax.value_and_grad(critic_loss)(learner.critics)
        updates, critic_opt = self.opt.update(grad, learner.critic_opt)
        critics = optax.apply_updates(learner.critics, updates)
        targets = optax.incremental_update(critics, learner.targets, c.target_rate)

        def actor_loss(params):
            control, log_c, prob, log_d = _sample_all(params, actor_key, batch.features, batch.mask)
            candidates = _critics(critics, batch.features, batch.mask, control)
            values = jnp.where((candidates[0].mean(axis=-1) <= candidates[1].mean(axis=-1))[..., None],
                               candidates[0], candidates[1])
            values = jnp.sort(values, axis=-1)
            score = _risk_score(values, c)
            entropy = _soft_terms(log_c, log_d, temperatures)
            soft = jnp.mean(jnp.sum(prob * (entropy - score), axis=-1))
            return soft + c.demo_weight*imitation_loss(params, demo), (log_c, prob, log_d)

        (loss_a, (log_c, prob, log_d)), grad = jax.value_and_grad(actor_loss, has_aux=True)(learner.actor)
        updates, actor_opt = self.opt.update(grad, learner.actor_opt)
        actor = optax.apply_updates(learner.actor, updates)

        valid_count = batch.mask.sum(axis=-1)
        discrete_active = valid_count > 1
        prob_drive = prob[:, DRIVE]
        entropy_c = jnp.sum(-log_c*prob_drive)/jnp.maximum(prob_drive.sum(), 1.)
        entropy_d = -jnp.sum(prob*log_d, axis=-1)
        discrete_error = jnp.sum(discrete_active*(entropy_d - 0.55*jnp.log(valid_count)))
        discrete_error /= jnp.maximum(discrete_active.sum(), 1)
        errors = jax.lax.stop_gradient(jnp.array((entropy_c + 1.8, discrete_error)))
        updates, temp_opt = self.temp_opt.update(errors, learner.temperature_opt)
        log_t = jnp.clip(optax.apply_updates(learner.log_temperatures, updates), -9., -2.)
        return Learner(actor, critics, targets, log_t, actor_opt, critic_opt, temp_opt), jnp.array(
            (loss_q, loss_a, temperatures[0], temperatures[1], entropy_c, entropy_d.mean())
        )
