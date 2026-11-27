"""Hybrid Soft Actor-Critic with twin quantile critics and two temperatures."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from env.resource_taxi import DRIVE, NUM_ACTIONS


@dataclass(frozen=True)
class AgentConfig:
    action_count: int = NUM_ACTIONS
    width: int = 128
    quantiles: int = 24
    learning_rate: float = 0.00025
    actor_learning_rate: float = 0.00005
    gradient_steps: int = 32
    minibatch_size: int = 256
    temperature_rate: float = 0.0003
    target_rate: float = 0.008
    risk_weight: float = 0.
    upside_weight: float = 0.
    tail_mass: float = 0.20
    demo_weight: float = 0.
    gamma: float = 0.997
    continuous_temperature: float = .0006
    discrete_temperature: float = .0012
    automatic_temperature: bool = True
    automatic_discrete_temperature: bool = False
    discrete_target_entropy_fraction: float = .55
    sample_target_action: bool = False
    multihead_critic: bool = False
    critic_layer_norm: bool = False
    separate_recurrent_encoders: bool = False
    recurrent_observation_skip: bool = False
    categorical_actor_update: str = 'reverse_kl'  # Keep original SAC; forward_kl is a labeled variant.


class Transition(NamedTuple):
    features: jax.Array
    mask: jax.Array
    control: jax.Array
    choice: jax.Array
    reward: jax.Array
    done: jax.Array
    next_features: jax.Array
    next_mask: jax.Array
    discount: jax.Array | None = None
    sequence_start: jax.Array | None = None


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


def _mlp(params, x, layer_norm=False):
    for w, b in params[:-1]:
        x = x @ w + b
        if layer_norm:
            x=(x-x.mean(axis=-1,keepdims=True))*jax.lax.rsqrt(x.var(axis=-1,keepdims=True)+1e-5)
        x = jax.nn.silu(x)
    w, b = params[-1]
    return x @ w + b


def _encode(features, mask):
    return jnp.concatenate((features, mask.astype(features.dtype)), axis=-1)


def actor_outputs(actor, features, mask):
    out = _mlp(actor, _encode(features, mask))
    logits = jnp.where(mask, out[..., 4:], -1e9)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    return 3.*jnp.tanh(out[..., :2]/3.), jnp.clip(out[..., 2:4], -4.5, 0.5), log_prob


def continuous_mask(count):
    if count == 26:
        return jnp.array(tuple(i not in (0, 23, 24) for i in range(26)))
    if count == NUM_ACTIONS:
        return jax.nn.one_hot(DRIVE, count, dtype=jnp.bool_)
    raise ValueError(f"Unknown hybrid action schema with {count} operations")


def sample(actor, key, features, mask, deterministic=False):
    key_control, key_choice = jax.random.split(key)
    mean, log_std, log_disc = actor_outputs(actor, features, mask)
    if deterministic:
        raw = mean
        choice = jnp.argmax(log_disc, axis=-1).astype(jnp.int32)
    else:
        raw = mean + jnp.exp(log_std)*jax.random.normal(key_control, mean.shape)
        choice = jax.random.categorical(key_choice, log_disc, axis=-1).astype(jnp.int32)
    return jnp.where(continuous_mask(log_disc.shape[-1])[choice][..., None], jnp.tanh(raw), 0.), choice


def _sample_all(actor, key, features, mask):
    mean, log_std, log_disc = actor_outputs(actor, features, mask)
    noise = jax.random.normal(key, mean.shape)
    raw = mean + jnp.exp(log_std) * noise
    log_normal = -0.5 * (noise**2 + jnp.log(2*jnp.pi)) - log_std
    log_jacobian = 2.*(jnp.log(2.) - raw - jax.nn.softplus(-2.*raw))
    log_control = (log_normal - log_jacobian).sum(axis=-1)
    return jnp.tanh(raw), log_control, jnp.exp(log_disc), log_disc


def _critics(critics, features, mask, control, layer_norm=False):
    # All branches use the parked control of zero except DRIVE.
    encoded = _encode(features, mask)
    count = mask.shape[-1]
    if critics[0][0][0].shape[0] == encoded.shape[-1]+2:
        def predict(critic):
            moving=_mlp(critic,jnp.concatenate((encoded,control),axis=-1),layer_norm).reshape((*encoded.shape[:-1],count,-1))
            parked=_mlp(critic,jnp.concatenate((encoded,jnp.zeros_like(control)),axis=-1),layer_norm).reshape(moving.shape)
            return jnp.where(continuous_mask(count)[...,None],moving,parked)
        return jnp.stack([predict(critic) for critic in critics])
    shape = (*encoded.shape[:-1], count)
    control = jnp.broadcast_to(control[..., None, :], (*shape, 2))*continuous_mask(count)[..., None]
    encoded = jnp.broadcast_to(encoded[..., None, :], (*shape, encoded.shape[-1]))
    actions = jnp.broadcast_to(jnp.eye(count), (*shape, count))
    inputs = jnp.concatenate((encoded, control, actions), axis=-1)
    return jnp.stack([_mlp(critic, inputs,layer_norm) for critic in critics])


def _taken(critics, batch, layer_norm=False):
    count = batch.mask.shape[-1]
    actions = jax.nn.one_hot(batch.choice, count)
    control = jnp.where(continuous_mask(count)[batch.choice][..., None], batch.control, 0.)
    if critics[0][0][0].shape[0] == batch.features.shape[-1]+count+2:
        inputs=jnp.concatenate((_encode(batch.features,batch.mask),control),axis=-1)
        def predict(critic):
            all_values=_mlp(critic,inputs,layer_norm).reshape((*inputs.shape[:-1],count,-1))
            return jnp.take_along_axis(all_values,batch.choice[...,None,None],axis=-2)[...,0,:]
        return jnp.stack([predict(critic) for critic in critics])
    inputs = jnp.concatenate((_encode(batch.features, batch.mask), control, actions), axis=-1)
    return jnp.stack([_mlp(critic, inputs,layer_norm) for critic in critics])


def _soft_terms(log_c, log_d, temperatures):
    alpha_c, alpha_d = temperatures
    return alpha_d*log_d + alpha_c*continuous_mask(log_d.shape[-1])*log_c[..., None]


def actor_objective(prob,log_d,log_c,score,temperatures,method='reverse_kl'):
    """Hybrid objective with an optional categorical forward-KL projection.

    Target probabilities come only from the frozen critic, never action labels.
    Forward CE is deliberately unscaled: alpha_d determines its target entropy,
    while a separate continuous objective retains reparameterized SAC gradients.
    This is an algorithm variant, not an algebraic rewrite of original SAC.
    """
    if method=='reverse_kl':
        return jnp.mean(jnp.sum(prob*(_soft_terms(log_c,log_d,temperatures)-score),axis=-1))
    if method!='forward_kl':raise ValueError('Unknown categorical actor update: '+method)
    alpha_c,alpha_d=temperatures
    continuous_cost=alpha_c*continuous_mask(log_d.shape[-1])*log_c[...,None]
    target_logits=(score-continuous_cost)/jnp.maximum(alpha_d,1e-8)
    target=jax.lax.stop_gradient(jax.nn.softmax(jnp.where(log_d>-1e8,target_logits,-1e9),axis=-1))
    categorical=-jnp.mean(jnp.sum(target*log_d,axis=-1))
    continuous=jnp.mean(jnp.sum(jax.lax.stop_gradient(prob)*(continuous_cost-score),axis=-1))
    return categorical+continuous


def bootstrap_distribution(learner, key, batch, config, next_actor_features=None):
    """Distributional soft target; optional categorical sampling is unbiased.

    The sampled path avoids constructing all action x quantile-pair targets.
    It samples the same mixture represented by exact categorical enumeration.
    """
    control_key, choice_key = jax.random.split(key)
    control, log_c, prob, log_d = _sample_all(
        learner.actor,control_key,batch.next_features if next_actor_features is None else next_actor_features,batch.next_mask)
    if config.sample_target_action:
        choice = jax.random.categorical(choice_key, log_d).astype(jnp.int32)
        following = batch._replace(features=batch.next_features, mask=batch.next_mask,
                                   control=control, choice=choice)
        candidates = _taken(learner.targets, following,config.critic_layer_norm)[:, :, None, :]
        entropy = jnp.take_along_axis(
            _soft_terms(log_c, log_d, jnp.exp(learner.log_temperatures)), choice[:, None], -1)
        target_prob = jnp.ones((prob.shape[0], 1))
    else:
        candidates = _critics(learner.targets, batch.next_features, batch.next_mask, control,config.critic_layer_norm)
        entropy = _soft_terms(log_c, log_d, jnp.exp(learner.log_temperatures))
        target_prob = prob
    target = jnp.where((candidates[0].mean(-1) <= candidates[1].mean(-1))[..., None],
                       candidates[0], candidates[1])
    discount = config.gamma if batch.discount is None else batch.discount[:, None, None]
    target = batch.reward[:, None, None]+discount*(~batch.done[:, None, None])*(target-entropy[..., None])
    return jax.lax.stop_gradient((target, target_prob))


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
        self.actor_optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adam(config.actor_learning_rate))
        self.temp_opt = optax.adam(config.temperature_rate)

    def init(self, key, feature_count):
        c = self.config
        key_a, key_q1, key_q2 = jax.random.split(key, 3)
        encoded = feature_count + c.action_count
        actor = _init_mlp(key_a, (encoded, c.width, c.width, 4 + c.action_count), 0.18)
        w, b = actor[-1]
        actor = (*actor[:-1], (w, b.at[2:4].set(-1.4)))
        critics = tuple(
            _init_mlp(k, (encoded+2 if c.multihead_critic else encoded+2+c.action_count,
                c.width,c.width,c.quantiles*c.action_count if c.multihead_critic else c.quantiles), 0.02)
            for k in (key_q1, key_q2)
        )
        log_t = jnp.log(jnp.array((c.continuous_temperature, c.discrete_temperature)))
        return Learner(actor, critics, critics, log_t, self.actor_optimizer.init(actor), self.opt.init(critics),
                       self.temp_opt.init(log_t))

    def imitate(self, learner, demo):
        loss, grad = jax.value_and_grad(imitation_loss)(learner.actor, demo)
        updates, state = self.actor_optimizer.update(grad, learner.actor_opt)
        return learner._replace(actor=optax.apply_updates(learner.actor, updates), actor_opt=state), loss

    def update(self, learner, key, batch, demo=None, actor_features=None, next_actor_features=None):
        c = self.config
        target_key, actor_key = jax.random.split(key)
        temperatures = jnp.exp(learner.log_temperatures)
        target, next_prob = bootstrap_distribution(learner,target_key,batch,c,next_actor_features)

        def critic_loss(params):
            return _distribution_loss(_taken(params, batch,c.critic_layer_norm), target, next_prob, c)

        loss_q, grad = jax.value_and_grad(critic_loss)(learner.critics)
        updates, critic_opt = self.opt.update(grad, learner.critic_opt)
        critics = optax.apply_updates(learner.critics, updates)
        targets = optax.incremental_update(critics, learner.targets, c.target_rate)

        def actor_loss(params):
            control, log_c, prob, log_d = _sample_all(params,actor_key,
                batch.features if actor_features is None else actor_features,batch.mask)
            candidates = _critics(critics, batch.features, batch.mask, control,c.critic_layer_norm)
            values = jnp.where((candidates[0].mean(axis=-1) <= candidates[1].mean(axis=-1))[..., None],
                               candidates[0], candidates[1])
            values = jnp.sort(values, axis=-1)
            score = _risk_score(values, c)
            soft = actor_objective(prob,log_d,log_c,score,temperatures,c.categorical_actor_update)
            return soft, (log_c, prob, log_d)

        (loss_a, (log_c, prob, log_d)), grad = jax.value_and_grad(actor_loss, has_aux=True)(learner.actor)
        updates, actor_opt = self.actor_optimizer.update(grad, learner.actor_opt)
        actor = optax.apply_updates(learner.actor, updates)

        valid_count = batch.mask.sum(axis=-1)
        discrete_active = valid_count > 1
        prob_drive = (prob*continuous_mask(prob.shape[-1])).sum(axis=-1)
        entropy_c = jnp.sum(-log_c*prob_drive)/jnp.maximum(prob_drive.sum(), 1.)
        entropy_d = -jnp.sum(prob*log_d, axis=-1)
        discrete_error = jnp.sum(discrete_active*(entropy_d - c.discrete_target_entropy_fraction*jnp.log(valid_count)))
        discrete_error /= jnp.maximum(discrete_active.sum(), 1)
        errors = jax.lax.stop_gradient(jnp.array((entropy_c + 1.8, discrete_error)))
        if c.automatic_temperature or c.automatic_discrete_temperature:
            updates, temp_opt = self.temp_opt.update(errors, learner.temperature_opt)
            log_t = jnp.clip(optax.apply_updates(learner.log_temperatures, updates), -12., -2.)
            if not c.automatic_temperature:
                log_t=log_t.at[0].set(learner.log_temperatures[0])
        else:
            log_t, temp_opt = learner.log_temperatures, learner.temperature_opt
        return Learner(actor, critics, targets, log_t, actor_opt, critic_opt, temp_opt), jnp.array(
            (loss_q, loss_a, temperatures[0], temperatures[1], entropy_c, entropy_d.mean())
        )
