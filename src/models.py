"""Shared hybrid-policy distributions, optional recurrent encoder, and snapshots."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.hybrid_dsac import _init_mlp, _mlp, actor_outputs, continuous_mask
from env import resource_taxi as env


WIDTH = 128
HISTORY = 32


def init_gru(key, inputs, width=WIDTH):
    a, b = jax.random.split(key)
    return (jax.random.normal(a, (inputs, 3*width))/inputs**.5,
            jax.random.normal(b, (width, 3*width))/width**.5, jnp.zeros(3*width))


def gru(params, x, hidden):
    wx, wh, bias = params
    xr, xz, xn = jnp.split(x@wx+bias, 3, axis=-1)
    hr, hz, hn = jnp.split(hidden@wh, 3, axis=-1)
    reset, update = jax.nn.sigmoid(xr+hr), jax.nn.sigmoid(xz+hz)
    candidate = jnp.tanh(xn+reset*hn)
    return update*hidden+(1-update)*candidate


def encode_history(encoder, history):
    def tick(hidden, observation):
        hidden = jnp.where(jnp.any(observation != 0, axis=-1, keepdims=True),
                           gru(encoder, observation, hidden), hidden)
        return hidden, None
    return jax.lax.scan(tick, jnp.zeros((history.shape[0], WIDTH)),
                        jnp.swapaxes(history, 0, 1))[0]


def recurrent_features(encoder,history,current_observation_skip=False):
    encoded=encode_history(encoder,history)
    return jnp.concatenate((history[:,-1],encoded),axis=-1) if current_observation_skip else encoded


def initial_memory(kind, count, feature_count):
    if kind == "dsac_gru":
        return jnp.zeros((count, HISTORY, feature_count))
    return jnp.zeros((count, WIDTH))


def init_ppo(key, feature_count, recurrent, action_count=env.NUM_ACTIONS):
    a, b, c = jax.random.split(key, 3)
    trunk = _init_mlp(a, (feature_count+action_count, WIDTH, WIDTH), 1.)
    head = _init_mlp(b, (WIDTH, 4+action_count+1), .03)
    w, bias = head[0]
    head = ((w, bias.at[2:4].set(-1.2)),)
    return {"trunk": trunk, "head": head,
            "gru": init_gru(c, WIDTH) if recurrent else ()}


def exploration_logprob(logp,mask,epsilon):
    if epsilon == 0.:
        return logp
    uniform=mask.astype(jnp.float32)/mask.sum(axis=-1,keepdims=True)
    mixture=(1-epsilon)*jnp.exp(logp)+epsilon*uniform
    return jnp.where(mask,jnp.log(jnp.maximum(mixture,1e-30)),-1e9)


def ppo_step(model, features, mask, hidden, exploration_epsilon=0.):
    x = jax.nn.silu(_mlp(model["trunk"], jnp.concatenate((features, mask.astype(jnp.float32)), axis=-1)))
    if model["gru"]:
        hidden = gru(model["gru"], x, hidden)
        x = hidden
    output = _mlp(model["head"], x)
    logp = jax.nn.log_softmax(jnp.where(mask, output[..., 4:-1], -1e9))
    logp = exploration_logprob(logp,mask,exploration_epsilon)
    return 3.*jnp.tanh(output[..., :2]/3.), jnp.clip(output[..., 2:4], -4.5, .5), logp, output[..., -1], hidden


def sample_hybrid(key, mean, logstd, logp, deterministic=False):
    a, b = jax.random.split(key)
    raw = mean if deterministic else mean+jnp.exp(logstd)*jax.random.normal(a, mean.shape)
    choice = jnp.argmax(logp, axis=-1) if deterministic else jax.random.categorical(b, logp)
    control = jnp.where(continuous_mask(logp.shape[-1])[choice][..., None], jnp.tanh(raw), 0.)
    joint, entropy_c, entropy_d = hybrid_logprob(mean, logstd, logp, raw, choice)
    return control, choice.astype(jnp.int32), raw, joint, entropy_c, entropy_d


def hybrid_logprob(mean, logstd, logp, raw, choice):
    normal = -.5*((raw-mean)*jnp.exp(-logstd))**2-logstd-.5*jnp.log(2*jnp.pi)
    jacobian = 2*(jnp.log(2.)-raw-jax.nn.softplus(-2*raw))
    continuous = (normal-jacobian).sum(axis=-1)
    discrete = jnp.take_along_axis(logp, choice[..., None], axis=-1)[..., 0]
    return discrete+jnp.where(continuous_mask(logp.shape[-1])[choice], continuous, 0.), -continuous, -(jnp.exp(logp)*logp).sum(axis=-1)


def hybrid_entropy(mean, logstd, logp, noise):
    """Reparameterized entropy of the current policy, with a fixed noise draw.

    Old rollout actions are valid for PPO likelihood ratios, but treating their
    negative log likelihood as differentiable entropy gives a zero expected
    entropy gradient at the behavior policy. Use a current-policy sample here.
    """
    sample = mean+jnp.exp(logstd)*noise
    log_jacobian = 2.*(jnp.log(2.)-sample-jax.nn.softplus(-2.*sample))
    continuous = (logstd+.5*jnp.log(2*jnp.pi*jnp.e)+log_jacobian).sum(axis=-1)
    discrete = -(jnp.exp(logp)*logp).sum(axis=-1)
    probability = (jnp.exp(logp)*continuous_mask(logp.shape[-1])).sum(axis=-1)
    return probability*continuous, discrete


def policy_step(model, kind, features, mask, hidden, key, deterministic=False, exploration_epsilon=0.):
    if kind.startswith("ppo"):
        mean, std, logp, value, hidden = ppo_step(model, features, mask, hidden,exploration_epsilon)
    elif kind == "dsac_gru":
        hidden = jnp.concatenate((hidden[:, 1:], features[:, None]), axis=1)
        encoded = encode_history(model["encoder"], hidden)
        expected=model['actor'][0][0].shape[0]-mask.shape[-1]
        if expected==encoded.shape[-1]+features.shape[-1]:
            encoded=jnp.concatenate((features,encoded),axis=-1)
        elif expected!=encoded.shape[-1]:
            raise ValueError('Recurrent actor input does not match its observation/memory schema')
        mean, std, logp = actor_outputs(model["actor"], encoded, mask)
        value = jnp.zeros(features.shape[:-1])
    else:
        mean, std, logp = actor_outputs(model, features, mask)
        value = jnp.zeros(features.shape[:-1])
    if not kind.startswith('ppo'):
        logp=exploration_logprob(logp,mask,exploration_epsilon)
    if key.ndim == 2:
        control, choice, raw, lp, hc, hd = jax.vmap(sample_hybrid, in_axes=(0, 0, 0, 0, None))(
            key, mean, std, logp, deterministic)
    else:
        control, choice, raw, lp, hc, hd = sample_hybrid(key, mean, std, logp, deterministic)
    return env.Action(control, choice), hidden, value, raw, lp, hc, hd


def save_policy(path, model, metadata):
    arrays = {}

    def encode(value):
        if isinstance(value, dict):
            return {"dict": {k: encode(v) for k, v in value.items()}}
        if isinstance(value, tuple):
            return {"tuple": [encode(v) for v in value]}
        name = f"array_{len(arrays)}"
        arrays[name] = np.asarray(value)
        return {"array": name}

    structure = encode(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, metadata=json.dumps(metadata), structure=json.dumps(structure), **arrays)
    temporary.replace(path)


def load_policy(path):
    with np.load(path, allow_pickle=False) as data:
        def decode(node):
            if "dict" in node:
                return {k: decode(v) for k, v in node["dict"].items()}
            if "tuple" in node:
                return tuple(decode(v) for v in node["tuple"])
            return jnp.asarray(data[node["array"]])
        return decode(json.loads(str(data["structure"]))), json.loads(str(data["metadata"]))
