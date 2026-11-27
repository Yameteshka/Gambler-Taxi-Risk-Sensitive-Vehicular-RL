"""Train a destination policy while leaving driving to the shared controller."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms.hybrid_dsac import _init_mlp, _mlp
import debt_strategy as strategy
from env import gamblers_taxi as taxi
from train import keys, summary


def config(urgency):
    return replace(taxi.TaxiConfig(), debt_time_weight=urgency, enable_impulse_override=False)


def initialize(key, params):
    _, state = taxi.GamblersTaxiEnv.reset(key, params)
    return _init_mlp(key, (strategy.features(state, params).size, 96, 96, len(strategy.NAMES)+1), 0.22)


def distribution(model, obs, allowed):
    output = _mlp(model, obs)
    logits = jnp.where(allowed, output[..., :-1], -1e6)
    return jax.nn.log_softmax(logits), output[..., -1]


def pick(model, state, params, key, mode):
    if mode == "work":
        return strategy.work_plan(state, params)
    if mode == "mixed":
        return strategy.mixed_plan(state, params)
    scores, _ = distribution(model, strategy.features(state, params), strategy.legal(state, params))
    if mode == "sample":
        return jax.random.categorical(key, scores).astype(jnp.int32)
    return jnp.argmax(scores).astype(jnp.int32)


def evaluate(model, params, seeds, mode="greedy", trace=False, record=None):
    def one(key):
        env = taxi.GamblersTaxiEnv
        _, state = env.reset(jax.random.fold_in(key, 1), params)

        def step(current, index):
            choice = pick(model, current, params, jax.random.fold_in(key, 2*index+2), mode)
            step_key = jax.random.fold_in(key, 2*index+3)
            if record is not None:
                new, _, rows = strategy.advance(step_key, current, choice, params, record)
                return new, (choice, rows)
            new, reward = strategy.advance(step_key, current, choice, params)
            return new, (current, choice, reward) if trace else None

        count = (params.config.horizon+strategy.STEPS-1)//strategy.STEPS
        return jax.lax.scan(step, state, jnp.arange(count))

    return jax.vmap(one)(seeds)


def demonstrations(params, expert, count=48):
    states, picked, _ = evaluate(None, params, keys(45000, count), expert, trace=True)[1]
    collect = jax.vmap(jax.vmap(lambda s: (strategy.features(s, params), strategy.legal(s, params))))
    observations, masks = collect(states)
    valid = (states.outcome == taxi.ACTIVE) & jnp.take_along_axis(masks, picked[..., None], -1)[..., 0]
    return observations[valid], masks[valid], picked[valid]


def save(path, model, metadata):
    payload = {f"{name}{i}": np.asarray(array) for i, layer in enumerate(model)
               for name, array in zip(("w", "b"), layer)}
    with path.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata), **payload)


def load(path):
    with np.load(path, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata"]))
        model = tuple((jnp.asarray(handle[f"w{i}"]), jnp.asarray(handle[f"b{i}"])) for i in range(3))
    return model, metadata


def train(urgency, seed, iterations, output, demos, params, expert):
    env = taxi.GamblersTaxiEnv
    random = jax.random.key(seed)
    model = initialize(random, params)
    opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(2.5e-4))
    opt_state = opt.init(model)
    demo_obs, demo_masks, demo_actions = demos
    count, rollout = 64, 24
    gamma = params.config.gamma**strategy.STEPS

    def update_with(loss, model, opt_state):
        value, grad = jax.value_and_grad(loss)(model)
        changes, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(model, changes), opt_state, value

    @jax.jit
    def pretrain(model, opt_state, rng):
        def batch(carry, key):
            model, opt_state = carry
            idx = jax.random.randint(key, (128,), 0, len(demo_actions))
            def loss(candidate):
                logp, _ = distribution(candidate, demo_obs[idx], demo_masks[idx])
                return -jnp.take_along_axis(logp, demo_actions[idx, None], -1).mean()
            model, opt_state, loss_value = update_with(loss, model, opt_state)
            return (model, opt_state), loss_value
        return jax.lax.scan(batch, (model, opt_state), jax.random.split(rng, 1200))[0]

    @jax.jit
    def iteration(model, opt_state, states, rng):
        def tick(states, key):
            act, sim, reset = jax.random.split(key, 3)
            features = jax.vmap(strategy.features, in_axes=(0, None))(states, params)
            masks = jax.vmap(strategy.legal, in_axes=(0, None))(states, params)
            logp, value = distribution(model, features, masks)
            action = jax.random.categorical(act, logp).astype(jnp.int32)
            selected = jnp.take_along_axis(logp, action[:, None], -1)[:, 0]
            new, reward = jax.vmap(strategy.advance, in_axes=(0, 0, 0, None))(
                jax.random.split(sim, count), states, action, params
            )
            done = new.outcome != taxi.ACTIVE
            start = jax.vmap(lambda k: env.reset(k, params)[1])(jax.random.split(reset, count))
            next_state = jax.tree.map(lambda a, b: jnp.where(
                done.reshape((count,) + (1,) * (a.ndim-1)), b, a
            ), new, start)
            return next_state, (features, masks, action, selected, value, reward, done)

        k_rollout, k_updates = jax.random.split(rng)
        states, (obs, masks, actions, old_logp, value, reward, done) = jax.lax.scan(
            tick, states, jax.random.split(k_rollout, rollout)
        )
        next_features = jax.vmap(strategy.features, in_axes=(0, None))(states, params)
        _, next_value = distribution(model, next_features, jax.vmap(strategy.legal, in_axes=(0, None))(states, params))
        following = jnp.concatenate((value[1:], next_value[None]))

        def gae(carry, x):
            r, terminal, v, vn = x
            continuation = gamma * ~terminal
            carry = r + continuation*vn - v + continuation*0.95*carry
            return carry, carry

        _, advantage = jax.lax.scan(gae, jnp.zeros_like(next_value), (reward, done, value, following), reverse=True)
        target = (advantage + value).ravel()
        advantage = ((advantage-advantage.mean())/(advantage.std()+1e-6)).ravel()
        obs, masks = obs.reshape((-1, obs.shape[-1])), masks.reshape((-1, masks.shape[-1]))
        actions, old_logp = actions.ravel(), old_logp.ravel()

        def batch(carry, key):
            model, opt_state = carry
            perm, chosen = key
            def loss(candidate):
                logp, critic = distribution(candidate, obs[perm], masks[perm])
                lp = jnp.take_along_axis(logp, actions[perm, None], -1)[:, 0]
                ratio = jnp.exp(lp-old_logp[perm])
                policy = -jnp.mean(jnp.minimum(ratio*advantage[perm], jnp.clip(ratio, .8, 1.2)*advantage[perm]))
                entropy = -jnp.sum(jnp.exp(logp)*logp, axis=-1).mean()
                imitation, _ = distribution(candidate, demo_obs[chosen], demo_masks[chosen])
                imitation = -jnp.take_along_axis(imitation, demo_actions[chosen, None], -1).mean()
                return policy + .5*jnp.mean((critic-target[perm])**2) - .015*entropy + .04*imitation
            model, opt_state, loss_value = update_with(loss, model, opt_state)
            return (model, opt_state), loss_value

        perm_key, demo_key = jax.random.split(k_updates)
        permutations = jax.vmap(lambda k: jax.random.permutation(k, count*rollout))(
            jax.random.split(perm_key, 3)
        ).reshape((-1, 192))
        demos = jax.random.randint(demo_key, (len(permutations), 64), 0, len(demo_actions))
        (model, opt_state), losses = jax.lax.scan(batch, (model, opt_state), (permutations, demos))
        return model, opt_state, states, losses.mean()

    evaluate_model = jax.jit(lambda model, seed_keys: evaluate(model, params, seed_keys)[0])
    model, opt_state = pretrain(model, opt_state, jax.random.fold_in(random, 7))
    metadata = {"seed": seed, "urgency": urgency, "budget": iterations, "expert": expert}
    save(output/"imitation.npz", model, {**metadata, "iteration": 0})
    states = jax.vmap(lambda k: env.reset(k, params)[1])(keys(110000+seed*count, count))
    best = None
    history = []
    for index in range(iterations+1):
        if index:
            model, opt_state, states, loss = iteration(model, opt_state, states, jax.random.fold_in(random, index+100))
        if index == 0 or index % 10 == 0 or index == iterations:
            metrics = summary(evaluate_model(model, keys(81000, 48)), params.config)
            if index and not np.isfinite(float(loss)):
                raise RuntimeError("Nonfinite PPO loss")
            history.append({"iteration": index, **metrics})
            score = (metrics["success"], metrics["debt_cleared"], -metrics["crash"], -metrics["mean_final_debt"])
            if best is None or score > best:
                best = score
                save(output/"selected.npz", model, {**metadata, "iteration": index, "validation": metrics})
            print(f"{urgency}/{seed} iter={index} win={metrics['success']:.3f} "
                  f"pay={metrics['debt_cleared']:.3f} crash={metrics['crash']:.3f}", flush=True)
    save(output/"last.npz", model, {**metadata, "iteration": iterations})
    (output/"validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 29, 47))
    parser.add_argument("--urgency", nargs="+", type=float, default=(0.25, 1.0))
    parser.add_argument("--expert", choices=("work", "mixed"), default="work")
    args = parser.parse_args()
    print(jax.devices(), flush=True)
    for urgency in args.urgency:
        params, _ = taxi.GamblersTaxiEnv.init_params(jax.random.key(0), config(urgency))
        demos = demonstrations(params, args.expert)
        for seed in args.seeds:
            prefix = "" if args.expert == "work" else "mixed_demos_"
            output = args.output/f"{prefix}urgency_{urgency:g}_seed_{seed}"
            output.mkdir(parents=True, exist_ok=True)
            train(urgency, seed, args.iterations, output, demos, params, args.expert)


if __name__ == "__main__":
    main()
