"""Registered headless experiments with full trajectory logs and resumable state."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import gzip
import shutil
from time import perf_counter
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms import hybrid_dsac as dsac
from algorithms.recurrent_dsac import RecurrentDSAC
from env import resource_taxi as env
from evaluate import evaluate
import models
import protocol
from telemetry import frame, write_chunk


ENVS, ROLLOUT, CAPACITY = 64, 64, 65536
OPT = optax.chain(optax.clip_by_global_norm(.5), optax.adam(2.5e-4))


@lru_cache(maxsize=64)
def compiled_collector(config, kind, scenario_count):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    design = {**protocol.load(), "training_scenario_count": scenario_count}
    return make_collector(params, kind, design)


@lru_cache(maxsize=32)
def compiled_ppo(config):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    return make_ppo_update(params, OPT)


@lru_cache(maxsize=16)
def dsac_agent(kind, config):
    return RecurrentDSAC(config) if kind == "dsac_gru" else dsac.HybridDSAC(config)


def reset_batch(params, design, counters):
    first, last = protocol.check_splits(design)
    ids = first+(counters % (last-first+1))
    return jax.vmap(env.ResourceTaxiEnv.reset, in_axes=(0, None))(ids, params)


def make_collector(params, kind, design):
    @eqx.filter_jit
    def collect(actor, states, hidden, episode_counters, rng, intrinsic):
        def tick(carry, index):
            states, hidden, counters, rng = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            obs = jax.vmap(env.ResourceTaxiEnv.get_observation, in_axes=(0, None))(states, params)
            initial_hidden = hidden if kind.startswith("ppo") else jnp.zeros((ENVS, 0))
            actions, hidden, value, raw, logp, entropy_c, entropy_d = models.policy_step(
                actor, kind, obs.features, obs.action_mask, hidden, action_key)
            next_obs, new, reward, done, info = jax.vmap(env.ResourceTaxiEnv.step, in_axes=(0, 0, 0, None))(
                jax.random.split(step_key, ENVS), states, actions, params)
            telemetry = frame(states, new, actions, reward, info)
            telemetry["entropy_c"] = entropy_c
            telemetry["entropy_d"] = entropy_d
            features, next_features = obs.features, next_obs.features
            bonus = jnp.zeros(ENVS)
            if intrinsic is not None:
                x = jnp.clip(obs.features[:, :16], -1., 1.)
                target = dsac._mlp(intrinsic["target"], x)
                guess = dsac._mlp(intrinsic["predictor"], x)
                bonus = .05*(1-params.config.gamma)*jnp.clip(
                    ((target-guess)**2).mean(axis=-1)/.01, 0., 1.)*intrinsic["weight"]
            transition = dsac.Transition(features, obs.action_mask, actions.control, actions.choice,
                                         reward+bonus, done, next_features, next_obs.action_mask)
            telemetry["intrinsic_reward"] = bonus
            counters = counters+ENVS*done
            _, starts = reset_batch(params, design, counters)
            states = jax.tree.map(lambda a, b: jnp.where(done.reshape(
                (ENVS,)+(1,)*(a.ndim-1)), b, a), new, starts)
            hidden = jnp.where(done.reshape((ENVS,)+(1,)*(hidden.ndim-1)), 0., hidden)
            return (states, hidden, counters, rng), (
                transition, value, raw, logp, done, telemetry, bonus,
                initial_hidden, entropy_c, entropy_d)

        return jax.lax.scan(tick, (states, hidden, episode_counters, rng), jnp.arange(ROLLOUT))
    return collect


def make_ppo_update(params, opt, observation_fn=env.ResourceTaxiEnv.get_observation):
    @eqx.filter_jit
    def update(model, opt_state, trace, end_states, end_hidden, rng):
        transitions, values, raw, old_logp, done, _, bonus, old_hidden, _, _ = trace
        obs = jax.vmap(observation_fn, in_axes=(0, None))(end_states, params)
        _, _, _, next_value, _ = models.ppo_step(model, obs.features, obs.action_mask, end_hidden)
        following = jnp.concatenate((values[1:], next_value[None]))
        resets = jnp.concatenate((jnp.zeros((1, ENVS), dtype=jnp.bool_), done[:-1]))

        def gae(carry, row):
            reward, terminal, value, next_value = row
            continuation = params.config.gamma*~terminal
            carry = reward+continuation*next_value-value+continuation*.95*carry
            return carry, carry
        _, advantage = jax.lax.scan(gae, jnp.zeros(ENVS),
            (transitions.reward, done, values, following), reverse=True)
        returns = advantage+values
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-6)

        def minibatch(carry, indices):
            model, opt_state = carry
            entropy_noise = jax.random.normal(jax.random.fold_in(rng, indices[0]+731), raw[:, indices].shape)

            def loss(candidate):
                def forward(hidden, row):
                    features, masks, reset = row
                    hidden = jnp.where(reset[:, None], 0., hidden)
                    mean, std, logp, value, hidden = models.ppo_step(candidate, features, masks, hidden)
                    return hidden, (mean, std, logp, value)
                _, (mean, std, logp, prediction) = jax.lax.scan(
                    forward, old_hidden[0, indices],
                    (transitions.features[:, indices], transitions.mask[:, indices], resets[:, indices]))
                lp, _, _ = models.hybrid_logprob(
                    mean, std, logp, raw[:, indices], transitions.choice[:, indices])
                entropy_c, entropy_d = models.hybrid_entropy(mean, std, logp, entropy_noise)
                ratio = jnp.exp(lp-old_logp[:, indices])
                objective = jnp.minimum(ratio*advantage[:, indices],
                                         jnp.clip(ratio, .8, 1.2)*advantage[:, indices])
                value_loss = jnp.mean((prediction-returns[:, indices])**2)
                entropy = .003*entropy_c.mean()+.01*entropy_d.mean()
                return -objective.mean()+.5*value_loss-entropy

            value, grad = jax.value_and_grad(loss)(model)
            changes, opt_state = opt.update(grad, opt_state)
            return (optax.apply_updates(model, changes), opt_state), value

        groups = jax.vmap(lambda k: jax.random.permutation(k, ENVS))(
            jax.random.split(rng, 3)).reshape((-1, 8))
        (model, opt_state), losses = jax.lax.scan(minibatch, (model, opt_state), groups)
        return model, opt_state, losses.mean()
    return update


def with_history(replay, batch, indices, lanes=ENVS, history=models.HISTORY):
    """Gather same-lane sequences and zero everything before the latest reset."""
    backward = (indices[:, None]-lanes*jnp.arange(history)[None]) % replay.features.shape[0]
    values = replay.features[backward]
    past_terminal = jnp.cumsum(replay.done[backward[:, 1:]], axis=1) > 0
    valid = jnp.concatenate((jnp.ones((indices.size, 1), dtype=jnp.bool_), ~past_terminal), axis=1)
    sequence = jnp.where(valid[..., None], values, 0.)[:, ::-1]
    following = jnp.concatenate((sequence[:, 1:], batch.next_features[:, None]), axis=1)
    return batch._replace(features=sequence, next_features=following)


@lru_cache(maxsize=16)
def make_dsac_update(agent, rollout=ROLLOUT):
    @eqx.filter_jit
    def update(learner, replay, cursor, size, transition, rng):
        transition = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), transition)
        indices = (cursor+jnp.arange(ENVS*rollout)) % CAPACITY
        replay = jax.tree.map(lambda a, b: a.at[indices].set(b), replay, transition)
        size = jnp.minimum(size+ENVS*rollout, CAPACITY)
        cursor = (cursor+ENVS*rollout) % CAPACITY

        def one(learner, key):
            sample_key, update_key = jax.random.split(key)
            if isinstance(agent, RecurrentDSAC):
                available = jnp.maximum(size-ENVS*(models.HISTORY-1), 1)
                age = jax.random.randint(sample_key, (agent.config.minibatch_size,), 0, available)
                idx = (cursor-1-age) % CAPACITY
            else:
                idx = jax.random.randint(sample_key, (agent.config.minibatch_size,), 0, size)
            batch = jax.tree.map(lambda x: x[idx], replay)
            if isinstance(agent, RecurrentDSAC):
                batch = with_history(replay, batch, idx)
            return agent.update(learner, update_key, batch)
        learner, losses = jax.lax.scan(one, learner, jax.random.split(rng, agent.config.gradient_steps))
        return learner, replay, cursor, size, losses.mean(axis=0)
    return update


def checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as file:
        pickle.dump(jax.device_get(payload), file, protocol=5)
    temporary.replace(path)


def train(row, steps=None, pilot=False, resume=False, stop_after=None):
    design = protocol.load()
    kind = ("dsac_gru" if row["algorithm"] == "hybrid_dsac" and row["architecture"] == "gru"
            else "dsac_mlp" if row["algorithm"] == "hybrid_dsac"
            else "ppo_gru" if row["architecture"] == "gru" else "ppo_mlp")
    final = protocol.run_config(row)
    budget = steps or int(row["physical_transition_budget"])
    if budget % (ENVS*ROLLOUT):
        raise ValueError(f"Training steps must be divisible by {ENVS*ROLLOUT}")
    validation_stride = max(32, budget//(ENVS*ROLLOUT*16))
    output = protocol.ROOT/("runs/pilots" if pilot else "artifacts/training")/row["run_id"]
    if pilot:
        import hashlib
        identity = {k: v for k, v in row.items() if k != "status"}
        digest = hashlib.sha256(json.dumps({"sources": protocol.source_hashes(), "run": identity,
            "budget": budget, "design": design}, sort_keys=True).encode()).hexdigest()[:10]
        output = output/f"{digest}_{budget}"
    output.mkdir(parents=True, exist_ok=True)
    final_params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), final)
    evaluation_config = replace(final, market="dynamic") if row["training_market"] == "fixed" else final
    evaluation_params = replace(final_params, config=evaluation_config)
    feature_count = env.ResourceTaxiEnv.reset(11, final_params)[0].features.size
    rng = jax.random.PRNGKey(int(row["initialization_seed"]))
    rng, init = jax.random.split(rng)
    opt = OPT
    is_dsac = kind.startswith("dsac")
    agent = None
    if is_dsac:
        cfg = dsac.AgentConfig(gamma=final.gamma, demo_weight=0.)
        agent = dsac_agent(kind, cfg)
        learner = agent.init(init, feature_count)
        actor = learner.actor
        opt_state = None
    else:
        actor = models.init_ppo(init, feature_count, kind == "ppo_gru")
        learner, opt_state = None, opt.init(actor)
    counters = jnp.arange(ENVS)
    hidden = models.initial_memory(kind, ENVS, feature_count)
    total, index, best, history, stage, replay = 0, 0, None, [], -1, None
    restored_stage = -1
    cursor, size = jnp.int32(0), jnp.int32(0)
    intrinsic = None
    intrinsic_opt = None
    if row["exploration"] == "rnd":
        a, b = jax.random.split(jax.random.fold_in(init, 55))
        intrinsic = {"target": dsac._init_mlp(a, (16, 32, 16), .2),
                     "predictor": dsac._init_mlp(b, (16, 32, 16), .2), "weight": jnp.float32(1.)}
        intrinsic_opt = opt.init(intrinsic["predictor"])

    @jax.jit
    def update_intrinsic(intrinsic, state, features, weight):
        def loss(predictor):
            x = jnp.clip(features[..., :16].reshape((-1, 16)), -1., 1.)
            return jnp.mean((dsac._mlp(predictor, x)-dsac._mlp(intrinsic["target"], x))**2)
        grad = jax.grad(loss)(intrinsic["predictor"])
        changes, state = opt.update(grad, state)
        return {**intrinsic, "predictor": optax.apply_updates(intrinsic["predictor"], changes),
                "weight": weight}, state

    algorithm_settings = asdict(agent.config) if is_dsac else {
        "learning_rate": .00025, "gae_lambda": .95, "clip_ratio": .2, "epochs": 3,
        "environment_minibatch": 8, "continuous_entropy": .003, "discrete_entropy": .01}
    metadata = {"kind": kind, "environment": asdict(final), "evaluation_environment": asdict(evaluation_config),
                "algorithm_settings": algorithm_settings,
                "actor_parameters": sum(x.size for x in jax.tree.leaves(actor)),
                "validation_stride_chunks": validation_stride,
                "run": row, "budget": budget,
                "pilot": pilot, "design": design, "source_hashes": protocol.source_hashes()}
    if resume and (output/"resume.pkl.gz").exists():
        with gzip.open(output/"resume.pkl.gz", "rb") as file:
            saved = pickle.load(file)
        if saved["metadata"]["budget"] != budget:
            raise ValueError("Resume budget differs from the saved run")
        if saved["metadata"]["source_hashes"] != metadata["source_hashes"]:
            raise ValueError("Source files changed since checkpoint; create a new registered run")
        for name in ("environment", "design"):
            if saved["metadata"][name] != metadata[name]:
                raise ValueError(f"Resume {name} differs from the saved experiment")
        old_identity = {k: v for k, v in saved["metadata"]["run"].items() if k != "status"}
        new_identity = {k: v for k, v in row.items() if k != "status"}
        if old_identity != new_identity:
            raise ValueError("Resume initialization or registered treatment changed")
        actor, learner, opt_state, states, hidden, counters, rng, replay = (
            saved[name] for name in ("actor", "learner", "opt_state", "states", "hidden", "counters", "rng", "replay"))
        total, index, best, history, cursor, size, stage = (
            saved[name] for name in ("total", "index", "best", "history", "cursor", "size", "stage"))
        intrinsic, intrinsic_opt = saved["intrinsic"], saved["intrinsic_opt"]
        log = output/"training.jsonl"
        if log.exists():
            records = [line for line in log.read_text(encoding="utf-8").splitlines()
                       if json.loads(line)["physical_transitions"] <= total]
            log.write_text(chr(10).join(records)+chr(10), encoding="utf-8")
        restored_stage = stage
        stage = -1  # Preserve restored states only within the same curriculum stage.
        resumed_states = True
    else:
        if (output/"config.json").exists():
            raise FileExistsError("Run already exists; resume it or use another registered run")
        (output/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for name in metadata["source_hashes"]:
            destination = output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protocol.ROOT/name, destination)
        shutil.copy2(protocol.ROOT/"requirements.txt", output/"source/requirements.txt")
        for name in ("protocol.json", "registry.csv"):
            destination = output/"source/experiments"/name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(protocol.ROOT/"experiments"/name, destination)
        models.save_policy(output/"initial.npz", actor, {**metadata, "physical_transitions": 0})
        states, resumed_states = None, False
    if not pilot:
        protocol.set_run_status(row["run_id"], "running")
    start = perf_counter()
    print(f"START {row['run_id']} {jax.devices()} budget={budget}", flush=True)
    while total < budget:
        fraction = total/budget
        current_stage = (7 if row["curriculum"] == "none" or fraction >= .5 else
                         4+min(int((fraction-.2)/.1), 2) if fraction >= .2 else min(int(fraction/.05), 3))
        if stage != current_stage:
            previous_stage = restored_stage if resumed_states else stage
            actual_change = previous_stage >= 0 and previous_stage != current_stage
            if actual_change:
                models.save_policy(output/f"stage_{previous_stage}.npz", actor,
                                   {**metadata, "physical_transitions": total})
                resumed_states = False
            config = protocol.curriculum(final, fraction, row["curriculum"] == "none",
                                         row.get("discount_curriculum") != "constant")
            if resumed_states:
                config = replace(config, loan_capacity=states.loan_balance.shape[-1])
            params = replace(final_params, config=config)
            if not resumed_states:
                _, states = reset_batch(params, design, counters)
                hidden = models.initial_memory(kind, ENVS, feature_count)
            resumed_states = False
            collector = compiled_collector(config, kind, design["training_scenario_count"])
            if is_dsac:
                agent = dsac_agent(kind, replace(cfg, gamma=config.gamma))
            update = make_dsac_update(agent) if is_dsac else compiled_ppo(config)
            # Replay is retained on resume and cleared at actual curriculum changes.
            if actual_change:
                replay, cursor, size = None, jnp.int32(0), jnp.int32(0)
            stage = current_stage
        rng, collect_key, update_key = jax.random.split(rng, 3)
        if intrinsic is not None:
            intrinsic = {**intrinsic, "weight": jnp.float32(max(0, 1.-fraction/.75))}
        if int(np.asarray((states.loan_balance > 0).sum(axis=-1)).max()) > params.config.loan_capacity-ROLLOUT:
            capacity = params.config.loan_capacity*2
            padding = capacity-params.config.loan_capacity
            states = replace(states, **{name: jnp.pad(getattr(states, name), ((0, 0), (0, padding)))
                for name in ("loan_ids", "loan_principal", "loan_balance", "loan_rates", "loan_opened")})
            params = replace(params, config=replace(params.config, loan_capacity=capacity))
            collector = compiled_collector(params.config, kind, design["training_scenario_count"])
        (states, hidden, counters, _), trace = collector(actor, states, hidden, counters, collect_key, intrinsic)
        transition = trace[0]
        if bool(np.any(np.asarray(trace[5]["capacity_needed"]))):
            raise RuntimeError("Loan capacity needs expansion before continuing this collector")
        if is_dsac:
            if replay is None:
                replay = jax.tree.map(lambda x: jnp.zeros((CAPACITY, *x.shape[2:]), x.dtype), transition)
            learner, replay, cursor, size, loss = update(learner, replay, cursor, size, transition, update_key)
            actor = learner.actor
        else:
            actor, opt_state, loss = update(actor, opt_state, trace, states, hidden, update_key)
        if not bool(jnp.all(jnp.isfinite(loss))):
            raise FloatingPointError("Nonfinite training loss")
        if intrinsic is not None:
            features = transition.features
            intrinsic, intrinsic_opt = update_intrinsic(intrinsic, intrinsic_opt, features,
                                                       jnp.float32(max(0, 1.-fraction/.75)))
        total += ENVS*ROLLOUT
        log_bytes = write_chunk(output/"traces"/f"chunk_{index:05}.npz", trace[5],
                                 {"stage": stage, "stage_configuration": asdict(params.config),
                                  "physical_transition_start": total-ENVS*ROLLOUT})
        index += 1
        loss_value = float(loss[0]) if is_dsac else float(loss)
        record = {"physical_transitions": total, "stage": stage, "loss": loss_value,
                  "loss_type": "quantile_critic" if is_dsac else "ppo_objective",
                  "elapsed_seconds": perf_counter()-start, "log_bytes": log_bytes,
                  "scenario_min": int(np.asarray(trace[5]["scenario_id"]).min()),
                  "scenario_max": int(np.asarray(trace[5]["scenario_id"]).max())}
        if is_dsac:
            names = ("critic_loss", "actor_loss", "continuous_temperature", "discrete_temperature",
                     "continuous_sample_entropy", "discrete_entropy", "encoder_loss")
            record.update({name: float(value) for name, value in zip(names, np.asarray(loss))})
        with (output/"training.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record)+chr(10))
        if index == 1 or index % validation_stride == 0 or total == budget or (stop_after and total >= stop_after):
            validation, _, _ = evaluate(actor, kind, evaluation_params, range(1000001, 1000065),
                                        output/"validation_traces"/f"step_{total:09}")
            history.append({"physical_transitions": total, **validation})
            score = (validation["mean_external_return"], validation["success_rate"], -(validation["starvation"]+validation["collision"]),
                     -validation["mean_remaining_debt"], -validation["mean_endpoint_adjusted_completion_tick"])
            if best is None or score > best:
                best = score
                models.save_policy(output/"selected.npz", actor,
                                   {**metadata, "physical_transitions": total, "validation": validation})
                checkpoint(output/"selected_state.pkl.gz", {"metadata": metadata, "total": total,
                    "actor": actor, "learner": learner, "training_configuration": asdict(params.config)})
            models.save_policy(output/"last.npz", actor, {**metadata, "physical_transitions": total})
            checkpoint(output/f"learner_{total:09}.pkl.gz", {"metadata": metadata, "total": total,
                "actor": actor, "learner": learner, "opt_state": opt_state,
                "training_configuration": asdict(params.config)})
            (output/"validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint(output/"resume.pkl.gz", {
                "metadata": metadata, "actor": actor, "learner": learner, "opt_state": opt_state,
                "states": states, "hidden": hidden, "counters": counters, "rng": rng, "replay": replay,
                "total": total, "index": index, "best": best, "history": history, "cursor": cursor,
                "size": size, "stage": stage, "intrinsic": intrinsic, "intrinsic_opt": intrinsic_opt})
            print(f"{row['run_id']} {total}/{budget} validation={validation['ownership']}/64 "
                  f"death={validation['starvation']} crash={validation['collision']} debt={validation['mean_remaining_debt']:.1f}",
                  flush=True)
        if stop_after and total >= stop_after and total < budget:
            return output
    if not pilot:
        protocol.set_run_status(row["run_id"], "trained")
    models.save_policy(output/"final.npz", actor, {**metadata, "physical_transitions": total})
    (output/"complete.json").write_text(json.dumps({"physical_transitions": total,
        "elapsed_seconds": perf_counter()-start, "best_validation": best}), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--group", choices=("main", "extension", "all"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Save a resumable checkpoint at this transition count")
    parser.add_argument("--init-seed", type=int, help="Override initialization for an explicitly labeled pilot only")
    args = parser.parse_args()
    if args.init_seed is not None and not args.pilot:
        parser.error("Initialization overrides are allowed only for --pilot")
    rows = protocol.registry()
    chosen = [r for r in rows if r["run_id"] == args.run_id] if args.run_id else [
        r for r in rows if args.group == "all" or r["group"] == args.group]
    if not chosen:
        parser.error("Select an existing --run-id or a --group")
    for row in chosen:
        if args.init_seed is not None:
            row = {**row, "initialization_seed": str(args.init_seed)}
        target = protocol.ROOT/"artifacts/training"/row["run_id"]/"complete.json"
        if target.exists() and not args.pilot:
            continue
        train(row, args.steps, args.pilot, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
