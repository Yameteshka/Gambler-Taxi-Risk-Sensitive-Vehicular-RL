"""Headless curriculum training, checkpoint selection and held-out evaluation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from algorithms import hybrid_dsac as dsac
from env import gamblers_taxi as taxi
import scenarios


class Replay(NamedTuple):
    data: dsac.Transition
    cursor: jax.Array
    size: jax.Array


class Collector(NamedTuple):
    key: jax.Array
    obs: taxi.TaxiObservation
    states: taxi.TaxiState
    replay: Replay
    learner: dsac.Learner


def keys(start, count):
    return jax.vmap(jax.random.key)(jnp.arange(start, start + count))


def load_actor(checkpoint):
    with np.load(checkpoint, allow_pickle=False) as data:
        meta = json.loads(str(data["config"]))
        actor = tuple(
            (jnp.asarray(data[f"w{i}"]), jnp.asarray(data[f"b{i}"])) for i in range(meta["layers"])
        )
    # Zero-pad features introduced after an older snapshot, before its action mask.
    first, bias = actor[0]
    added = taxi.DEPENDENCE_FEATURES * ("enable_dependence" not in meta["environment"])
    added += taxi.ANXIETY_FEATURES * ("anxiety_steering" not in meta["environment"])
    added += taxi.ECONOMY_FEATURES * ("operating_cost" not in meta["environment"])
    added += taxi.DEBT_CAMPAIGN_FEATURES * ("starting_debt" not in meta["environment"])
    if added:
        split = first.shape[0] - taxi.NUM_ACTIONS
        first = jnp.concatenate((first[:split], jnp.zeros((added, first.shape[1])),
                                 first[split:]), axis=0)
        actor = ((first, bias), *actor[1:])
    environment = {
        "blackjack_win_commission": 0., "operating_cost": 0., "casino_open": 0,
        "casino_close": meta["environment"]["day"], "casino_round_interval": 0,
        "timeout_shortfall_penalty": 0., "enable_dependence": False, "starting_debt": 0.,
        "debt_time_weight": 0., **meta["environment"],
    }
    return actor, taxi.TaxiConfig(**environment), meta


def save_actor(path, actor, params, agent, stage, metrics, seed):
    meta = {"environment": asdict(params.config), "agent": asdict(agent), "stage": stage,
            "metrics": metrics, "layers": len(actor), "seed": seed}
    arrays = {name: np.asarray(array) for i, layer in enumerate(actor)
              for name, array in zip((f"w{i}", f"b{i}"), layer)}
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as file:
        np.savez_compressed(file, config=json.dumps(meta), **arrays)
    tmp.replace(path)


def summary(states, config):
    states = jax.device_get(states)
    capital = np.asarray(taxi._balance(states) + config.buyout*(states.outcome == taxi.SUCCESS))
    tail = np.sort(capital)[:max(1, int(np.ceil(0.2*capital.size)))]
    count = {name: float(np.asarray(states.counters)[..., index].mean())
             for index, name in enumerate(taxi.COUNTER_NAMES)}
    success = states.outcome == taxi.SUCCESS
    paid = np.asarray(states.debt_free_at) >= 0
    return {
        "episodes": int(capital.size),
        "success": float(success.mean()),
        "success_after_repayment": int((success & (states.counters[..., taxi.REPAYMENTS] > 0)).sum()),
        "debt_cleared": float(paid.mean()),
        "mean_debt_free_tick": float(np.asarray(states.debt_free_at)[paid].mean()) if paid.any() else None,
        "mean_final_debt": float(states.debt.mean()),
        "crash": float((states.outcome == taxi.CRASH).mean()),
        "timeout": float((states.outcome == taxi.TIMEOUT).mean()),
        "mean_capital": float(capital.mean()),
        "tail20_capital": float(tail.mean()),
        "mean_finish_time": float(states.time.mean()),
        "mean_habit": float(states.habit.mean()),
        "mean_anxiety": float(states.anxiety.mean()),
        "repeat_gambling": float((states.counters[..., taxi.SPINS] + states.counters[..., taxi.HANDS] >= 2).mean()),
        **count,
    }


def evaluate_cohorts(actor, params, seeds, strategy=-1):
    evaluate = eqx.filter_jit(evaluate_policy)
    states = evaluate(actor, params, seeds, strategy=strategy)

    @eqx.filter_jit
    def starts(keys, params):
        reset = lambda key: taxi.GamblersTaxiEnv.reset(jax.random.split(key)[1], params)[1].time
        return jax.vmap(reset)(keys)

    late = np.asarray(starts(seeds, params)) >= 228
    groups = {}
    for name, mask in (("early", ~late), ("late", late)):
        if np.any(mask):
            idx = np.flatnonzero(mask)
            groups[name] = summary(jax.tree.map(lambda x: x[idx], states), params.config)
    return {"all": summary(states, params.config), **groups}


def evaluate_policy(actor, params, seeds, strategy=-1, deterministic=True):
    env = taxi.GamblersTaxiEnv

    def reset(seed):
        key, reset_key = jax.random.split(seed)
        return key, *env.reset(reset_key, params)

    def one(key, obs, state):
        key, policy_key, env_key = jax.random.split(key, 3)
        if strategy == 5:
            action = scenarios.mixed_expert(state, obs, params)
        elif strategy >= 0:
            action = scenarios.scripted_policy(state, obs, params, strategy)
        else:
            action = taxi.TaxiAction(*dsac.sample(actor, policy_key, obs.features, obs.action_mask, deterministic))
        new_obs, new, _, _, _ = env.step(env_key, state, action, params)
        keep = lambda new, old: jnp.where(state.outcome == taxi.ACTIVE, new, old)
        return key, jax.tree.map(keep, new_obs, obs), jax.tree.map(keep, new, state)

    def body(carry, _):
        return jax.vmap(one)(*carry), None

    (_, _, states), _ = jax.lax.scan(body, jax.vmap(reset)(seeds), None, length=params.config.horizon)
    return states


def make_demonstrations(params, seed, consistent_expert=False, settlement_demos=0):
    scripted_trace = eqx.filter_jit(scenarios.scripted_trace)
    traces = [
        scripted_trace(keys(seed + 1000*i, count), params, jnp.int32(i))[1]
        for i, count in enumerate((40, 20, 16, 16, 16))
    ]
    data, states, reset_pool = [], [], []
    for transitions, history, active in traces:
        active = np.asarray(active)
        rows = active.nonzero()
        data.append(jax.tree.map(lambda x: x[rows], transitions))
        states.append(jax.tree.map(lambda x: x[rows], history))
        playable = active & (np.asarray(history.activity) == taxi.DRIVE)
        # Near-subgoal and ready-to-buy starts expose the terminal reward early.
        centers = np.asarray(params.zone_centers)
        dist = np.linalg.norm(np.asarray(history.agent_pos)[..., None, :] - centers, axis=-1)
        near = dist.min(axis=-1) <= 4.5
        rows = (playable & near & ((np.arange(active.shape[0])[:, None] % 7) == 0)).nonzero()
        reset_pool.append(jax.tree.map(lambda x: x[rows], history))
    concatenate = lambda groups: jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *groups)
    demos = concatenate(data)
    history = concatenate(states)
    start_pool = concatenate(reset_pool)
    # Imitate reliable driving, while giving the replay buffer a range of financial outcomes.
    buyable = np.asarray(taxi._balance(history)) >= params.config.buyout
    free_driving = np.asarray(demos.choice) == taxi.DRIVE
    work_prefix = np.arange(demos.choice.size) < data[0].choice.size
    mixed_driving = (~work_prefix & free_driving & (np.asarray(history.time) >= 228)
                     & (np.asarray(history.counters)[:, taxi.PAID] == 0))
    if params.config.late_start_probability:
        mixed_driving &= np.asarray(history.counters)[:, taxi.FIRED] == 0
    else:
        mixed_driving &= False
    clean = np.concatenate((np.arange(data[0].choice.size),
                            np.flatnonzero(~work_prefix & buyable & free_driving),
                            np.flatnonzero(mixed_driving & ~buyable),
                            np.flatnonzero(~work_prefix & ~free_driving)))
    imitation = jax.tree.map(lambda x: x[clean], demos)
    if consistent_expert:
        transitions, history, active = scripted_trace(keys(seed + 9500, 96), params, jnp.int32(5))[1]
        positions = (np.asarray(active) & (np.asarray(history.activity) == taxi.DRIVE)).nonzero()
        all_valid = np.asarray(active).nonzero()
        # Give consistent supervision on both routes. Extra free-driving rows reduce
        # the proportion of repeated, deterministic waiting transitions.
        imitation = jax.tree.map(lambda x: jnp.concatenate((x[all_valid], x[positions])), transitions)
    if settlement_demos:
        finals, (transitions, history, active) = scripted_trace(
            keys(seed + 12000, settlement_demos), params, jnp.int32(4)
        )
        settled = np.asarray((finals.outcome == taxi.SUCCESS) & (finals.counters[:, taxi.REPAYMENTS] > 0))
        solvent = np.asarray(taxi._balance(history) >= params.config.buyout)
        rows = (np.asarray(active) & settled[None, :] & solvent).nonzero()
        if not rows[0].size:
            raise RuntimeError("No successful bank settlement in the scenario demonstrations")
        examples = jax.tree.map(lambda x: x[rows], transitions)
        demos = concatenate([demos, examples])
        repetitions = max(1, imitation.choice.size // (3 * examples.choice.size))
        imitation = concatenate([imitation, *([examples] * repetitions)])
        start_pool = concatenate([start_pool, jax.tree.map(lambda x: x[rows], history)])
    return demos, imitation, start_pool


def _append(replay, block):
    count = block.choice.shape[0]
    capacity = replay.data.choice.size
    idx = (replay.cursor + jnp.arange(count)) % capacity
    data = jax.tree.map(lambda buf, val: buf.at[idx].set(val), replay.data, block)
    return Replay(data, (replay.cursor + count) % capacity, jnp.minimum(replay.size + count, capacity))


def _draw(data, key, size, count):
    idx = jax.random.randint(key, (count,), 0, size)
    return jax.tree.map(lambda x: x[idx], data)


def make_collector(agent, params, demos, imit, pool, count, rollout_steps, batch, updates):
    env = taxi.GamblersTaxiEnv

    def reset_one(key, curriculum):
        key_reset, key_sample, key_mix = jax.random.split(key, 3)
        _, init = env.reset(key_reset, params)
        idx = jax.random.randint(key_sample, (), 0, pool.time.size)
        near = jax.tree.map(lambda x: x[idx], pool)
        use_near = jax.random.uniform(key_mix) < curriculum
        state = jax.tree.map(lambda a, b: jnp.where(use_near, a, b), near, init)
        return env.get_observation(state, params), state

    reset = jax.vmap(reset_one, in_axes=(0, None))

    def collect(carry, curriculum):
        def tick(carry, _):
            key, obs, states, actor = carry
            key, key_action, key_env, key_reset = jax.random.split(key, 4)
            control, choice = dsac.sample(actor, key_action, obs.features, obs.action_mask)
            next_obs, new, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(key_env, count), states, taxi.TaxiAction(control, choice), params
            )
            block = dsac.Transition(obs.features, obs.action_mask, control, choice, reward, done,
                                    next_obs.features, next_obs.action_mask)
            reset_obs, reset_states = reset(jax.random.split(key_reset, count), curriculum)
            def select(new, reset_value):
                axes = (count,) + (1,) * (new.ndim - 1)
                return jnp.where(done.reshape(axes), reset_value, new)
            return (key, jax.tree.map(select, next_obs, reset_obs),
                    jax.tree.map(select, new, reset_states), actor), (block, new.outcome)

        (key, obs, states, _), (transitions, outcomes) = jax.lax.scan(
            tick, (carry.key, carry.obs, carry.states, carry.learner.actor), None, length=rollout_steps
        )
        block = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), transitions)
        return Collector(key, obs, states, _append(carry.replay, block), carry.learner), outcomes

    def iteration(carry, curriculum):
        collected, outcomes = collect(carry, curriculum)

        def learn(carry, _):
            key, learner = carry
            key, k_sample, k_demo, k_imit, k_update = jax.random.split(key, 5)
            regular = _draw(collected.replay.data, k_sample, collected.replay.size, batch*3//4)
            expert = _draw(demos, k_demo, demos.choice.size, batch-batch*3//4)
            mixed = jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), regular, expert)
            model_demo = _draw(imit, k_imit, imit.choice.size, batch//2)
            learner, metrics = agent.update(learner, k_update, mixed, model_demo)
            return (key, learner), metrics

        (key, learner), metrics = jax.lax.scan(
            learn, (collected.key, collected.learner), None, length=updates
        )
        return collected._replace(key=key, learner=learner), (metrics.mean(axis=0), jnp.bincount(
            outcomes.ravel(), length=len(taxi.OUTCOME_NAMES)
        ))

    return jax.jit(iteration, donate_argnums=(0,)), jax.jit(reset)


def warmup(agent, learner, data, key, steps, batch):
    def loop(learner, keys):
        def body(model, key):
            mini = _draw(data, key, data.choice.size, batch)
            return agent.imitate(model, mini)

        return jax.lax.scan(body, learner, keys)

    return jax.jit(loop)(learner, jax.random.split(key, steps))


def corrective_demonstrations(actor, params, seeds, expert_fraction):
    """Label states actually visited by the current policy with scenario actions."""
    env = taxi.GamblersTaxiEnv

    def reset(key):
        key, reset_key = jax.random.split(key)
        return key, *env.reset(reset_key, params)

    def one(key, obs, state):
        key, expert_key, policy_key, env_key = jax.random.split(key, 4)
        expert = scenarios.mixed_expert(state, obs, params) if (
            params.config.late_start_probability or params.config.starting_debt
        ) else (
            scenarios.scripted_policy(state, obs, params, 0)
        )
        control, choice = dsac.sample(actor, policy_key, obs.features, obs.action_mask, True)
        use_expert = jax.random.bernoulli(expert_key, expert_fraction)
        action = taxi.TaxiAction(jnp.where(use_expert, expert.control, control),
                                 jnp.where(use_expert, expert.choice, choice))
        next_obs, new, reward, done, _ = env.step(env_key, state, action, params)
        active = state.outcome == taxi.ACTIVE
        # Corrective labels train only the actor. Their outcomes came from the executed action.
        row = dsac.Transition(obs.features, obs.action_mask, expert.control, expert.choice, reward, done,
                              next_obs.features, next_obs.action_mask)
        keep = lambda new, old: jnp.where(active, new, old)
        return (key, jax.tree.map(keep, next_obs, obs), jax.tree.map(keep, new, state)), (row, active)

    def body(carry, _):
        return jax.vmap(one)(*carry)

    _, trace = jax.lax.scan(body, jax.vmap(reset)(seeds), None, length=params.config.horizon)
    return trace


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(args.seed)
    c = taxi.TaxiConfig(late_start_probability=args.late_starts,
                       enable_dependence=args.dependence)
    agent_config = dsac.AgentConfig(width=args.width, quantiles=args.quantiles, gamma=c.gamma,
                                   risk_weight=args.risk, demo_weight=args.demo_weight,
                                   learning_rate=args.learning_rate, upside_weight=args.upside)
    params, state = taxi.GamblersTaxiEnv.init_params(key, c)
    obs = taxi.GamblersTaxiEnv.get_observation(state, params)
    print(f"backend={jax.devices()} features={obs.features.size} seed={args.seed}", flush=True)
    agent = dsac.HybridDSAC(agent_config)
    start = perf_counter()
    demos, imit, pool = make_demonstrations(params, 30000 + args.seed, args.consistent_expert,
                                           args.settlement_demos)
    print(f"demos={demos.choice.size} imitation={imit.choice.size} starts={pool.time.size} "
          f"seconds={perf_counter()-start:.2f}", flush=True)
    key, init_key, warm_key, reset_key = jax.random.split(key, 4)
    learner = agent.init(init_key, obs.features.size)
    if args.initialize:
        actor, _, _ = load_actor(args.initialize)
        learner = learner._replace(actor=actor)
    if args.imitation:
        learner, loss = warmup(agent, learner, imit, warm_key, args.imitation, args.batch)
        print(f"imitation={args.imitation} loss={float(loss[-1]):.5f}", flush=True)
    if args.correction_rounds:
        correct = eqx.filter_jit(corrective_demonstrations)
        for index in range(args.correction_rounds):
            frac = jnp.float32(0.40 / (index + 1))
            inputs = keys(40000 + args.seed + index*1000, 32)
            rows, active = correct(learner.actor, params, inputs, frac)
            active = np.asarray(active)
            sampled = active & ((np.indices(active.shape).sum(axis=0) % 3) == 0)
            ix = sampled.nonzero()
            correction = jax.tree.map(lambda x: x[ix], rows)
            imit = jax.tree.map(lambda a, b: jnp.concatenate((a, b), axis=0), imit, correction)
            key, k = jax.random.split(key)
            learner, loss = warmup(agent, learner, imit, k, args.correction_steps, args.batch)
            print(f"correction={index+1} samples={correction.choice.size} loss={float(loss[-1]):.5f}", flush=True)
    evaluate = eqx.filter_jit(evaluate_policy)
    val_keys = keys(10000, args.val_episodes)
    best = None

    def validate(actor, stage):
        nonlocal best
        metrics = summary(evaluate(actor, params, val_keys), c)
        result = (metrics["success"], metrics["debt_cleared"],
                  -(metrics["mean_debt_free_tick"] or c.horizon),
                  -metrics["crash"], metrics["mean_capital"])
        row = {"step": stage, **metrics}
        print("validation " + json.dumps(row), flush=True)
        with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        if best is None or result > best:
            best = result
            save_actor(output / "best.npz", actor, params, agent_config, stage, metrics, args.seed)

    validate(learner.actor, 0)
    if args.iterations:
        iteration, reset = make_collector(agent, params, demos, imit, pool, args.envs, args.rollout,
                                          args.batch, args.updates)
        first_obs, states = reset(jax.random.split(reset_key, args.envs), 0.75)
        replay = Replay(jax.tree.map(lambda x: jnp.zeros((args.capacity, *x.shape[1:]), x.dtype), demos),
                        jnp.int32(0), jnp.int32(0))
        carry = Collector(key, first_obs, states, replay, learner)
        chunk = perf_counter()
        chunk_steps = 0
        for i in range(1, args.iterations + 1):
            curriculum = jnp.float32(0.70 * max(0.0, 1.-i/max(1, args.iterations*0.75)))
            carry, (losses, outcomes) = iteration(carry, curriculum)
            chunk_steps += args.envs*args.rollout
            if i == 1 or i % args.log_every == 0 or i == args.iterations:
                values = np.asarray(losses)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite optimization metrics: {values}")
                row = {"iteration": i, "steps": i*args.envs*args.rollout,
                       "steps_per_second": chunk_steps/(perf_counter()-chunk),
                       "loss": values.tolist(), "outcomes": np.asarray(outcomes).tolist()}
                print("training " + json.dumps(row), flush=True)
                with (output / "training.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                chunk, chunk_steps = perf_counter(), 0
            if i % args.validate_every == 0 or i == args.iterations:
                validate(carry.learner.actor, i)
                chunk, chunk_steps = perf_counter(), 0
        save_actor(output / "last.npz", carry.learner.actor, params, agent_config, args.iterations, {}, args.seed)
    actor, _, meta = load_actor(output / "best.npz")
    test_keys = keys(20000, args.test_episodes)
    ablation = replace(params, config=replace(c, enable_emotion_effects=False))
    result = {
        "device": str(jax.devices()[0]), "jax": jax.__version__, "training_seconds": perf_counter()-start,
        "chosen_checkpoint": meta, "test_seed_start": 20000, "validation_seed_start": 10000,
        "policy": summary(evaluate(actor, params, test_keys), c),
        "emotion_off": summary(evaluate(actor, ablation, test_keys), c),
    }
    for strategy, name in enumerate(("work_sleep", "conditional_gambling", "slots_first",
                                      "blackjack_first", "bank_and_gamble", "mixed_expert",
                                      "chase_losses", "fast_cash")):
        result[name] = summary(evaluate(None, params, test_keys, strategy=strategy), c)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("test " + json.dumps(result["policy"]), flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--imitation", type=int, default=1400)
    parser.add_argument("--correction-rounds", type=int, default=0)
    parser.add_argument("--correction-steps", type=int, default=850)
    parser.add_argument("--consistent-expert", action="store_true")
    parser.add_argument("--settlement-demos", type=int, default=0,
                        help="extra successful credit routes for configurations where they are reachable")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=12)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=160000)
    parser.add_argument("--quantiles", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--risk", type=float, default=0.)
    parser.add_argument("--upside", type=float, default=0., help="weight on the highest return quantiles")
    parser.add_argument("--late-starts", type=float, default=0.)
    parser.add_argument("--dependence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=48)
    parser.add_argument("--test-episodes", type=int, default=128)
    args = parser.parse_args()
    if min(args.risk, args.upside) < 0 or args.risk + args.upside > 1:
        parser.error("--risk and --upside must be nonnegative and sum to at most 1")
    return args


if __name__ == "__main__":
    run(parse())
