"""Held-out debt cases, voluntary chasing, and paired input interventions."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms import hybrid_dsac
import debt_learning as learning
from debt_navigation import guarded_control
import debt_strategy as strategy
from env import city
from env import gamblers_taxi as taxi
from train import keys, load_actor, summary


ENV = taxi.GamblersTaxiEnv
TEST_START = 91000
TEST_COUNT = 128
PROBE_START = 97000
PROBE_COUNT = 256
EMOTIONS = ("mood", "tilt", "anxiety", "habit")
CASES = ("first_win", "first_failure", "history_with_both_games", "first_repeat_after_loss",
         "work_sleep_and_repeat", "win_work_sleep_and_repeat")


def record(old, new, obs, info):
    return {
        "active": old.outcome == taxi.ACTIVE, "time": old.time, "position": old.agent_pos,
        "debt": old.debt, "cash": old.cash, "fatigue": old.fatigue, "habit": old.habit,
        "anxiety": old.anxiety, "employed": old.employed,
        "requested": info["requested_choice"], "executed": info["choice"],
        "slots": obs.action_mask[taxi.SPIN], "blackjack": obs.action_mask[taxi.DEAL],
        "resolved": info["game_resolved"], "net": info["gambling_net"],
        "counters": new.counters, "next_outcome": new.outcome,
    }


@eqx.filter_jit
def original_rollouts(model, params, inputs, guard):
    def one(seed):
        rng = jax.random.fold_in(seed, 0)
        _, start = ENV.reset(jax.random.fold_in(seed, 1), params)

        def step(carry, _):
            rng, old = carry
            rng, policy_key, env_key = jax.random.split(rng, 3)
            obs = ENV.get_observation(old, params)
            control, choice = hybrid_dsac.sample(model, policy_key, obs.features, obs.action_mask, True)
            nominal = control
            if guard:
                control = jax.lax.cond(choice == taxi.DRIVE, lambda: guarded_control(old, params, control),
                                       lambda: control)
            _, new, _, _, info = ENV.step(env_key, old, taxi.TaxiAction(control, choice), params)
            active = old.outcome == taxi.ACTIVE
            new = jax.tree.map(lambda a, b: jnp.where(active, a, b), new, old)
            row = record(old, new, obs, info)
            row["intervened"] = active & jnp.any(control != nominal)
            row["drive"] = control
            row["offered_work"] = obs.action_mask[taxi.SHIFT]
            row["offered_repay"] = obs.action_mask[taxi.REPAY]
            row["alt_action_probability"] = jnp.exp(
                hybrid_dsac.actor_outputs(model, obs.features, obs.action_mask)[2]
            )
            return (rng, new), row

        (_, end), trace = jax.lax.scan(step, (rng, start), None, length=params.config.horizon)
        return end, trace

    return jax.vmap(one)(inputs)


@eqx.filter_jit
def destination_rollouts(model, params, inputs, mode):
    states, (options, traces) = learning.evaluate(model, params, inputs, mode, record=record)
    traces = jax.tree.map(lambda a: a.reshape((a.shape[0], -1, *a.shape[3:])), traces)
    traces["option"] = jnp.repeat(options, strategy.STEPS, axis=1)
    return states, traces


def observed_chasing(trace, index, cfg):
    """Measure the first legal next-round decision, and any repeat in a 155-tick window."""
    active = trace["active"][index]
    event_ticks = np.flatnonzero(active & (trace["resolved"][index] != 0))
    time = trace["time"][index]
    req, act = trace["requested"][index], trace["executed"][index]
    both = trace["slots"][index] | trace["blackjack"][index]
    voluntary = ((req == taxi.SPIN) | (req == taxi.DEAL)) & (req == act)
    actual_bet = (act == taxi.SPIN) | (act == taxi.DEAL)
    events = []
    for tick in event_ticks:
        # Stop at actual initiation of the next bet, horizon, or window expiry.
        later = np.arange(tick+1, min(len(active), tick+cfg.day+1))
        later = later[active[later]]
        wager = later[actual_bet[later]]
        if wager.size:
            later = later[later <= wager[0]]
        eligible = later[both[later]]
        first = int(eligible[0]) if eligible.size else None
        repeat = later[voluntary[later]]
        recorded_net = float(trace["net"][index, tick])
        events.append({
            "tick": int(time[tick])+1, "game": taxi.ACTION_NAMES[trace["resolved"][index, tick]],
            "net": recorded_net, "result": ("loss" if recorded_net < 0 else "win" if recorded_net > 0 else "push"),
            "first_eligible_tick": int(time[first]) if first is not None else None,
            "voluntary_at_first": bool(voluntary[first]) if first is not None else None,
            "voluntary_repeat_tick": int(time[repeat[0]]) if repeat.size else None,
        })
    return events


def chase_summary(episodes):
    results = {}
    for label in ("loss", "win", "push"):
        events = [event for row in episodes for event in row["chase_events"] if event["result"] == label]
        legal = [event for event in events if event["first_eligible_tick"] is not None]
        results[label] = {
            "resolved_rounds": len(events), "episodes": sum(any(
                event["result"] == label for event in row["chase_events"]
            ) for row in episodes),
            "eligible": len(legal), "voluntary_at_first": sum(e["voluntary_at_first"] for e in legal),
            "voluntary_within_one_day": sum(e["voluntary_repeat_tick"] is not None for e in events),
        }
    return results


def episode_rows(states, trace, cfg):
    rows = []
    for index in range(TEST_COUNT):
        outcome = int(states.outcome[index])
        active = trace["active"][index]
        repayment = trace["executed"][index] == taxi.REPAY
        events = observed_chasing(trace, index, cfg)
        row = {
            "seed": TEST_START+index, "outcome": taxi.OUTCOME_NAMES[outcome], "time": int(states.time[index]),
            "cash": float(states.cash[index]), "debt": float(states.debt[index]),
            "debt_free_tick": int(states.debt_free_at[index]), "terminal_pos": states.agent_pos[index].tolist(),
            "first_repay_tick": next((int(v) for v in trace["time"][index][active & repayment]), None),
            "start_tick": int(trace["time"][index, 0]),
            "max_fatigue": float(max(trace["fatigue"][index][active].max(), states.fatigue[index])),
            "min_debt": float(min(trace["debt"][index][active].min(), states.debt[index])),
            "chase_events": events,
            **dict(zip(taxi.COUNTER_NAMES, states.counters[index].tolist(), strict=True)),
        }
        for field in ("intervened", "offered_work", "offered_repay"):
            if field in trace:
                row[field] = int((trace[field][index] & active).sum())
        rows.append(row)
    return rows


def diagnostic_summary(rows):
    return {
        "episodes_with_work_offer": sum(row.get("offered_work", 0) > 0 for row in rows),
        "episodes_with_repay_offer": sum(row.get("offered_repay", 0) > 0 for row in rows),
        "episodes_with_shift": sum(row["paid"] > 0 for row in rows),
        "episodes_with_repay": sum(row["repayments"] > 0 for row in rows),
        "episodes_with_gamble": sum(row["spins"] + row["hands"] > 0 for row in rows),
        "episodes_with_intervention": sum(row.get("intervened", 0) > 0 for row in rows),
        "median_paid": float(np.median([row["paid"] for row in rows])),
        "median_terminal_tick": float(np.median([row["time"] for row in rows])),
        "crashes_before_first_pay": sum(row["paid"] == 0 and row["outcome"] == "crash" for row in rows),
        "crashes_after_first_pay": sum(row["paid"] > 0 and row["outcome"] == "crash" for row in rows),
    }


def save_cases(output, name, rows, trace):
    repeat = lambda row: any(e["result"] == "loss" and e["voluntary_repeat_tick"] is not None
                             for e in row["chase_events"])
    candidates = (
        lambda row: row["outcome"] == "success",
        lambda row: row["outcome"] != "success",
        lambda row: row["spins"] > 0 and row["hands"] > 0,
        repeat,
        lambda row: row["paid"] >= 2 and row["sleeps"] >= 1 and repeat(row),
        lambda row: row["outcome"] == "success" and row["paid"] >= 2 and row["sleeps"] >= 1 and repeat(row),
    )
    picked = {}
    for label, match in zip(CASES, candidates, strict=True):
        index = next((i for i, row in enumerate(rows) if match(row)), None)
        if index is None:
            continue
        row = rows[index]
        picked[label] = row["seed"]
        mask = trace["active"][index]
        np.savez_compressed(output/f"{name}_{label}_{row['seed']}.npz", **{
            field: values[index][mask] for field, values in trace.items()
        })
    return picked


def _after_slot(seed, params, won):
    c = params.config
    fr = jax.random.uniform(jax.random.fold_in(seed, 19), (6,))
    _, state = ENV.reset(seed, params)
    day = jnp.int32(1 + 6*fr[0])
    tilt, mood, fatigue, habit, base_debt = fr[1:]
    pending = replace(
        state, time=day*c.day+c.casino_open+jnp.int32(fr[1]*7),
        agent_pos=params.zone_centers[city.SLOTS], activity=jnp.int32(taxi.SPIN),
        remaining=jnp.int32(1), speed=jnp.float32(0), cash=jnp.float32(130), stake=jnp.float32(c.slot_stake),
        debt=260.+850.*base_debt, fatigue=.20+.48*fatigue, habit=.15+.50*habit,
        mood=-.43+.80*mood, tilt=.08+.68*tilt, anxiety=.46+.38*base_debt,
        shift_claimed=jnp.bool_(True), abstinence=jnp.int32(0), next_round=jnp.int32(0),
    )
    force = replace(params, config=replace(c, slot_probability=float(won)))
    step = lambda p, k, s, choice: ENV.step(k, s, taxi.TaxiAction(jnp.array([-1., 0.]), choice), p)[1]
    settled = step(force, seed, pending, jnp.int32(taxi.WAIT))

    def wait(i, state):
        return step(params, jax.random.fold_in(seed, i), state, jnp.int32(taxi.DRIVE))

    return jax.lax.fori_loop(0, c.casino_round_interval, wait, settled)


@eqx.filter_jit
def probes(model, params, inputs, upper):
    def probability(state):
        if upper:
            logp, _ = learning.distribution(model, strategy.features(state, params), strategy.legal(state, params))
            return jnp.exp(logp[strategy.SLOTS]), jnp.argmax(logp) == strategy.SLOTS
        obs = ENV.get_observation(state, params)
        _, _, logp = hybrid_dsac.actor_outputs(model, obs.features, obs.action_mask)
        return jnp.exp(logp[taxi.SPIN]), jnp.argmax(logp) == taxi.SPIN

    def one(seed):
        loss, win = _after_slot(seed, params, False), _after_slot(seed, params, True)
        unadjusted_win, _ = probability(win)
        win = replace(win, cash=loss.cash)
        legal = taxi._legal_actions(loss, params)[taxi.SPIN] & taxi._legal_actions(win, params)[taxi.SPIN]
        p_loss, a_loss = probability(loss)
        p_win, a_win = probability(win)
        moved = {field: probability(replace(loss, **{field: getattr(win, field)}))[0] - p_loss
                 for field in EMOTIONS}
        for field, value in (
            ("debt_minus_250", replace(loss, debt=loss.debt-250.)),
            ("cash_plus_100", replace(loss, cash=loss.cash+100.)),
            ("fatigue_plus_20pp", replace(loss, fatigue=jnp.minimum(loss.fatigue+.20, 1.))),
            ("two_days_later", replace(loss, time=loss.time+2*params.config.day,
                                      next_round=loss.next_round+2*params.config.day)),
        ):
            moved[field] = probability(value)[0] - p_loss
        return {"valid": legal, "loss": p_loss, "win": p_win, "a_loss": a_loss, "a_win": a_win,
                "win_unadjusted": unadjusted_win, **moved}

    return jax.vmap(one)(inputs)


def probe_summary(values):
    good = values["valid"]
    delta = values["loss"][good]-values["win"][good]
    if not delta.size:
        raise ValueError("No matched legal post-outcome slot decisions")
    bootstrap = np.random.default_rng(139).choice(delta, size=(5000, delta.size), replace=True).mean(axis=1)
    return {
        "pairs": int(good.sum()), "loss_mean": float(values["loss"][good].mean()),
        "win_mean_matched": float(values["win"][good].mean()),
        "win_mean_with_payout": float(values["win_unadjusted"][good].mean()),
        "loss_minus_win": float(delta.mean()), "paired_bootstrap_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
        "modal_loss": int(values["a_loss"][good].sum()), "modal_win": int(values["a_win"][good].sum()),
        "loss_feature_changes": {f: {"mean_delta": float(x[good].mean()), "mean_absolute_delta": float(np.abs(x[good]).mean())}
                                 for f, x in values.items() if f in EMOTIONS or f.endswith(("250", "100", "20pp", "later"))},
    }


def save_json(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--destination-only", action="store_true")
    args = parser.parse_args()
    path = args.output
    path.mkdir(parents=True, exist_ok=True)
    cases = path/"cases"
    cases.mkdir(exist_ok=True)
    test = keys(TEST_START, TEST_COUNT)
    probe_keys = keys(PROBE_START, PROBE_COUNT)
    root = Path(__file__).resolve().parent
    current = path/"results.json"
    results = json.loads(current.read_text(encoding="utf-8")) if current.exists() else {}
    results["protocol"] = {
        "test_seed_start": TEST_START, "test_count": TEST_COUNT,
        "probe_seed_start": PROBE_START, "probe_count": PROBE_COUNT,
        "backend": str(jax.devices()[0]), "jax": jax.__version__,
        "numpy": np.__version__, "optax": optax.__version__,
        "source_sha256": {name: file_sha(root/name) for name in (
            "debt_learning.py", "debt_strategy.py", "debt_navigation.py", "debt_research.py", "debt_decisions.py",
            "env/base.py", "env/gamblers_taxi.py", "env/city.py", "algorithms/hybrid_dsac.py",
            "scenarios.py", "train.py",
        )},
    }

    def assess(name, state_and_trace, params, model, upper, checkpoint=None, metadata=None):
        states, trace = jax.device_get(state_and_trace)
        rows = episode_rows(states, trace, params.config)
        if any(row["impulses"] for row in rows):
            raise AssertionError("Nonvoluntary bet in the research evaluation")
        for row in rows:
            if row["outcome"] == "success" and not (row["debt"] == 0 and 0 <= row["debt_free_tick"] < row["time"]):
                raise AssertionError("Buyout must follow full bank repayment")
        horizon = params.config.horizon
        out = {"summary": {
            **summary(states, params.config),
            "mean_debt_free_tick_or_horizon": float(np.mean([
                row["debt_free_tick"] if row["debt_free_tick"] >= 0 else horizon for row in rows
            ])),
            "mean_buyout_tick_or_horizon": float(np.mean([
                row["time"] if row["outcome"] == "success" else horizon for row in rows
            ])),
            "buyouts_after_playing": sum(row["outcome"] == "success" and row["spins"]+row["hands"] > 0 for row in rows),
        }, "diagnostics": diagnostic_summary(rows),
               "environment": asdict(params.config), "chase": chase_summary(rows),
               "cases": save_cases(cases, name, rows, trace)}
        save_json(path/f"{name}_episodes.json", rows)
        if model is not None:
            values = jax.device_get(probes(model, params, probe_keys, upper))
            np.savez_compressed(path/f"{name}_probes.npz", **values)
            out["probes"] = probe_summary(values)
        if checkpoint is not None:
            out["checkpoint"] = {"file": str(checkpoint.relative_to(path.parent)), "sha256": file_sha(checkpoint),
                                 "metadata": metadata}
        results[name] = out
        save_json(current, results)
        print(name, out["summary"]["success"], out["summary"]["debt_cleared"], out["summary"]["crash"],
              out["chase"], flush=True)

    if not args.destination_only:
        for kind in ("neutral", "upside"):
            checkpoint = path.parent/f"debt_{kind}_actor.npz"
            model, cfg, metadata = load_actor(checkpoint)
            params, _ = ENV.init_params(jax.random.key(0), replace(cfg, enable_impulse_override=False))
            for guard in (False, True):
                name = f"legacy_{kind}" + ("_guarded" if guard else "")
                assess(name, original_rollouts(model, params, test, guard), params, model, False, checkpoint, metadata)

    if not args.legacy_only:
        params, _ = ENV.init_params(jax.random.key(0), learning.config(.65))
        for mode in ("work", "mixed"):
            assess(f"scripted_{mode}", destination_rollouts(None, params, test, mode), params, None, True)
        for checkpoint in sorted((path/"models").glob("*/selected.npz")):
            for candidate in (checkpoint, checkpoint.with_name("imitation.npz")):
                if not candidate.is_file():
                    continue
                model, metadata = learning.load(candidate)
                params, _ = ENV.init_params(jax.random.key(0), learning.config(metadata["urgency"]))
                suffix = "_imitation" if candidate.stem == "imitation" else ""
                assess(checkpoint.parent.name+suffix, destination_rollouts(model, params, test, "greedy"),
                       params, model, True, candidate, metadata)


if __name__ == "__main__":
    main()
