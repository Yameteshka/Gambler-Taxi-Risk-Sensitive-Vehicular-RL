"""Frozen-policy evaluation with compressed traces and bounded observation windows."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env import resource_taxi as t
import models
import navigation
from telemetry import frame, write_chunk


def summary(states, external_returns, window):
    states = jax.device_get(states)
    outcomes = np.asarray(states.outcome)
    cash = np.asarray(states.cash)
    debt = np.asarray(states.car_debt+t.micro_debt(states))
    returns = np.asarray(external_returns)
    return {
        "episodes": int(outcomes.size),
        **{name: int((outcomes == i).sum()) for i, name in enumerate(t.OUTCOME_NAMES)},
        "censored": int((outcomes == t.ACTIVE).sum()),
        "success_rate": float((outcomes == t.SUCCESS).mean()),
        "mean_external_return": float(returns.mean()),
        "return_percentiles": dict(zip(("p05", "p10", "p50", "p90", "p95"), np.quantile(returns, [.05, .1, .5, .9, .95]).tolist())),
        "mean_cash": float(cash.mean()), "mean_remaining_debt": float(debt.mean()),
        "mean_net_worth": float(np.asarray(t.net_worth(states)).mean()),
        "outside_financial_linear_region": int((np.abs(np.asarray(t.net_worth(states))) > 2000).sum()),
        "car_cleared": int((np.asarray(states.car_debt) == 0).sum()),
        "all_debt_cleared": int((debt == 0).sum()),
        "mean_endpoint_adjusted_completion_tick": float(np.where(outcomes == t.SUCCESS, states.time, window).mean()),
        "mean_borrowed": float(np.asarray(states.borrowed).mean()),
        "mean_interest": float(np.asarray(states.interest).mean()),
        **{name: float(np.asarray(states.counters)[..., i].mean()) for i, name in enumerate(t.COUNTER_NAMES)},
    }


@lru_cache(maxsize=64)
def compiled_block(kind, deterministic, controller, count, window):
    @eqx.filter_jit
    def block(states, hidden, returns, start, model, params):
        def tick(carry, index):
            states, hidden, returns = carry
            obs = jax.vmap(t.ResourceTaxiEnv.get_observation, in_axes=(0, None))(states, params)
            key = jax.vmap(lambda scenario, time: jax.random.fold_in(
                jax.random.fold_in(jax.random.PRNGKey(9181), scenario), time))(states.scenario_id, states.time)
            if kind == "reference":
                actions = jax.vmap(navigation.reference_policy, in_axes=(0, 0, None, None))(
                    states, obs, params, controller)
                requested_control = actions.control
            else:
                actions, hidden, _, _, _, _, _ = models.policy_step(
                    model, kind, obs.features, obs.action_mask, hidden, key, deterministic)
                requested_control = actions.control
                if controller == "guard":
                    controls = jax.vmap(navigation.guard, in_axes=(0, None, 0))(states, params, actions.control)
                    actions = t.Action(jnp.where((actions.choice == t.DRIVE)[:, None], controls, actions.control),
                                       actions.choice)
            _, new, reward, _, info = jax.vmap(t.ResourceTaxiEnv.step, in_axes=(0, 0, 0, None))(
                key, states, actions, params)
            within = states.time < window
            new = jax.tree.map(lambda a, b: jnp.where(
                within.reshape((count,)+(1,)*(a.ndim-1)), a, b), new, states)
            returns += params.config.gamma**states.time*info["external_reward"]*within
            trace = frame(states, new, actions, reward, info)
            trace["requested_control"] = requested_control
            trace["observation"] = obs.features
            trace["action_mask"] = obs.action_mask
            trace["active"] &= within
            return (new, hidden, returns), trace
        return jax.lax.scan(tick, (states, hidden, returns), jnp.arange(64))

    return block


RESET = eqx.filter_jit(jax.vmap(t.ResourceTaxiEnv.reset, in_axes=(0, None)))


def evaluate(model, kind, params, scenario_ids, output=None, deterministic=False, controller="learned", interface="physical"):
    if interface == "strategy":
        from evaluate_strategy import evaluate as strategy_evaluate
        return strategy_evaluate(model,kind,params,scenario_ids,output,deterministic,controller)
    window = params.config.horizon or 30*params.config.day
    # At most one loan can be opened per tick; capacity follows the observation window.
    # This is storage allocation, not a lending limit in the task.
    params = replace(params, config=replace(params.config,
        loan_capacity=max(params.config.loan_capacity, window+1)))
    ids = jnp.asarray(scenario_ids, dtype=jnp.int32)
    obs, states = RESET(ids, params)
    hidden = models.initial_memory(kind, len(ids), obs.features.shape[-1])
    returns = jnp.zeros(len(ids))

    block = compiled_block(kind, deterministic, controller, len(ids), window)

    chunk = 0
    while int(np.asarray(states.time).max()) < window and bool(np.any(np.asarray(states.outcome) == t.ACTIVE)):
        (states, hidden, returns), trace = block(states, hidden, returns, jnp.int32(chunk*64), model, params)
        if bool(np.any(np.asarray(trace["capacity_needed"]))):
            raise RuntimeError("Increase loan_capacity before this evaluation window")
        if output is not None:
            write_chunk(Path(output)/f"trace_{chunk:05}.npz", trace)
        chunk += 1
    result = summary(states, returns, window)
    if output is not None:
        path = Path(output)
        payload = {**result, "config": asdict(params.config), "kind": kind,
                   "scenario_ids": list(map(int, scenario_ids)), "deterministic": deterministic,
                   "controller": controller}
        (path/"evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(path/"endpoints.npz", scenario_id=np.asarray(ids),
            external_return=np.asarray(returns), outcome=np.asarray(states.outcome),
            time=np.asarray(states.time), cash=np.asarray(states.cash), car_debt=np.asarray(states.car_debt),
            micro_debt=np.asarray(t.micro_debt(states)), counters=np.asarray(states.counters),
            loan_ids=np.asarray(states.loan_ids), loan_principal=np.asarray(states.loan_principal),
            loan_balance=np.asarray(states.loan_balance), loan_rates=np.asarray(states.loan_rates),
            loan_opened=np.asarray(states.loan_opened))
        from statistics_report import uncertainty
        (path/"uncertainty.json").write_text(json.dumps(uncertainty(np.asarray(returns),
            np.asarray(states.outcome)), indent=2), encoding="utf-8")
        marker = path/"complete.tmp"
        marker.write_text(json.dumps({"episodes": len(ids), "window": window}), encoding="utf-8")
        marker.replace(path/"complete.json")
    return result, states, returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", choices=("reserve", "opportunistic"), default="reserve")
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax and mean controls instead of the trained stochastic policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, metadata = models.load_policy(args.checkpoint) if args.checkpoint else (None, {})
    config = t.Config(**metadata.get("evaluation_environment", metadata.get("environment", {}))) if metadata else t.Config()
    params, _ = t.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    result, _, _ = evaluate(model, metadata.get("kind", "reference"), params,
                            range(args.scenario_start, args.scenario_start+args.episodes), args.output,
                            deterministic=args.deterministic, interface=metadata.get("interface","physical"),
                            controller="learned" if model is not None else args.reference)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
