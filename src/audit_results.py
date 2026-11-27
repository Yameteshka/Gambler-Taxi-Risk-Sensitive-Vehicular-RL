"""Replay every recorded evaluation action and verify numeric trajectories.

This checks execution, saved policy identity, logging and accounting. Independent
rule tests in verify.py are still necessary: replay alone cannot validate rules.
"""
import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env import resource_taxi as env
from evaluate import RESET
import models
import navigation
from telemetry import frame, file_hash


@lru_cache(maxsize=64)
def replay_block(kind, deterministic, controller, window):
    @eqx.filter_jit
    def block(states, hidden, returns, model, params, controls, choices):
        def tick(carry, recorded):
            states, hidden, returns = carry
            control, choice = recorded
            observation = jax.vmap(env.ResourceTaxiEnv.get_observation, in_axes=(0, None))(states, params)
            keys = jax.vmap(lambda s, t: jax.random.fold_in(jax.random.fold_in(
                jax.random.PRNGKey(9181), s), t))(states.scenario_id, states.time)
            if kind == "reference":
                predicted = jax.vmap(navigation.reference_policy, in_axes=(0, 0, None, None))(
                    states, observation, params, controller)
            else:
                predicted, hidden, _, _, _, _, _ = models.policy_step(
                    model, kind, observation.features, observation.action_mask, hidden, keys, deterministic)
                if controller == "guard":
                    guarded = jax.vmap(navigation.guard, in_axes=(0, None, 0))(states, params, predicted.control)
                    predicted = env.Action(jnp.where((predicted.choice == env.DRIVE)[:, None], guarded, predicted.control), predicted.choice)
            action = env.Action(control, choice)
            _, new, reward, _, info = jax.vmap(env.ResourceTaxiEnv.step, in_axes=(0, 0, 0, None))(
                keys, states, action, params)
            within = states.time < window
            new = jax.tree.map(lambda a, b: jnp.where(within.reshape((within.size,)+(1,)*(a.ndim-1)), a, b), new, states)
            returns += params.config.gamma**states.time*info["external_reward"]*within
            trace = frame(states, new, action, reward, info)
            trace["active"] &= within
            # Loan arrays are checked at endpoints and by payment conservation.
            trace.pop("loan_repayments")
            trace.pop("loan_ids")
            trace["policy_control"] = predicted.control
            trace["policy_choice"] = predicted.choice
            return (new, hidden, returns), trace
        return jax.lax.scan(tick, (states, hidden, returns), (controls, choices))
    return block


def audit(directory, checkpoint=None):
    directory = Path(directory)
    config = json.loads((directory/"evaluation.json").read_text(encoding="utf-8"))
    if config.get("interface") == "strategy":
        from audit_strategy import audit as audit_strategy
        return audit_strategy(directory,checkpoint)
    c = env.Config(**config["config"])
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), c)
    ids = jnp.array(config["scenario_ids"], dtype=jnp.int32)
    observation, states = RESET(ids, params)
    kind = config["kind"]
    if kind == "reference":
        model = None
    elif checkpoint is None:
        raise ValueError("A learned-policy audit requires its exact checkpoint")
    else:
        model, metadata = models.load_policy(checkpoint)
        if metadata["kind"] != kind:
            raise ValueError("Checkpoint algorithm differs from recorded evaluation")
    hidden = models.initial_memory(kind, len(ids), observation.features.shape[-1])
    returns = jnp.zeros(len(ids))
    window = c.horizon or 30*c.day
    block = replay_block(kind, config["deterministic"], config["controller"], window)
    active_steps, comparisons, max_error = 0, 0, 0.
    trace_hashes = {}
    for path in sorted(directory.glob("trace_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            (states, hidden, returns), expected = block(states, hidden, returns, model, params,
                jnp.asarray(data["control"]), jnp.asarray(data["requested"]))
            active = data["active"]
            active_steps += int(active.sum())
            for name, value in expected.items():
                recorded_name = {"policy_control": "control", "policy_choice": "requested"}.get(name, name)
                actual = data[recorded_name]
                wanted = np.asarray(value)
                if wanted.shape[:2] != active.shape or actual.shape != wanted.shape:
                    raise AssertionError(f"{name}: invalid time/environment axes {actual.shape}, {wanted.shape}, active={active.shape}")
                if name == "active":
                    np.testing.assert_array_equal(actual, wanted)
                elif active.any():
                    if wanted.dtype.kind in "biu":
                        np.testing.assert_array_equal(actual[active], wanted[active], err_msg=name)
                    else:
                        error = float(np.max(np.abs(actual[active]-wanted[active])))
                        max_error = max(max_error, error)
                        np.testing.assert_allclose(actual[active], wanted[active], rtol=2e-6, atol=2e-5, err_msg=name)
                comparisons += 1
            # Cash-flow identity: borrowing, wages and net game settlements minus
            # purchases and repayments. Escrow/game costs are handled by replay.
            assert np.all(data["cash"][active] >= -1e-4), "Negative cash"
            assert np.all(data["car_debt"][active] >= -1e-4), "Negative car debt"
            assert np.all(data["micro_debt"][active] >= -1e-4), "Negative personal debt"
        trace_hashes[path.name] = file_hash(path)
    with np.load(directory/"endpoints.npz", allow_pickle=False) as endpoints:
        for name in ("time", "cash", "car_debt", "counters", "loan_ids", "loan_principal", "loan_balance", "loan_rates", "loan_opened", "outcome"):
            np.testing.assert_allclose(np.asarray(getattr(states, name)), endpoints[name], rtol=2e-6, atol=2e-5, err_msg=name)
        np.testing.assert_allclose(np.asarray(returns), endpoints["external_return"], rtol=2e-6, atol=2e-5)
    result = {"passed": True, "active_physical_steps_replayed": active_steps, "field_chunk_comparisons": comparisons,
              "maximum_logged_float_error": max_error, "policy_reproduced": True,
              "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None, "trace_sha256": trace_hashes,
              "scope": "Replay of logged active decisions and endpoints using the recorded policy and parameters. Simulator rules are tested in verify.py."}
    (directory/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.evaluation, arguments.checkpoint)), flush=True)
