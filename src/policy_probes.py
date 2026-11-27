"""Matched public-event probes and voluntary loss-following choices."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env import city, resource_taxi as env
from algorithms.hybrid_dsac import actor_outputs
import models
import strategy


def probabilities(model, kind, features, mask, memory):
    if kind.startswith("ppo"):
        return jnp.exp(models.ppo_step(model, features, mask, memory)[2])
    if kind == "dsac_gru":
        history = jnp.concatenate((memory[:, 1:], features[:, None]), axis=1)
        return jnp.exp(actor_outputs(model["actor"], models.encode_history(model["encoder"], history), mask)[2])
    return jnp.exp(actor_outputs(model, features, mask)[2])


def matched(model, kind, config, interface="physical"):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    # Synthetic probes use a development scenario, never the held-out test bank.
    reset = strategy.reset if interface == "strategy" else env.ResourceTaxiEnv.reset
    observe = strategy.get_observation if interface == "strategy" else env.ResourceTaxiEnv.get_observation
    _, state = reset(6, params)
    if interface == "strategy":
        state = replace(state,previous_option=jnp.int32(strategy.SLOTS_MEDIUM))
    states = []
    labels = []
    for cash in (30., 100., 300.):
        for event in (-25., 0., 25.):
            states.append(replace(state, agent_pos=params.zone_centers[city.SLOTS],
                time=jnp.int32(110), cash=jnp.float32(cash), previous_event=jnp.float32(event),
                previous_action=jnp.int32(env.SPIN_MEDIUM)))
            labels.append({"probe": "public_event", "cash": cash, "previous_public_event": event})
    for field, values in (("satiety", (.15, .5, .85)), ("fuel", (1., 8., 20.)),
                          ("fatigue", (.1, .6, 1.)), ("time", (95, 110, 140))):
        for value in values:
            changed = replace(state, agent_pos=params.zone_centers[city.SLOTS],
                time=jnp.int32(110), cash=jnp.float32(100.), previous_event=jnp.float32(0.),
                previous_action=jnp.int32(env.WAIT))
            changed = replace(changed, **{field: jnp.int32(value) if field == "time" else jnp.float32(value)})
            states.append(changed)
            labels.append({"probe": "resource_or_time", "intervention": field, "value": value})
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    obs = jax.vmap(observe, in_axes=(0, None))(batch, params)
    memory = models.initial_memory(kind, len(states), obs.features.shape[-1])
    prob = probabilities(model, kind, obs.features, obs.action_mask, memory)
    for label, p in zip(labels, np.asarray(prob)):
        if interface == "strategy":
            label["gamble_probability"] = float(p[strategy.SLOTS_SMALL:strategy.BLACKJACK_LARGE+1].sum())
            label["travel_intention_probability"] = float(p[list(strategy.CONTINUOUS)].sum())
            label["wait_probability"] = float(p[strategy.WAIT])
            label["work_probability"] = float(p[strategy.WORK])
            label["meal_probability"] = float(p[strategy.DINER:strategy.RESTAURANT+1].sum())
        else:
            label["gamble_probability"] = float(p[env.SPIN_SMALL:env.DEAL_LARGE+1].sum())
            label["drive_probability"] = float(p[env.DRIVE])
    return labels


def loss_following(directory):
    pending, opportunities, continued = {}, 0, 0
    windows = []
    for path in sorted(Path(directory).glob("trace_*.npz")):
        with np.load(path) as d:
            for tick in range(d["time"].shape[0]):
                for lane in range(d["time"].shape[1]):
                    if not d["active"][tick, lane]:
                        continue
                    scenario = int(d["scenario_id"][tick, lane])
                    eligible = bool(d["slots_legal"][tick, lane] or d["blackjack_legal"][tick, lane])
                    if "decision_start" in d:
                        eligible &= bool(d["decision_start"][tick,lane])
                    action = int(d["executed"][tick, lane])
                    if scenario in pending and eligible:
                        opportunities += 1
                        chosen = env.SPIN_SMALL <= action <= env.DEAL_LARGE
                        continued += int(chosen)
                        windows.append({"scenario_id": scenario, "loss_tick": pending.pop(scenario),
                            "decision_tick": int(d["time"][tick, lane]), "voluntary_wager": bool(chosen)})
                    if d["game_resolved"][tick, lane] and d["gambling_net"][tick, lane] < 0:
                        pending[scenario] = int(d["time"][tick, lane])
    return {"eligible_follow_up_decisions": opportunities, "voluntary_wagers": continued,
        "losses_without_observed_eligible_follow_up": len(pending),
        "rate": continued/opportunities if opportunities else None,
        "interpretation": "First legal wagering opportunity at a policy decision after a loss; continuation requires an actual new wager on that physical step. Descriptive; liquidity, cooldown, decision cadence and travel affect eligibility.",
        "windows": windows}


def recurrent_reset_probe(model, kind, directory):
    if kind not in ("ppo_gru", "dsac_gru"):
        return {"status": "not applicable to a feedforward policy"}
    memory = None
    sampled = []

    @eqx.filter_jit
    def block(memory, features, masks, active):
        def tick(memory, row):
            features, masks, active = row
            original = probabilities(model, kind, features, masks, memory)
            reset = probabilities(model, kind, features, masks, jnp.zeros_like(memory))
            _, new_memory, _, _, _, _, _ = models.policy_step(model, kind, features, masks,
                memory, jax.random.PRNGKey(81), True)
            memory = jnp.where(active.reshape((active.size,)+(1,)*(memory.ndim-1)), new_memory, memory)
            distance = .5*jnp.abs(original-reset).sum(axis=-1)
            return memory, distance
        return jax.lax.scan(tick, memory, (features, masks, active))

    paths = sorted(Path(directory).glob("decision_*.npz")) or sorted(Path(directory).glob("trace_*.npz"))
    for path in paths:
        with np.load(path) as data:
            features, masks = data["observation"], data["action_mask"]
            if memory is None:
                memory = models.initial_memory(kind, features.shape[1], features.shape[2])
            memory, distance = block(memory, jnp.asarray(features), jnp.asarray(masks), jnp.asarray(data["active"]))
            eligible = data["active"] & (data["time"] > 8) & (masks.sum(axis=-1) > 1)
            sampled.extend(np.asarray(distance)[eligible].tolist())
    return {"decisions": len(sampled), "mean_action_total_variation_after_memory_reset": float(np.mean(sampled)) if sampled else None,
            "scope": "Recorded decision distribution probe, not a counterfactual episode outcome."}


def generate(checkpoint, directory):
    model, metadata = models.load_policy(checkpoint)
    result = {"matched_public_event_probe": matched(model, metadata["kind"], env.Config(**metadata.get("evaluation_environment", metadata["environment"])), metadata.get("interface","physical")),
              "loss_following": loss_following(directory),
              "history_reset": recurrent_reset_probe(model, metadata["kind"], directory),
              "claim_limit": "These are behavioral sensitivities, not thoughts, attention explanations, or evidence of human addiction."}
    (Path(directory)/"policy_probes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    generate(args.checkpoint, args.evaluation)
