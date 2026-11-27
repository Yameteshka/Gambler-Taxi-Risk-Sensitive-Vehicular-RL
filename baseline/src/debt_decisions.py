"""Compare policy outputs at reachable kinds of financial decision points."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from algorithms import hybrid_dsac
import debt_learning as learning
import debt_strategy as strategy
from env import city
from env import gamblers_taxi as taxi
from train import keys, load_actor


ENV = taxi.GamblersTaxiEnv
N = 256
CONDITIONS = ("base", "earlier", "later", "fatigue_low", "fatigue_high", "cash_plus_200", "debt_minus_250")


@eqx.filter_jit
def choices(actor, params, zone, upper):
    c = params.config

    def one(key):
        _, state = ENV.reset(key, params)
        fr = jax.random.uniform(jax.random.fold_in(key, 6), (4,))
        night = zone == city.HOME
        phase = c.casino_open+5 if night else 29
        state = replace(
            state, agent_pos=params.zone_centers[zone], time=jnp.int32(4*c.day+phase),
            cash=110.+340.*fr[0], debt=410.+650.*fr[1], fatigue=.15+.77*fr[2],
            anxiety=.49+.32*fr[1], mood=-.66+.58*fr[3], tilt=.26*fr[3],
            habit=.12+.44*fr[3], shift_claimed=jnp.bool_(night), abstinence=jnp.int32(56),
        )
        variants = (
            state, replace(state, time=state.time-2*c.day), replace(state, time=state.time+2*c.day),
            replace(state, fatigue=jnp.float32(.15)), replace(state, fatigue=jnp.float32(.95)),
            replace(state, cash=state.cash+200.), replace(state, debt=state.debt-250.),
        )

        def probability(example):
            if upper:
                logp, _ = learning.distribution(actor, strategy.features(example, params),
                                                strategy.legal(example, params))
            else:
                obs = ENV.get_observation(example, params)
                logp = hybrid_dsac.actor_outputs(actor, obs.features, obs.action_mask)[2]
            return jnp.exp(logp), jnp.argmax(logp)

        return jax.vmap(probability)(jax.tree.map(lambda *items: jnp.stack(items), *variants))

    return jax.vmap(one)(keys(98000, N))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output
    results = {"seeds": [98000, 98000+N-1], "policy": {}}
    research = json.loads((path/"results.json").read_text(encoding="utf-8"))
    for name, data in research.items():
        if not isinstance(data, dict) or "checkpoint" not in data or name.endswith(("_guarded", "_imitation")):
            continue
        checkpoint = path.parent/data["checkpoint"]["file"]
        upper = not name.startswith("legacy")
        actor = learning.load(checkpoint)[0] if upper else load_actor(checkpoint)[0]
        params, _ = ENV.init_params(jax.random.key(0), taxi.TaxiConfig(**data["environment"]))
        labels = strategy.NAMES if upper else taxi.ACTION_NAMES
        by_zone = results["policy"][name] = {}
        for zone in (city.WORK, city.HOME, city.BANK):
            probabilities, modal = jax.device_get(choices(actor, params, zone, upper))
            by_zone[city.ZONE_NAMES[zone]] = {
                variant: {"probability": dict(zip(labels, probabilities[:, i].mean(axis=0).tolist())),
                          "modal": dict(zip(labels, np.bincount(modal[:, i], minlength=len(labels)).tolist())),
                          "mean_absolute_delta": dict(zip(labels, np.abs(
                              probabilities[:, i]-probabilities[:, 0]
                          ).mean(axis=0).tolist()))}
                for i, variant in enumerate(CONDITIONS)
            }
    (path/"decision_probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
