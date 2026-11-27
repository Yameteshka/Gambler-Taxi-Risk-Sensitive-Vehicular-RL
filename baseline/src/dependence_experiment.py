"""Paired evaluation of the same actor and scripted driver with habit dynamics on/off."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import equinox as eqx
import jax

from env.gamblers_taxi import GamblersTaxiEnv
from train import evaluate_policy, keys, load_actor, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dependence.json"))
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--episodes", type=int, default=256)
    args = parser.parse_args()
    actor, config, meta = load_actor(args.checkpoint)
    env = GamblersTaxiEnv
    params, _ = env.init_params(jax.random.key(args.seed), config)
    seeds = keys(args.seed, args.episodes)
    evaluate = eqx.filter_jit(evaluate_policy)
    result = {"checkpoint": args.checkpoint.name, "checkpoint_step": meta["stage"], "seed": args.seed,
              "episodes_per_condition": args.episodes}
    for name, strategy, weights in (("actor", -1, actor), ("bank_and_gamble", 4, None)):
        result[name] = {}
        for enabled in (False, True):
            current = replace(params, config=replace(config, enable_dependence=enabled))
            result[name]["on" if enabled else "off"] = summary(
                evaluate(weights, current, seeds, strategy=strategy), current.config
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
