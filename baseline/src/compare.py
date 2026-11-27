"""Compare selected checkpoints on common seeds and save the project results."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from dataclasses import replace

import jax

from env.gamblers_taxi import GamblersTaxiEnv
from train import evaluate_cohorts, keys, load_actor


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"test_seed_start": args.seed, "test_episodes": args.episodes, "models": {}, "baselines": {}}
    configs = []
    val_rank = {}
    seeds = keys(args.seed, args.episodes)
    prefix = "mixed_" if args.previous else ""
    for name, run in (("neutral", args.neutral), ("risk", args.risk)):
        checkpoint = run / "best.npz"
        actor, config, meta = load_actor(checkpoint)
        configs.append(config)
        params, _ = GamblersTaxiEnv.init_params(jax.random.key(args.seed), config)
        off = replace(params, config=replace(config, enable_emotion_effects=False))
        val = meta["metrics"]
        val_rank[name] = (val["success"], -val["crash"], val["mean_capital"])
        row = {"checkpoint": meta, "test": evaluate_cohorts(actor, params, seeds),
               "emotion_off": evaluate_cohorts(actor, off, seeds)}
        if (run / "evaluation.json").exists():
            raw = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
            row["training_seconds"] = raw["training_seconds"]
            row["device"] = raw["device"]
        if (run / "training.jsonl").exists():
            records = [json.loads(line) for line in (run / "training.jsonl").read_text(encoding="utf-8").splitlines()]
            rate = sorted(r["steps_per_second"] for r in records[1:])
            row["training_steps_per_second_median"] = rate[len(rate)//2] if rate else None
            row["training_steps"] = records[-1]["steps"]
            row["first_training_chunk_steps_per_second"] = records[0]["steps_per_second"]
        result["models"][name] = row
        copyfile(checkpoint, args.output / f"{prefix}{name}.npz")
        for log in ("training", "validation"):
            if (run / f"{log}.jsonl").exists():
                copyfile(run / f"{log}.jsonl", args.output / f"{prefix}{name}_{log}.jsonl")
        print(name, json.dumps(row["test"]), flush=True)
    if configs[0] != configs[1]:
        raise ValueError("Checkpoints must use the same environment configuration")
    params, _ = GamblersTaxiEnv.init_params(jax.random.key(args.seed), configs[0])
    for i, name in enumerate(("work_sleep", "conditional_gambling", "slots_first",
                              "blackjack_first", "bank_and_gamble", "mixed_expert")):
        result["baselines"][name] = evaluate_cohorts(None, params, seeds, strategy=i)
        print(name, json.dumps(result["baselines"][name]["all"]), flush=True)
    result["selected"] = max(val_rank, key=val_rank.get)
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        result["initial_mixed_experiment"] = previous["models"]
    if args.early:
        result["early_curriculum"] = {}
        for name, run in (("neutral", args.early), ("risk", args.early_risk)):
            if run is None:
                continue
            actor, config, meta = load_actor(run / "best.npz")
            params, _ = GamblersTaxiEnv.init_params(jax.random.key(args.seed), config)
            off = replace(params, config=replace(config, enable_emotion_effects=False))
            result["early_curriculum"][name] = {
                "checkpoint": meta, "test": evaluate_cohorts(actor, params, seeds),
                "emotion_off": evaluate_cohorts(actor, off, seeds),
            }
            copyfile(run / "best.npz", args.output / f"early_{name}.npz")
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"selected={result['selected']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--early", type=Path)
    parser.add_argument("--early-risk", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=128)
    compare(parser.parse_args())
