"""Execute registered training and disjoint frozen-policy comparisons locally."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import traceback
import importlib.metadata
import platform

import jax

from env import resource_taxi as env
from evaluate import evaluate
import models
import protocol
from policy_probes import generate
from critic_certificate import analyze as critic_certificate
from audit_results import audit
from telemetry import file_hash
from strategy_training import train


def variants(config):
    yield "standard", config, "test_standard", "learned"
    yield "deterministic", config, "test_standard", "learned"
    for days, label in ((5, "5d"), (10, "10d"), (0, "open")):
        yield label, replace(config, days=days), "test_standard", "learned"
    for label, changes, bank in (
        ("weakened_control", {"control_strength": .5}, "test_standard"),
        ("actuation_noise", {"actuation_noise": .15}, "test_standard"),
        ("low_fuel", {"starting_fuel": 1.}, "test_low_fuel"),
        ("low_satiety", {"starting_satiety": .25}, "test_low_satiety"),
        ("low_liquidity", {"starting_cash": 10.}, "test_low_liquidity"),
        ("low_income", {"salary": config.salary*.65}, "test_low_income"),
        ("fuel_price_shock", {"fuel_price_multiplier": 2.}, "test_fuel_price_shock"),
        ("personal_debt_start", {"starting_microloan": 75.}, "test_personal_debt"),
        ("adverse_combination", {"starting_fuel": 1., "starting_satiety": .25,
            "starting_cash": 10., "starting_microloan": 75., "salary": config.salary*.65}, "test_adverse_combination"),
        ("meal_mean_only", {"meals": "meal_mean_only"}, "test_standard"),
        ("meal_high_variance", {"meals": "meal_high_variance"}, "test_standard"),
        ("food_site_identity_swap", {"food_swap": True}, "test_standard")):
        if bank != "test_standard":
            yield label+"_control", config, bank, "learned"
        yield label, replace(config, **changes), bank, "learned"
    yield "driving_guard", config, "test_standard", "guard"


def freeze():
    design = protocol.load()
    payload = {"design": design, "sources": protocol.source_hashes(),
        "registered_conditions": [{k: v for k, v in row.items() if k != "status"} for row in protocol.registry()],
        "dependencies_sha256": file_hash(protocol.ROOT/"requirements.txt"),
        "device": str(jax.devices()), "initial_conditions": asdict(env.Config()),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    path = protocol.ROOT/"experiments/freeze.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (previous["sources"] != payload["sources"] or previous["design"] != design or
                previous.get("registered_conditions") != payload["registered_conditions"]):
            raise RuntimeError("Frozen source or design changed; register a protocol revision")
    else:
        for name in ("environment_verification", "strategy_verification", "learning_verification"):
            gate = json.loads((protocol.ROOT/f"experiments/{name}.json").read_text(encoding="utf-8"))
            if gate["source_hashes"] != payload["sources"]:
                raise RuntimeError(f"Repeat {name} for the current source before freezing")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (protocol.ROOT/"experiments/runtime-requirements.txt").write_text(
            chr(10).join(f"{name}=={version}" for name, version in sorted(payload["packages"].items()))+chr(10), encoding="utf-8")


def evaluate_run(row, full=False):
    source = protocol.ROOT/"artifacts/training"/row["run_id"]/"selected.npz"
    model, metadata = models.load_policy(source)
    config = env.Config(**metadata.get("evaluation_environment", metadata["environment"]))
    design = protocol.load()
    cases = list(variants(config)) if full and row["training_horizon"] == "10d" and row["group"] == "main" else list(variants(config))[:2]
    for label, changed, bank_name, controller in cases:
        output = protocol.ROOT/"artifacts/evaluation"/row["run_id"]/label
        if (output/"complete.json").exists():
            if not (output/"audit.json").exists():
                audit(output, source)
            if label == "standard" and not (output/"policy_probes.json").exists():
                generate(source, output)
            if label == "standard" and metadata["kind"].startswith("dsac") and not (output/"critic_equivalence.json").exists():
                snapshot = sorted(source.parent.glob("learner_*.pkl.gz"))[-1]
                critic_certificate(snapshot, output/"critic_equivalence.json")
            continue
        bank = design["scenario_banks"][bank_name]
        params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), changed)
        result, _, _ = evaluate(model, metadata["kind"], params, range(bank["first"], bank["last"]+1),
                                output, controller=controller, deterministic=label == "deterministic", interface=metadata.get("interface","physical"))
        (output/"provenance.json").write_text(json.dumps({"checkpoint_sha256": file_hash(source),
            "source_hashes": protocol.source_hashes(), "case": label, "bank": bank_name}, indent=2), encoding="utf-8")
        print(f"EVAL {row['run_id']} {label}: {result['ownership']}/{result['episodes']}", flush=True)
        audit(output, source)
        if label == "standard":
            generate(source, output)
            if metadata["kind"].startswith("dsac"):
                snapshot = sorted(source.parent.glob("learner_*.pkl.gz"))[-1]
                critic_certificate(snapshot, output/"critic_equivalence.json")
    protocol.set_run_status(row["run_id"], "evaluated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--run-id")
    parser.add_argument("--full-frozen-suite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    freeze()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be within shard-count")
    # Interleave algorithms and repetitions on the ten-day main task first.
    # This changes queue order only; every registered condition is preserved.
    rows = sorted(protocol.registry(), key=lambda r:(r["group"] != "main",
        {"10d":0,"5d":1,"open":2}[r["training_horizon"]],r["repetition"],
        r["algorithm"] != "hybrid_dsac",r["architecture"] != "mlp",r["run_id"]))
    for index, row in enumerate(rows):
        if index % args.shard_count != args.shard_index:
            continue
        if args.run_id and row["run_id"] != args.run_id:
            continue
        folder = protocol.ROOT/"artifacts/training"/row["run_id"]
        try:
            if args.phase != "evaluate" and not (folder/"complete.json").exists():
                if shutil.disk_usage(protocol.ROOT).free < 4*1024**3:
                    raise RuntimeError("Less than 4 GiB free; preserve existing logs before continuing")
                train(row, resume=(folder/"resume.pkl.gz").exists())
            if args.phase != "train":
                evaluate_run(row, args.full_frozen_suite)
            if args.shard_index == 0:
                from build_reports import main as build_reports
                build_reports()
        except Exception:
            protocol.set_run_status(row["run_id"], "failed")
            text = traceback.format_exc()
            (protocol.ROOT/"experiments/last_failure.txt").write_text(text, encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
