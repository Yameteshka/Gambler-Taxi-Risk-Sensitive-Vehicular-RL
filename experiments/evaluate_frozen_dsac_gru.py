"""Evaluate the selected DSAC GRU on the original, registered final conditions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/study"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(source / "src"), str(source / "experiments")]

    import jax
    import numpy as np
    import debt500_study
    import driver_diagnostics
    import models
    from audit_results import audit
    from env import resource_taxi as env
    from evaluate import evaluate

    version = read_json(STUDY / "data/cohort/freeze.json")
    for filename, expected in version["sources"].items():
        # The publication archive refreshes the report templates.
        if filename == "src/build_reports.py":
            continue
        if sha256(source / filename) != expected:
            raise ValueError(f"Frozen source differs from original: {filename}")
    wrapper = source / "experiments/resource_study.py"
    if sha256(wrapper) != version["wrapper_sha256"]:
        raise ValueError("Frozen runner wrapper differs from the original")

    selected = next(item for item in read_json(STUDY / "checkpoints/manifest.json")
                    if item["model"] == "dsac_gru")
    checkpoint = STUDY / selected["path"]
    if sha256(checkpoint) != selected["sha256"]:
        raise ValueError("DSAC GRU checkpoint hash differs from the study record")
    model, metadata = models.load_policy(checkpoint)
    kind = metadata["kind"]
    config = env.Config(**metadata["evaluation_environment"])
    variants = list(debt500_study.variants(config))
    with (STUDY / "data/frozen_conditions.csv").open(newline="", encoding="utf-8") as stream:
        registry = list(csv.DictReader(stream))
    if len(variants) != len(registry) or len(variants) != 30:
        raise ValueError("Frozen runner and final condition register differ in size")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"NVIDIA GPU required, found {jax.devices()}")

    banks = {"test_standard": 3_000_001}
    cases = []
    for row, (label, changed, bank, controller) in zip(registry, variants):
        if (label, bank, controller) != (row["case"], row["bank"], row["controller"]):
            raise ValueError(f"Frozen runner differs from the register at {row['condition']}")
        if bank not in banks:
            banks[bank] = 3_100_001 + 100_000 * (len(banks) - 1)
        cases.append({"condition": row["condition"], "case": label, "bank": bank,
                      "scenario_start": banks[bank], "episodes": int(row["planned_per_policy"]),
                      "controller": controller, "deterministic": label == "deterministic"})
    manifest = {"checkpoint_sha256": selected["sha256"], "policy": kind,
                "source_sha256": sha256(STUDY / "checkpoints/frozen_source.zip"),
                "cases": cases}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        archived_hash = read_json(STUDY / "data/final_dsac_gru/manifest.json")["source_sha256"]
        source_matches = existing["source_sha256"] in (manifest["source_sha256"], archived_hash)
        if not source_matches or {**existing, "source_sha256": manifest["source_sha256"]} != manifest:
            raise ValueError("Existing output belongs to a different evaluation")
    else:
        write_json(manifest_path, manifest)
    print(f"GPU {jax.devices()} | {len(cases)} conditions | {sum(c['episodes'] for c in cases)} episodes",
          flush=True)

    rows = []
    for case, (_, changed, _, _) in zip(cases, variants):
        label = case["case"]
        start = case["scenario_start"]
        ids = list(range(start, start + case["episodes"]))
        folder = output / label
        begun = time.monotonic()
        write_json(output / "status.json", {"state": "running", "completed": len(rows), "current": case})
        print(f"{case['condition']} {label} [{ids[0]}–{ids[-1]}]", flush=True)
        if not (folder / "complete.json").is_file():
            if folder.exists():
                folder.rename(folder.with_name(f"{label}.interrupted-{time.time_ns()}"))
            params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), changed)
            evaluate(model, kind, params, ids, folder, deterministic=case["deterministic"],
                     controller=case["controller"], interface="strategy")
        result = read_json(folder / "evaluation.json")
        if (result["scenario_ids"] != ids or result["kind"] != kind
                or result["deterministic"] != case["deterministic"]):
            raise ValueError(f"Output failed registered scenario checks: {label}")
        with np.load(folder / "endpoints.npz", allow_pickle=False) as endpoint:
            np.testing.assert_array_equal(endpoint["scenario_id"], ids)
            if int((endpoint["outcome"] == env.SUCCESS).sum()) != result["ownership"]:
                raise ValueError(f"Outcome summary disagrees with endpoints: {label}")
        finished_eval = time.monotonic()
        print(f"  evaluated: {result['ownership']}/{len(ids)} ownership; {finished_eval - begun:.1f}s",
              flush=True)
        audit_path = folder / "audit.json"
        if not audit_path.is_file():
            audit(folder, checkpoint)
        receipt = read_json(audit_path)
        if not receipt["passed"] or receipt["checkpoint_sha256"] != selected["sha256"]:
            raise ValueError(f"Replay audit failed or used a different checkpoint: {label}")
        diagnostic_path = folder / "driver_diagnostics.json"
        diagnostic = read_json(diagnostic_path) if diagnostic_path.is_file() else driver_diagnostics.diagnose(folder)
        row = {**case, **{key: result[key] for key in (
            "ownership", "starvation", "collision", "deadline", "censored", "success_rate",
            "mean_external_return", "mean_remaining_debt", "paid", "loans", "meals", "refills",
            "sleeps", "spins", "hands")}, "flagged_driver_stalls": diagnostic["flagged_episodes"],
            "elapsed_seconds": round(time.monotonic() - begun, 2)}
        rows.append(row)
        pending = output / "summary.tmp"
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(output / "summary.csv")
        print(f"  audited: {len(rows)}/{len(cases)} conditions; {row['elapsed_seconds']:.1f}s", flush=True)
    write_json(output / "status.json", {"state": "complete", "completed": len(rows)})


if __name__ == "__main__":
    main()
