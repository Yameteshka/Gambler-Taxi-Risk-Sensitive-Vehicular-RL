"""Post-evaluation summaries; never used for fitting or checkpoint selection.

Run after the registered suite: python reports/analyze_study.py
Dependencies: NumPy; Matplotlib for figures. All inputs remain inspectable.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from statistics_report import lower_cvar


def interval(matrix, statistic="mean", seed=51451):
    """Crossed resampling of training repetitions and common scenario columns.

    Tail estimands are the mean of per-policy tails, not a pooled-mixture tail.
    """
    x = np.asarray(matrix, dtype=float)
    def estimate(a):
        if statistic == "cvar05":
            return float(lower_cvar(a).mean())
        if statistic == "p05":
            return float(np.quantile(a, .05, axis=-1).mean())
        return float(a.mean())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        rows = rng.integers(x.shape[0], size=x.shape[0])
        columns = rng.integers(x.shape[1], size=x.shape[1])
        samples.append(estimate(x[np.ix_(rows, columns)]))
    return {"estimate": estimate(x), "crossed_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "training_repetitions": x.shape[0], "matched_scenarios": x.shape[1]}


def tail_difference(a, b, seed=51611):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Unmatched tail comparison")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        r = rng.integers(a.shape[0], size=a.shape[0])
        c = rng.integers(a.shape[1], size=a.shape[1])
        samples.append(float((lower_cvar(a[np.ix_(r, c)]) - lower_cvar(b[np.ix_(r, c)])).mean()))
    return {"mean_per_policy_cvar05_difference": float((lower_cvar(a)-lower_cvar(b)).mean()),
            "paired_crossed_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "training_repetitions": a.shape[0], "scenarios": a.shape[1]}


def load_evaluations(rows):
    records, catalog, hashes = [], [], {}
    for row in rows:
        folder = ROOT / "artifacts/training" / row["run_id"]
        entry = {k: row[k] for k in ("run_id", "group", "algorithm", "architecture", "repetition",
                                    "initialization_seed", "training_horizon", "comparison_ids", "status")}
        entry["training_completed"] = (folder / "complete.json").exists()
        entry["policies"] = [str(p.relative_to(ROOT)).replace(chr(92), "/") for p in sorted(folder.glob("*.npz"))]
        if (folder / "selected.npz").exists():
            with np.load(folder / "selected.npz", allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
            entry["selected_at_transitions"] = metadata["physical_transitions"]
            hashes[str((folder / "selected.npz").relative_to(ROOT))] = hashlib.sha256((folder / "selected.npz").read_bytes()).hexdigest()
        catalog.append(entry)
        for path in sorted((ROOT / "artifacts/evaluation" / row["run_id"]).glob("*/complete.json")):
            case = path.parent.name
            audit_path = path.parent / "audit.json"
            if not audit_path.exists() or not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
                continue
            with np.load(path.parent / "endpoints.npz", allow_pickle=False) as data:
                arrays = {k: np.array(data[k]) for k in ("scenario_id", "external_return", "outcome", "cash", "car_debt", "micro_debt", "time", "counters")}
            result = json.loads((path.parent / "evaluation.json").read_text(encoding="utf-8"))
            probe = path.parent / "policy_probes.json"
            records.append({"row": row, "case": case, "data": arrays, "result": result,
                            "probes": json.loads(probe.read_text(encoding="utf-8")) if probe.exists() else {}})
    return records, catalog, hashes


def group_key(record):
    r = record["row"]
    return (r["group"], r["algorithm"], r["architecture"], r["training_horizon"], r["comparison_ids"], record["case"])


def main():
    with (ROOT / "experiments/registry.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    records, catalog, hashes = load_evaluations(rows)
    groups = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)
    summary = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda r: r["row"]["repetition"])
        for item in items[1:]:
            np.testing.assert_array_equal(items[0]["data"]["scenario_id"], item["data"]["scenario_id"])
        returns = np.stack([r["data"]["external_return"] for r in items])
        outcomes = np.stack([r["data"]["outcome"] for r in items])
        fields = ("group", "algorithm", "architecture", "horizon", "comparison_ids", "case")
        entry = dict(zip(fields, key))
        entry.update(return_mean=interval(returns), ownership=interval(outcomes == 1),
                     per_policy_p05=interval(returns, "p05"), per_policy_cvar05=interval(returns, "cvar05"),
                     collision_rate=float((outcomes == 2).mean()), starvation_rate=float((outcomes == 4).mean()),
                     censored_rate=float((outcomes == 0).mean()), deadline_rate=float((outcomes == 3).mean()),
                     mean_remaining_debt=float(np.mean([r["data"]["car_debt"]+r["data"]["micro_debt"] for r in items])),
                     mean_shifts=float(np.mean([r["result"]["paid"] for r in items])),
                     mean_gambles=float(np.mean([r["result"]["spins"]+r["result"]["hands"] for r in items])),
                     mean_meals=float(np.mean([r["result"]["meals"] for r in items])),
                     mean_sleeps=float(np.mean([r["result"]["sleeps"] for r in items])),
                     mean_loans=float(np.mean([r["result"]["loans"] for r in items])))
        followups = [r["probes"].get("loss_following", {}) for r in items]
        n = sum(p.get("eligible_follow_up_decisions", 0) for p in followups)
        wagers = sum(p.get("voluntary_wagers", 0) for p in followups)
        entry["loss_following"] = {"eligible_decisions": n, "voluntary_wagers": wagers,
                                    "descriptive_rate": wagers/n if n else None}
        summary.append(entry)
    tails = {}
    anchor = {r["row"]["repetition"]: r for r in records if r["row"]["group"] == "main"
              and r["row"]["algorithm"] == "hybrid_dsac" and r["row"]["training_horizon"] == "10d" and r["case"] == "standard"}
    for treatment in ("T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10", "T11"):
        selected = {r["row"]["repetition"]: r for r in records if treatment in r["row"]["comparison_ids"].split("|") and r["case"] == "standard"}
        common = sorted(anchor.keys() & selected.keys())
        if not common:
            continue
        for repetition in common:
            np.testing.assert_array_equal(selected[repetition]["data"]["scenario_id"], anchor[repetition]["data"]["scenario_id"])
        tails[treatment] = tail_difference(np.stack([selected[r]["data"]["external_return"] for r in common]),
                                           np.stack([anchor[r]["data"]["external_return"] for r in common]))
    out = ROOT / "reports"
    (out / "policy_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    payload = {"groups": summary, "paired_tail_comparisons": tails,
               "estimand": "Mean per-trained-policy statistic across independent initializations, with common scenario columns. Not the tail of a pooled policy mixture.",
               "warning": "Exploratory intervals; no multiplicity correction. Five initializations and about thirteen tail scenarios per policy limit precision.",
               "postprocessing_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashes}
    (out / "study_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Policy catalog and empirical analysis", "", f"Completed training: {sum(r['training_completed'] for r in catalog)}/{len(rows)}. Completed evaluation cases: {len(records)}.", "",
             
             "[Machine-readable policy catalog](policy_catalog.json) and [Estimates and uncertainty](study_analysis.json)", "",
             "| Algorithm / architecture | Horizon | Treatment | Case | Runs | Ownership | Mean return [95% CI] | Mean CVaR05 | Work / gambles / meals / sleep |", "|---|---|---|---|---:|---:|---|---:|---|"]
    for s in summary:
        lo, hi = s["return_mean"]["crossed_bootstrap_95ci"]
        lines.append(f"| {s['algorithm']} / {s['architecture']} | {s['horizon']} | {s['comparison_ids']} | {s['case']} | {s['return_mean']['training_repetitions']} | {s['ownership']['estimate']:.1%} | {s['return_mean']['estimate']:.4f} [{lo:.4f}, {hi:.4f}] | {s['per_policy_cvar05']['estimate']:.4f} | {s['mean_shifts']:.2f} / {s['mean_gambles']:.2f} / {s['mean_meals']:.2f} / {s['mean_sleeps']:.2f} |")
    lines += ["", payload["estimand"], "", payload["warning"], "",
              "Zero loss-following opportunities means the behavior was not observed; it is not a measured zero propensity to chase losses. A higher gambling count is not by itself evidence of addiction. Policy sensitivity probes do not reveal thoughts."]
    (out / "study_analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plots(records, summary)
    print(f"Analyzed {len(records)} completed cases; catalog contains {len(catalog)} registered runs.")


def plots(records, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not records:
        return
    target = ROOT / "media/figures"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for algorithm, architecture in (("hybrid_dsac", "mlp"), ("hybrid_ppo", "mlp"), ("hybrid_ppo", "gru")):
        selected = [r for r in records if r["row"]["group"] == "main" and r["row"]["training_horizon"] == "10d" and r["row"]["algorithm"] == algorithm and r["row"]["architecture"] == architecture and r["case"] == "standard"]
        if not selected:
            continue
        label = f"{algorithm} / {architecture}"
        for r in selected:
            x = np.sort(r["data"]["external_return"])
            axes[0].step(x, np.arange(1, len(x)+1)/len(x), where="post", alpha=.25, linewidth=1)
        x = np.sort(np.concatenate([r["data"]["external_return"] for r in selected]))
        axes[0].step(x, np.arange(1, len(x)+1)/len(x), where="post", label=label)
        s = next(s for s in summary if s["group"] == "main" and s["horizon"] == "10d" and s["algorithm"] == algorithm and s["architecture"] == architecture and s["case"] == "standard")
        i = len(axes[1].lines)
        est = s["per_policy_cvar05"]["estimate"]
        lo, hi = s["per_policy_cvar05"]["crossed_bootstrap_95ci"]
        axes[1].errorbar(est, i, xerr=[[max(0., est-lo)], [max(0., hi-est)]], fmt="o", label=label)
    axes[0].set(xlabel="Discounted external return", ylabel="Empirical cumulative probability", title="Ten-day return distributions")
    axes[1].set(xlabel="Mean per-policy lower CVaR05", yticks=[], title="Tail uncertainty across runs and scenarios")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(target / "return_tails.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
