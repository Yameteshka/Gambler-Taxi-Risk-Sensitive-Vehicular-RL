"""Finite-sample uncertainty; bootstrap units are scenarios and training repetitions."""

import argparse
import json
import csv
from pathlib import Path

import numpy as np


def wilson(successes, n, z=1.959963984540054):
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def lower_cvar(values, mass=.05):
    """Mean of the lowest mass, using a fractional boundary observation."""
    x = np.sort(np.asarray(values), axis=-1)
    count = x.shape[-1]*mass
    whole = int(count)
    return (x[..., :whole].sum(axis=-1)+(count-whole)*x[..., whole])/count


def bootstrap(values, statistic=np.mean, repetitions=2000, seed=70031):
    x = np.asarray(values)
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(x.size, size=(repetitions, x.size))]
    estimates = statistic(sampled, axis=-1)
    return np.quantile(estimates, [.025, .975]).tolist()


def uncertainty(returns, outcomes):
    x = np.asarray(returns)
    return {"n": len(x), "success_rate_95ci": wilson(int((np.asarray(outcomes) == 1).sum()), len(x)),
            "mean_return_95ci": bootstrap(x), "p05": float(np.quantile(x, .05)),
            "p05_95ci": bootstrap(x, lambda a, axis: np.quantile(a, .05, axis=axis)),
            "lower_cvar05": float(lower_cvar(x)),
            "lower_cvar05_95ci": bootstrap(x, lambda a, axis: lower_cvar(a)),
            "effective_tail_observations": float(.05*len(x)),
            "interpretation": "Conditional scenario-sample uncertainty for this trained policy; not a proof or training-seed uncertainty."}


def paired_hierarchical(a, b, repeats=2000):
    """Matched [training repetition, scenario] outcomes, crossed resampling."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Provide matched repetition-by-scenario arrays")
    rng = np.random.default_rng(81131)
    delta = a-b
    estimates = []
    for _ in range(repeats):
        rows = rng.integers(delta.shape[0], size=delta.shape[0])
        columns = rng.integers(delta.shape[1], size=delta.shape[1])
        estimates.append(delta[np.ix_(rows, columns)].mean())
    return {"mean_difference": float(delta.mean()), "paired_crossed_bootstrap_95ci": np.quantile(estimates, [.025, .975]).tolist(),
            "training_repetitions": a.shape[0], "scenarios": a.shape[1]}


def registered_comparisons(root):
    root = Path(root)
    with (root/"experiments/registry.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    def select(algorithm, architecture="mlp", horizon="10d", comparison=None):
        return {r["repetition"]: r for r in rows if r["algorithm"] == algorithm and
            r["architecture"] == architecture and r["training_horizon"] == horizon and
            ((comparison in r["comparison_ids"].split("|")) if comparison else r["group"] == "main")}

    def compare(left, right, left_case="standard", right_case="standard"):
        pairs = []
        for repetition in sorted(left.keys() & right.keys()):
            paths = [root/"artifacts/evaluation"/r["run_id"]/case/"endpoints.npz"
                     for r, case in ((left[repetition], left_case), (right[repetition], right_case))]
            if not all(p.exists() for p in paths):
                continue
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                np.testing.assert_array_equal(a["scenario_id"], b["scenario_id"])
                pairs.append(((a["outcome"] == 1).copy(), (b["outcome"] == 1).copy(),
                              a["external_return"].copy(), b["external_return"].copy()))
        if not pairs:
            return {"status": "pending", "matched_training_repetitions": 0}
        return {"status": "complete" if len(pairs) == 5 else "partial",
                "ownership_difference": paired_hierarchical(np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])),
                "return_difference": paired_hierarchical(np.stack([p[2] for p in pairs]), np.stack([p[3] for p in pairs]))}

    anchor = select("hybrid_dsac")
    result = {"M01_dsac_minus_ppo": compare(anchor, select("hybrid_ppo")),
              "M02_ppo_gru_minus_mlp": compare(select("hybrid_ppo", "gru"), select("hybrid_ppo"))}
    for horizon in ("5d", "open"):
        for algorithm, architecture in (("hybrid_dsac", "mlp"), ("hybrid_ppo", "mlp"), ("hybrid_ppo", "gru")):
            result[f"H01_{algorithm}_{architecture}_{horizon}_minus_10d"] = compare(
                select(algorithm, architecture, horizon), select(algorithm, architecture))
    for comparison in ("T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10", "T11"):
        result[comparison+"_treatment_minus_reference"] = compare(
            select("hybrid_dsac", "gru" if comparison == "T06" else "mlp", comparison=comparison), anchor)
    for comparison in ("X01", "X02"):
        for architecture in ("mlp", "gru"):
            public = anchor if architecture == "mlp" else select("hybrid_dsac", "gru", comparison="T06")
            result[f"{comparison}_{architecture}"] = compare(
                select("hybrid_dsac", architecture, comparison=comparison), public)
    result["X03_accelerated_minus_standard_calendar"] = compare(
        select("hybrid_dsac", horizon="5d", comparison="X03"), select("hybrid_dsac", horizon="5d"))
    for algorithm, architecture in (("hybrid_dsac", "mlp"), ("hybrid_ppo", "mlp"), ("hybrid_ppo", "gru")):
        selected = select(algorithm, architecture)
        for case in ("weakened_control", "actuation_noise", "driving_guard", "low_fuel", "low_satiety",
                     "low_liquidity", "low_income", "fuel_price_shock", "personal_debt_start", "adverse_combination",
                     "meal_mean_only", "meal_high_variance", "food_site_identity_swap", "5d", "open"):
            control = case+"_control" if case in ("low_fuel", "low_satiety", "low_liquidity", "low_income",
                "fuel_price_shock", "personal_debt_start", "adverse_combination") else "standard"
            result[f"frozen_{algorithm}_{architecture}_{case}"] = compare(selected, selected, case, control)
    (root/"reports/comparisons.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    with np.load(args.evaluation/"endpoints.npz") as data:
        result = uncertainty(data["external_return"], data["outcome"])
    (args.evaluation/"uncertainty.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
