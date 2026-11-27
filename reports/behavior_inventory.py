"""Describe all audited standard trajectories and flag cases for investigation.

Flags describe observed actions, not a diagnosis of motives or learning bugs.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def inspect(folder):
    metadata = json.loads((folder/"evaluation.json").read_text(encoding="utf-8"))
    ids = metadata["scenario_ids"]
    strategic = metadata.get("interface") == "strategy"
    data = {s: {"scenario_id": s, "active_steps": 0, "moving_steps": 0, "free_stationary_steps": 0,
                "braking_at_rest": 0, "drive_decisions": 0, "meals": 0, "shifts": 0,
                "gambles": 0, "sleeps": 0, "loans_opened": 0, "car_payments": 0,
                "micro_payments": 0, "service_decisions": 0, "voluntary_wait_decisions": 0,
                "travel_decisions": 0, "position_min": [float("inf")]*2,
                "position_max": [float("-inf")]*2} for s in ids}
    food = []
    # Public-feature offsets are checked against the active explicit schema.
    # No future realized portion is used to rank an available alternative.
    for path in sorted(folder.glob("trace_*.npz")):
        with np.load(path, allow_pickle=False) as d:
            active = d["active"]
            for lane, scenario in enumerate(ids):
                m = active[:, lane]
                if not m.any():
                    continue
                record = data[scenario]
                speed, command = d["speed"][:, lane], d["executed"][:, lane]
                record["active_steps"] += int(m.sum())
                record["moving_steps"] += int((m & (speed > .05)).sum())
                record["drive_decisions"] += int((m & (command == 0)).sum())
                record["free_stationary_steps"] += int((m & (speed <= .05) & (d["activity"][:, lane] == 0)).sum())
                record["braking_at_rest"] += int((m & (speed <= .001) & (command == 0) & (d["applied_control"][:, lane, 0] < 0)).sum())
                if strategic:
                    decision = m & d["decision_start"][:, lane]
                    record["service_decisions"] += int(decision.sum())
                    voluntary = decision & (d["macro_executed"][:, lane] == 0) & (d["activity"][:, lane] == 0)
                    record["voluntary_wait_decisions"] += int(voluntary.sum())
                    record["travel_decisions"] += int((decision & (d["macro_executed"][:, lane] != 0)).sum())
                for field, index in (("shifts", 0), ("loans_opened", 2), ("car_payments", 3),
                                     ("micro_payments", 4), ("sleeps", 10), ("meals", 11)):
                    record[field] += int(d["events"][m, lane, index].sum())
                record["gambles"] += int(d["events"][m, lane, 5:7].sum())
                record["position_min"] = np.minimum(record["position_min"], d["position"][m, lane].min(axis=0)).tolist()
                record["position_max"] = np.maximum(record["position_max"], d["position"][m, lane].max(axis=0)).tolist()
            if strategic:
                for tick, lane in np.argwhere(active & (d["executed"] == 4)):
                    food.append({"scenario_id": ids[lane], "tick": int(d["time"][tick,lane]),
                        "physical_venue": int(d["meal_venue"][tick,lane]),
                        "cash_before_order": float(d["cash_before"][tick,lane]),
                        "satiety_before_order": float(d["satiety_before"][tick,lane]),
                        "paid_price": float(d["service_price"][tick,lane]),
                        "scope":"Actual physical purchase receipt. Public inputs at intention selection are retained in decision_*.npz; no future realized portion ranks an alternative."})
            elif "observation" in d:
                for tick, lane in np.argwhere(active & (d["executed"] == 4)):
                    observation = d["observation"][tick, lane]
                    venue = int(d["meal_venue"][tick, lane])
                    food.append({"scenario_id": ids[lane], "tick": int(d["time"][tick, lane]),
                        "physical_venue": venue, "cash_before_order": float(observation[68]*1000),
                        "satiety_before_order": float(observation[74]),
                        "public_food_quotes": (observation[97:100]*50).tolist(),
                        "public_route_distances_to_food": (observation[43:46]*50).tolist(),
                        "scope": "Public decision inputs. Quotes can change during travel."})
            if "loan_receipt_location" in d:
                sums = np.zeros(active.shape, dtype=np.float64)
                locations = d["loan_receipt_location"]
                np.add.at(sums, (locations[:, 0], locations[:, 1]), d["loan_receipt_amount"])
                np.testing.assert_allclose(sums, d["micro_payment"], rtol=2e-6, atol=2e-5)
                assert np.all(d["loan_receipt_id"] > 0)
            else:
                assert not np.any(d["micro_payment"] > 0), "Missing individual repayment receipts"
    with np.load(folder/"endpoints.npz", allow_pickle=False) as endpoints:
        for lane, scenario in enumerate(ids):
            record = data[scenario]
            record["outcome"] = int(endpoints["outcome"][lane])
            record["position_span"] = float(np.linalg.norm(np.subtract(record["position_max"], record["position_min"])))
            record["moving_fraction"] = record["moving_steps"]/max(record["active_steps"], 1)
            record["flags"] = []
            if record["outcome"] == 4 and record["position_span"] < 1 and record["shifts"] == 0 and record["gambles"] == 0:
                record["flags"].append("Stationary starvation without work or wagering: inspect learning and feasibility; do not label a deliberate strategy.")
            if not strategic and record["drive_decisions"] and record["braking_at_rest"]/record["drive_decisions"] > .9:
                record["flags"].append("Over 90% of Drive decisions apply braking at rest: inspect policy means and critic preferences.")
            if strategic and record["drive_decisions"] and record["braking_at_rest"]/record["drive_decisions"] > .9:
                record["flags"].append("Shared driver applies braking at rest on over 90% of physical Drive commands: investigate route blocking or controller limitations before attributing inactivity to the economic policy.")
            np.testing.assert_equal(record["active_steps"], int(endpoints["time"][lane]))
    return {"run_id": folder.parent.name, "case": folder.name, "episodes": list(data.values()), "food_orders": food,
            "flags_are": "Trajectory flags for repeated actions and resource depletion."}


def main():
    records = []
    for audit in sorted((ROOT/"artifacts/evaluation").glob("*/*/audit.json")):
        if audit.parent.name != "standard" or not json.loads(audit.read_text(encoding="utf-8")).get("passed"):
            continue
        records.append(inspect(audit.parent))
    (ROOT/"reports/behavior_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = ["# Behavior and sanity-check inventory", "", 
            "| Policy | Episodes | Mean moving fraction | Flagged episodes | Work | Wagers | Meals | Sleep | New loans |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        episodes = r["episodes"]
        values = [np.mean([e[k] for e in episodes]) for k in ("shifts", "gambles", "meals", "sleeps", "loans_opened")]
        rows.append(f"| {r['run_id']} | {len(episodes)} | {np.mean([e['moving_fraction'] for e in episodes]):.1%} | {sum(bool(e['flags']) for e in episodes)} | "+" | ".join(f"{v:.2f}" for v in values)+" |")
    rows += ["", "[Episode-level evidence, public meal-decision inputs and flags](behavior_inventory.json)", "",
             "Moving fraction excludes parked service time only through its explicit speed definition; service-related standing is expected. A stationary death and an early crash are investigated rather than assigned psychological explanations."]
    (ROOT/"reports/behavior_inventory.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(f"Inspected {len(records)} audited standard policy evaluations.")


if __name__ == "__main__":
    main()
