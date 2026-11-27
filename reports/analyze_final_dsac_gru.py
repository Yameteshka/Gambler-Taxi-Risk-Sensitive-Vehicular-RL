"""Extract and reconcile the audited final evaluation and its $1,000 buyouts."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/dsac_gru_final"
OUTPUT = ROOT / "reports/study/data/final_dsac_gru"
OUTCOME_OWNERSHIP = 1
FIELDS = ("paid", "missed", "loans", "car_payments", "micro_payments",
          "spins", "hands", "wins", "losses", "pushes", "sleeps", "meals",
          "refills", "fees")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def casino_intentions(folder, metadata):
    names = metadata["macro_names"]
    indexes = [i for i, name in enumerate(names) if name.startswith(("Slots ", "Blackjack "))]
    count = 0
    episodes = set()
    for path in sorted(folder.glob("decision_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            requested = data["active"] & np.isin(data["choice"], indexes)
            count += int(requested.sum())
            episodes.update(int(i) for i in data["scenario_id"][requested])
    return count, len(episodes)


def collect_field(traces, field):
    return np.concatenate([trace[field] for trace in traces])


def reconcile_buyouts(folder, metadata):
    with np.load(folder / "endpoints.npz", allow_pickle=False) as endpoint:
        winners = np.flatnonzero(endpoint["outcome"] == OUTCOME_OWNERSHIP)
        scenario_ids = endpoint["scenario_id"][winners]
        final_ticks = endpoint["time"][winners]
        final_cash = endpoint["cash"][winners]
        counters = endpoint["counters"][winners]
    traces = [[] for _ in winners]
    fields = ("time", "wages", "food_spent", "fuel_spent", "car_payment",
              "micro_payment", "borrowed_total", "gambling_net", "fees_total",
              "interest_total", "cash", "car_debt", "micro_debt",
              "car_cleared_at", "all_cleared_at")
    for path in sorted(folder.glob("trace_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            active = data["active"][:, winners]
            values = {field: data[field][:, winners] for field in fields}
            for lane, trace in enumerate(traces):
                mask = active[:, lane]
                if mask.any():
                    trace.append({key: value[:, lane][mask] for key, value in values.items()})

    rows, events = [], []
    config = metadata["config"]
    for lane, trace in enumerate(traces):
        v = {field: collect_field(trace, field) for field in fields}
        tick = v["time"]
        np.testing.assert_array_equal(tick, np.arange(1, final_ticks[lane] + 1))
        assert not v["car_debt"][-1] and not v["micro_debt"][-1]
        np.testing.assert_equal(v["cash"][-1], final_cash[lane])
        sid = int(scenario_ids[lane])
        count = dict(zip(FIELDS, map(int, counters[lane])))
        increments = {name: np.diff(v[name], prepend=0) for name in (
            "wages", "food_spent", "fuel_spent", "borrowed_total", "fees_total")}
        shifts = increments["wages"][increments["wages"] > 0]
        np.testing.assert_equal(shifts.size, count["paid"])
        payments = float(v["car_payment"].sum(dtype=np.float64))
        micro_payments = float(v["micro_payment"].sum(dtype=np.float64))
        gambling = float(v["gambling_net"].sum(dtype=np.float64))
        cash_residual = (config["starting_cash"] + v["wages"][-1] + v["borrowed_total"][-1]
                         + gambling - v["food_spent"][-1] - v["fuel_spent"][-1]
                         - payments - micro_payments - final_cash[lane])
        debt_residual = config["starting_debt"] + v["interest_total"][-1] + v["fees_total"][-1] - payments
        if abs(cash_residual) > .02 or (not count["loans"] and abs(debt_residual) > .02):
            raise ValueError(f"S28 cash or debt does not reconcile for {sid}")
        average_wage = config["salary"] * 1.03
        rows.append({
            "scenario_id": sid, "ownership_tick": int(tick[-1]),
            "ownership_day": int((tick[-1] - 1) // config["day"] + 1),
            "all_debt_cleared_tick": int(v["all_cleared_at"][-1]),
            "shifts": count["paid"], "shift_wages": "/".join(f"{n:.2f}" for n in shifts),
            "total_wages": float(v["wages"][-1]), "food_spent": float(v["food_spent"][-1]),
            "fuel_spent": float(v["fuel_spent"][-1]), "car_payments": round(payments, 4),
            "interest": float(v["interest_total"][-1]), "fees": float(v["fees_total"][-1]),
            "meals": count["meals"], "refills": count["refills"], "sleeps": count["sleeps"],
            "loans": count["loans"], "slots": count["spins"], "blackjack": count["hands"],
            "gambling_net": gambling, "cash_at_ownership": float(final_cash[lane]),
            "same_cost_cash_at_flat_base_wage": round(float(final_cash[lane] + count["paid"] * config["salary"] - v["wages"][-1]), 4),
            "same_cost_cash_at_expected_wage": round(float(final_cash[lane] + count["paid"] * average_wage - v["wages"][-1]), 4),
            "cash_residual": round(float(cash_residual), 6),
        })
        event_values = {
            "shift paid": increments["wages"], "meal purchased": increments["food_spent"],
            "fuel purchased": increments["fuel_spent"], "car payment": v["car_payment"],
            "late fee": increments["fees_total"], "new loan": increments["borrowed_total"],
            "personal loan payment": v["micro_payment"], "gambling settlement": v["gambling_net"],
        }
        for event, amounts in event_values.items():
            for i in np.flatnonzero(np.abs(amounts) > .001):
                events.append({"scenario_id": sid, "tick": int(tick[i]),
                               "day": int((tick[i] - 1) // config["day"] + 1), "event": event,
                               "amount": round(float(amounts[i]), 4),
                               "cash": round(float(v["cash"][i]), 4),
                               "car_debt": round(float(v["car_debt"][i]), 4)})
        events.append({"scenario_id": sid, "tick": int(tick[-1]),
                       "day": int((tick[-1] - 1) // config["day"] + 1), "event": "ownership",
                       "amount": 0.0, "cash": round(float(v["cash"][-1]), 4), "car_debt": 0.0})
    return rows, sorted(events, key=lambda row: (row["scenario_id"], row["tick"]))


def main():
    status = load(RUN / "status.json")
    manifest = load(RUN / "manifest.json")
    with (RUN / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    assert status == {"state": "complete", "completed": 30}
    assert [row["condition"] for row in summary] == [case["condition"] for case in manifest["cases"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    behavior, result_sets = [], {}
    failure_choices = {}
    active_ticks, exact_audits, starts = 0, 0, 0
    for case, row in zip(manifest["cases"], summary):
        folder = RUN / case["case"]
        audit = load(folder / "audit.json")
        assert audit["passed"] and audit["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        active_ticks += audit["active_physical_steps_replayed"]
        exact_audits += audit["maximum_logged_float_error"] == 0
        evaluation = load(folder / "evaluation.json")
        with np.load(folder / "endpoints.npz", allow_pickle=False) as endpoint:
            ids = endpoint["scenario_id"].copy()
            counters = endpoint["counters"].copy()
            won = endpoint["outcome"] == OUTCOME_OWNERSHIP
            np.testing.assert_array_equal(ids, np.arange(case["scenario_start"], case["scenario_start"] + case["episodes"]))
            assert int(won.sum()) == int(row["ownership"])
            assert np.all(endpoint["car_debt"][won] == 0)
            assert np.all(endpoint["micro_debt"][won] == 0)
        requests, request_episodes = casino_intentions(folder, evaluation)
        inherited_loan = int(evaluation["config"]["starting_microloan"] > 0)
        new_loan_counts = counters[:, 2] - inherited_loan
        assert np.all(new_loan_counts >= 0)
        behavior.append({
            "condition": case["condition"], "case": case["case"], "episodes": len(ids),
            "ownership": int(won.sum()), "casino_wager_intentions": requests,
            "episodes_with_casino_intention": request_episodes,
            **dict(zip(FIELDS, map(int, counters.sum(axis=0)))),
            "episodes_with_started_wager": int((counters[:, 5:7].sum(axis=1) > 0).sum()),
            "inherited_loans": inherited_loan * len(ids),
            "new_loans": int(new_loan_counts.sum()),
            "episodes_with_new_loan": int((new_loan_counts > 0).sum()),
        })
        starts += len(ids)
        result_sets[case["case"]] = (ids, won)
        portable = OUTPUT / case["condition"]
        portable.mkdir(exist_ok=True)
        for name in ("evaluation.json", "uncertainty.json", "audit.json", "driver_diagnostics.json", "endpoints.npz"):
            shutil.copyfile(folder / name, portable / name)
        if case["case"] in ("low_fuel", "adverse_combination"):
            choice_counts = np.zeros(len(evaluation["macro_names"]), dtype=np.int64)
            for path in sorted(folder.glob("decision_*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    choice_counts += np.bincount(data["choice"][data["active"]], minlength=len(choice_counts))
            failure_choices[case["case"]] = {
                "choices": dict(zip(evaluation["macro_names"], map(int, choice_counts))),
                "episodes_with_paid_shift": int((counters[:, 0] > 0).sum()),
                "episodes_with_meal": int((counters[:, 11] > 0).sum()),
                "episodes_with_refill": int((counters[:, 12] > 0).sum()),
            }

    paired = []
    for case in manifest["cases"]:
        label = case["case"]
        if label == "standard" or (case["bank"] != "test_standard" and label.endswith("_control")):
            continue
        control = "standard" if case["bank"] == "test_standard" else label + "_control"
        control_ids, baseline = result_sets[control]
        ids, outcomes = result_sets[label]
        np.testing.assert_array_equal(ids, control_ids)
        improvement = int((outcomes & ~baseline).sum())
        decline = int((baseline & ~outcomes).sum())
        paired.append({"condition": case["condition"], "case": label, "control": control,
                       "episodes": len(ids), "case_ownership": int(outcomes.sum()),
                       "control_ownership": int(baseline.sum()), "treatment_only": improvement,
                       "control_only": decline,
                       "difference_percentage_points": round(100 * (improvement - decline) / len(ids), 4)})

    assert len(paired) == 22
    target = RUN / "initial_car_debt_1000"
    winners, events = reconcile_buyouts(target, load(target / "evaluation.json"))
    assert len(winners) == len([e for e in events if e["event"] == "ownership"])
    assert len(winners) == int(next(row for row in summary if row["condition"] == "S28")["ownership"])
    write_csv("s28_buyouts.csv", winners)
    write_csv("s28_buyout_events.csv", events)
    write_csv("behavior.csv", behavior)
    write_csv("paired_ownership.csv", paired)
    for name in ("summary.csv", "manifest.json", "status.json"):
        shutil.copyfile(RUN / name, OUTPUT / name)
    receipt = {"audited_conditions": len(behavior), "episodes": starts,
               "active_physical_ticks_replayed": active_ticks, "exact_replay_conditions": exact_audits,
               "ownership_with_debt_violations": 0,
               "casino_wager_intentions": sum(row["casino_wager_intentions"] for row in behavior),
               "episodes_with_casino_intention": sum(row["episodes_with_casino_intention"] for row in behavior),
               "slot_spins": sum(row["spins"] for row in behavior),
               "blackjack_hands": sum(row["hands"] for row in behavior),
               "inherited_loans": sum(row["inherited_loans"] for row in behavior),
               "new_loans": sum(row["new_loans"] for row in behavior),
               "episodes_with_new_loan": sum(row["episodes_with_new_loan"] for row in behavior)}
    (OUTPUT / "failure_choices.json").write_text(
        json.dumps(failure_choices, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "audit_summary.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("S28 reconciliation:", [(row["scenario_id"], row["shifts"], row["total_wages"],
                                   row["cash_at_ownership"], row["same_cost_cash_at_flat_base_wage"]) for row in winners])


if __name__ == "__main__":
    main()
