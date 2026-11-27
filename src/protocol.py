"""Validate scenario splits and resolve registered experiment settings."""

import csv
from dataclasses import replace
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from env.resource_taxi import Config


ROOT = Path(__file__).resolve().parents[1]


def load():
    design = json.loads((ROOT/"experiments/protocol.json").read_text(encoding="utf-8"))
    check_splits(design)
    return design


def check_splits(design):
    count = design["training_scenario_count"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("training_scenario_count must be a positive integer")
    first = design["scenario_banks"]["train"]["first"]
    last = first+count-1
    reserved = [v for n, v in design["scenario_banks"].items() if n != "train"]
    if first <= 10 or any(not (last < v["first"] or first > v["last"]) for v in reserved):
        raise ValueError("Training scenarios overlap a reserved evaluation bank")
    return first, last


def registry():
    with (ROOT/"experiments/registry.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_config(row):
    horizon = {"5d": 5, "10d": 10, "open": 0}[row["training_horizon"]]
    return Config(days=horizon, starting_debt=float(row.get('starting_debt') or 1000.), market=row["training_market"],
                  meals=row["food_restoration"], installments=row["contract"] == "scheduled",
                  casino=row["casino"] == "enabled",
                  hidden_casino_regime=row.get("casino_regime") == "hidden",
                  reduced_quotes=row.get("quote_information") == "reduced",
                  accelerated_schedule=row.get("payment_calendar") == "accelerated",
                  financial_progress=row.get("financial_progress", "enabled") != "disabled",
                  navigation_cue=row.get("navigation_cue", "enabled") != "disabled",
                  public_food_menu=row.get("public_food_menu") == "enabled",
                  reward_shaping=row.get('reward_shaping','enabled') != 'disabled',
                  predictive_driver=row.get('predictive_driver') == 'enabled',
                  predictive_escape=row.get('predictive_escape') == 'enabled',
                  ppo_native_exploration=row.get('ppo_native_exploration')=='enabled',
                  standard_start_rehearsal=float(row.get('standard_start_rehearsal') or 0.),
                  resource_observation=row.get('resource_observation')=='enabled',
                  observation_history=row.get('observation_history')=='enabled',
                  ppo_epochs=int(row.get('ppo_epochs') or 3),
                  ppo_target_kl=float(row.get('ppo_target_kl') or 0.),
                  ppo_clip=float(row.get('ppo_clip') or .2),
                  ppo_value_head_lr_multiplier=float(row.get('ppo_value_head_lr_multiplier') or 1.),
                  ppo_normalize_value_loss=row.get('ppo_normalize_value_loss')=='enabled',
                  ppo_exploration_decay=row.get('ppo_exploration_decay')=='enabled',
                  unavailable_wait_ticks=int(row.get('unavailable_wait_ticks') or 1))


def curriculum(final, fraction, disabled=False, temporal=True, strategic=False):
    if disabled or fraction >= .5:
        return final
    if strategic and fraction < .2:
        level = min(int(fraction/.05),3)
        discount = (.995,.997,.998,.999)[level] if temporal else final.gamma
        return replace(final,starting_debt=200.,hunger_tick=final.hunger_tick*.55,
                       fuel_distance=final.fuel_distance*.6,fuel_work=final.fuel_work*.6,
                       market="fixed",gamma=discount)
    if strategic:
        return replace(final,starting_debt=500.,hunger_tick=final.hunger_tick*.8,
                       intermediate_starts=True,
                       intermediate_home_fraction=.25+.25*min(int((fraction-.2)/.1),2))
    if fraction < .2:
        level = min(int(fraction/.05), 3)
        discount = (.97, .985, .995, .997)[level] if temporal else final.gamma
        return replace(final, starting_debt=150., hunger_tick=final.hunger_tick*.55,
                       fuel_distance=final.fuel_distance*.6, fuel_work=final.fuel_work*.6,
                       market="fixed", accessible_start=True, school_level=level, gamma=discount)
    return replace(final, starting_debt=500., hunger_tick=final.hunger_tick*.8,
                   intermediate_starts=True,
                   intermediate_home_fraction=.25+.25*min(int((fraction-.2)/.1), 2))


def source_hashes():
    return {str(p.relative_to(ROOT)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/"src").rglob("*.py"))}


@contextmanager
def experiment_lock():
    with (ROOT/"experiments/registry.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def set_run_status(run_id, status, **changes):
    with experiment_lock():
        _set_run_status(run_id, status, **changes)


def _set_run_status(run_id, status, **changes):
    rows = registry()
    found = False
    for row in rows:
        if row["run_id"] == run_id:
            row.update(status=status, **changes)
            found = True
    if not found:
        raise KeyError(run_id)
    path = ROOT/"experiments/registry.csv"
    temp = path.with_suffix(".tmp")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
