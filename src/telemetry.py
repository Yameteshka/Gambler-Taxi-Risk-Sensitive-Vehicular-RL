"""Compressed numeric trajectories and exact loan-payment receipts."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from env import resource_taxi as env


def frame(old, new, action, reward, info):
    return {
        "scenario_id": new.scenario_id, "time": new.time, "position": new.agent_pos,
        "forward": new.agent_forward_dir, "traffic": new.kinematic_obstacles,
        "speed": new.speed, "cash": new.cash, "car_debt": new.car_debt,
        "micro_debt": env.micro_debt(new), "loan_count": (new.loan_balance > 0).sum(axis=-1),
        "fuel": new.fuel, "satiety": new.satiety, "fatigue": new.fatigue,
        "cash_before": old.cash, "satiety_before": old.satiety, "fuel_before": old.fuel,
        "activity": new.activity, "remaining": new.remaining, "outcome": new.outcome,
        "requested": info["requested"], "executed": info["executed"], "control": action.control,
        "applied_control": info["applied_control"],
        "reward": reward, "external_reward": info["external_reward"], "shaping": info["shaping"],
        "financial_reward": info["financial_reward"], "net_worth": env.net_worth(new),
        "events": info["events"], "game_resolved": info["game_resolved"],
        "gambling_net": info["gambling_net"], "car_payment": info["car_payment"],
        "micro_payment": info["micro_payment"], "loan_opened": info["loan_opened"],
        "borrowed_total": new.borrowed, "interest_total": new.interest, "fees_total": new.fees,
        "food_spent": new.food_spent, "fuel_spent": new.fuel_spent, "wages": new.wages,
        "push_ticks": new.push_ticks, "car_cleared_at": new.car_cleared_at,
        "all_cleared_at": new.all_cleared_at, "hand": new.hand, "dealer": new.dealer[..., 0],
        "meal_venue": new.service_venue, "service_value": new.service_value, "service_price":new.service_price,
        "slots_legal": info["slots_legal"], "blackjack_legal": info["blackjack_legal"],
        "loan_repayments": info["loan_repayments"], "loan_ids": old.loan_ids,
        "capacity_needed": info["capacity_needed"],
        "active": old.outcome == env.ACTIVE,
    }


def write_chunk(path, trace, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    host = {key: np.asarray(value) for key, value in trace.items()
            if key not in ("loan_repayments", "loan_ids")}
    shape = host["active"].shape
    for name, value in host.items():
        if value.shape[:len(shape)] != shape:
            raise ValueError(f"Trajectory field {name} lost its time/environment axes: {value.shape} vs {shape}")
    payment_mask = host["micro_payment"] > 0
    if payment_mask.any():
        repayment = np.asarray(trace["loan_repayments"])[payment_mask]
        ids = np.asarray(trace["loan_ids"])[payment_mask]
        rows, columns = np.nonzero(repayment > 0)
        locations = np.argwhere(payment_mask)
        host["loan_receipt_location"] = locations[rows]
        host["loan_receipt_id"] = ids[rows, columns]
        host["loan_receipt_amount"] = repayment[rows, columns]
    with path.open("wb") as file:
        np.savez_compressed(file, metadata=json.dumps(metadata or {}), **host)
    return path.stat().st_size


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
