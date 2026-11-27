"""Export one measured destination-policy case to the existing video renderer."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import debt_learning as learning
from debt_research import TEST_COUNT, TEST_START
from env import gamblers_taxi as taxi
from train import keys


ENV = taxi.GamblersTaxiEnv


def frame(state, obs, params):
    return {
        "tick": state.time, "position": state.agent_pos, "forward": state.agent_forward_dir,
        "traffic": state.kinematic_obstacles, "rays": obs.collision_rays, "outcome": state.outcome,
        "status": jnp.array([state.speed, state.cash, state.debt, state.stake, state.fatigue, state.mood,
                             state.tilt, state.employed, state.activity, state.remaining]),
        "hand": state.hand, "dealer_upcard": state.dealer[0], "anxiety": state.anxiety,
        "dependence": jnp.array([state.habit, taxi._urge(state, params)]),
    }


@eqx.filter_jit
def run(model, params, seeds, index):
    def record(old, new, obs, info):
        return frame(old, obs, params), {
            "events": info["events"], "choice": info["choice"], "requested_choice": info["requested_choice"],
            "daily_cost": info["daily_cost"], "gambling_net": info["gambling_net"],
            "game_resolved": info["game_resolved"],
        }

    ends, (_, (frames, transitions)) = learning.evaluate(model, params, seeds, record=record)
    end = jax.tree.map(lambda x: x[index], ends)
    flatten = lambda a: a[index].reshape((-1, *a.shape[3:]))
    return (jax.tree.map(flatten, frames), jax.tree.map(flatten, transitions),
            frame(end, ENV.get_observation(end, params), params))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.research/"results.json").read_text(encoding="utf-8"))[args.policy]
    episodes = json.loads((args.research/f"{args.policy}_episodes.json").read_text(encoding="utf-8"))
    index = args.seed-TEST_START
    if not 0 <= index < TEST_COUNT:
        parser.error("seed is outside the saved test bank")
    expected = episodes[index]
    actor, meta = learning.load(args.research.parent/result["checkpoint"]["file"])
    cfg = taxi.TaxiConfig(**result["environment"])
    params, _ = ENV.init_params(jax.random.key(0), cfg)
    frames, transitions, last = jax.device_get(run(actor, params, keys(TEST_START, TEST_COUNT), index))
    count = int(last["tick"]-frames["tick"][0])
    outcome = taxi.OUTCOME_NAMES[int(last["outcome"])]
    if outcome != expected["outcome"] or last["tick"] != expected["time"]:
        raise RuntimeError("Replay differs from the original evaluated trajectory")
    np.testing.assert_allclose(last["status"][1:3], (expected["cash"], expected["debt"]), rtol=1e-6)
    header = {**asdict(cfg), "seed": args.seed, "policy": args.policy, "checkpoint_step": meta["iteration"],
              "actions": taxi.ACTION_NAMES, "events": taxi.COUNTER_NAMES, "dependence": cfg.enable_dependence}
    arrays = {name: np.concatenate((a[:count], last[name][None])) for name, a in frames.items()}
    arrays.update({name: np.concatenate((np.zeros_like(a[:1]), a[:count])) for name, a in transitions.items()})
    args.log.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.log, header=json.dumps(header), **arrays)
    print(f"{args.log} seed={args.seed} outcome={outcome} frames={count+1}")


if __name__ == "__main__":
    main()
