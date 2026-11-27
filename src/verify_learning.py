"""Check resumed training against an uninterrupted run and smoke-test all learners."""

import json
import pickle
import gzip

import jax
import numpy as np

import protocol
from strategy_training import train
from strategy_training import with_history
from algorithms.hybrid_dsac import Transition
import jax.numpy as jnp
import models


def main():
    # A nearly unsquashed Gaussian's entropy must increase with log standard
    # deviation. A fixed-old-action score estimator incorrectly averages to zero.
    epsilon = jax.random.normal(jax.random.PRNGKey(381), (8192, 2))
    logp = jnp.full((8192, 25), -1e9).at[:, 0].set(0.)
    def entropy(scale):
        return models.hybrid_entropy(jnp.zeros_like(epsilon), jnp.full_like(epsilon, scale), logp, epsilon)[0].mean()
    entropy_gradient = float(jax.grad(entropy)(jnp.float32(-2.)))
    assert 1.8 < entropy_gradient < 2., entropy_gradient
    # A terminal in one lane must not leak into another lane's replay history.
    features = jnp.arange(16, dtype=jnp.float32)[:, None]
    done = jnp.zeros(16, dtype=jnp.bool_).at[4].set(True)
    fake = Transition(features, jnp.ones((16, 25), dtype=jnp.bool_), jnp.zeros((16, 2)),
        jnp.zeros(16, dtype=jnp.int32), jnp.zeros(16), done, features+2, jnp.ones((16, 25), dtype=jnp.bool_))
    indices = jnp.array((10, 13))
    batch = jax.tree.map(lambda x: x[indices], fake)
    history = with_history(fake, batch, indices, lanes=2, history=4)
    np.testing.assert_array_equal(history.features[..., 0], ((0, 6, 8, 10), (7, 9, 11, 13)))
    np.testing.assert_array_equal(history.next_features[..., 0], ((6, 8, 10, 12), (9, 11, 13, 15)))
    row = protocol.registry()[0]
    interrupted = {**row, "run_id": "resume_equivalence_interrupted", "initialization_seed": "9091", "curriculum":"none"}
    uninterrupted = {**row, "run_id": "resume_equivalence_uninterrupted", "initialization_seed": "9091", "curriculum":"none"}
    a = train(interrupted, 65536, pilot=True, stop_after=16384)
    assert not (a/"complete.json").exists(), "The interrupted check must actually stop before completion"
    train(interrupted, 65536, pilot=True, resume=True)
    b = train(uninterrupted, 65536, pilot=True)
    with gzip.open(a/"resume.pkl.gz", "rb") as file:
        left = pickle.load(file)
    with gzip.open(b/"resume.pkl.gz", "rb") as file:
        right = pickle.load(file)
    for name in ("actor", "learner", "states", "hidden", "counters", "rng", "replay", "cursor", "size"):
        for x, y in zip(jax.tree.leaves(left[name]), jax.tree.leaves(right[name])):
            np.testing.assert_array_equal(x, y, err_msg=f"Resume mismatch: {name}")
    tested = []
    for algorithm, architecture in (("hybrid_ppo", "mlp"), ("hybrid_ppo", "gru"), ("hybrid_dsac", "gru")):
        candidate = next(r for r in protocol.registry() if r["algorithm"] == algorithm and r["architecture"] == architecture
                         and r["training_horizon"] == "10d" and r["repetition"] == "B")
        output = train({**candidate, "run_id": "smoke_"+candidate["run_id"], "initialization_seed": "9092", "curriculum":"none"}, 65536, pilot=True)
        tested.append(str(output.relative_to(protocol.ROOT)))
    result = {"resume_bitwise_equivalent": True, "continuous_entropy_gradient_check": entropy_gradient, "smoke_runs": tested,
              "source_hashes": protocol.source_hashes(), "scope": "Execution and persistence, not learning convergence."}
    (protocol.ROOT/"experiments/learning_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("All learners execute; resumed DSAC state is bitwise equivalent", flush=True)


if __name__ == "__main__":
    main()
