"""One real collection and DSAC update, not a task-learning experiment."""
from pathlib import Path
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import progressive_curriculum as curriculum
import protocol
import strategy
import strategy_training as training
import models
from env import resource_taxi as env
from algorithms import hybrid_dsac as dsac

final = env.Config(starting_debt=500., public_food_menu=True, loan_capacity=16)
config = curriculum.configuration(final, 0)
params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
design = protocol.load()
obs, states = training.reset_batch(params, design, jnp.arange(training.ENVS))
agent = dsac.HybridDSAC(dsac.AgentConfig(gamma=config.gamma, demo_weight=0., action_count=strategy.COUNT))
learner = agent.init(jax.random.key(301), obs.features.shape[-1])
hidden = models.initial_memory('dsac_mlp', training.ENVS, obs.features.shape[-1])
collector = training.make_collector(params, 'dsac_mlp', design)
_, trace = collector(learner.actor, states, hidden, jnp.arange(training.ENVS), jax.random.key(7), None)
transitions = trace[0]
duration = trace[5]['active'].sum(axis=-1)
# Match the environment's float32 gamma rather than exponentiating a float64
# Python literal; their difference compounds over a 128-tick action.
expected_discount = np.power(np.float32(config.gamma), np.asarray(duration), dtype=np.float32)
np.testing.assert_allclose(np.asarray(transitions.discount), expected_discount, rtol=1e-6)
assert np.isfinite(np.asarray(transitions.reward)).all()
batch = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:]))[:256], transitions)
updated, losses = agent.update(learner, jax.random.key(8), batch)
assert np.isfinite(np.asarray(losses)).all()
change = float(jnp.sqrt(sum(jnp.sum((a-b)**2) for a, b in zip(
    jax.tree.leaves(updated.actor), jax.tree.leaves(learner.actor)))))
assert change > 0
result = {'passed': True, 'devices': [str(device) for device in jax.devices()],
    'active_physical_steps': int(np.asarray(duration).sum()), 'feature_count': int(obs.features.shape[-1]),
    'actor_parameter_change': change, 'losses': np.asarray(losses).tolist(),
    'scope': 'One real collection and parameter update; no evidence of improved task learning.'}
(ROOT/'experiments/progressive_collection_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
