"""Verify the new observation and PPO path, and recheck the frozen GRU."""
from dataclasses import replace
from pathlib import Path
import os
import json
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import budget_curriculum as curriculum
import models
import protocol
import strategy
import strategy_training as training
from env import resource_taxi as env
from evaluate import evaluate

base=env.Config(starting_debt=500.,public_food_menu=True,resource_observation=True,
    standard_start_rehearsal=.25,ppo_native_exploration=True,ppo_exploration_decay=True,
    ppo_epochs=12,ppo_target_kl=.02,ppo_clip=.15,ppo_normalize_value_loss=True)
assert curriculum.configuration(base,12).exploration_epsilon>0
assert curriculum.configuration(base,13).exploration_epsilon==0
config=curriculum.configuration(replace(base,observation_history=True),4)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
obs,state=strategy.reset(11,params)
assert obs.features.shape==(370,)
next_obs,_,_,_,_=eqx.filter_jit(strategy.advance)(state,
    env.Action(jnp.zeros(2),jnp.int32(strategy.WAIT)),params)
np.testing.assert_allclose(next_obs.features[172:344],obs.features[:172])
assert float(next_obs.features[-26+strategy.WAIT])==1.

keys=jax.random.split(jax.random.key(301),3)
counters=jnp.arange(training.ENVS)
ppo=[]
for kind in ('ppo_mlp','ppo_gru'):
    recurrent=kind=='ppo_gru'
    config=curriculum.configuration(replace(base,observation_history=not recurrent,
        ppo_value_head_lr_multiplier=3. if recurrent else 1.),4)
    params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
    obs,states=training.reset_batch(params,protocol.load(),counters)
    actor=models.init_ppo(keys[0],obs.features.shape[-1],recurrent,strategy.COUNT)
    memory=models.initial_memory(kind,training.ENVS,obs.features.shape[-1])
    (end,hidden,_,_),trace=training.make_collector(params,kind,protocol.load())(
        actor,states,memory,counters,keys[1],None)
    new,_,metrics=training.make_ppo_update(params,training.OPT,diagnostics=True)(
        actor,training.OPT.init(actor),trace,end,hidden,keys[2])
    values=np.asarray(metrics)
    assert np.isfinite(values).all() and 0<values[1]<=96
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(actor),jax.tree.leaves(new)))
    row={'model':kind,'input_width':obs.features.shape[-1],
         'loss':float(values[0]),'applied_updates':int(values[1]),'maximum_minibatch_kl':float(values[2])}
    print(row,flush=True)
    ppo.append(row)

reference=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training/dsac_gru/selected.npz'
actor,metadata=models.load_policy(reference)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(**metadata['evaluation_environment']))
summary,_,_=evaluate(actor,'dsac_gru',params,range(1000001,1000065),deterministic=True,interface='strategy')
result={'passed':True,'ppo_updates':ppo,'frozen_gru_argmax':summary,'source_hashes':protocol.source_hashes()}
out=ROOT/'reports/local_food_recovery/tuned_preflight.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('FROZEN_GRU_ARGMAX',summary['ownership'],'/ 64',flush=True)
