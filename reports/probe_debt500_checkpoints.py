"""Matched public-state preferences across PPO checkpoints, exploration disabled."""
from pathlib import Path
import os
from dataclasses import replace
import sys,json
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import models,strategy
from env import resource_taxi as env,city
BASE=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'debt500_study/output/artifacts/debt500_study/training'
OUT=ROOT/'reports/debt500_diagnosis';OUT.mkdir(exist_ok=True)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(starting_debt=500.,public_food_menu=True,loan_capacity=8))
_,state=strategy.reset(6,params)
states=[];labels=[]
for zone in (city.HOME,city.BANK,city.DINER,city.CAFE,city.RESTAURANT):
 for satiety in (.05,.2,.8):
  s=replace(state,agent_pos=params.zone_centers[zone],satiety=jnp.float32(satiety),
    cash=jnp.float32(100.),time=jnp.int32(100),shift_claimed=jnp.bool_(True))
  states.append(s);labels.append({'place':city.NAMES[zone],'satiety':satiety})
batch=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
obs=jax.vmap(strategy.get_observation,in_axes=(0,None))(batch,params)
records=[]
for kind in ('ppo_mlp','ppo_gru'):
 for name in ('initial','stage_0','stage_1','stage_2','stage_3','selected','final'):
  path=BASE/kind/(name+'.npz')
  if not path.exists():continue
  actor,meta=models.load_policy(path)
  memory=models.initial_memory(kind,len(states),194)
  p=np.asarray(jnp.exp(models.ppo_step(actor,obs.features,obs.action_mask,memory)[2]))
  entries=[{**label,'meal_probability':float(x[3:6].sum()),'repay_probability':float(x[11:17].sum()),
    'borrow_probability':float(x[8:11].sum()),'work_probability':float(x[1])} for label,x in zip(labels,p)]
  records.append({'kind':kind,'checkpoint':name,'training_steps':meta.get('physical_transitions'),
    'scope':'Synthetic matched states; zero recurrent history, exploration mixture disabled. Policy sensitivity, not a counterfactual episode result.',
    'contexts':entries})
(OUT/'checkpoint_preferences.json').write_text(json.dumps(records,indent=2))
for r in records:
 hungry=[x for x in r['contexts'] if x['satiety']==.05]
 print(r['kind'],r['checkpoint'],'hungry food probabilities',[(x['place'],round(x['meal_probability'],5)) for x in hungry],flush=True)
# Verify what the embedded economic route heuristic prioritizes.
s=replace(state,satiety=jnp.float32(.05),cash=jnp.float32(200),car_debt=jnp.float32(100))
useful=np.asarray(env.useful_targets(s,params))
print('Hungry but solvent route targets:',[city.NAMES[i] for i in np.flatnonzero(useful)],flush=True)
