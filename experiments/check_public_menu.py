"""Check public menu terms against actual transactions, with no training."""
from pathlib import Path
from dataclasses import replace
import sys,json
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env,city
import strategy

checks=[]
for meals in ('reference_uniform','meal_mean_only','meal_high_variance'):
 for swap in (False,True):
  config=env.Config(public_food_menu=True,meals=meals,food_swap=swap,loan_capacity=8)
  params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
  _,state=strategy.reset(6,params)
  state=replace(state,cash=jnp.float32(500.),satiety=jnp.float32(.1))
  menu=env.public_food_menu(state,params)
  features=strategy.get_observation(state,params).features
  assert features.size==194 # 153 original + 15 public terms + 26 intention indicators
  old=strategy.get_observation(state,replace(params,config=replace(config,public_food_menu=False))).features
  np.testing.assert_array_equal(features[:153],old[:153])
  masked=env.public_food_menu(state,replace(params,config=replace(config,reduced_quotes=True)))
  np.testing.assert_array_equal(masked[:,0],np.zeros(3))
  np.testing.assert_array_equal(masked[:,1:],menu[:,1:])
  for venue in range(3):
   parked=replace(state,agent_pos=params.zone_centers[city.DINER+venue])
   _,new,_,_,_=env.ResourceTaxiEnv.step(jax.random.key(0),parked,env.Action(jnp.zeros(2),jnp.int32(env.EAT)),params)
   assert menu[venue,1]-1e-6<=new.service_value<=menu[venue,3]+1e-6
   np.testing.assert_allclose(new.service_price,menu[venue,0])
   np.testing.assert_allclose(new.remaining,menu[venue,4]-1)
   # Changing the future purchase ordinal changes the draw, not public terms.
   np.testing.assert_array_equal(menu,env.public_food_menu(replace(state,meal_count=state.meal_count+7),params))
  checks.append({'meals':meals,'identity_swap':swap,'passed':True})
result={'passed':True,'checks':checks,'scope':'Observation and transaction terms only; no policy is trained or resumed.'}
(ROOT/'experiments/public_menu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
