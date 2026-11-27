"""Check exploration likelihoods, gradients and mechanically valid start states."""
from pathlib import Path
from dataclasses import replace
import sys,json
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import models,strategy
from env import resource_taxi as env,city
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(starting_debt=50.,public_food_menu=True,exploration_starts=True,exploration_epsilon=.8,loan_capacity=8))
states=[]
for scenario in range(11,31):
 _,s=strategy.reset(scenario,params);states.append(s)
 zone=scenario%10
 np.testing.assert_allclose(s.agent_pos,params.zone_centers[zone])
 if zone==city.GARAGE:assert s.car_debt==0 and env.micro_debt(s)==0
 if zone in (city.SLOTS,city.BLACKJACK):assert s.time==params.config.casino_open
 if zone>=city.DINER:assert s.satiety<=.2+1e-6
batch=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
obs=jax.vmap(strategy.get_observation,in_axes=(0,None))(batch,params)
model=models.init_ppo(jax.random.key(301),194,False,strategy.COUNT)
memory=models.initial_memory('ppo_mlp',20,194)
actions,_,_,raw,saved_logp,_,_=models.policy_step(model,'ppo_mlp',obs.features,obs.action_mask,memory,jax.random.key(37),exploration_epsilon=.8)
mean,std,logp,_,_=models.ppo_step(model,obs.features,obs.action_mask,memory,.8)
recomputed,_,_=models.hybrid_logprob(mean,std,logp,raw,actions.choice)
np.testing.assert_allclose(saved_logp,recomputed,atol=1e-6)
p=jnp.exp(logp);np.testing.assert_allclose(p.sum(-1),1.,atol=1e-6)
assert bool(jnp.all(p>=.8*obs.action_mask/obs.action_mask.sum(-1,keepdims=True)-1e-7))
def loss(candidate):
 lp=models.ppo_step(candidate,obs.features,obs.action_mask,memory,.8)[2]
 return -jnp.take_along_axis(lp,actions.choice[:,None],-1).mean()
gradient=jax.grad(loss)(model)
norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(gradient))))
assert norm>1e-7 and np.isfinite(norm)
result={'passed':True,'on_policy_log_probability_matches':True,'gradient_norm':norm,'start_scenarios':[11,30],'no_expert_action_labels':True}
(ROOT/'experiments/cold_start_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
