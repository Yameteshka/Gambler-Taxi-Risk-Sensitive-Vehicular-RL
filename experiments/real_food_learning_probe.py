"""A three-intention food-to-ownership test using cached real simulator transitions.

Enumerate every action, without action labels, demonstrations, new rewards, or
an oracle teacher. Cache three macro transitions to make neural learning fast.
The benchmark stops after three intentions; it is not full-game evidence.
Continuous driving style is held at zero while constructing this cache.
"""
from pathlib import Path
from dataclasses import replace,asdict
from typing import NamedTuple
from types import SimpleNamespace
import argparse,json,sys
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import strategy,models
from env import resource_taxi as env,city
import local_learning_probe as probe
cache_sources=probe.protocol.source_hashes()

parser=argparse.ArgumentParser();parser.add_argument('--model',default='both',
    choices=('both','all','dsac_mlp','ppo_mlp','dsac_gru','ppo_gru'))
parser.add_argument('--iterations',type=int,default=240)
parser.add_argument('--escape',action='store_true');args=parser.parse_args()
out=ROOT/('reports/local_learning_probe/real_food_learning_release' if args.escape else 'reports/local_learning_probe/real_food_learning');out.mkdir(parents=True,exist_ok=True)
c=env.Config(starting_debt=500.,public_food_menu=True,reward_shaping=False,
    navigation_cue=False,predictive_driver=True,predictive_escape=args.escape,loan_capacity=16)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),c)
(out/'cache_metadata.json').write_text(json.dumps({'config':asdict(c),'source_hashes':cache_sources,
    'macro_steps':strategy.STEPS,'fixed_control':[0.,0.],'depth':3},indent=2))
ids=jnp.arange(11,17)
_,starts=jax.vmap(strategy.reset,in_axes=(0,None))(ids,params)
starts=replace(starts,agent_pos=jnp.broadcast_to(params.zone_centers[city.RESTAURANT],(6,2)),
    car_debt=jnp.zeros(6),car_paid=jnp.full(6,500.),satiety=jnp.full(6,.04),
    car_cleared_at=jnp.zeros(6,dtype=jnp.int32),all_cleared_at=jnp.zeros(6,dtype=jnp.int32))

@eqx.filter_jit
def expand(states):
    many=jax.tree.map(lambda x:jnp.repeat(x,26,axis=0),states)
    choices=jnp.tile(jnp.arange(26,dtype=jnp.int32),states.time.size)
    obs,new,reward,done,trace=jax.vmap(lambda s,a:strategy.advance(s,a,params))(
        many,env.Action(jnp.zeros((choices.size,2)),choices))
    duration=trace['active'].sum(axis=1)
    events=trace['events'].sum(axis=1)[:,jnp.array((env.PAID,env.MEALS,env.REFILLS,env.SLEEPS,env.LOANS,env.MICRO_PAYMENTS))]
    return obs,new,reward,done,c.gamma**duration,events

first_obs,first,first_reward,first_done,first_discount,first_events=expand(starts)
initial_obs=jax.vmap(strategy.get_observation,in_axes=(0,None))(starts,params)
eligible=np.asarray(initial_obs.action_mask).reshape(-1)&~np.asarray(first_done)
indices=np.flatnonzero(eligible)
children=jax.tree.map(lambda x:x[indices],first)
last_obs,last,last_reward,last_done,last_discount,last_events=expand(children)
child_count=len(indices)
grand_indices=np.flatnonzero(np.asarray(first_obs.action_mask)[indices].reshape(-1)&~np.asarray(last_done))
grandchildren=jax.tree.map(lambda x:x[grand_indices],last)
third_obs,third,third_reward,third_done,third_discount,third_events=expand(grandchildren)
grand_count=len(grand_indices);root_count=6;total=root_count+child_count+grand_count+1;absorbing=total-1
features=np.zeros((total,probe.BASE.size),np.float32);masks=np.zeros((total,26),bool)
features[:6]=np.asarray(initial_obs.features);masks[:6]=np.asarray(initial_obs.action_mask)
features[6:6+child_count]=np.asarray(first_obs.features)[indices];masks[6:6+child_count]=np.asarray(first_obs.action_mask)[indices]
features[6+child_count:-1]=np.asarray(last_obs.features)[grand_indices];masks[6+child_count:-1]=np.asarray(last_obs.action_mask)[grand_indices]
masks[-1,0]=True
following=np.full((total,26),absorbing,np.int32)
rewards=np.zeros((total,26),np.float32);discounts=np.zeros_like(rewards)
done=np.ones((total,26),bool);wins=np.zeros_like(done);events=np.zeros((total,26,6),np.float32)
rewards[:6]=np.asarray(first_reward).reshape(6,26);discounts[:6]=np.asarray(first_discount).reshape(6,26)
wins[:6]=(np.asarray(first.outcome)==env.SUCCESS).reshape(6,26)
events[:6]=np.asarray(first_events).reshape(6,26,6)
for child,flat in enumerate(indices):
    row,action=divmod(int(flat),26);following[row,action]=6+child;done[row,action]=False
rewards[6:6+child_count]=np.asarray(last_reward).reshape(child_count,26)
discounts[6:6+child_count]=np.asarray(last_discount).reshape(child_count,26)
wins[6:6+child_count]=(np.asarray(last.outcome)==env.SUCCESS).reshape(child_count,26)
events[6:6+child_count]=np.asarray(last_events).reshape(child_count,26,6)
for grand,flat in enumerate(grand_indices):
    row,action=divmod(int(flat),26);following[6+row,action]=6+child_count+grand;done[6+row,action]=False
rewards[6+child_count:-1]=np.asarray(third_reward).reshape(grand_count,26)
discounts[6+child_count:-1]=np.asarray(third_discount).reshape(grand_count,26)
wins[6+child_count:-1]=(np.asarray(third.outcome)==env.SUCCESS).reshape(grand_count,26)
events[6+child_count:-1]=np.asarray(third_events).reshape(grand_count,26,6)
np.savez_compressed(out/'transitions.npz',features=features,masks=masks,next_node=following,
    rewards=rewards,discounts=discounts,done=done,wins=wins,events=events)
# Verify the intended dependency against actual transitions, not an action label.
assert wins[:6].sum()==0
reachable=[bool(wins[following[i,strategy.RESTAURANT],strategy.BUY] or
    wins[following[following[i,strategy.RESTAURANT],strategy.BUY],strategy.BUY]) for i in range(6)]
(out/'cache_checks.json').write_text(json.dumps({'passed':all(reachable),'reachable_by_meal_then_ownership':reachable,'depth':3},indent=2))
assert all(reachable),'No verified meal-to-ownership continuation within three intentions'
print('REAL_FOOD_CACHE',json.dumps({'nodes':total,'initial_ownership':int(wins[:6].sum()),
    'three_intention_ownership_transitions':int(wins.sum()),'unlabeled':True}),flush=True)
features,masks,following,rewards,discounts,done,wins,events=map(jnp.asarray,
    (features,masks,following,rewards,discounts,done,wins,events))

class State(NamedTuple):
    node:jax.Array
    done:jax.Array
    won:jax.Array
    counters:jax.Array

def reset(key,size=64,case=None):
    node=jax.random.randint(key,(size,),0,6) if case is None else jnp.full(size,case)
    return State(node,jnp.zeros(size,bool),jnp.zeros(size,bool),jnp.zeros((size,6)))
def observation(s,unused=None):return SimpleNamespace(features=features[s.node],action_mask=masks[s.node])
def step(s,a,key):
    index=s.node,a.choice
    return State(following[index],done[index],wins[index],s.counters+events[index]),rewards[index],done[index]
def discount(s,a):return discounts[s.node,a.choice]
probe.reset=reset;probe.observation=observation;probe.step=step;probe.GAMMA=c.gamma
probe.CASES=tuple('real_food_scenario_'+str(int(i)) for i in ids)
settings={'actor_learning_rate':.0001,'learning_rate':.0003,'gradient_steps':64,
    'minibatch_size':128,'target_rate':.005,'automatic_temperature':False,
    'continuous_temperature':3e-6,'discrete_temperature':3e-6,
    'sample_target_action':True,'critic_layer_norm':True,
    'separate_recurrent_encoders':True,'recurrent_observation_skip':True}
kinds=('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru') if args.model=='all' else ('dsac_mlp','ppo_mlp') if args.model=='both' else (args.model,)
results={}
for kind in kinds:
    result=probe.run(kind,settings if kind.startswith('dsac') else {},args.iterations,out/kind,.2,
        'Three-intention ownership benchmark; actual real-simulator rewards, observations and variable-duration discounts. Fixed driving style, cached scenario realizations, artificial three-intention cutoff. Not full-game performance.',discount)
    results[kind]=result['passed'];print('REAL_FOOD_LEARNED',kind,result['passed'],flush=True)
summary={'passed':all(results.values()),'models':results,'scenario_ids':np.asarray(ids).tolist(),
    'config':asdict(c),'satiety':.04,'remaining_car_debt':0.,'initial_cash':100.,
    'no_action_labels':True,'scope':'Local causal meal-to-ownership learning only; no training weights are promoted.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
if not summary['passed']:raise SystemExit(2)
