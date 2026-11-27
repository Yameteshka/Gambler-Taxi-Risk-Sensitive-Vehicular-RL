"""Real mechanics checks and a one-intention real-transition learning benchmark.

The neural benchmark is contextual, not a full-game ownership evaluation. Its
reward table comes from real simulator transitions, without action labels.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json,sys,shutil
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import strategy,resource_curriculum
from env import resource_taxi as env
import local_learning_probe as probe
final=env.Config(starting_debt=500.,public_food_menu=True,loan_capacity=16,
                 reward_shaping=False,navigation_cue=False)
mechanics=[]
for level,action,field in ((2,strategy.WORK,'wages'),(3,strategy.RESTAURANT,'satiety'),
                         (4,strategy.FUEL_FULL,'fuel'),(5,strategy.SLEEP,'fatigue'),
                         (6,strategy.MICRO_ALL,'micro_debt')):
    config=replace(resource_curriculum.configuration(final,level),training_rehearsal=0.)
    params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
    _,state=strategy.reset(11,params)
    _,new,_,_,rows=eqx.filter_jit(strategy.advance)(state,env.Action(jnp.zeros(2),jnp.int32(action)),params)
    if field=='micro_debt':before=float(env.micro_debt(state));after=float(env.micro_debt(new))
    else:before=float(getattr(state,field));after=float(getattr(new,field))
    assert after<before if field in ('fatigue','micro_debt') else after>before
    mechanics.append({'context':level,'action':strategy.NAMES[action],'field':field,'before':before,'after':after})

# Six actual garage states, differing in exogenous scenario streams.
config=replace(resource_curriculum.configuration(final,0),training_rehearsal=0.)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
obs,states=jax.vmap(strategy.reset,in_axes=(0,None))(jnp.arange(11,17),params)
expanded=jax.tree.map(lambda x:jnp.repeat(x,26,axis=0),states)
choices=jnp.tile(jnp.arange(26,dtype=jnp.int32),6)
action=env.Action(jnp.zeros((156,2)),choices)
@eqx.filter_jit
def table(states,action):
    _,new,_,_,rows=jax.vmap(lambda s,a:strategy.advance(s,a,params,controller='fixed'))(states,action)
    reward=(rows['external_reward']*config.gamma**jnp.arange(strategy.STEPS)).sum(axis=1)
    counters=rows['events'].sum(axis=1)[:,jnp.array((env.PAID,env.MEALS,env.REFILLS,env.SLEEPS,env.LOANS,env.MICRO_PAYMENTS))]
    return reward.reshape((6,26)),(new.outcome==env.SUCCESS).reshape((6,26)),counters.reshape((6,26,6))
rewards,wins,events=table(expanded,action)
assert np.asarray(wins[:,strategy.BUY]).all()
def observation(s,unused=None):
    return SimpleNamespace(features=obs.features[s.case],action_mask=obs.action_mask[s.case])
def step(s,action,key):
    index=(s.case,action.choice)
    new=s._replace(done=jnp.ones_like(s.done),won=wins[index],counters=events[index])
    return new,rewards[index],new.done
probe.observation=observation;probe.step=step;probe.GAMMA=final.gamma
probe.CASES=tuple(f'real_state_{n}' for n in range(6))
settings={'actor_learning_rate':.0001,'learning_rate':.0003,'gradient_steps':64,'minibatch_size':128,
    'target_rate':.005,'critic_layer_norm':True,'automatic_temperature':False,
    'continuous_temperature':3e-6,'discrete_temperature':3e-6,'sample_target_action':True}
out=ROOT/'reports/local_learning_probe/real_context'
out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'real_transition_table.npz',features=np.asarray(obs.features),
    masks=np.asarray(obs.action_mask),external_rewards=np.asarray(rewards),ownership=np.asarray(wins),events=np.asarray(events))
shutil.copy2(Path(__file__),out/'real_context_source.py')
results={kind:probe.run(kind,settings if kind.startswith('dsac') else {},120,out/kind,.2,
    'One-intention contextual learning using a fixed-style real simulator transition table. Not full-game performance.')
    for kind in ('dsac_mlp','ppo_mlp')}
result={'passed':all(row['passed'] for row in results.values()),'mechanics':mechanics,
    'learned_real_transition_contexts':{kind:row['passed'] for kind,row in results.items()},
    'one_intention_termination_is_benchmark_only':True,
    'scope':'Mechanical resource recovery in real states plus actual-reward contextual learning. Long-horizon transfer is not established.'}
(out/'summary.json').write_text(json.dumps(result,indent=2));print('REAL_PROBE '+json.dumps(result),flush=True)
if not result['passed']:raise SystemExit(2)
