"""Check recovery reset invariants, rejection gates and real unavailable-service timing."""
from pathlib import Path
from dataclasses import replace
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
from env import resource_taxi as env,city
import strategy,budget_curriculum as curriculum,protocol
from strategy_training import behavior_epsilon

out=ROOT/'reports/resource_study_review_2026-11-23/recovery_checks'
out.mkdir(parents=True,exist_ok=True)
base=env.Config(starting_debt=500.,public_food_menu=True,reward_shaping=False,
    navigation_cue=False,predictive_driver=True,predictive_escape=True,loan_capacity=16)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),base)
rows=[]
for stage,lesson in enumerate(curriculum.LESSONS):
    config=curriculum.configuration(base,stage)
    p=replace(params,config=config)
    reset=eqx.filter_jit(jax.vmap(strategy.reset,in_axes=(0,None)))
    obs,states=reset(jnp.arange(11,43),p)
    assert obs.features.shape==(32,194) and np.isfinite(obs.features).all()
    assert np.allclose(np.asarray(states.car_debt+states.car_paid),500.)
    assert np.all(np.asarray(states.cash)>0) and np.all(np.asarray(states.satiety)>0)
    assert np.all(np.asarray(states.loan_balance)>=0)
    row={'stage':stage,'lesson':lesson.name,'satiety_min':float(states.satiety.min()),
         'satiety_max':float(states.satiety.max()),'cash_min':float(states.cash.min()),
         'cash_max':float(states.cash.max()),'distinct_positions':len(np.unique(np.asarray(states.agent_pos),axis=0))}
    if stage in (7,8,9):
        p=replace(params,config=curriculum.evaluation_configuration(config))
        _,fixed=reset(jnp.arange(11,43),p)
        assert np.allclose(np.asarray(fixed.agent_pos),np.asarray(params.zone_centers[(city.WORK,city.BANK,city.HOME)[stage-7]]))
        assert np.ptp(np.asarray(fixed.cash))>20 and np.ptp(np.asarray(fixed.satiety))>.05
    rows.append(row)

# A rare lucky win or aggregate meals on unsuccessful episodes must not pass.
summary={'episodes':16,'ownership':4}
assert not curriculum.gate(6,summary,4,True,successful_meal_episodes=16)['passed']
summary={'episodes':16,'ownership':16}
assert not curriculum.gate(6,summary,0,True,successful_meal_episodes=1)['passed']
assert curriculum.gate(6,summary,0,True,successful_meal_episodes=16)['passed']
assert not curriculum.gate(9,summary,1,True,successful_meal_episodes=16)['passed']
assert curriculum.gate(9,summary,16,True,successful_meal_episodes=16)['passed']
assert not curriculum.gate(10,summary,16,True,successful_refuel_episodes=0)['passed']
assert curriculum.gate(10,summary,16,True,successful_refuel_episodes=16)['passed']
native=replace(base,ppo_native_exploration=True,exploration_epsilon=.2)
assert behavior_epsilon(native,'ppo_mlp')==0.
assert behavior_epsilon(native,'dsac_mlp')==.2
assert len(list(curriculum.additional_checks(base,17)))==15

# The 25% option samples complete standard resets, including debt and cash.
mixed=curriculum.configuration(replace(base,standard_start_rehearsal=.25),9)
reset_ids=jnp.arange(11,267)
batch_reset=eqx.filter_jit(jax.vmap(strategy.reset,in_axes=(0,None)))
_,mixed_states=batch_reset(reset_ids,replace(params,config=mixed))
standard_mask=np.asarray(mixed_states.car_debt)==500.
assert .15<float(standard_mask.mean())<.35
assert np.all(np.asarray(mixed_states.cash)[standard_mask]==100.)
assert np.allclose(np.asarray(mixed_states.satiety)[standard_mask],base.starting_satiety)
_,gate_states=batch_reset(reset_ids,replace(params,config=curriculum.evaluation_configuration(mixed)))
assert np.all(np.asarray(gate_states.car_debt)==150.)

timing=[]
for wait in (1,8):
    p=replace(params,config=replace(base,unavailable_wait_ticks=wait))
    _,s=strategy.reset(11,p)
    s=replace(s,agent_pos=params.zone_centers[city.WORK],shift_claimed=jnp.bool_(True))
    _,new,reward,_,trace=eqx.filter_jit(strategy.advance)(s,env.Action(jnp.zeros(2),jnp.int32(strategy.WORK)),p)
    duration=int(trace['active'].sum())
    assert duration==wait and int(new.time-s.time)==wait
    assert int(new.counters[env.PAID])==0
    assert np.isclose(float(s.satiety-new.satiety),wait*base.hunger_tick,atol=1e-6)
    assert np.isclose(float(reward),float((trace['reward']*trace['active']*base.gamma**jnp.arange(128)).sum()),atol=1e-6)
    timing.append({'unavailable_wait_ticks':wait,'physical_duration':duration,'reward':float(reward)})
result={'passed':True,'reset_checks':rows,'unavailable_service_checks':timing,
        'standard_home_rehearsal_share':float(standard_mask.mean()),
        'source_hashes':protocol.source_hashes(),
        'scope':'Reset, transaction and curriculum gate checks.'}
(out/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('RECOVERY_CHECKS',json.dumps(result),flush=True)
