"""Check reset accounting, invariant rules and short real ownership paths.

The hand-chosen paths are engineering feasibility witnesses only. They are
never supplied to a training actor or replay buffer.
"""
from pathlib import Path
from dataclasses import replace,asdict
import sys,json
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env
import strategy,ownership_curriculum as curriculum,protocol
final=env.Config(starting_debt=500.,public_food_menu=True,reward_shaping=False,
    navigation_cue=False,predictive_driver=True,predictive_escape=True,loan_capacity=16)
records=[];witnesses=[]
for level,lesson in enumerate(curriculum.LESSONS):
    c=curriculum.evaluation_configuration(curriculum.configuration(final,level))
    changes={k for k,v in asdict(c).items() if v!=asdict(final)[k]}
    assert changes<={'ownership_reset_level','training_rehearsal','exploration_epsilon'}
    params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),c)
    _,s=strategy.reset(11,params)
    assert float(s.cash)==100. and abs(float(s.car_debt+s.car_paid)-500.)<1e-4
    assert bool(jnp.isfinite(strategy.get_observation(s,params).features).all())
    assert strategy.get_observation(s,params).features.size==194
    assert int(s.time)==(100 if level==5 else 0)
    assert float(env.micro_debt(s))==(50. if level==9 else 0.)
    records.append({'level':level,'lesson':lesson.name,'cash':float(s.cash),'remaining_car_debt':float(s.car_debt),
        'micro_debt':float(env.micro_debt(s)),'time':int(s.time),'satiety':float(s.satiety)})
    if level not in (3,4,5,6):continue
    ids=jnp.arange(11,17);_,states=jax.vmap(strategy.reset,in_axes=(0,None))(ids,params)
    advance=eqx.filter_jit(jax.vmap(lambda s,a:strategy.advance(s,a,params)))
    phases=[(strategy.RESTAURANT,lambda s:s.counters[:,env.MEALS]>0)]
    if level==5:phases.append((strategy.WORK,lambda s:s.counters[:,env.PAID]>0))
    if level in (4,5,6):phases.append((strategy.CAR_ALL,lambda s:s.car_debt==0))
    phases.append((strategy.BUY,lambda s:s.outcome==env.SUCCESS))
    for option,condition in phases:
        for _ in range(4):
            live=~condition(states)&(states.outcome==env.ACTIVE)
            if not bool(live.any()):break
            _,new,_,_,_=advance(states,env.Action(jnp.zeros((6,2)),jnp.full(6,option,jnp.int32)))
            states=jax.tree.map(lambda x,y:jnp.where(live.reshape((6,)+(1,)*(x.ndim-1)),x,y),new,states)
    row={'level':level,'scenario_ids':np.asarray(ids).tolist(),'ownership':int((np.asarray(states.outcome)==env.SUCCESS).sum()),
        'outcomes':np.asarray(states.outcome).tolist(),'times':np.asarray(states.time).tolist(),
        'meals':np.asarray(states.counters[:,env.MEALS]).tolist(),'paid_shifts':np.asarray(states.counters[:,env.PAID]).tolist(),
        'manual_diagnostic_only':True}
    witnesses.append(row);print('FOOD_PATH_WITNESS '+json.dumps(row),flush=True)
result={'passed':all(r['ownership']==6 for r in witnesses),'reset_accounting':records,'food_path_witnesses':witnesses,
    'source_hashes':protocol.source_hashes(),'scope':'Accounting invariants and manually chosen feasibility witnesses; not learned performance.'}
(ROOT/'experiments/ownership_curriculum_verification.json').write_text(json.dumps(result,indent=2))
print('OWNERSHIP_CURRICULUM_CHECK',result['passed'],flush=True)
if not result['passed']:raise SystemExit(2)
