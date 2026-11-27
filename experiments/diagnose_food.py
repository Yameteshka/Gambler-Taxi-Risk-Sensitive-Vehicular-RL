"""Food execution unit probes, never demonstrations or training examples."""
from pathlib import Path
from dataclasses import replace
import sys,json
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import strategy
from env import resource_taxi as env,city

params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(loan_capacity=8))
cases=[(seed,start,option) for seed in range(1,11) for start in (city.HOME,city.WORK)
       for option in (strategy.DINER,strategy.CAFE,strategy.RESTAURANT)]
ids,starts,options=jnp.asarray(cases).T
_,states=jax.vmap(strategy.reset,in_axes=(0,None))(ids,params)
states=replace(states,agent_pos=params.zone_centers[starts])
@eqx.filter_jit
def probe(states):
 def tick(carry,index):
  state,first_meal=carry
  _,new,_,_,rows=jax.vmap(strategy.advance,in_axes=(0,0,None))(
      state,env.Action(jnp.zeros((len(cases),2)),options),params)
  delivered=new.counters[:,env.MEALS]>0
  first_meal=jnp.where((first_meal<0)&delivered,new.time,first_meal)
  frozen=(state.counters[:,env.MEALS]>0)|(state.outcome!=env.ACTIVE)
  new=jax.tree.map(lambda a,b:jnp.where(frozen.reshape((len(cases),)+(1,)*(a.ndim-1)),b,a),new,state)
  return (new,first_meal),(new.counters[:,env.MEALS],rows['active'].sum(axis=1))
 return jax.lax.scan(tick,(states,jnp.full(len(cases),-1)),jnp.arange(8))
(end,times),(meals,durations)=probe(states)
records=[]
for i,(seed,start,option) in enumerate(cases):
 records.append(dict(scenario=seed,start=city.NAMES[start],destination=strategy.NAMES[option],
     first_intention_delivered=bool(meals[0,i]>0),delivered_with_repeated_target=bool(end.counters[i,env.MEALS]>0),
     completion_tick=int(times[i]),outcome=int(end.outcome[i]),cash=float(end.cash[i])))
result={'scope':'Development-only execution tests of requested food destinations. Fixed repeated commands test mechanics; no learned weights are changed and no examples enter training.',
        'cases':records,'one_intention_deliveries':sum(r['first_intention_delivered'] for r in records),
        'repeated_target_deliveries':sum(r['delivered_with_repeated_target'] for r in records),'total_cases':len(records)}
(ROOT/'experiments/food_execution_after_fix.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
