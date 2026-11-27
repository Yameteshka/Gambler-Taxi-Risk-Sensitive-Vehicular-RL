"""Matched development route checks for the existing and candidate shared drivers."""
from pathlib import Path
from dataclasses import replace
import sys,json,time,argparse
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env,city
import navigation
parser=argparse.ArgumentParser();parser.add_argument('--driver',choices=('brake','predictive'),required=True);args=parser.parse_args()
pairs=((2,1),(1,2),(2,3),(0,1),(0,9),(9,1),(0,6),(6,1),(9,2))
cases=[(seed,a,b) for seed in range(1,11) for a,b in pairs]
ids,start,goal=jnp.array(cases).T
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(days=0,loan_capacity=8,
    hunger_tick=0.,hunger_drive=0.,predictive_driver=args.driver=='predictive',reward_shaping=False))
_,states=jax.vmap(env.ResourceTaxiEnv.reset,in_axes=(0,None))(ids,params)
states=replace(states,agent_pos=params.zone_centers[start])
@eqx.filter_jit
def run(states):
 def tick(carry,index):
  s,arrived=carry
  def command(one,target):
   if args.driver=='predictive':return navigation.predictive_guard(one,params,target,.9,1.)
   return navigation.guard(one,params,navigation.nominal_to_zone(one,params,target,.9,1.))
  controls=jax.vmap(command)(s,goal)
  _,new,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
      jax.random.split(jax.random.key(0),len(cases)),s,env.Action(controls,jnp.zeros(len(cases),dtype=jnp.int32)),params)
  reached=(jnp.linalg.norm(new.agent_pos-params.zone_centers[goal],axis=-1)<=params.config.zone_radius)&(new.speed<params.config.parking_speed)&(new.outcome==env.ACTIVE)
  freeze=arrived|(s.outcome!=env.ACTIVE)
  new=jax.tree.map(lambda a,b:jnp.where(freeze.reshape((len(cases),)+(1,)*(a.ndim-1)),b,a),new,s)
  return (new,arrived|reached),None
 return jax.lax.scan(tick,(states,jnp.zeros(len(cases),dtype=jnp.bool_)),jnp.arange(256))[0]
started=time.perf_counter();final,arrived=run(states);jax.block_until_ready(arrived)
result={'driver':args.driver,'routes':len(cases),'arrivals':int(np.asarray(arrived).sum()),
 'collisions':int((np.asarray(final.outcome)==env.CRASH).sum()),'elapsed_seconds':time.perf_counter()-started,
 'by_pair':{},'scope':'90 matched development routes, fixed initial heading. Not a full-game success rate.'}
for a,b in pairs:
 ix=np.array([(x==a and y==b) for _,x,y in cases]);result['by_pair'][city.NAMES[a]+' -> '+city.NAMES[b]]={
  'arrivals':int(np.asarray(arrived)[ix].sum()),'collisions':int((np.asarray(final.outcome)[ix]==env.CRASH).sum())}
folder=ROOT/'reports/local_learning_probe';folder.mkdir(parents=True,exist_ok=True)
(folder/f'driver_{args.driver}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
