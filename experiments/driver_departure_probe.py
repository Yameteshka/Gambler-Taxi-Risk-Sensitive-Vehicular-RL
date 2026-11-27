"""Replay preserved physical actions, then compare actual bank-departure states.

Validation data are development data here. Fuel/hunger are equalized only for
this navigation diagnostic; full study dynamics retain all resource costs.
"""
from pathlib import Path
import os
from dataclasses import replace
import sys,json,time,argparse
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env,city
import navigation
parser=argparse.ArgumentParser();parser.add_argument('--escape',action='store_true');args=parser.parse_args()
run=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'progressive_curriculum_pilot/output/artifacts/progressive_curriculum_pilot/training/dsac_mlp'
folder=sorted((run/'validation_traces').glob('*'))[-1]
record=json.loads((folder/'evaluation.json').read_text())
c=replace(env.Config(**record['config']),reward_shaping=False,predictive_escape=args.escape)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),c)
ids=jnp.array(record['scenario_ids'],dtype=jnp.int32)
_,states=jax.vmap(env.ResourceTaxiEnv.reset,in_axes=(0,None))(ids,params)
@eqx.filter_jit
def replay(carry,controls,choices,active,start,macros,styles):
 def tick(carry,row):
  states,grabbed,found,style=carry
  control,choice,live,begin,macro,requested_style=row
  bank=jnp.linalg.norm(states.agent_pos-params.zone_centers[city.BANK],axis=-1)<c.zone_radius+.1
  capture=live&begin&(macro==1)&bank&~found
  grabbed=jax.tree.map(lambda old,new:jnp.where(capture.reshape((ids.size,)+(1,)*(new.ndim-1)),new,old),grabbed,states)
  style=jnp.where(capture[:,None],requested_style,style)
  _,new,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
      jax.random.split(jax.random.key(0),ids.size),states,env.Action(control,choice),params)
  states=jax.tree.map(lambda old,new:jnp.where(live.reshape((ids.size,)+(1,)*(new.ndim-1)),new,old),states,new)
  return (states,grabbed,found|capture,style),None
 return jax.lax.scan(tick,carry,(controls,choices,active,start,macros,styles))[0]
carry=(states,states,jnp.zeros(ids.size,dtype=jnp.bool_),jnp.zeros((ids.size,2)))
for path in sorted(folder.glob('trace_*.npz')):
 with np.load(path) as data:
  carry=replay(carry,*[jnp.asarray(data[key]) for key in ('control','requested','active','decision_start','macro_executed','macro_control')])
replayed,captured,found,styles=carry
with np.load(folder/'endpoints.npz') as endpoints:
 for field in ('time','cash','car_debt','loan_balance','counters','outcome'):
  np.testing.assert_allclose(np.asarray(getattr(replayed,field)),endpoints[field],rtol=2e-6,atol=3e-5,err_msg=field)
ix=np.flatnonzero(np.asarray(found));assert len(ix)>0
captured=jax.tree.map(lambda value:value[ix],captured);styles=styles[ix]
# Original headings, steering, fatigue, traffic positions and velocities persist.
# Isolate routing by providing both controllers the same survival resources.
diagnostic=replace(c,hunger_tick=0.,hunger_drive=0.,hunger_work=0.,hunger_push=0.)
params=replace(params,config=diagnostic)
captured=replace(captured,satiety=jnp.ones(len(ix)),fuel=jnp.full(len(ix),25.))
def compare(predictive):
 @eqx.filter_jit
 def run(states):
  def tick(carry,index):
   states,arrived=carry
   def command(state,style):
    if predictive:return navigation.predictive_guard(state,params,city.WORK,.9+.1*style[0],1.+.1*style[1])
    return navigation.guard(state,params,navigation.nominal_to_zone(state,params,city.WORK,.9+.1*style[0],1.+.1*style[1]))
   controls=jax.vmap(command)(states,styles)
   _,new,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
     jax.random.split(jax.random.key(0),len(ix)),states,env.Action(controls,jnp.zeros(len(ix),dtype=jnp.int32)),params)
   reached=(jnp.linalg.norm(new.agent_pos-params.zone_centers[city.WORK],axis=-1)<=c.zone_radius)&(new.speed<c.parking_speed)&(new.outcome==env.ACTIVE)
   freeze=arrived|(states.outcome!=env.ACTIVE)
   new=jax.tree.map(lambda a,b:jnp.where(freeze.reshape((len(ix),)+(1,)*(a.ndim-1)),b,a),new,states)
   return (new,arrived|reached),None
  return jax.lax.scan(tick,(states,jnp.zeros(len(ix),dtype=jnp.bool_)),jnp.arange(256))[0]
 started=time.perf_counter();end,arrived=run(captured);jax.block_until_ready(arrived)
 return {'arrivals':int(np.asarray(arrived).sum()),'collisions':int((np.asarray(end.outcome)==env.CRASH).sum()),
    'elapsed_seconds':time.perf_counter()-started,'arrived_scenarios':np.asarray(captured.scenario_id)[np.asarray(arrived)].tolist()}
result={'states':len(ix),'escape':args.escape,'recorded_endpoint_replay_passed':True,'source_validation':folder.name,'brake':compare(False),'predictive':compare(True),
 'scope':'Matched bank-departure states reconstructed by real-action replay. Resources equalized for navigation only; not an ownership evaluation.'}
out=ROOT/('reports/local_learning_probe/actual_bank_departures_escape.json' if args.escape else 'reports/local_learning_probe/actual_bank_departures.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
