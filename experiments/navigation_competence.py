"""Development-only shared-driver benchmark over all directed service pairs."""
from dataclasses import replace
from pathlib import Path
import sys,json,time,hashlib,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env,city
import navigation,protocol

parser=argparse.ArgumentParser()
parser.add_argument("--horizon",type=int,default=24)
parser.add_argument("--hold",type=int,default=4)
args=parser.parse_args()

cases=[(seed,a,b) for seed in range(1,11) for a in range(10) for b in range(10) if a!=b]
ids,start,goal=jnp.asarray(cases).T
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(days=0,loan_capacity=8,hunger_tick=0.,hunger_drive=0.))
_,states=jax.vmap(env.ResourceTaxiEnv.reset,in_axes=(0,None))(ids,params)
states=replace(states,agent_pos=params.zone_centers[start])

@eqx.filter_jit
def run(states):
 def tick(carry,index):
  state,arrived=carry
  def command(s,target):
   return navigation.guard(s,params,navigation.nominal_to_zone(s,params,target,.9,1.),horizon=args.horizon,hold=args.hold)
  controls=jax.vmap(command)(state,goal)
  _,new,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
      jax.random.split(jax.random.PRNGKey(0),len(cases)),state,env.Action(controls,jnp.zeros(len(cases),dtype=jnp.int32)),params)
  reached=(jnp.linalg.norm(new.agent_pos-params.zone_centers[goal],axis=-1)<=params.config.zone_radius)&(new.speed<params.config.parking_speed)&(new.outcome==env.ACTIVE)
  freeze=arrived|(state.outcome!=env.ACTIVE)
  new=jax.tree.map(lambda a,b:jnp.where(freeze.reshape((len(cases),)+(1,)*(a.ndim-1)),b,a),new,state)
  return (new,arrived|reached),(new.agent_pos,new.speed)
 return jax.lax.scan(tick,(states,jnp.zeros(len(cases),dtype=jnp.bool_)),jnp.arange(256))

started=time.perf_counter();(final,arrived),traces=run(states);jax.block_until_ready(arrived)
ok=np.asarray(arrived);failures=[]
for i in np.flatnonzero(~ok):
 seed,a,b=cases[i]
 failures.append({"scenario":seed,"from":city.NAMES[a],"to":city.NAMES[b],"outcome":int(final.outcome[i]),"position":np.asarray(final.agent_pos[i]).tolist()})
result={"scope":"Navigation only: all directed service pairs, fixed initial east heading, ten development traffic scenarios. Arrival is not a full-game ownership result.",
        "routes":len(cases),"arrivals":int(ok.sum()),"rate":float(ok.mean()),"collisions":int((np.asarray(final.outcome)==env.CRASH).sum()),"horizon":args.horizon,"hold":args.hold,
        "unfinished":int((~ok&(np.asarray(final.outcome)==env.ACTIVE)).sum()),"max_physical_ticks":256,"elapsed_seconds":time.perf_counter()-started,
        "failures":failures,"source_hashes":protocol.source_hashes(),"benchmark_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
folder=Path(__file__).resolve().parent
(folder/f"navigation_h{args.horizon}_hold{args.hold}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
(folder/"navigation_competence.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
np.savez_compressed(folder/"navigation_competence.npz",cases=np.asarray(cases),arrived=ok,position=np.asarray(traces[0]),speed=np.asarray(traces[1]))
print(json.dumps({k:v for k,v in result.items() if k not in ("failures","source_hashes")}),flush=True)
