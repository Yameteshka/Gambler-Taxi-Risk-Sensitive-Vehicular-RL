"""Development-only route competence and throughput checks for shared driving."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env
import strategy

params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0))
_,states=jax.vmap(strategy.reset,in_axes=(0,None))(jnp.arange(1,11),params)
commands=env.Action(jnp.zeros((10,2)),jnp.full(10,strategy.WORK,dtype=jnp.int32))

@eqx.filter_jit
def block(states):
 def tick(states,_):
  _,new,_,_,rows=jax.vmap(strategy.advance,in_axes=(0,0,None))(states,commands,params)
  return new,rows
 return jax.lax.scan(tick,states,jnp.arange(16))

started=time.perf_counter();final,rows=block(states);jax.block_until_ready(final.cash)
compile_elapsed=time.perf_counter()-started
started=time.perf_counter();_,_=block(states);jax.block_until_ready(_)
runtime=time.perf_counter()-started
result={"development_scenarios":list(range(1,11)),"test":"Repeated explicit GoWork intention; once per decision, physical work eligibility applies",
 "time":np.asarray(final.time).tolist(),"work_completed":np.asarray(final.counters[:,env.PAID]).tolist(),
 "outcome":np.asarray(final.outcome).tolist(),"positions":np.asarray(final.agent_pos).tolist(),
 "compile_and_first_run_seconds":compile_elapsed,"cached_run_seconds":runtime,
 "scheduled_physical_steps":16*strategy.STEPS*10,"device":str(jax.devices()),
 "mean_actual_step_rate":16*strategy.STEPS*10/runtime}
target=Path(__file__).resolve().parent/"shared_driver_check.json";target.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
