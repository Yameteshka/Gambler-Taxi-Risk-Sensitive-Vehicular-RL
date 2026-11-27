"""Evaluate service policies through shared driving; retain every physical step."""
from dataclasses import replace, asdict
from functools import lru_cache
from pathlib import Path
import json
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env, city
import strategy, models
from telemetry import write_chunk

RESET=eqx.filter_jit(jax.vmap(strategy.reset,in_axes=(0,None)))


@lru_cache(maxsize=64)
def compiled_block(kind,deterministic,controller,window):
 @eqx.filter_jit
 def block(states,hidden,returns,model,params):
  def tick(carry,index):
   states,hidden,returns=carry
   obs=jax.vmap(strategy.get_observation,in_axes=(0,None))(states,params)
   keys=jax.vmap(lambda s,t:jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(9181),s),t))(states.scenario_id,states.time)
   actions,hidden,_,raw,lp,hc,hd=models.policy_step(model,kind,obs.features,obs.action_mask,hidden,keys,deterministic)
   decisions={"scenario_id":states.scenario_id,"time":states.time,"active":(states.outcome==env.ACTIVE)&(states.time<window),
              "observation":obs.features,"action_mask":obs.action_mask,"choice":actions.choice,"control":actions.control,"raw_control":raw,
              "log_probability":lp,"entropy_c":hc,"entropy_d":hd}
   _,new,_,_,physical=jax.vmap(strategy.advance,in_axes=(0,0,None,None,None))(states,actions,params,controller,window)
   decisions["duration"]=physical["active"].sum(axis=1)
   external=(physical["external_reward"]*physical["active"]*params.config.gamma**jnp.arange(strategy.STEPS)[None]).sum(axis=1)
   returns+=params.config.gamma**states.time*external
   return (new,hidden,returns),(physical,decisions)
  return jax.lax.scan(tick,(states,hidden,returns),jnp.arange(8))
 return block


def flatten_physical(trace):
 # Collector/evaluator arrays are [decision, environment, physical_tick, ...].
 return jax.tree.map(lambda x:jnp.swapaxes(x,1,2).reshape((-1,x.shape[1],*x.shape[3:])),trace)


def evaluate(model,kind,params,scenario_ids,output=None,deterministic=False,controller="shared"):
 from evaluate import summary
 from statistics_report import uncertainty
 controller="shared" if controller=="learned" else controller
 window=params.config.horizon or 30*params.config.day
 params=replace(params,config=replace(params.config,loan_capacity=max(params.config.loan_capacity,(window+params.config.loan_duration-1)//params.config.loan_duration+2)))
 ids=jnp.asarray(scenario_ids,dtype=jnp.int32)
 obs,states=RESET(ids,params)
 hidden=models.initial_memory(kind,len(ids),obs.features.shape[-1])
 returns=jnp.zeros(len(ids));index=0
 block=compiled_block(kind,deterministic,controller,window)
 while bool(np.any((np.asarray(states.outcome)==env.ACTIVE)&(np.asarray(states.time)<window))):
  (states,hidden,returns),(physical,decisions)=block(states,hidden,returns,model,params)
  physical=flatten_physical(physical)
  if np.any(np.asarray(physical["capacity_needed"])):raise RuntimeError("Insufficient loan allocation")
  if output is not None:
   folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
   write_chunk(folder/f"trace_{index:05}.npz",physical,{"interface":"strategy","maximum_physical_steps_per_decision":strategy.STEPS})
   np.savez_compressed(folder/f"decision_{index:05}.npz",**jax.device_get(decisions))
  index+=1
 result=summary(states,returns,window)
 if output is not None:
  folder=Path(output)
  payload={**result,"config":asdict(params.config),"kind":kind,"interface":"strategy","macro_names":strategy.NAMES,
           "scenario_ids":list(map(int,scenario_ids)),"deterministic":deterministic,"controller":controller,
           "maximum_physical_steps_per_decision":strategy.STEPS,"decision_duration":"variable; see decision duration arrays",
           "macro_zones":np.asarray(strategy.ZONES).tolist(),"service_centers":city.ZONES}
  (folder/"evaluation.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
  np.savez_compressed(folder/"endpoints.npz",scenario_id=np.asarray(ids),external_return=np.asarray(returns),
      **{k:np.asarray(getattr(states,k)) for k in ("outcome","time","cash","car_debt","counters","loan_ids","loan_principal","loan_balance","loan_rates","loan_opened")},
      micro_debt=np.asarray(env.micro_debt(states)))
  (folder/"uncertainty.json").write_text(json.dumps(uncertainty(np.asarray(returns),np.asarray(states.outcome)),indent=2),encoding="utf-8")
  (folder/"complete.json").write_text(json.dumps({"episodes":len(ids),"window":window,"interface":"strategy"}),encoding="utf-8")
 return result,states,returns
