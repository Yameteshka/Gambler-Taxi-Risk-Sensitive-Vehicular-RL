"""Reproduce saved service-policy decisions and every underlying physical step."""
from pathlib import Path
import json
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env
import models
from evaluate_strategy import RESET,compiled_block,flatten_physical
from telemetry import file_hash


def compare_fields(recorded,expected,active,skip=()):
 maximum=0.
 for name,value in expected.items():
  if name in skip:continue
  actual=recorded[name];wanted=np.asarray(value)
  if actual.shape!=wanted.shape or wanted.shape[:2]!=active.shape:
   raise AssertionError(f"{name}: invalid batch/time shape {actual.shape} vs {wanted.shape}")
  if name=="active":np.testing.assert_array_equal(actual,wanted)
  elif active.any():
   if wanted.dtype.kind in "biu":np.testing.assert_array_equal(actual[active],wanted[active],err_msg=name)
   else:
    maximum=max(maximum,float(np.max(np.abs(actual[active]-wanted[active]))))
    np.testing.assert_allclose(actual[active],wanted[active],rtol=2e-6,atol=2e-5,err_msg=name)
 return maximum


def audit(directory,checkpoint):
 directory=Path(directory)
 meta=json.loads((directory/"evaluation.json").read_text(encoding="utf-8"))
 params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(**meta["config"]))
 ids=jnp.array(meta["scenario_ids"],dtype=jnp.int32)
 obs,states=RESET(ids,params)
 model,saved=models.load_policy(checkpoint)
 if saved.get("interface")!="strategy":raise ValueError("A strategy checkpoint is required")
 kind=meta["kind"];window=params.config.horizon or 30*params.config.day
 hidden=models.initial_memory(kind,len(ids),obs.features.size//len(ids))
 returns=jnp.zeros(len(ids));physical_count=decision_count=0;maximum=0.;hashes={}
 replay=compiled_block(kind,meta["deterministic"],meta["controller"],window)
 for path in sorted(directory.glob("trace_*.npz")):
  (states,hidden,returns),(physical,decisions)=replay(states,hidden,returns,model,params)
  physical=flatten_physical(physical)
  with np.load(path,allow_pickle=False) as data:
   physical_count+=int(data["active"].sum())
   maximum=max(maximum,compare_fields(data,physical,data["active"],("loan_ids","loan_repayments")))
   mask=np.asarray(physical["micro_payment"])>0
   if mask.any():
    payments=np.asarray(physical["loan_repayments"])[mask];loan_ids=np.asarray(physical["loan_ids"])[mask]
    r,c=np.nonzero(payments>0)
    np.testing.assert_array_equal(data["loan_receipt_location"],np.argwhere(mask)[r])
    np.testing.assert_array_equal(data["loan_receipt_id"],loan_ids[r,c])
    np.testing.assert_allclose(data["loan_receipt_amount"],payments[r,c],rtol=2e-6,atol=2e-5)
  decision_path=directory/path.name.replace("trace_","decision_")
  with np.load(decision_path,allow_pickle=False) as data:
   decision_count+=int(data["active"].sum())
   maximum=max(maximum,compare_fields(data,decisions,data["active"]))
  hashes[path.name]=file_hash(path);hashes[decision_path.name]=file_hash(decision_path)
 with np.load(directory/"endpoints.npz",allow_pickle=False) as data:
  for name in ("time","cash","car_debt","counters","loan_ids","loan_principal","loan_balance","loan_rates","loan_opened","outcome"):
   np.testing.assert_allclose(np.asarray(getattr(states,name)),data[name],rtol=2e-6,atol=2e-5,err_msg=name)
  np.testing.assert_allclose(np.asarray(returns),data["external_return"],rtol=2e-6,atol=2e-5)
 result={"passed":True,"interface":"strategy","active_physical_steps_replayed":physical_count,
         "active_service_decisions_replayed":decision_count,"maximum_logged_float_error":maximum,
         "policy_reproduced":True,"checkpoint_sha256":file_hash(checkpoint),"trace_sha256":hashes,
         "scope":"Full policy-and-controller replay and exact per-loan receipt checks."}
 (directory/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 return result
