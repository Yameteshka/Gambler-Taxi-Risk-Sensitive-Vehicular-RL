"""Sequential, from-scratch training of four agents on one shared curriculum."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess
from datetime import datetime,timezone
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import protocol,models
from strategy_training import train
from env import resource_taxi as env
from evaluate import evaluate
from audit_results import audit
from run_suite import variants

STUDY=ROOT/'artifacts/generalist_study'
REPORTS=ROOT/'reports/generalist_study'
ORDER=(('hybrid_ppo','mlp'),('hybrid_ppo','gru'),('hybrid_dsac','mlp'),('hybrid_dsac','gru'))
def stamp():return datetime.now(timezone.utc).isoformat()
def status(**changes):
 p=STUDY/'status.json';d=json.loads(p.read_text()) if p.exists() else {}
 d.update(changes,updated_at=stamp());p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def inspect(output):
 with np.load(output/'endpoints.npz') as d:
  counts=d['counters'].sum(axis=0);outcomes=d['outcome']
 return {'episodes':len(outcomes),'ownership':int((outcomes==env.SUCCESS).sum()),
   'observed_completed_actions':dict(zip(env.COUNTER_NAMES,map(int,counts))),
   'scope':'Completed actions by the frozen selected policy.'}
def main():
 STUDY.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
 design={'method':'Four independent architectures, common ten-day training distribution; sequential execution; frozen-policy scenario interventions.',
  'order':ORDER,'initialization_seed':301,'budget_per_model':4194304,'economic_demonstrations':False,
  'curriculum':{'fractions':[.25,.25,.25,.25],'car_principal':[50,300,700,1000],
   'home_start_probability':[.25,.5,.75,1.],'hunger_multiplier':[.65,.8,1,1],
   'market':['fixed','dynamic','dynamic','dynamic'],'physical_discount':.999},
  'training_scenarios':[11,10010],'selection_validation':[1000001,1000064],
  'new_standard_final_scenarios':[3000001,3000256],
  'new_adverse_banks':'3100001 + 100000*k, each 256 scenarios; one matched control per bank',
  'limitations':'One initialization per architecture is exploratory; scenario replication is not independent training replication. Five-day and continuing evaluation are horizon transfer, not separate horizon-specific retraining.',
  'sources':protocol.source_hashes(), 'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 design=json.loads(json.dumps(design))
 frozen=STUDY/'freeze.json'
 if frozen.exists():
  if json.loads(frozen.read_text())!=design:raise RuntimeError('Study source or design changed; preserve and register a separate study')
 else:
  frozen.write_text(json.dumps(design,indent=2),encoding='utf-8')
  import shutil
  shutil.copytree(ROOT/'src',STUDY/'source',ignore=shutil.ignore_patterns('__pycache__'))
 baseline=protocol.registry()[0];skills={}
 for algorithm,architecture in ORDER:
  kind=('ppo' if algorithm=='hybrid_ppo' else 'dsac')+'_'+architecture
  row={**baseline,'run_id':kind,'algorithm':algorithm,'architecture':architecture,
   'initialization_seed':'301','curriculum':'generalist','training_horizon':'10d','exploration':'standard',
   'training_market':'dynamic','casino':'enabled','physical_transition_budget':'4194304',
   'protocol_revision':'shared_curriculum_01','status':'planned'}
  status(state='training',current_model=kind)
  folder=STUDY/'training'/kind
  if not (folder/'complete.json').exists():
   train(row,resume=(folder/'resume.pkl.gz').exists(),output_root='artifacts/generalist_study/training')
  model,meta=models.load_policy(folder/'selected.npz');params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(**meta['evaluation_environment']))
  output=STUDY/'readiness'/kind
  status(state='checking_readiness',current_model=kind)
  if not (output/'audit.json').exists():
   evaluate(model,kind,params,range(1000001,1000065),output,interface='strategy')
   audit(output,folder/'selected.npz')
  skills[kind]=inspect(output)
  (REPORTS/'readiness.json').write_text(json.dumps(skills,indent=2),encoding='utf-8')
  print('READINESS '+kind+' '+json.dumps(skills[kind]),flush=True)
 # Do not burn a new final test bank to discover a learner is not ready.
 if any(v['ownership']==0 or any(v['observed_completed_actions'][key]==0 for key in ('meals','sleeps','paid','refills')) for v in skills.values()):
  status(state='needs_learning_diagnosis',reason='At least one learner lacks validation ownership or an observed essential resource action (meal, sleep, work, refill). New final banks remain untouched.',readiness=skills)
  return
 for algorithm,architecture in ORDER:
  kind=('ppo' if algorithm=='hybrid_ppo' else 'dsac')+'_'+architecture
  model,meta=models.load_policy(STUDY/'training'/kind/'selected.npz')
  config=env.Config(**meta['evaluation_environment']);banks={'test_standard':3000001}
  for label,changed,bank,controller in variants(config):
   if bank not in banks:banks[bank]=3100001+100000*(len(banks)-1)
   output=STUDY/'evaluation'/kind/label;status(state='evaluating',current_model=kind,current_case=label)
   if (output/'audit.json').exists():continue
   params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),changed)
   evaluate(model,kind,params,range(banks[bank],banks[bank]+256),output,
      deterministic=label=='deterministic',controller=controller,interface='strategy')
   audit(output,STUDY/'training'/kind/'selected.npz')
 status(state='evaluations_complete',readiness=skills)

if __name__=='__main__':
 try:main()
 except BaseException as error:
  if STUDY.exists():status(state='stopped',error=repr(error))
  raise
