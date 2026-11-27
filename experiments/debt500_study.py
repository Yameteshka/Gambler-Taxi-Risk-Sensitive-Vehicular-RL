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
from run_suite import variants as previous_variants

def variants(config):
 yield from previous_variants(config)
 for debt in (100.,250.,1000.,1500.):
  from dataclasses import replace
  yield f"initial_car_debt_{int(debt)}",replace(config,starting_debt=debt),"test_standard","learned"
 from dataclasses import replace
 yield "casino_unavailable",replace(config,casino=False),"test_standard","learned"

STUDY=ROOT/'artifacts/debt500_study'
REPORTS=ROOT/'reports/debt500_study'
ORDER=(('hybrid_ppo','mlp'),('hybrid_ppo','gru'),('hybrid_dsac','mlp'),('hybrid_dsac','gru'))
CURRICULUM='cold_debt500'
ROW_EXTRAS={}
STUDY_METADATA={}
TRAINING_ONLY=False
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
  'order':ORDER,'initialization_seed':301,'budget_per_model':4194304,'economic_demonstrations':False,'public_food_menu':True,'final_training_debt':500,'cold_start_fraction':.1,'cold_exploration_epsilon':.8,'service_observation_features':194,
  'curriculum':{'fractions':[.1,.15,.25,.25,.25],'car_principal':[50,50,150,300,500],
   'home_start_probability':[0,.25,.5,.75,1.],'hunger_multiplier':[.65,.65,.8,1,1],
   'market':['fixed','fixed','dynamic','dynamic','dynamic'],'physical_discount':.999},
  'training_scenarios':[11,10010],'selection_validation':[1000001,1000064],
  'new_standard_final_scenarios':[3000001,3000256],
  'new_adverse_banks':'3100001 + 100000*k, each 256 scenarios; one matched control per bank',
  'limitations':'One initialization per architecture is exploratory; scenario replication is not independent training replication. Five-day and continuing evaluation are horizon transfer, not separate horizon-specific retraining.',
  'sources':protocol.source_hashes(), 'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 design=json.loads(json.dumps(design))
 if ROW_EXTRAS:
  design.update(treatment=ROW_EXTRAS,**STUDY_METADATA)
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
   'initialization_seed':'301','curriculum':'cold_debt500','starting_debt':'500','training_horizon':'10d','exploration':'standard',
   'training_market':'dynamic','casino':'enabled','physical_transition_budget':'4194304',
   'protocol_revision':'cold_debt500_01','public_food_menu':'enabled','status':'planned',
   **ROW_EXTRAS,'curriculum':CURRICULUM}
  status(state='training',current_model=kind)
  folder=STUDY/'training'/kind
  if not (folder/'complete.json').exists():
   try:
    train(row,resume=(folder/'resume.pkl.gz').exists(),output_root=(STUDY/'training').relative_to(ROOT))
   except RuntimeError:
    failure=folder/'curriculum_failure.json'
    if CURRICULUM not in ('resource_frontier','ownership_frontier') or not failure.exists():raise
    skills[kind]={'ownership':0,'training_readiness_passed':False,'training_failure':json.loads(failure.read_text())}
    (REPORTS/'readiness.json').write_text(json.dumps(skills,indent=2),encoding='utf-8')
    status(state='model_learning_failure',current_model=kind,readiness=skills)
    print('LEARNING_FAILURE '+kind+' '+json.dumps(skills[kind]),flush=True)
    continue
  model,meta=models.load_policy(folder/'selected.npz');params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(**meta['evaluation_environment']))
  output=STUDY/'readiness'/kind
  status(state='checking_readiness',current_model=kind)
  if not (output/'audit.json').exists():
   evaluate(model,kind,params,range(1000001,1000065),output,interface='strategy')
   audit(output,folder/'selected.npz')
  skills[kind]=inspect(output)
  if CURRICULUM=='ownership_frontier':
   from driver_diagnostics import diagnose
   skills[kind]['driver_diagnostics']=diagnose(output)
  coverage=folder/'cold_start_coverage.json'
  if coverage.exists():
   skills[kind]['cold_start_coverage']=json.loads(coverage.read_text())
   skills[kind]['training_readiness_passed']=skills[kind]['cold_start_coverage']['passed']
  else:
   gates=[json.loads(line) for line in (folder/'curriculum_gates.jsonl').read_text().splitlines()]
   skills[kind]['training_readiness_passed']=gates[-1]['passed'] and gates[-1]['lesson']=='standard_home_actor'
  if CURRICULUM in ('resource_frontier','ownership_frontier'):
   from dataclasses import replace
   import ownership_curriculum
   skills[kind]['retained_resource_contexts']={}
   contexts=ownership_curriculum.RETAINED_CONTEXTS if CURRICULUM=='ownership_frontier' else ((2,'earning'),(3,'food'),(4,'fuel'),(6,'personal_debt'))
   for level,label in contexts:
    reset_field={'ownership_reset_level':level,'training_reset_level':-1} if CURRICULUM=='ownership_frontier' else {'training_reset_level':level}
    context=replace(params.config,**reset_field,training_rehearsal=0.,exploration_epsilon=0.)
    context_params=replace(params,config=context)
    result,_,_=evaluate(model,kind,context_params,range(1000001,1000017),STUDY/'retained_skills'/kind/label,interface='strategy')
    skills[kind]['retained_resource_contexts'][label]=result
   skills[kind]['training_readiness_passed'] &= all(
       result['ownership']>=4 for result in skills[kind]['retained_resource_contexts'].values())
   if CURRICULUM=='ownership_frontier':
    result=json.loads((output/'evaluation.json').read_text())
    stalls=skills[kind]['driver_diagnostics']['flagged_episodes']
    skills[kind]['standard_readiness']={'passed':result['ownership']>=32 and result['starvation']<=16 and stalls<=4,
        'required_ownership':32,'maximum_starvation':16,'maximum_potential_driver_stalls':4,'episodes':64}
    skills[kind]['training_readiness_passed'] &= skills[kind]['standard_readiness']['passed']
  (REPORTS/'readiness.json').write_text(json.dumps(skills,indent=2),encoding='utf-8')
  print('READINESS '+kind+' '+json.dumps(skills[kind]),flush=True)
 # Do not burn a new final test bank to discover a learner is not ready.
 if any(v['ownership']==0 or not v['training_readiness_passed'] for v in skills.values()):
  status(state='needs_learning_diagnosis',reason='At least one learner lacks validation ownership or an observed essential resource action (meal, sleep, work, refill). New final banks remain untouched.',readiness=skills)
  return
 if TRAINING_ONLY:
  status(state='training_complete_evaluation_deferred',readiness=skills,
      reason='Keep final banks untouched until all four architectures are frozen and ready.')
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
   if CURRICULUM=='ownership_frontier':
    from driver_diagnostics import diagnose
    diagnose(output)
 status(state='evaluations_complete',readiness=skills)

if __name__=='__main__':
 try:main()
 except BaseException as error:
  if STUDY.exists():status(state='stopped',error=repr(error))
  raise
