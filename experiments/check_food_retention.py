"""Evaluate frozen lesson checkpoints on one matched, food-required development context.

No optimizer updates, rewards, action labels, new training or final-bank access.
Imports the original downloaded source snapshot to preserve execution semantics.
"""
from pathlib import Path
import os
from dataclasses import replace
import json,sys,hashlib,time

ROOT=Path(__file__).resolve().parents[1]
STUDY=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study'
SNAPSHOT=STUDY/'training/dsac_gru/source'
sys.path.insert(0,str(SNAPSHOT/'src'))
import jax,numpy as np
import models,ownership_curriculum as curriculum
from env import resource_taxi as env
from evaluate import evaluate

OUT=ROOT/'reports/resource_study_review_2026-11-23/food_retention'
OUT.mkdir(parents=True,exist_ok=True)
cases={'ppo_mlp':('lesson_06_passed','lesson_09_passed','lesson_11_passed','last'),
       'ppo_gru':('lesson_06_passed','lesson_09_passed','lesson_11_passed','last'),
       'dsac_mlp':('lesson_02_passed','last'),
       'dsac_gru':('selected',)}
results=[]
print('DEVICES',jax.devices(),flush=True)
for kind,labels in cases.items():
    for label in labels:
        checkpoint=STUDY/'training'/kind/(label+'.npz')
        raw=checkpoint.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        actor,metadata=models.load_policy(checkpoint)
        for name,expected in metadata['source_hashes'].items():
            if hashlib.sha256((SNAPSHOT/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Frozen source mismatch: '+name)
        config=env.Config(**metadata['evaluation_environment'])
        config=curriculum.evaluation_configuration(curriculum.configuration(config,6))
        params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
        started=time.monotonic()
        summary,states,returns=evaluate(actor,kind,params,curriculum.VALIDATION_IDS,
            output=None,interface='strategy',controller='shared',deterministic=False)
        row={'model':kind,'checkpoint':label,'checkpoint_sha256':digest,
             'physical_training_ticks':metadata.get('physical_transitions'),
             'context':'food_approach','scenario_ids':list(curriculum.VALIDATION_IDS),
             'elapsed_seconds':time.monotonic()-started,'summary':summary,
             'scope':'Matched development scenarios evaluated with the actor and exploration override disabled.'}
        np.savez_compressed(OUT/(kind+'_'+label+'.npz'),
            outcome=np.asarray(states.outcome),counters=np.asarray(states.counters),
            satiety=np.asarray(states.satiety),cash=np.asarray(states.cash),
            time=np.asarray(states.time),external_return=np.asarray(returns))
        results.append(row)
        temporary=OUT/'results.tmp'
        temporary.write_text(json.dumps(results,indent=2),encoding='utf-8')
        temporary.replace(OUT/'results.json')
        print('RETENTION',json.dumps(row),flush=True)
