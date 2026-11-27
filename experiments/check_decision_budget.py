"""Exercise real training, checkpoint resume and the decision-budget boundary.

An intentionally tiny engineering run must stop as unqualified, never become
a completed scientific training. Its weights are never promoted.
"""
from pathlib import Path
import json,sys,gzip,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import protocol,strategy_training as training,ownership_curriculum
from resource_study import SETTINGS
row={**protocol.registry()[0],'run_id':'engineering_decision_boundary','algorithm':'hybrid_dsac',
    'architecture':'mlp','initialization_seed':'301','curriculum':ownership_curriculum.NAME,
    'starting_debt':'500','training_horizon':'10d','exploration':'standard','training_market':'dynamic',
    'casino':'enabled','physical_transition_budget':'131072','network_decision_budget':'1024',
    'public_food_menu':'enabled','reward_shaping':'disabled','navigation_cue':'disabled',
    'predictive_driver':'enabled','predictive_escape':'enabled','replay_warmup':'0',
    'dsac_settings':json.dumps(SETTINGS,sort_keys=True),'status':'engineering_check'}
folder=ROOT/'runs/engineering_checks'/row['run_id']
if folder.exists():raise RuntimeError('Preserve the existing engineering run instead of overwriting it')
out=training.train(row,stop_after=1,output_root='runs/engineering_checks')
with gzip.open(out/'resume.pkl.gz','rb') as f:first=pickle.load(f)
assert first['index']==1 and first['total']<131072
try:training.train(row,resume=True,output_root='runs/engineering_checks')
except RuntimeError as error:
    assert 'Total budget exhausted before all actor-only curriculum gates passed' in str(error)
else:raise AssertionError('Tiny curriculum run was incorrectly declared complete')
with gzip.open(out/'resume.pkl.gz','rb') as f:last=pickle.load(f)
assert last['index']==2 and last['total']<131072
failure=json.loads((out/'curriculum_failure.json').read_text())
assert failure['network_decisions']==1024 and not (out/'complete.json').exists()
config=json.loads((out/'config.json').read_text())
result={'passed':True,'resumed_from_decisions':512,'stopped_at_decisions':1024,
    'physical_ticks':last['total'],'not_declared_trained':True,'scope':'Engineering smoke only, no promoted weights.'}
(ROOT/'experiments/decision_budget_verification.json').write_text(json.dumps(result,indent=2))
print('DECISION_BUDGET_VERIFIED '+json.dumps(result),flush=True)
