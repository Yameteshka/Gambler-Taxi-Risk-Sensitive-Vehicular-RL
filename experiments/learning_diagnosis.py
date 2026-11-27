"""Recompute decision/update accounting and entropy bounds from preserved runs."""
from pathlib import Path
import os
import json,math,numpy as np
ROOT=Path(__file__).resolve().parents[1]
run=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'progressive_curriculum_pilot/output/artifacts/progressive_curriculum_pilot/training/dsac_mlp'
rows=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
physical=decisions=0
for path in sorted((run/'traces').glob('chunk_*.npz')):
    with np.load(path) as data:
        physical+=int(data['active'].sum())
        decisions+=int((data['active']&data['decision_start']).sum())
def bound(ac,ad,gamma):
    return (ac*math.log(4)+ad*math.log(26))/(1-gamma)
result={'physical_steps':physical,'network_decisions':decisions,'optimizer_updates':len(rows)*32,
    'mean_physical_steps_per_decision':physical/decisions,'updates_per_decision':len(rows)*32/decisions,
    'reward_scale':'Ownership is approximately +1; finite unsuccessful base return is -0.1.',
    'entropy_upper_bound_initial_config':bound(.0006,.0012,.999),
    'entropy_upper_bound_proposed_fixed_config':bound(3e-6,3e-6,.999),
    'bound_assumptions':'Discounted entropy return, at most one decision per physical tick, categorical <= log(26), tanh control in [-1,1]^2 with differential entropy <= log(4). This is a conservative mathematical upper bound, NOT observed entropy reward and NOT a proof of the failure cause.'}
out=ROOT/'reports/learning_recovery';out.mkdir(parents=True,exist_ok=True)
(out/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
