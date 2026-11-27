"""NumPy-only DSAC MLP actor probabilities on saved food-lesson inputs."""
from pathlib import Path
import os
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
old=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training/dsac_mlp'
gates=[json.loads(x) for x in (old/'curriculum_gates.jsonl').read_text().splitlines()]
last=gates[-1]
folder=old/'curriculum_validation'/f"stage_{last['stage']:02d}_step_{last['physical_transitions']:09d}"
with np.load(next(folder.glob('decision_*.npz'))) as d:
    features=d['observation'][0];mask=d['action_mask'][0]
def load(path):
    with np.load(path) as d:
        def decode(node):
            if 'dict' in node:return {k:decode(v) for k,v in node['dict'].items()}
            if 'tuple' in node:return tuple(decode(v) for v in node['tuple'])
            return d[node['array']].copy()
        return decode(json.loads(str(d['structure'])))
def score(actor):
    x=np.concatenate((features,mask.astype(features.dtype)),axis=-1)
    for w,b in actor[:-1]:
        x=x@w+b;x=x/(1+np.exp(-np.clip(x,-80,80)))
    x=x@actor[-1][0]+actor[-1][1]
    logits=np.where(mask,x[:,4:],-1e9);v=logits-logits.max(-1,keepdims=True)
    probs=np.exp(v);probs/=probs.sum(-1,keepdims=True)
    return {'food_probability_mean':float(probs[:,3:6].sum(-1).mean()),
        'ownership_probability_mean':float(probs[:,25].mean()),
        'entropy_mean':float(-(probs*np.log(np.maximum(probs,1e-30))).sum(-1).mean()),
        'mean_probabilities':probs.mean(0).tolist()}
paths=[old/'lesson_02_passed.npz',old/'last.npz']
paths+=sorted((ROOT/'artifacts/local_food_recovery').glob('dsac_mlp_*/final.npz'))
rows=[]
for path in paths:
    row={'checkpoint':path.name,**score(load(path))};rows.append(row)
    print(json.dumps(row),flush=True)
out=ROOT/'reports/local_food_recovery';out.mkdir(parents=True,exist_ok=True)
(out/'actor_concentration.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
