"""Read saved DSAC critic estimates on the recorded near-restaurant hunger state."""
from pathlib import Path
import os,sys,json,gzip,pickle
os.environ['JAX_PLATFORMS']='cpu'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax.numpy as jnp,numpy as np
from algorithms import hybrid_dsac as dsac
old=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training/dsac_mlp'
gates=[json.loads(x) for x in (old/'curriculum_gates.jsonl').read_text().splitlines()]
g=gates[-1];folder=old/'curriculum_validation'/f"stage_{g['stage']:02d}_step_{g['physical_transitions']:09d}"
with np.load(next(folder.glob('decision_*.npz'))) as d:
    features=jnp.asarray(d['observation'][0]);mask=jnp.asarray(d['action_mask'][0])
tick=next(x['physical_transitions'] for x in gates if x['stage']==2 and x['passed'])
paths=[old/f'learner_{tick:09d}.pkl.gz']
paths+=sorted((ROOT/'artifacts/local_food_recovery').glob('dsac_mlp_*/learner.pkl.gz'))
rows=[]
for path in paths:
    with gzip.open(path,'rb') as f:data=pickle.load(f)
    learner=data['learner']
    q=np.asarray(dsac._critics(learner.critics,features,mask,jnp.zeros((len(features),2)),True)).mean(-1).min(0)
    mean,_,logp=dsac.actor_outputs(learner.actor,features,mask)
    actual=np.asarray(dsac._critics(learner.critics,features,mask,jnp.tanh(mean),True)).mean(-1).min(0)
    row={'checkpoint':path.name,
         'mean_min_twin_q':dict(zip(('Wait','Work','Sleep','Diner','Cafe','Restaurant','Buy'),
            map(float,q[:,[0,1,2,3,4,5,25]].mean(0)))),
         'at_actor_mean_style':dict(zip(('Wait','Work','Sleep','Diner','Cafe','Restaurant','Buy'),
            map(float,actual[:,[0,1,2,3,4,5,25]].mean(0)))),
         'actor_food_probability':float(np.exp(np.asarray(logp))[:,3:6].sum(-1).mean())}
    rows.append(row);print(json.dumps(row),flush=True)
out=ROOT/'reports/local_food_recovery/critic_estimates.json'
out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
