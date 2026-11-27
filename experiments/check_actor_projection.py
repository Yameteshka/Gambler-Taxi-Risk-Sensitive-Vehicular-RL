"""CPU-only gradient checks on frozen logs. No simulator rollout or agent training."""
from pathlib import Path
import os,sys,json,pickle,gzip
os.environ['JAX_PLATFORMS']='cpu'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax,jax.numpy as jnp,numpy as np
from algorithms import hybrid_dsac as a
folder=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training/dsac_mlp'
g=json.loads((folder/'curriculum_gates.jsonl').read_text().splitlines()[-1])
trace=folder/'curriculum_validation'/f"stage_{g['stage']:02d}_step_{g['physical_transitions']:09d}"
with np.load(next(trace.glob('decision_*.npz'))) as d:
    x=jnp.asarray(d['observation'][0]);mask=jnp.asarray(d['action_mask'][0])
with gzip.open(ROOT/'artifacts/local_food_recovery/dsac_mlp_control/learner.pkl.gz','rb') as f:
    learner=pickle.load(f)['learner']
mean,_,logp=a.actor_outputs(learner.actor,x,mask)
q=a._critics(learner.critics,x,mask,jnp.tanh(mean),True).mean(-1).min(0)
q=jax.lax.stop_gradient(q)
temperatures=jnp.array((3e-6,3e-6))
def loss(z,method):
    ld=jax.nn.log_softmax(jnp.where(mask,z,-1e9),axis=-1)
    return a.actor_objective(jnp.exp(ld),ld,jnp.zeros(len(x)),q,temperatures,method)
old=jax.grad(lambda z:loss(z,'reverse_kl'))(logp)
new=jax.grad(lambda z:loss(z,'forward_kl'))(logp)
best=jnp.argmax(jnp.where(mask,q,-jnp.inf),axis=-1)
assert np.isfinite(old).all() and np.isfinite(new).all()
assert np.max(np.abs(np.asarray(new)[~np.asarray(mask)]))==0
assert np.all(np.asarray(new)[np.arange(len(x)),np.asarray(best)]<0)
legacy=(jnp.exp(logp)*(a._soft_terms(jnp.zeros(len(x)),logp,temperatures)-q)).sum(-1).mean()
assert np.allclose(loss(logp,'reverse_kl'),legacy)
updated=logp-new*len(x)
after=jax.nn.softmax(jnp.where(mask,updated,-1e9),axis=-1)
before=jnp.exp(logp)
assert np.all(np.asarray(after)[np.arange(len(x)),np.asarray(best)]>
              np.asarray(before)[np.arange(len(x)),np.asarray(best)])
r={'passed':True,'scope':'Frozen-critic logit gradient only; no full-game learning claim.',
   'best_legal_actions':np.asarray(best).tolist(),
   'restaurant_q_mean':float(q[:,5].mean()),'buy_q_mean':float(q[:,25].mean()),
   'restaurant_reverse_gradient_mean':float(old[:,5].mean()),
   'restaurant_forward_gradient_mean':float(new[:,5].mean()),
   'gradient_magnitude_ratio_restaurant':float(jnp.abs(new[:,5]).mean()/jnp.maximum(jnp.abs(old[:,5]).mean(),1e-30)),
   'restaurant_probability_before':float(before[:,5].mean()),
   'restaurant_probability_after_one_logit_step':float(after[:,5].mean())}
out=ROOT/'reports/local_food_recovery/actor_projection_check.json'
out.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r),flush=True)
