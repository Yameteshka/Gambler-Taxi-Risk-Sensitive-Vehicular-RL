"""Certify an action-equivalence error bound in a stationary, noiseless state."""
from pathlib import Path
import sys,json,gzip,pickle,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env
from algorithms import hybrid_dsac as dsac
import models

def analyze(path,output):
 with gzip.open(path,'rb') as file: saved=pickle.load(file)
 meta=saved['metadata']; kind=meta['kind']; config=env.Config(**meta['environment'])
 if config.actuation_noise: raise ValueError('Exact equivalence assumes noiseless actuation')
 params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
 obs,state=env.ResourceTaxiEnv.reset(6,params)
 controls=jnp.array(((-1.,0.),(-.5,0.),(0.,0.)))
 states=jax.tree.map(lambda x:jnp.broadcast_to(x,(3,*x.shape)),state)
 _,next_states,rewards,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
  jax.random.split(jax.random.PRNGKey(0),3),states,env.Action(controls,jnp.zeros(3,dtype=jnp.int32)),params)
 for value in jax.tree.leaves(next_states): np.testing.assert_array_equal(np.asarray(value[0]),np.asarray(value[1])); np.testing.assert_array_equal(np.asarray(value[1]),np.asarray(value[2]))
 np.testing.assert_array_equal(np.asarray(rewards),np.repeat(np.asarray(rewards)[0],3))
 learner=saved['learner']; f=jnp.broadcast_to(obs.features,(3,obs.features.size)); mask=jnp.broadcast_to(obs.action_mask,(3,env.NUM_ACTIONS))
 if kind=='dsac_gru':
  history=models.initial_memory(kind,3,f.shape[-1]).at[:,-1,:].set(f)
  f=models.encode_history(learner.encoder,history); critics=learner.inner.critics
 else: critics=learner.critics
 q=np.asarray(dsac._critics(critics,f,mask,controls))[:,:,env.DRIVE,:]
 records=[]
 for i,values in enumerate(q):
  means=values.mean(axis=-1); ordered=np.sort(values,axis=-1)
  w1=max(float(np.abs(ordered[a]-ordered[b]).mean()) for a in range(3) for b in range(3))
  gap=float(np.ptp(means))
  records.append({'critic':i+1,'predicted_means':means.tolist(),'mean_gap':gap,'certified_max_absolute_mean_error_lower_bound':gap/2,'pairwise_empirical_w1_max':w1,'certified_max_w1_error_lower_bound':w1/2})
 result={'checkpoint':str(path),'kind':kind,'physical_transitions':saved.get('total'),
  'probe':'Stationary standard home reset, scenario 6; all actions Drive with steering zero and acceleration -1/-0.5/0',
  'verified_identical_next_state_and_reward':True,'critics':records,
  'proof':'Identical reward and next state imply identical true Q=r+gamma*V for a fixed continuation policy. A prediction gap g implies at least one absolute mean error >=g/2. The triangle inequality gives the analogous W1 bound.',
  'scope':'A pointwise critic approximation-error bound.'}
 Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('checkpoint',type=Path); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); analyze(args.checkpoint,args.output)
