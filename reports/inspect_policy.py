from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('src').resolve()))
import jax,jax.numpy as jnp,numpy as np
from dataclasses import replace
from env import city,resource_taxi as e
from algorithms.hybrid_dsac import actor_outputs
import models
records=[]
for root in sorted(Path('runs/pilots').glob('*/*_4194304')):
 if not (root/'last.npz').exists(): continue
 model,meta=models.load_policy(root/'last.npz'); kind=meta['kind']
 params,_=e.ResourceTaxiEnv.init_params(jax.random.key(0),e.Config(**meta['environment']))
 _,s=e.ResourceTaxiEnv.reset(6,params)
 cases=[('home',s),('work',replace(s,agent_pos=params.zone_centers[city.WORK])),('hungry_diner',replace(s,agent_pos=params.zone_centers[city.DINER],satiety=jnp.float32(.2))),('solvent_bank',replace(s,agent_pos=params.zone_centers[city.BANK],cash=jnp.float32(1500)))]
 for name,state in cases:
  obs=e.ResourceTaxiEnv.get_observation(state,params); f=obs.features[None]; mask=obs.action_mask[None]
  if kind.startswith('ppo'): mean,std,lp,_,_=models.ppo_step(model,f,mask,models.initial_memory(kind,1,f.shape[-1]))
  else: mean,std,lp=actor_outputs(model,f,mask)
  p=np.exp(np.asarray(lp[0])); records.append({'pilot':str(root),'kind':kind,'steps':meta['physical_transitions'],'state':name,'cue':np.asarray(f[0,-3:]).tolist(),'mean_control':np.tanh(np.asarray(mean[0])).tolist(),'raw_std':np.exp(np.asarray(std[0])).tolist(),'probabilities':{e.ACTION_NAMES[i]:float(p[i]) for i in np.argsort(-p)[:5]}})
Path('reports/pilot_policy_outputs.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
