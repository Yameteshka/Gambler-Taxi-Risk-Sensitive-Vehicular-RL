"""Analyze observed failures without changing a frozen training run."""
from pathlib import Path
import os
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'debt500_study/output/artifacts/debt500_study'
OUTPUT=ROOT/'reports/debt500_diagnosis';OUTPUT.mkdir(exist_ok=True)
result={}
for folder in sorted((BASE/'readiness').glob('*')):
 if not (folder/'audit.json').exists():continue
 meta=json.loads((folder/'evaluation.json').read_text());config=meta['config']
 with np.load(folder/'endpoints.npz') as d:
  ids=d['scenario_id'].tolist()
  episodes={s:{'scenario':s,'outcome':int(d['outcome'][i]),'ticks':int(d['time'][i]),
   'cash':float(d['cash'][i]),'car_debt':float(d['car_debt'][i]),'micro_debt':float(d['micro_debt'][i]),
   'counters':d['counters'][i].tolist(),'payments_draining_cash':0,'hungry_free_decisions':0,
   'hungry_food_choices':0,'last_free_choice':None} for i,s in enumerate(ids)}
 previous_position=np.broadcast_to(np.array((5.,15.)),(len(ids),2)).copy()
 previous_activity=np.zeros(len(ids),int);max_hunger_error=0.;squared=[]
 for file in sorted(folder.glob('trace_*.npz')):
  with np.load(file) as d:
   for tick in range(d['active'].shape[0]):
    active=d['active'][tick];position=d['position'][tick]
    movement=np.linalg.norm(position-previous_position,axis=-1)>1e-6
    executed=d['executed'][tick]
    working=(previous_activity==2)|(executed==2)
    pushing=(executed==0)&(d['fuel_before'][tick]<=0)&(d['speed'][tick]>0)
    hunger=config['hunger_tick']+config['hunger_work']*working+config['hunger_drive']*movement+config['hunger_push']*pushing
    delivered=d['events'][tick,:,11]>0
    expected=np.clip(d['satiety_before'][tick]-hunger+delivered*d['service_value'][tick],0.,1.)
    if active.any():max_hunger_error=max(max_hunger_error,float(np.max(np.abs(expected[active]-d['satiety'][tick,active]))))
    for lane in np.flatnonzero(active):
     episodes[ids[lane]]['wages']=float(d['wages'][tick,lane])
     episodes[ids[lane]]['food_spent']=float(d['food_spent'][tick,lane])
     if d['car_payment'][tick,lane]>0 and d['cash'][tick,lane]<4:
      episodes[ids[lane]]['payments_draining_cash']+=1
    previous_position=np.where(active[:,None],position,previous_position)
    previous_activity=np.where(active,d['activity'][tick],previous_activity)
 for file in sorted(folder.glob('decision_*.npz')):
  with np.load(file) as d:
   for tick,lane in np.argwhere(d['active']):
    f=d['observation'][tick,lane];choice=int(d['choice'][tick,lane]);r=episodes[ids[lane]]
    if f[100]<.5:continue # forced wait during an already-running service
    if f[74]<.25:
     r['hungry_free_decisions']+=1;r['hungry_food_choices']+=int(3<=choice<=5)
    # An optimistic travel lower bound uses straight-line distance and vmax,
    # ignoring obstacles and braking; no realized future portion is consulted.
    menu=f[153:168].reshape(3,5);cash=float(f[68]*1000)
    vectors=f[16:36].reshape(10,2)[7:]*40
    distance=np.maximum(np.linalg.norm(vectors,axis=1)-config['zone_radius'],0)
    travel=distance/(config['max_speed']*config['dt'])
    food_impossible=f[74]<=config['hunger_tick']*(travel+menu[:,4]*10)
    affordable=cash>=menu[:,0]*50
    r['last_free_choice']={'tick':int(d['time'][tick,lane]),'choice':choice,
       'satiety':float(f[74]),'cash':cash,'all_food_unaffordable':bool(not affordable.any()),
       'no_affordable_food_reachable_even_under_optimistic_bound':bool(np.all(~affordable|food_impossible))}
 groups={}
 for outcome,label in ((1,'ownership'),(4,'starvation'),(2,'collision')):
  e=[r for r in episodes.values() if r['outcome']==outcome]
  groups[label]={'episodes':len(e),'mean_ticks':float(np.mean([r['ticks'] for r in e])) if e else None,
   'mean_wages':float(np.mean([r.get('wages',0) for r in e])) if e else None,
   'cash_at_least_40':sum(r['cash']>=40 for r in e),'cash_below_4':sum(r['cash']<4 for r in e),
   'episodes_with_car_payment_draining_cash':sum(r['payments_draining_cash']>0 for r in e),
   'car_already_paid':sum(r['car_debt']==0 for r in e),
   'hungry_free_decisions':sum(r['hungry_free_decisions'] for r in e),
   'hungry_food_choices':sum(r['hungry_food_choices'] for r in e),
   'too_late_at_last_free_choice':sum(bool(r['last_free_choice'] and r['last_free_choice']['no_affordable_food_reachable_even_under_optimistic_bound']) for r in e)}
 history=json.loads((BASE/'training'/folder.name/'validation.json').read_text())
 result[folder.name]={'groups':groups,'maximum_independent_hunger_update_error':max_hunger_error,
   'validation_history':[{k:r[k] for k in ('physical_transitions','ownership','mean_external_return','starvation','collision')} for r in history],
   'episodes':list(episodes.values())}
(OUTPUT/'trajectory_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{'groups':v['groups'],'max_hunger_error':v['maximum_independent_hunger_update_error']} for k,v in result.items()},indent=2))
