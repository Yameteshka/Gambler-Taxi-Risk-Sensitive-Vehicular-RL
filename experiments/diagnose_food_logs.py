"""Read-only decision/physical-trace reconciliation on saved development data."""
from pathlib import Path
import os
from collections import Counter
import json,csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/resource_study_review_2026-11-23/food_diagnosis'
OUT.mkdir(parents=True,exist_ok=True)
review=json.loads((OUT.parent/'review.json').read_text())
names=('Wait','Work','Sleep','Diner','Cafe','Restaurant','Refill5','FillTank','Borrow25','Borrow50','Borrow100','Car25','Car100','CarAll','Micro25','Micro100','MicroAll','Slots10','Slots25','Slots55','Blackjack10','Blackjack30','Blackjack90','Hit','Stand','Buy')
results=[];allrows=[]
cases=[]
for model in review['models']:
    cases.append((model,'selected',model['selected_standard_development']['source']))
    latest=model['latest_saved_standard_validation']['source']
    if latest!=model['selected_standard_development']['source']:
        cases.append((model,'latest_validation',latest))
    cases.append((model,'last_lesson_gate',model['last_gate']['source']))
for model,label,source in cases:
    kind=model['model'];folder=ROOT/source
    with np.load(folder/'endpoints.npz') as d:
        endpoints={k:d[k] for k in d.files}
    config=json.loads((folder/'evaluation.json').read_text())['config']
    records=[]
    for path in sorted(folder.glob('decision_*.npz')):
        with np.load(path) as f:d={k:f[k] for k in f.files}
        with np.load(folder/path.name.replace('decision_','trace_')) as f:
            keys=('active','executed','events','satiety_before','cash_before','satiety','cash','position','time','food_spent','outcome','remaining','car_payment')
            p={k:f[k] for k in keys}
        for i,lane in zip(*np.where(d['active'])):
            ix=np.flatnonzero(p['active'][i*128:(i+1)*128,lane])+i*128
            if not len(ix):continue
            first,last=ix[0],ix[-1];choice=int(d['choice'][i,lane])
            obs=d['observation'][i,lane];menu=obs[-41:-26].reshape(3,5)
            rec={'model':kind,'checkpoint_context':label,'scenario':int(d['scenario_id'][i,lane]),'time':int(d['time'][i,lane]),
                'choice':choice,'action':names[choice],'duration':int(len(ix)),
                'satiety':float(p['satiety_before'][first,lane]),'cash':float(p['cash_before'][first,lane]),
                'cash_after':float(p['cash'][last,lane]),'satiety_after':float(p['satiety'][last,lane]),
                'food_legal':bool(d['action_mask'][i,lane,3:6].all()),
                'discrete_entropy':float(d['entropy_d'][i,lane]),
                'diner_price':float(menu[0,0]*50),
                'restaurant_price':float(menu[2,0]*50),
                'meal_started':bool(np.any(p['executed'][ix,lane]==4)),
                'meal_completed':int(p['events'][ix,lane,11].sum()),
                'work_started':bool(np.any(p['executed'][ix,lane]==2)),
                'work_completed':int(p['events'][ix,lane,0].sum()),
                'car_payment':float(p['car_payment'][ix,lane].sum()),
                'terminal':int(p['outcome'][last,lane]),
                'distance_moved':float(np.linalg.norm(np.diff(p['position'][ix,lane],axis=0),axis=1).sum())}
            records.append(rec)
    starved=set(map(int,endpoints['scenario_id'][endpoints['outcome']==4]))
    low=[r for r in records if r['satiety']<.2]
    food=[r for r in records if 3<=r['choice']<=5]
    work=[r for r in records if r['choice']==1]
    affordable_low=[r for r in low if r['cash']>=r['restaurant_price']]
    histories=[]
    for sid in sorted(starved):
        rows=[r for r in records if r['scenario']==sid]
        histories.append({'scenario':sid,'decisions':len(rows),'last_decisions':rows[-8:]})
    gatespath=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training'/kind/'curriculum_gates.jsonl'
    gates=[json.loads(s) for s in gatespath.read_text().splitlines()]
    passed=[{'stage':g['stage'],'lesson':g['lesson'],'ownership':g['summary']['ownership'],
             'meals':g['summary']['meals']} for g in gates if g['passed']]
    result={'model':kind,'checkpoint_context':label,'source':folder.name,
            'config':{k:config[k] for k in ('public_food_menu','reward_shaping','financial_progress','hunger_tick','hunger_work','hunger_drive')},
            'decisions':len(records),'low_satiety_decisions':len(low),
            'low_satiety_actions':dict(Counter(r['action'] for r in low)),
            'low_satiety_food_legal_decisions':sum(r['food_legal'] for r in low),
            'restaurant_affordable_low_satiety_decisions':len(affordable_low),
            'affordable_low_actions':dict(Counter(r['action'] for r in affordable_low)),
            'mean_discrete_entropy':float(np.mean([r['discrete_entropy'] for r in records])),
            'car_payments_leaving_less_than_diner_price':sum(r['car_payment']>0 and r['cash_after']<r['diner_price'] for r in records),
            'work_intentions':len(work),'work_intentions_without_start':sum(not r['work_started'] for r in work),
            'work_intentions_no_start_and_duration_1':sum(not r['work_started'] and r['duration']==1 for r in work),
            'food_intentions':food,'passed_curriculum':passed,'starvation_histories':histories}
    results.append(result);allrows+=records
    print(json.dumps({k:v for k,v in result.items() if k not in ('food_intentions','starvation_histories','passed_curriculum')},indent=2),flush=True)
    print('FOOD',json.dumps(food),flush=True)
    print('PASSED',json.dumps(passed),flush=True)
    print('EXAMPLE',json.dumps(histories[0] if histories else None),flush=True)
(OUT/'diagnosis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
with (OUT/'decisions.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
