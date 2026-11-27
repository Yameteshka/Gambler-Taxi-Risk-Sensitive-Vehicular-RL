"""Sequential local followups after the four registered baseline trials finish."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/local_food_recovery'
names=('dsac_mlp_control','dsac_mlp_entropy','ppo_mlp_context','ppo_gru_context')
deadline=time.monotonic()+1800
while not all((OUT/name/'result.json').exists() for name in names):
    if time.monotonic()>deadline:raise TimeoutError('Baseline trials did not finish; no followup started')
    time.sleep(5)
parent=OUT/'dsac_mlp_control/learner.pkl.gz'
commands=[
    ['--case','dsac_mlp_control','--head-scale','.2','--food-only','--parent',str(parent),'--decisions','8192'],
    ['--case','dsac_mlp_control','--food-only','--parent',str(parent),'--decisions','8192']]
for kind in ('ppo_mlp','ppo_gru'):
    baseline=json.loads((OUT/(kind+'_context')/'result.json').read_text())
    if not baseline['passed_short_development']:
        commands.append(['--case',kind+'_context','--head-scale','.2','--decisions','16384'])
for arguments in commands:
    print('FOLLOWUP_START',arguments,flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'experiments/local_recovery_trials.py'),*arguments],cwd=ROOT,check=True)
print('FOLLOWUPS_FINISHED',flush=True)
