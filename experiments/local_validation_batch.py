"""Run GPU checks sequentially and retain complete output without interleaving."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'reports/local_learning_probe';folder.mkdir(parents=True,exist_ok=True)
jobs=[('target_regression',['experiments/check_dsac_targets.py']),
      ('target_regression_normalized',['experiments/check_dsac_targets.py','--normalized']),
      ('ppo_resource_learning',['experiments/local_learning_probe.py','--model','ppo_mlp','--variant','normalized_bounded','--exploration','.2','--iterations','400']),
      ('driver_brake',['experiments/local_driver_probe.py','--driver','brake']),
      ('driver_predictive',['experiments/local_driver_probe.py','--driver','predictive'])]
status=[]
for name,args in jobs:
    started=time.perf_counter();print('CHECK_START '+name,flush=True)
    with (folder/(name+'.log')).open('w') as output:
        result=subprocess.run([sys.executable,'-B','-u',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    status.append({'check':name,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started})
    (folder/'local_checks.json').write_text(json.dumps(status,indent=2))
    print('CHECK_DONE '+json.dumps(status[-1]),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
