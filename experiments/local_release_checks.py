"""Run the final common engineering checks before releasing the cluster task."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/local_learning_probe'
checks=(('check_driver_forecast.py',()),('check_ownership_curriculum.py',()),
    ('driver_departure_probe.py',('--escape',)),('all_routes_probe.py',('--escape',)),
    ('check_dsac_targets.py',('--normalized',)))
rows=[]
for script,args in checks:
    started=time.perf_counter();print('RELEASE_CHECK_START',script,flush=True)
    with (out/(script.removesuffix('.py')+'_release.log')).open('w') as log:
        result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    rows.append({'script':script,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-started})
    (out/'release_checks.json').write_text(json.dumps(rows,indent=2));print('RELEASE_CHECK_END',json.dumps(rows[-1]),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
