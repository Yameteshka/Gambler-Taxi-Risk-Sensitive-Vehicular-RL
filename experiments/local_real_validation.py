"""Retain output from the final local real-state checks."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'reports/local_learning_probe'
results=[]
for name,script in (('departure_replay','driver_departure_probe.py'),('real_context','real_context_probe.py')):
    print('REAL_CHECK_START '+name,flush=True);start=time.perf_counter()
    with (folder/(name+'.log')).open('w') as output:
        result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments'/script)],cwd=ROOT,
            stdout=output,stderr=subprocess.STDOUT)
    results.append({'check':name,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start})
    (folder/'real_checks.json').write_text(json.dumps(results,indent=2));print('REAL_CHECK_DONE '+json.dumps(results[-1]),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
