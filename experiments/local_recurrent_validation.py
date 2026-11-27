"""Apply the same synthetic skill check to the two recurrent agents."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'reports/local_learning_probe'
rows=[]
for kind in ('dsac_gru','ppo_gru'):
    print('RECURRENT_CHECK_START '+kind,flush=True);started=time.perf_counter()
    with (folder/(kind+'_learning.log')).open('w') as output:
        job=subprocess.run([sys.executable,'-B','-u','experiments/local_learning_probe.py','--model',kind,
            '--variant','normalized_bounded','--exploration','.2','--iterations','600'],cwd=ROOT,
            stdout=output,stderr=subprocess.STDOUT)
    rows.append({'kind':kind,'exit_code':job.returncode,'elapsed_seconds':time.perf_counter()-started})
    (folder/'recurrent_checks.json').write_text(json.dumps(rows,indent=2))
    print('RECURRENT_CHECK_DONE '+json.dumps(rows[-1]),flush=True)
    if job.returncode:raise SystemExit(job.returncode)
