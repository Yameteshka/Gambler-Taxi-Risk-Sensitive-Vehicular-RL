"""Local unattended study queue with durable logs and an inspectable status file.

Run with Windows Python from the project folder. No uploads or scheduled tasks.
"""
from pathlib import Path
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT/'experiments/queue_status.json'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(state):
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8',newline='') as file:
        rows=list(csv.DictReader(file))
    counts={}
    for row in rows:counts[row['status']]=counts.get(row['status'],0)+1
    state.update(updated_at=stamp(),registered_runs=len(rows),run_status_counts=counts,
                 free_disk_gib=round(shutil.disk_usage(ROOT).free/1024**3,2))
    temp=STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    temp.replace(STATUS)
    lines=["# Local study queue", "", f"State: **{state['state']}**", "",
        f"Updated: {state['updated_at']}", "", f"Phase: {state.get('phase','')}", "",
        f"Registered runs: {len(rows)}. Free disk: {state['free_disk_gib']:.2f} GiB.", "",
        "| Run status | Count |", "|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name,count in sorted(counts.items()))
    lines.extend(["", "## Worker logs", ""])
    for worker in state.get('workers',[]):
        relative=Path(worker['log']).relative_to(ROOT).as_posix()
        lines.append(f"- [Worker {worker['shard']}]({relative}): exit code {worker['exit_code']}")
    lines.extend(["", "A planned or running row is not a result. Reports use only completed evaluations with passed replay audits.", ""])
    (ROOT/'OVERNIGHT_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')


def wsl(arguments):
    # All arguments here are fixed project-owned CLI switches, never user text.
    command=('export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda '
             'PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; '
             '../.venv-wsl/bin/python '+arguments)
    return ['wsl','-d','Ubuntu','--','bash','-lc',command]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps({'workers':args.workers,'commands':[wsl(
            f'src/run_suite.py --phase all --full-frozen-suite --shard-index {i} --shard-count {args.workers}')
            for i in range(args.workers)],'status_file':str(STATUS)},indent=2));return
    if not (ROOT/'experiments/freeze.json').exists():
        raise RuntimeError('Complete verification and source freeze before starting the queue')
    if STATUS.exists():
        previous=json.loads(STATUS.read_text(encoding='utf-8'))
        pid=previous.get('pid')
        if previous.get('state')=='running' and pid:
            # Never silently launch duplicate workers over the same run folders.
            result=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True)
            if f'"{pid}"' in result.stdout:raise RuntimeError(f'A queue already runs as process {pid}')
    folder=ROOT/'runs/queue'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    state={'state':'running','pid':os.getpid(),'started_at':stamp(),'workers':[],
           'phase':'registered training, frozen evaluation and replay audits',
           'log_directory':folder.relative_to(ROOT).as_posix(),'postprocessing':[]}
    processes=[]
    handles=[]
    # Keep the machine awake for the authorized overnight computation while
    # allowing its display to sleep. Restore the default when this process ends.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for i in range(args.workers):
            log=folder/f'worker_{i}.log';handle=log.open('w',encoding='utf-8');handles.append(handle)
            process=subprocess.Popen(wsl(f'src/run_suite.py --phase all --full-frozen-suite --shard-index {i} --shard-count {args.workers}'),
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process)
            state['workers'].append({'shard':i,'pid':process.pid,'log':log.relative_to(ROOT).as_posix(),'exit_code':None})
        while True:
            for record,process in zip(state['workers'],processes):record['exit_code']=process.poll()
            write_status(state)
            if all(p.poll() is not None for p in processes):break
            time.sleep(15)
        if any(p.returncode for p in processes):
            state.update(state='needs_investigation',phase='one or more workers stopped; preserved logs identify the failure')
            return
        state['phase']='English reports, scientific figures and offline replay gallery'
        write_status(state)
        environment={**os.environ,'PYTHONPATH':str(ROOT.parent/'.deps')+os.pathsep+str(ROOT/'src'),
                     'PYTHONIOENCODING':'utf-8','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}
        for script,arguments in (
            ('src/build_reports.py',[]),('reports/design_figures.py',[]),
            ('reports/analyze_study.py',[]),('reports/behavior_inventory.py',[]),
            ('reports/replay_gallery.py',['--render','--workers','4'])):
            entry={'script':script,'started_at':stamp()};state['postprocessing'].append(entry);write_status(state)
            with (folder/(Path(script).stem+'.log')).open('w',encoding='utf-8') as handle:
                result=subprocess.run([sys.executable,str(ROOT/script),*arguments],cwd=ROOT,env=environment,
                    stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            entry.update(exit_code=result.returncode,finished_at=stamp());write_status(state)
            if result.returncode:raise RuntimeError(f'Postprocessing failed: {script}')
        state.update(state='complete',phase='all registered runs, frozen tests, audits, reports and requested replay videos completed')
    except Exception as error:
        state.update(state='needs_investigation',error=repr(error))
        raise
    finally:
        for handle in handles:handle.close()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        write_status(state)


if __name__=='__main__':main()
