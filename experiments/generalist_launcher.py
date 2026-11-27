"""Run the sequential local GPU study independently of the chat window."""
from pathlib import Path
import ctypes,subprocess,json,time,os
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/generalist_study'
folder.mkdir(parents=True,exist_ok=True)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
 with (folder/'execution.log').open('a',encoding='utf-8') as log:
  process=subprocess.Popen(['wsl','-d','Ubuntu','--','bash','-lc',
   'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=src OMP_NUM_THREADS=4; ../.venv-wsl/bin/python experiments/generalist_study.py'],
   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  (folder/'launcher.json').write_text(json.dumps({'pid':os.getpid(),'worker_pid':process.pid,'state':'running'},indent=2))
  code=process.wait()
  (folder/'launcher.json').write_text(json.dumps({'pid':os.getpid(),'worker_pid':process.pid,'state':'finished' if code==0 else 'stopped','exit_code':code},indent=2))
finally:
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
