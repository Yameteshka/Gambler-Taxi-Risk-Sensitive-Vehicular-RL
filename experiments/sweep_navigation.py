from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parents[1]
for horizon in (6,12,24):
 for hold in (2,4):
  subprocess.run([sys.executable,str(root/'experiments/navigation_competence.py'),'--horizon',str(horizon),'--hold',str(hold)],cwd=root,check=True)
