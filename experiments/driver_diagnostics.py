"""Flag prolonged blocked driving separately from a policy choosing to wait."""
from pathlib import Path
import json
import numpy as np

def diagnose(folder,threshold=64):
    folder=Path(folder);record=json.loads((folder/'evaluation.json').read_text())
    ids=np.asarray(record['scenario_ids']);n=len(ids)
    streak=np.zeros(n,np.int64);longest=streak.copy();previous=None
    for path in sorted(folder.glob('trace_*.npz')):
        with np.load(path) as d:
            active=d['active'];positions=d['position'];speed=d['speed']
            # Physical action 0 is DRIVE. WAIT is 1 and is deliberately excluded.
            attempting=(d['requested']==0)&(d['activity']==0)&active
        for i in range(len(active)):
            stopped=np.zeros(n,bool) if previous is None else np.linalg.norm(positions[i]-previous,axis=-1)<1e-4
            blocked=attempting[i]&stopped&(np.abs(speed[i])<.05)
            streak=np.where(blocked,streak+1,0);longest=np.maximum(longest,streak);previous=positions[i]
    flagged=longest>=threshold
    result={'episodes':n,'threshold_physical_ticks':threshold,'flagged_episodes':int(flagged.sum()),
        'scenario_ids':ids[flagged].tolist(),'longest_stopped_drive_ticks':dict(zip(map(str,ids),map(int,longest))),
        'scope':'Prolonged low-speed episodes while the policy requested driving. Voluntary WAIT is excluded.'}
    (folder/'driver_diagnostics.json').write_text(json.dumps(result,indent=2))
    return result
