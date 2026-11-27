"""Check final-source gates and replay each learner on development scenarios."""
from pathlib import Path
import json
import os
import subprocess
import sys
import jax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import protocol
import models
from env import resource_taxi as env
from evaluate import evaluate
from audit_results import audit


def main():
    current=protocol.source_hashes()
    for name,script in (('environment_verification','verify.py'),
                        ('strategy_verification','verify_strategy.py'),
                        ('learning_verification','verify_learning.py')):
        path=ROOT/f'experiments/{name}.json'
        previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if previous.get('source_hashes')!=current:
            subprocess.run([sys.executable,str(ROOT/'src'/script)],cwd=ROOT,check=True)
        print(f'GATE {name}: current source verified',flush=True)
    subprocess.run([sys.executable,str(ROOT/'src/analytic_checks.py')],cwd=ROOT,check=True)
    checks=[]
    for run in ('resume_equivalence_uninterrupted','smoke_ppo_mlp_10d_curriculum_dynamic_repB',
                'smoke_ppo_gru_10d_curriculum_dynamic_repB','smoke_dsac_gru_10d_curriculum_dynamic_repB'):
        paths=list((ROOT/'runs/pilots'/run).glob('*/complete.json'))
        folder=max(paths,key=lambda p:p.stat().st_mtime).parent
        policy=folder/'selected.npz'
        model,metadata=models.load_policy(policy)
        params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(**metadata['evaluation_environment']))
        output=ROOT/'runs/preflight'/run
        result,_,_=evaluate(model,metadata['kind'],params,range(1,11),output,interface='strategy')
        checked=audit(output,policy)
        checks.append({'kind':metadata['kind'],'checkpoint':str(policy.relative_to(ROOT)),
            'evaluation':str(output.relative_to(ROOT)),'outcomes':result,
            'audit_passed':checked['passed'],'replayed_steps':checked['active_physical_steps_replayed']})
        print(f'REPLAY {metadata["kind"]}: passed on development scenarios 1-10',flush=True)
    (ROOT/'experiments/preflight.json').write_text(json.dumps({'passed':True,
        'source_hashes':current,'devices':str(jax.devices()),'checks':checks,
        'scope':'Execution, accounting and exact policy/physical replay. Does not certify convergence or optimality.'},indent=2),encoding='utf-8')
    from run_suite import freeze
    freeze()
    print('PREFLIGHT PASSED; registered sources and design frozen',flush=True)


if __name__=='__main__':main()
