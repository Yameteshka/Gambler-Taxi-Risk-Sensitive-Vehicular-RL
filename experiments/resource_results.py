"""Export inspectable English tables from this task's recorded results only."""
from pathlib import Path
import csv,json

def write_results(root,study_name='resource_study'):
    study=root/'artifacts'/study_name;out=root/'reports'/study_name
    out.mkdir(parents=True,exist_ok=True)
    state=json.loads((study/'status.json').read_text())
    rows=[]
    for path in sorted((study/'evaluation').glob('*/*/evaluation.json')):
        d=json.loads(path.read_text());audit=path.with_name('audit.json')
        diagnostic=path.with_name('driver_diagnostics.json')
        rows.append({'model':path.parent.parent.name,'condition':path.parent.name,
            **{k:d[k] for k in ('episodes','ownership','starvation','collision','deadline','success_rate','mean_external_return','mean_remaining_debt')},
            'return_p05':d['return_percentiles']['p05'],'return_p50':d['return_percentiles']['p50'],
            'potential_driver_stalls':json.loads(diagnostic.read_text())['flagged_episodes'] if diagnostic.exists() else None,
            'audit_record_present':audit.exists(),'record':path.relative_to(root).as_posix()})
    if rows:
        with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Recorded ownership-study results','',f"Task state: `{state['state']}`.",'',
        'These tables summarize saved results. One initialization per architecture is exploratory; scenario replications do not estimate training-seed variance. Curriculum wins are not standard-game success rates.','',
        '## Frozen-policy evaluations','']
    if rows:
        lines+=['| Model | Condition | Episodes | Ownership | Starvation | Collision | Mean return | Return p05 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['model']} | {r['condition']} | {r['episodes']} | {r['ownership']} | {r['starvation']} | {r['collision']} | {r['mean_external_return']:.4f} | {r['return_p05']:.4f} |")
        lines+=['','See `results.csv` for remaining debt, potential driver stalls and record paths. Per-condition directories retain uncertainty estimates, policy decisions, physical trajectories, individual-loan endpoints and audits. Stalled driving is distinguished from a policy choosing WAIT; flagged episodes remain in raw outcome statistics and require separate interpretation.']
    else:lines+=['No final-bank results were generated. Inspect readiness and curriculum failure records before interpreting task completion.']
    lines+=['','## Training accounting','',
        '| Model | Decisions | Physical ticks | Optimizer updates | Final logged lesson |','|---|---:|---:|---:|---:|']
    for folder in sorted((study/'training').glob('*')):
        path=folder/'training.jsonl'
        if not path.exists():continue
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if not records:continue
        last=records[-1];updates=sum(r.get('optimizer_updates_this_chunk',r.get('updates_this_chunk',0)) for r in records)
        decisions=last.get('network_decisions_total',last.get('network_decisions','unrecorded'))
        lines.append(f"| {folder.name} | {decisions} | {last['physical_transitions']} | {updates} | {last['stage']} |")
    lines+=['','The submitted design and source freeze are in `artifacts/'+study_name+'/freeze.json`. Readiness is in `reports/'+study_name+'/readiness.json`.']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'result_index.json').write_text(json.dumps({'task_state':state['state'],'evaluation_records':len(rows),
        'complete_120_condition_records':len(rows)==120,'tables':'results.csv' if rows else None},indent=2))

if __name__=='__main__':write_results(Path(__file__).resolve().parents[1])
