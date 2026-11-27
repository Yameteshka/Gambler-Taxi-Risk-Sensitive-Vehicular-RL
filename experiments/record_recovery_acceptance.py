"""Record actual local evidence; never infer an untested architecture passed."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'reports/local_learning_probe'
def read(path):return json.loads(path.read_text()) if path.exists() else None
mechanics=read(base/'real_context/summary.json')
targets=read(ROOT/'experiments/dsac_target_verification_normalized.json')
driver=read(base/'actual_bank_departures_escape.json')
routes=read(base/'all_routes_clock_escape_True.json')
forecast=read(ROOT/'experiments/driver_forecast_verification.json')
curriculum=read(ROOT/'experiments/ownership_curriculum_verification.json')
budget=read(ROOT/'experiments/decision_budget_verification.json')
food=read(base/'real_food_learning_release/summary.json')
release=read(base/'release_checks.json')
release_ready=bool(release and len(release)==5 and all(row['exit_code']==0 for row in release))
route_ready=bool(routes and routes['nonterminal_origin_arrivals']>=.99*routes['nonterminal_origin_routes'])
common=bool(mechanics and mechanics['passed'] and targets and targets['passed'] and driver and
    driver.get('recorded_endpoint_replay_passed') and driver['predictive']['collisions']==0 and
    driver['predictive']['arrivals']==driver['states'] and route_ready and
    forecast and forecast['passed'] and curriculum and curriculum['passed'] and budget and budget['passed'] and
    food and food['passed'] and len(curriculum.get('food_path_witnesses',[]))==4 and release_ready)
qualified={};evidence={}
for kind in ('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru'):
    variants=('separate_memory_skip_extended',) if kind=='dsac_gru' else ('normalized_bounded',)
    candidates=[base/variant/kind/'result.json' for variant in variants]
    result_path=next((path for path in candidates if (read(path) or {}).get('passed')),candidates[-1])
    result=read(result_path)
    qualified[kind]=bool(common and result and result['passed'])
    causal=read(base/'real_food_learning_release'/kind/'result.json')
    qualified[kind] &= bool(causal and causal['passed'] and not causal.get('source_changed_during_run',False))
    if kind.endswith('mlp'):
        qualified[kind] &= bool(mechanics and mechanics['learned_real_transition_contexts'].get(kind))
    evidence[kind]={'qualified':qualified[kind],'learning_result':str(result_path.relative_to(ROOT)),
        'causal_food_result':f'reports/local_learning_probe/real_food_learning_release/{kind}/result.json',
        'scope':result['scope'] if result else 'No completed local learning check.'}
sources={}
for path in sorted((ROOT/'src').rglob('*.py')):
    raw=path.read_bytes();ast.parse(raw,filename=str(path))
    sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
result={'qualified_models':qualified,'evidence':evidence,'common_checks_passed':common,
    'driver_route_summary':{key:routes[key] for key in ('routes','arrivals','collisions','unfinished','nonterminal_origin_routes','nonterminal_origin_arrivals')} if routes else None,
    'routing_acceptance':'At least 99% arrival on the development matrix excluding terminal-only garage departures; all failures remain in the route record. Not a safety guarantee.',
    'evidence_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'experiments/ownership_curriculum_verification.json',ROOT/'experiments/driver_forecast_verification.json',
            ROOT/'experiments/decision_budget_verification.json',base/'actual_bank_departures_escape.json',
            base/'all_routes_clock_escape_True.json',base/'real_food_learning_release/summary.json') if p.exists()},
    'source_hashes':sources,'scope':'Local calibration acceptance only; not full-game convergence or final-test performance.'}
(ROOT/'experiments/recovery_acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'qualified_models':qualified,'common_checks_passed':common}),flush=True)
