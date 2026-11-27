"""Build the development evidence summary without inventing game outcomes."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'reports/local_learning_probe'
def read(p):return json.loads(p.read_text())
paths={kind:base/('separate_memory_skip_extended' if kind=='dsac_gru' else 'normalized_bounded')/kind/'result.json'
    for kind in ('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru')}
lines=['# Local learning validation and transfer diagnosis','',
    'The calibration uses fixed development cases. The main study trains independent checkpoints.','',
    '## Learned behavior checks','',
    '| Architecture | Synthetic contexts: worst final-three success | Synthetic decisions | Actual-transition food benchmark: worst final-three success | Real-cache decisions |',
    '|---|---:|---:|---:|---:|']
series={}
for kind,p in paths.items():
    synthetic=read(p);real=read(base/'real_food_learning_release'/kind/'result.json')
    worst=lambda d:min(min(row['success_rates'].values()) for row in d['history'][-3:])
    lines.append(f"| {kind} | {worst(synthetic):.1%} | {synthetic['history'][-1]['transitions']:,} | {worst(real):.1%} | {real['history'][-1]['transitions']:,} |")
    series[kind]={'synthetic':synthetic['history'],'real':real['history']}
lines+=['','Each check evaluates 128 sampled episodes in each of six contexts. Success is a point estimate on fixed development contexts, not independent training replication. The synthetic task abstracts travel, calendar and blackjack. The actual-transition cache holds driving style at zero and truncates after three intentions. Actual simulator observations, external rewards and variable-duration discounts are retained. No action labels, demonstration loss or automatic purchases are used.','',
    'The causal food test starts with satiety 0.04 and cleared car debt. A meal is needed before the driver can reach ownership. Its data include every available action; the learner chooses the action sequence. Reaching the game objective in this small task is distinct from winning a ten-day campaign.','',
    '## Reproduced implementation and curriculum problems','',
    '- A former bank-departure controller crashed in 18 of 62 reconstructed development states. Candidate controllers and their selected-state scope are recorded separately. These are not 62 complete campaigns.',
    '- A long nominal forecast could indefinitely postpone departure from the restaurant. The former forecast also held clock, fuel and fatigue fixed. The selected implementation advances these quantities and falls back to a short stopping guard when all nominal continuations fail.',
    '- An early sustained-steering candidate with a frozen clock was rejected after failing the meal-to-garage witness; the selected planner combines sustained candidates with consistent clock/fuel/fatigue propagation. A traffic-margin candidate was rejected after increasing failures. A direct hungry-home curriculum candidate was also rejected after its proposed short witness missed the work schedule..',
    '- The failed partial MLP pilot consumed 1,581,742 DSAC and 1,281,099 PPO physical ticks before failing its food gate. These stopped runs were not promoted and final banks were not consumed.',
    '- The new food frontier separates meal-to-ownership, meal-to-settlement, meal-to-earning and a short approach to food. The main training budget now counts 262,144 network decisions per architecture, with physical ticks and optimizer updates reported separately.','',
    '## Common release checks','']
accept=read(ROOT/'experiments/recovery_acceptance.json')
lines+=['Release qualified architectures: '+', '.join(k for k,v in accept['qualified_models'].items() if v)+'.','']
for name in ('ownership_curriculum_verification','driver_forecast_verification','decision_budget_verification'):
    d=read(ROOT/'experiments'/f'{name}.json');lines.append(f"- `{name}`: {'passed' if d['passed'] else 'failed'}.")
bank=read(base/'actual_bank_departures_escape.json');routes=read(base/'all_routes_clock_escape_True.json')
lines+=['',f"Reconstructed bank departures: {bank['predictive']['arrivals']}/{bank['states']} arrivals, {bank['predictive']['collisions']} collisions. Resources were equalized for routing isolation.",
    '',f"Expanded route matrix: {routes['arrivals']}/{routes['routes']} arrivals, {routes['collisions']} collisions, {routes['unfinished']} unfinished. Excluding terminal-only garage departures: {routes['nonterminal_origin_arrivals']}/{routes['nonterminal_origin_routes']} arrivals. All failed cases remain in `all_routes_clock_escape_True.json`; this is not a safety guarantee.",
    '', 'Hand-chosen food-path witnesses verify feasibility and accounting only; they are not learned performance and never enter replay. The deliberately tiny budget/resume run correctly stops as unqualified after 1,024 decisions rather than being marked trained.',
    '', '## Reproduction and next stage','',
    'See [the active protocol](../../LEARNING_PROTOCOL.md) for the complete curriculum, architecture settings, budget, source freeze, training/validation/final splits, readiness criteria and automatic 30-condition evaluation. [Primary-source review](literature.md) distinguishes quantile DSAC, Hybrid SAC and Gaussian DSAC-T. The cluster study starts fresh and must demonstrate standard-game ownership and retained resource skills before consuming final banks.','',
    'Calibration is one initialization with multiple rejected development attempts. Its curves cannot be used as an unbiased algorithm ranking. A complete paper requires final-game results and independent training replications.']
out=ROOT/'reports/learning_recovery';out.mkdir(exist_ok=True)
(out/'local_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'calibration_series.json').write_text(json.dumps(series))
print('LOCAL_VALIDATION_REPORT_WRITTEN',flush=True)
