# Debt-500 mortality diagnosis

## Findings

Both selected PPO policies achieved 15/64 ownership completions. Their winning scenario sets overlap in **15 of 15 cases**. The two policies succeed on the same starts, which is consistent with a wage-dependent strategy.

| Policy | Ownership | Starvation | Collision | Starved after cash-draining car payment | Starved with less than $4 | Mean wages: ownership / starvation |
|---|---:|---:|---:|---:|---:|---|
| ppo_gru | 15 | 48 | 1 | 47 | 45 | $439.83 / $306.98 |
| ppo_mlp | 15 | 47 | 2 | 44 | 43 | $439.83 / $304.20 |

The association with cash-draining repayment is observational, not a randomized causal effect. Successful policies also use all-cash payments, but they obtain enough wages to finish before resources run out. With lower observed wages, the same general pattern leaves unpaid debt and no food money. The final choices are often already too late or unaffordable; this points to planning before repayment rather than a last-moment Eat action.

## Checks that passed

An independent NumPy reconstruction of every active hunger update used baseline/work/movement/pushing consumption and delivered meal portions. The maximum absolute discrepancy was 2.41e-8. No hunger arithmetic defect was found in these traces. Existing full replay audits passed. Model selection by external return selected a 15-success checkpoint for both architectures, matching the highest observed validation ownership count; a wrong checkpoint is not the explanation here.

## Concrete design problems and hypotheses

1. **Cold-start exposure is not a learned skill.** The behavior distribution includes 80% uniform exploration. Hundreds of observed meals under that mixture cannot establish that the actor itself chooses food. Matched probes with exploration disabled show PPO MLP food probability at a critically hungry restaurant context falling from about 1.66% at the cold boundary to 0.031% at the selected checkpoint. These are synthetic state probes. GRU probes start with zero memory and require more qualification.
2. **The early objective can be solved without wages or sustained survival.** Early debt is $50 while initial cash is $100, and some garage contexts already have zero debt. This exposes successful terminal mechanics but does not teach the complete earn/eat/refuel/repay chain. The cold gate checked completed actions, not transfer of an autonomous survival policy.
3. **A hand-written economic priority remains inside shaping and an observation cue.** `useful_targets` forces Bank when settlement is affordable and Garage when debt is cleared, even at critical hunger. A direct probe at satiety 0.05, cash $200 and debt $100 returns only Bank. Nearest-service potential uses that selection. It can create locally misleading feedback for buying food. The potential still telescopes at terminal states. The target rule nevertheless supplies Bank as an explicit cue in this hunger state.
4. **Resource practice vanishes as the curriculum returns to normal starts.** The final stage has fixed initial reserves, and replay is cleared at curriculum transitions. This may encourage a short wage/repayment pattern and suppress previously sampled resource actions. A controlled retained-resource-context comparison is needed.
5. **SAC entropy is per decision, while actions have different physical durations.** Short bank operations supply more entropy opportunities per physical time than travel or work. Measure durations/temperatures and test annealed or physically normalized entropy before attributing SAC loops to financial reasoning.

## Next controlled repairs

Keep the current frozen run as the baseline. First compare removal of the hand-written economic route cue and its geometric reward term, keeping public maps/menu and driving unchanged. Separately test maintaining diverse resource-critical training contexts, with unchanged external rewards and no expert actions. Add actor-only validation of resource decisions after cold start rather than treating exploratory visits as mastery. Do not impose automatic food purchases, protected cash reserves, forced evaluation wagers or manual schedules.

Use one PPO diagnostic comparison before repeating four full trainings. Report ownership, survival, cash exhausted by repayments, resource use and actual episode traces. Use held-out ownership and survival to evaluate the change.

## Evidence

- [Episode-level trajectory diagnosis](trajectory_diagnosis.json)
- [Matched checkpoint preference probes](checkpoint_preferences.json)
- [Trace analysis implementation](../diagnose_debt500.py)
- [Checkpoint probe implementation](../probe_debt500_checkpoints.py)
- [Economic route heuristic and potential](../../src/env/resource_taxi.py)
