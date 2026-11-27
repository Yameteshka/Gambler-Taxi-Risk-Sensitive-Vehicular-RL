# Debt repayment with broad cold-start exploration

The cold-start baseline changes several reset settings together. The [gradual curriculum](PROGRESSIVE_CURRICULUM.md) changes one factor per lesson and uses actor-only advancement checks. 

All four architectures receive the same training distribution, public menu, physical experience budget and curriculum. This is a fresh study; previous $1000 runs and their checkpoints remain separate.

## Training schedule

| Experience fraction | Car principal | Initial states | Exploration mixture |
|---|---:|---|---:|
| 0–10% | $50 | Balanced service contexts, low satiety at food, low fuel at station, opening time at casinos; debt-free garage contexts and personal-debt bank contexts also occur | 80% uniform categorical, 20% actor |
| 10–25% | $50 | Mixed services and home | 20% uniform categorical, 80% actor |
| 25–50% | $150 | Mixed services and home | Actor |
| 50–75% | $300 | Mostly home | Actor |
| 75–100% | $500 | Standard home start | Actor |

The external objective is unchanged. No action-label imitation, casino bonus or automatic bank loan is introduced. Actions have real cash, time, hunger and fuel consequences. Each architecture gets 4,194,304 active physical steps, including cold start; initialization seed 301 and training scenarios 11–10010 are shared. This single initialization per architecture is exploratory, not a measure of training variance.

For PPO, collection and the clipped likelihood ratio both use the same mixture distribution. Forced actions are not mislabeled as ordinary on-policy samples. SAC can train off-policy on the exploratory transitions; its target actor remains the learned actor. At evaluation, the exploration mixture is disabled, and the frozen actor is sampled as in the previous protocol.

At the end of cold start, actual completed work, sleep, meals, fuel purchases, slots, blackjack, borrowing and both kinds of repayment must all be present. Actual car-loan closure, microloan closure and ownership must also occur. The coverage report is based on environment events, not requested actions. If coverage is incomplete, the run stops for diagnosis rather than silently proceeding. Tutorial ownership is not counted as full-task evaluation success.

After training, validation must include an actual $500 ownership completion before final tests are consumed. Cold-start coverage proves exposure, not optimality or spontaneous economic behavior in the final policy. Non-use in an evaluation is reported; it is not hidden by injecting an action.

## Frozen scenarios

[The explicit 30-condition registry](experiments/debt500_scenarios.json) contains condition IDs S01–S30 and references to the earlier comparison IDs. Each frozen policy gets 256 matched episodes per condition. These include different deadlines, resources, markets, personal debt, meal conditions, driving perturbations, initial car debts $100/$250/$1000/$1500, and unavailable casino. The reference debt is $500. Some conditions are paired controls, and standard/ten-day presentations share the reference configuration; they are not 30 independent populations.

The original training-ablation IDs are preserved in [EXPERIMENTS.md](EXPERIMENTS.md). Frozen interventions cannot substitute for retraining ablations such as curriculum/no curriculum.

## Artifacts

- `artifacts/debt500_study/training/<kind>/cold_start_coverage.json`: mechanics actually experienced.
- `artifacts/debt500_study/readiness/<kind>/`: frozen validation and replay audits.
- `experiments/cold_start_verification.json`: mixture likelihood, gradient and start-state checks.

Training uses the existing $500 contract with milestones proportional to principal and the same ten-day calendar. All final tests use the untouched banks beginning at 3000001. No existing final test results are relabeled as validation data.
