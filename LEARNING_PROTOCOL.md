# Ownership learning and automatic evaluation

Four independent policies train once: DSAC MLP, PPO MLP, DSAC GRU and PPO GRU. They share the game, scenario bank, reset curriculum, exploration schedule and decision budget. Their collected trajectories and adaptive lesson durations naturally differ. The shared driver executes the policy's chosen destination; every economic intention is chosen by the policy.

One training workflow performs all training, development readiness checks, and then all 30 frozen evaluation conditions. Failed readiness saves diagnostics and reserves the final banks.

## Diagnosis and local evidence

The failed gradual pilot recorded 614,296 physical ticks but only 9,216 network decisions and 576 optimizer updates: [accounting](reports/learning_recovery/diagnosis.json). Those are different units. The subsequent partial MLP pilot learned endpoint, settlement and earning contexts but failed its food lesson. Short synthetic success was insufficient evidence of transfer. Its submitted sources and policies remain preserved.

A real driver standstill was reproduced on restaurant-to-garage travel. When all nominal continuations predicted later collisions, the long forecast could repeatedly postpone motion. The selected repair uses a short stopping guard in this case and advances clock, fuel and fatigue consistently with actual driving. An early sustained-turn candidate with an inconsistent clock was rejected. The corrected forecast evaluates both four-tick and twelve-tick steering continuations and cumulative route-distance cost, rather than terminal distance alone. A traffic-margin candidate was rejected after increasing route failures. The driver never selects food, employment, games, borrowing or repayment.

Local acceptance checks are development evidence, not full-game statistics:

- Six synthetic ownership contexts cover earning, food, fuel, fatigue, personal-debt repayment and checkout. Each context must exceed 70% success in each of the final three checks, with 128 sampled evaluations per context. Abstractions of travel, calendar and blackjack are explicitly documented. The four selected implementations passed; their initial failures remain in the histories.
- Mechanical resource checks verify wages, meals, fuel, sleep and personal-loan accounting in actual simulator states.
- A three-intention food benchmark enumerates actual simulator transitions for six scenario realizations. It retains actual observations, external rewards and variable-duration discounts. Driving style is fixed when constructing the cache; branches stop after three intentions. All four selected implementations passed. This is a narrow causal learning check, not a complete campaign or independent training replication. No action labels or demonstrations are supplied.
- Forecast propagation is compared with consecutive actual driving steps across day boundaries, fuel exhaustion and high fatigue.
- Matched driver checks use reconstructed bank-departure states and all directed venue pairs under multiple headings and clock phases. Food depletion is disabled for navigation isolation. Unfinished routes are reported explicitly, including direct departures from the terminal-only garage centre.
- Reset checks verify original principal, paid principal, cash, individual-loan records and unchanged transition/reward parameters. Hand-chosen trajectories verify reset feasibility and are kept outside the training replay buffer.
- Quantile tests compare sampled versus enumerated targets, terminal masking, fixed temperatures and recurrent history reset boundaries.
- A small real training run checks checkpoint resume and the network-decision stopping boundary. It must remain marked unqualified rather than become a completed scientific training.

The release decision is [recovery_acceptance.json](experiments/recovery_acceptance.json). Submission rejects unqualified models or changes after that release freeze. Histories, transition caches, rejected candidates and numeric evidence are retained in `reports/local_learning_probe/`. No calibration weights are promoted: the study starts all four policies fresh.

## Game and objective

Actual acceleration, steering, bicycle dynamics, friction, lidar, traffic, collisions, hunger, fuel, fatigue, work windows, interest and individual loans remain in the simulator. Training starts with $100 cash and a $500 original car principal. Curriculum paid principal is recorded consistently; it does not grant new cash.

The external ownership reward, time preference, financial increment, casino odds, wages and meal distributions remain unchanged. The former hand-designed potential and economic route cue are disabled for this treatment. Global venues, route information and public menus remain visible. There is no automatic meal, automatic loan, casino-visit bonus or economic behavior cloning. 

## Reset curriculum

Only initial states and the eventual exploration coefficient change. Earlier contexts are sampled with probability 0.2 for rehearsal. Actor-only lesson checks use the current context without rehearsal or extra exploration. All transitions use the same original contract and reward kernel, allowing replay retention. Earlier failed curriculum definitions remain available for historical replay.

| Lesson | Start | Remaining car debt | Special initial condition |
|---|---|---:|---|
| 0 | Garage | $0 | Ownership endpoint |
| 1 | Bank | $50 | Settlement before ownership |
| 2 | Work | $150 | Earnings required |
| 3 | Restaurant | $0 | Satiety 0.04: meal before ownership |
| 4 | Restaurant | $50 | Satiety 0.04: meal and settlement |
| 5 | Restaurant | $150 | Satiety 0.12, clock 100: meal and earnings |
| 6 | Restaurant approach (31, 17) | $50 | Satiety 0.12: travel to food before settlement |
| 7 | Fuel station | $150 | Empty tank |
| 8 | Home | $150 | Fatigue 0.95; sleep remains voluntary |
| 9 | Bank | $150 | Separate $50 personal loan blocks ownership |
| 10 | Home | $250 | Longer earning horizon |
| 11 | Home | $350 | Longer earning horizon |
| 12 | Home | $500 | Full obligation |
| 13 | Standard home | $500 | Extra exploration disabled |

Traffic is advanced to a shifted reset clock. Ordinary gates require four ownership completions from 16 development scenarios; the final standard lesson requires eight. Earning-dependent contexts require two successful episodes with paid work. Food and empty-tank contexts check their corresponding completed action. Curriculum ownership is never reported as standard-game performance.

Each lesson receives at least 65,536 physical ticks and 2,048 decisions before advancement. Checks occur at 65,536-physical-tick intervals. A diagnostic cap triggers only after BOTH 786,432 physical ticks and 65,536 decisions. This prevents long travel from consuming the learning allowance before the optimizer has enough experience. The rule is common to all architectures.

## Models and training budget

Every policy starts fresh at initialization seed 301 and receives **262,144 network decisions**. The physical safety ceiling is 33,554,432 ticks: at most 128 ticks per decision. This replaces the earlier 4,194,304-physical-tick budget for the new study only. Logs retain decisions, physical ticks, optimizer updates, lesson exposure, wall time and failures. Standard validation is scheduled every 16,384 decisions.

DSAC uses twin 24-quantile critics, width 128, nonaffine normalization in critic hidden layers, critic learning rate 3e-4, actor learning rate 1e-4, batch 128, 64 updates per 512 collected decisions, target rate 0.005 and replay warmup 2,048 transitions. Both temperatures are fixed at 3e-6. A conservative positive entropy-return upper bound at gamma 0.999 is below 0.014; this bounds an incentive rather than proving the historical failure cause. Sampled categorical targets estimate the same quantile-mixture loss as enumeration.

DSAC GRU has separate actor/critic memories and a current-observation bypass. Shared-memory and separate-memory-only failures are retained. This adapts principles from [Ni et al., ICML 2022](https://proceedings.mlr.press/v162/ni22a.html) and the authors' [recurrent actor](https://github.com/twni2016/pomdp-baselines/blob/main/policies/models/recurrent_actor.py); it is not an exact reproduction. PPO retains its clipped objective, value learning, likelihood-correct exploration mixture and variable-duration discounts. Both algorithms collect with epsilon 0.2 until the final lesson; evaluations disable that mixture. Principal DSAC risk/upside coefficients remain zero.

The [literature review](reports/learning_recovery/literature.md) distinguishes quantile DSAC from Gaussian DSAC-T and links Hybrid SAC and reverse-curriculum sources. 

## Readiness and automatic experiments

Selected frozen actors must pass the final curriculum and retained earning, food-approach, fuel and personal-debt checks. On the 64-scenario standard development bank, release requires at least 32 ownership completions and at most 16 starvation outcomes. Failures preserve diagnostics and keep final banks untouched.

Qualified actors then run **S01-S30, 256 episodes per condition and architecture**, without scenario-specific retraining. Training uses scenarios 11-10010; development selection uses 1000001-1000064. The standard final bank is 3000001-3000256; adverse banks follow the frozen registry. Five-day and continuing conditions are horizon transfer. One initialization per architecture remains exploratory: scenario replications do not measure between-training-seed variation or prove mathematical hard limits.

The same training workflow saves numeric trajectories, decisions, resource events, individual-loan endpoints, checkpoints, source hashes, audits and result tables, and automatically enters evaluation after readiness. Training and simulation are headless; videos are rendered later from frozen-policy traces.

## Execution

On Linux or WSL with the documented GPU environment:

```bash
export PYTHONPATH=src
export GT_STUDY=resource_study
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python experiments/resource_study.py
```

The [reproduction guide](reports/study/reproduce.md) documents how to run the four learners with the saved settings.
