# Financial Deadlines, Survival Costs and Voluntary Risk in a Driving Task

Iuliia Kurnaeva  
2026-11-19 04:08 UTC

## Abstract and current evidence

We implement a JAX driving task coupling navigation, work, sleep, food, fuel and two classes of debt. The driver has $100 of initial liquidity and must repay a $1000 car obligation while maintaining survival resources. A registered study compares hybrid distributional SAC and PPO, feedforward and recurrent representations, and five-day, ten-day and continuing tasks. Further comparisons separate training interventions from frozen-policy stress tests. Negative-expectation games offer faster, uncertain cash flows. The objective is held constant rather than directly rewarding gambling.

Eighteen of 125 registered runs completed training and received standard evaluations. 

## Motivation and research questions

Expected money, probability of repaying a deadline-bound obligation, and speed of repayment are different criteria. An action with negative financial expectation may still increase the chance of crossing a payment threshold before time runs out. Resource costs complicate the decision: keeping an emergency cash reserve can delay debt repayment but prevent a high-interest liquidity spiral. Cheap food saves travel time yet is expensive per unit of satiety. These tradeoffs motivate measurable questions: which algorithms learn ownership, how horizon alters risk, whether memory helps under predictive uncertainty, and which failures arise from economic planning versus vehicle control.

The project is a synthetic stress-test environment. Possible practical uses include evaluating planning methods under liquidity constraints, testing robustness to operating-cost shocks, studying rare-failure reporting and comparing exploration in mixed action spaces. It is not calibrated to household finances, taxi businesses, real casinos or human dependence. Its results cannot directly recommend lending or gambling behavior.

## Task construction and development record

The starting design attached economic choices to a required vehicular environment. Earlier iterations mixed emotional variables, imposed gambling impulses and changing reward definitions, making behavioral interpretations difficult. The current study removes those mechanisms, retains one external objective, separates personal debt from car debt and assigns every experimental condition an ID. Historical artifacts are preserved with a checksum manifest and are not pooled with current results.

Development uses only scenario IDs 1–10. Accounting tests exposed the need to distinguish loan proceeds from income and interest/fees from repaid principal. Service timing must consume hunger before a meal can rescue the driver. Navigation checks enforce the sum of two unit-circle radii and distinguish safe parking from merely entering an interaction radius. Feasibility pilots compare nearby low-restoration meals against distant larger portions and batch bank payments to reduce travel. The salary was calibrated from a provisional $145 to $175 before the frozen study. Earlier calibration outcomes under different geometry or salaries are engineering history, not interchangeable estimates of the final environment.

## Formal objective and analytical results

Let cash be C, outstanding car debt D_c, personal debt D_p and reserved stake E. Financial net worth is W=C+E-D_c-D_p. Loan disbursement adds equal cash and principal. Repayment subtracts equal cash and liability; neither creates wealth. Interest and late fees lower W. A successfully delivered shift raises W, while purchases and casino losses lower it. This accounting prevents credit from becoming a reward loophole.

The external reward includes a bounded secondary wealth-utility increment as well as terminal ownership/survival and time costs. Its clipping can affect behavior outside the linear wealth range; T09 measures its contribution. The external reward and potential are defined in [Assignment 1](assignment1.md). The discounted running cost is bounded by 0.1. The base return of every unsuccessful finite trajectory is -0.1, independent of duration, removing a pure failure-delay incentive. Potential shaping telescopes to -Phi(start) at true terminal states. These are mathematical statements about the specified reward. In contrast, the held-out empirical tail is an estimate with uncertainty.

Slots return pM-1=-0.290909 dollars per dollar wagered. An exact dynamic program enumerating replacement-card ranks and optimal hit/stand decisions yields blackjack EV=-0.06975294 per dollar under the implemented rules. Even the favorable latent slot regime has negative expectation (-0.054545 per dollar). The games have negative expected income. A wager can still raise the probability of reaching a fixed cash threshold in some states.

Ratios of expected food price to raw portion are 100, 100 and about 53.85 dollars per satiety unit for diner, cafe and restaurant. These are planning benchmarks, not a proof that the restaurant is optimal. Travel, clipping, remaining lifetime, service time, fuel, missed shifts and cash on hand can reverse the preference. Higher-variance portions have the same nominal means but a different starvation tail. Comparisons against an expected-value benchmark must condition on the actual task objective and opportunity costs.

## Experimental design and reproducibility

All external quotes and outcomes are generated from scenario-specific named random streams. Initialization seeds A–E (201–205) control learning randomness, separately from scenario IDs. Development IDs 1–10 are never used for training, reset curricula, demonstrations, or replay. Training uses 11–10010; validation uses 1000001–1000064; the standard final bank is 2000001–2000256. Adverse banks have separate non-overlapping ranges in `experiments/protocol.json`. Increasing the training scenario count does not move reserved banks; overlap is rejected before training.

Each trained policy is evaluated on the same 256 final scenarios. A fifth percentile uses approximately 13 tail observations. Each evaluation records Wilson intervals for ownership and a scenario bootstrap for mean return, p05 and lower-tail CVaR05. Training variability is assessed across five fresh initializations. Matched comparisons use crossed resampling over initialization and scenario; treating all 1280 outcomes as independent would understate uncertainty. 

The complete comparison matrix is [EXPERIMENTS.md](../EXPERIMENTS.md). The [policy catalog and crossed uncertainty analysis](study_analysis.md) and [calibration catalog](calibration_catalog.json) keep main evidence separate from engineering history. M01/M02/H01 comprise 45 main runs. T01–T11 add 55 training runs, changing one declared factor against the ten-day DSAC reference. H02 and F01–F12 reuse frozen weights to change horizon, control strength/noise, driving guard, initial fuel/satiety/cash, wages, fuel price, initial personal debt, food variance and service identity. No driving-controller intervention is silently described as the original learned policy. X01–X03 have implementation flags for latent casino regimes, reduced public quotes and an accelerated calendar, and 25 registered matched training arms.

P01 measures the first eligible voluntary wagering decision after a loss, reports its denominator and censored follow-ups, and probes public-event sensitivity at matched cash/time/resources. P02 compares recurrent action distributions with preserved versus reset memory on recorded observations. Synthetic matched probes are labeled as such. Differences do not reveal internal verbal thoughts or establish psychological motives. Near-zero gambling participation leaves loss-chasing undefined rather than evidence of zero dependence.

Each artifact retains source/config hashes, scenario IDs, checkpoint hash and requested/executed decisions. Scenario RNG streams are separated from policy sampling; daily prices use absolute day, portions use venue and purchase ordinal, games use game stream and resolved-round ordinal, and traffic uses its own initialization stream. Different policies need not consume the same outcome sequence at the same time; the matching convention is explicit. Sampling the trained stochastic policy is primary, using a separate policy RNG keyed by scenario and absolute tick. Argmax/mean extraction is a separately labeled diagnostic. Training transitions and evaluation episodes are different experimental units.

## Observed results

| Run | Case | Episodes | Ownership | Crashes | Starvation | Debt |
|---|---|---:|---:|---:|---:|---:|
| dsac_mlp_10d_curriculum_dynamic_repA | standard | 256 | 0 | 3 | 253 | $1012.38 |
| dsac_mlp_10d_curriculum_dynamic_repB | standard | 256 | 0 | 85 | 171 | $1007.46 |
| dsac_mlp_10d_curriculum_dynamic_repC | standard | 256 | 0 | 49 | 207 | $889.29 |
| dsac_mlp_10d_curriculum_dynamic_repD | standard | 256 | 0 | 97 | 159 | $1004.68 |
| dsac_mlp_10d_curriculum_dynamic_repE | standard | 256 | 0 | 96 | 160 | $1006.08 |
| dsac_mlp_5d_curriculum_dynamic_repA | standard | 256 | 0 | 95 | 161 | $1006.65 |
| ppo_gru_10d_curriculum_dynamic_repA | standard | 256 | 0 | 0 | 256 | $1007.68 |
| ppo_gru_10d_curriculum_dynamic_repB | standard | 256 | 0 | 0 | 256 | $1012.75 |
| ppo_gru_10d_curriculum_dynamic_repC | standard | 256 | 0 | 0 | 256 | $1012.71 |
| ppo_gru_10d_curriculum_dynamic_repD | standard | 256 | 0 | 0 | 256 | $987.69 |
| ppo_gru_10d_curriculum_dynamic_repE | standard | 256 | 0 | 2 | 254 | $727.70 |
| ppo_gru_5d_curriculum_dynamic_repA | standard | 256 | 0 | 0 | 256 | $1012.75 |
| ppo_mlp_10d_curriculum_dynamic_repA | standard | 256 | 0 | 0 | 256 | $1004.19 |
| ppo_mlp_10d_curriculum_dynamic_repB | standard | 256 | 0 | 0 | 256 | $725.01 |
| ppo_mlp_10d_curriculum_dynamic_repC | standard | 256 | 0 | 1 | 255 | $1010.01 |
| ppo_mlp_10d_curriculum_dynamic_repD | standard | 256 | 0 | 12 | 244 | $625.80 |
| ppo_mlp_10d_curriculum_dynamic_repE | standard | 256 | 0 | 8 | 248 | $602.63 |
| ppo_mlp_5d_curriculum_dynamic_repA | standard | 256 | 0 | 5 | 251 | $569.57 |

The machine-readable source of every row is `artifacts/evaluation/<run_id>/<case>/evaluation.json`, with endpoint arrays, uncertainty intervals and complete compressed traces. [results.md](results.md) lists all completed cases and completion status. Until the suite is complete, partial algorithm rankings and differences between horizons are provisional. Selection is validation-only; evaluating many frozen stress cases does not authorize retuning on them and reusing the same final bank as untouched evidence.

## Interpretation, limitations and contribution

The reusable contribution currently established is the auditable environment, hybrid-learning implementation, separated accounting, registered condition matrix and preserved trajectories. A quantitative algorithmic or behavioral contribution requires completed replicated comparisons against relevant existing work. General human findings such as greater risk under pressure cannot be claimed as new discoveries from this simulation alone. A conference-level claim would additionally need broader baselines, learning-budget sensitivity, stronger external validity and a literature-positioned novelty argument.

Five training repetitions provide a limited estimate of optimization variation, and 256 scenarios give imprecise extreme tails. Bootstrap intervals can degenerate when every outcome is identical. The continuing evaluation right-censors episodes that are still active at the time limit. Mean return and CVaR summarize different parts of the return distribution.

## Reproduce and extend

The [README](../README.md) specifies setup, exact run IDs, source freeze, checkpoints, replay and report generation. Execute the full registered suite with `python src/run_suite.py --phase all --full-frozen-suite`. Preserve `experiments/freeze.json`, registry, dependency record, complete checkpoints and trajectory chunks together. Increasing scenario count requires validating disjoint banks. Any post-freeze environment/reward change requires a new protocol and separated results, rather than overwriting an experiment.

## References

- [InnoRL IAEML course starter](https://github.com/InnoRL/IAEML-assignment).
- [Delalleau et al., Hybrid SAC poster: Discrete and Continuous Action Representation for Practical RL in Video Games](https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/6DcUzo72uIDbw5Piijo2AQ/2bcd7b560e172b5411cb1b86cf0dfa5b/AAAI_2020_FINAL.pdf).
- [Ma et al., DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning](https://arxiv.org/abs/2004.14547).
- [Burda et al., Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894).
- [Ng, Harada and Russell, Policy Invariance under Reward Transformations](https://aima.cs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- [Dabney et al., Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044).
