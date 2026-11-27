# Local learning validation and transfer diagnosis

## Learned behavior checks

| Architecture | Synthetic contexts: worst final-three success | Synthetic decisions | Actual-transition food benchmark: worst final-three success | Real-cache decisions |
|---|---:|---:|---:|---:|
| dsac_mlp | 100.0% | 307,200 | 100.0% | 122,880 |
| ppo_mlp | 99.2% | 204,800 | 100.0% | 122,880 |
| dsac_gru | 100.0% | 409,600 | 100.0% | 122,880 |
| ppo_gru | 100.0% | 307,200 | 100.0% | 122,880 |

Each check evaluates 128 sampled episodes in each of six contexts. Success is a point estimate on fixed development contexts, not independent training replication. The synthetic task abstracts travel, calendar and blackjack. The actual-transition cache holds driving style at zero and truncates after three intentions. Actual simulator observations, external rewards and variable-duration discounts are retained. No action labels, demonstration loss or automatic purchases are used.

The causal food test starts with satiety 0.04 and cleared car debt. A meal is needed before the driver can reach ownership. Its data include every available action; the learner chooses the action sequence. Reaching the game objective in this small task is distinct from winning a ten-day campaign.

## Reproduced implementation and curriculum problems

- A former bank-departure controller crashed in 18 of 62 reconstructed development states. Candidate controllers and their selected-state scope are recorded separately. These are not 62 complete campaigns.
- A long nominal forecast could indefinitely postpone departure from the restaurant. The former forecast also held clock, fuel and fatigue fixed. The selected implementation advances these quantities and falls back to a short stopping guard when all nominal continuations fail.
- An early sustained-steering candidate with a frozen clock was rejected after failing the meal-to-garage witness; the selected planner combines sustained candidates with consistent clock/fuel/fatigue propagation. A traffic-margin candidate was rejected after increasing failures. A direct hungry-home curriculum candidate was also rejected after its proposed short witness missed the work schedule..
- The failed partial MLP pilot consumed 1,581,742 DSAC and 1,281,099 PPO physical ticks before failing its food gate. These stopped runs were not promoted and final banks were not consumed.
- The new food frontier separates meal-to-ownership, meal-to-settlement, meal-to-earning and a short approach to food. The main training budget now counts 262,144 network decisions per architecture, with physical ticks and optimizer updates reported separately.

## Common release checks

Release qualified architectures: dsac_mlp, ppo_mlp, dsac_gru, ppo_gru.

- `ownership_curriculum_verification`: passed.
- `driver_forecast_verification`: passed.
- `decision_budget_verification`: passed.

Reconstructed bank departures: 62/62 arrivals, 0 collisions. Resources were equalized for routing isolation.

Expanded route matrix: 512/540 arrivals, 1 collisions, 27 unfinished. Excluding terminal-only garage departures: 486/486 arrivals. All failed cases remain in `all_routes_clock_escape_True.json`; this is not a safety guarantee.

Hand-chosen food-path witnesses verify feasibility and accounting only; they are not learned performance and never enter replay. The deliberately tiny budget/resume run correctly stops as unqualified after 1,024 decisions rather than being marked trained.

## Reproduction and next stage

See [the active protocol](../../LEARNING_PROTOCOL.md) for the complete curriculum, architecture settings, budget, source freeze, training/validation/final splits, readiness criteria and automatic 30-condition evaluation. [Primary-source review](literature.md) distinguishes quantile DSAC, Hybrid SAC and Gaussian DSAC-T. The study starts fresh and must demonstrate standard-game ownership and retained resource skills before consuming final banks.

Calibration is one initialization with multiple rejected development attempts. Its curves cannot be used as an unbiased algorithm ranking. A complete paper requires final-game results and independent training replications.
