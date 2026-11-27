# Gambler’s Taxi: Implementation plan

Iuliia Kurnaeva  
18 November 2026

## 1. Scope and intended outcome

Build a driving and resource-management task in which an agent starts with exactly $100, repays a car loan, and maintains enough cash for fuel and meals. Every voluntary personal loan is a separate, more expensive liability. Compare five-day, ten-day, and open-ended campaigns. The central question is how a learned policy adapts to changing income, survival needs, liquidity pressure, and financial deadlines.

The task includes sleep, fuel consumption, hunger, three restaurants and separate car and personal liabilities. A daily shift window replaces the earlier job center.

## 2. Assignment requirements and deliverables

The copied assignment PDF specifies the following requirements.

| Assignment requirement | Gambler’s Taxi implementation or deliverable |
|---|---|
| A new JAX vehicle environment derived from BaseEnv | New `src/env/resource_taxi.py` with `init_params`, `reset`, `step`, and `get_observation` |
| State, observation, and action spaces | Vehicle, accounts, individual loans, resources, clock, prices, market indicators, operation eligibility, and documented hidden variables |
| Acceleration, steering, and friction | Bicycle dynamics, traction/braking, drag, lidar, static and moving circular obstacles |
| Assignment 1 report | Signed English Markdown with dynamics, rewards, reflection, video, and local reproduction |
| RL setup and understanding | Explain the learned policy, returns, actor/critic updates, exploration, and evaluation |
| Computational optimization | Batched JAX rollouts, fixed shapes, compiled updates, measured throughput |
| Assignment 2 report | Signed English Markdown with setup, optimization, reflection, video, and model checkpoint |
| Submission | Repository link and branch named `Submission`. Deadline 20 November 2026 at 23:59. |

The PDF describes PPO as an example and permits other RL algorithms. A learned continuous driving policy will be included in the main experiment. A scenario driver can establish feasibility and support diagnostic comparisons. Its contribution will be recorded in each result.

The reports will describe completed implementation and actual results. The implementation plan itself supplies no new experiment findings.

## 3. Financial rules

### Accounts and ownership

Represent funds and liabilities explicitly.

| Quantity | Meaning |
|---|---|
| `cash` | Money available to spend on living, fuel, wagers, and voluntary payments |
| `car_debt` | Remaining car principal and accrued charges |
| `microloans` | Individual expensive loans with IDs, principal, interest, dates, payments, and status |
| `personal_debt` | Derived sum of all outstanding microloan balances |
| `car_paid_total` | Payments credited to the car loan |
| `fees_total`, `interest_total` | Auditable costs by debt type |
| `escrow` | Reserved wager until the game settles |

The confirmed standard start is exactly $100 cash, $1,000 car debt, and no microloans. Cash can subsequently exceed $100. Wages, game settlements, meal purchases, and fuel all use this freely spendable balance. Loan payments remain explicit choices at the bank.

The implemented ownership condition is `car_debt == 0`, `personal_debt == 0`, and successful parking at the garage while alive, within the deadline for finite campaigns. Signing takes one service operation. The proposal replaces the historical baseline's additional $115 purchase fee with loan completion, so the new car cost is defined once. There is no additional ownership fee in the implemented task.

### Borrowing and repayment

Provisional daily rates are 0.5% for car debt and 3% for personal debt. Rates convert to each simulation step as `(1 + daily_rate) ** (1 / ticks_per_day) - 1`. This makes the rate meaningful when simulation resolution changes.

A bank visit offers microloan draws of $25, $50, or $100. Each accepted draw opens a new ledger entry, even when older loans remain unpaid. Its ID, opening date, original principal, daily rate, accumulated interest, payments, remaining balance, and paid status are retained. The actor can pay the car or personal liabilities partially or in full while keeping an operating reserve. Loan proceeds increase cash and the new principal by the same amount. Repayment reduces cash and the chosen liability. Interest and fees are separate expenses. Every positive microloan balance blocks ownership.

The game panel shows the car loan separately, followed by a scrollable list of microloans and a total outstanding count and balance. The machine log retains individual payments and closed loans. The default payment-to-microloans action allocates to the highest interest rate first, then the oldest ID, and displays that rule. The actor chooses how much to allocate between personal and car debt. A later individual-loan selector can test whether the ordering rule matters.

JAX state uses padded arrays for active loan entries. Finite campaigns can allocate the theoretical maximum draws permitted by the horizon and minimum loan-service duration. The open-ended campaign grows capacity between compiled rollout chunks when necessary. Closed entries can be archived in the event log and their slots reused. Capacity pressure must preserve every outstanding liability and its terms and cannot silently cap borrowing or discard a debt.

The ten-day car contract has scheduled payments at days 3, 6, and 10 with provisional cumulative principal targets of $300, $600, and $1,000. Missing a target generates a quoted, once-per-deadline fee, provisionally $10, and leaves the balance payable. Fees do not repeatedly fire every tick. The schedule and remaining amounts appear in observations and the panel. Paying early reduces future amounts due. The main horizon comparison holds these payment dates fixed at days 3, 6, and 10 and changes only the game completion deadline. A five-day agent therefore attempts early full repayment. A separate accelerated-contract comparison can use days 2, 4, and 5 with the same cumulative targets. Its results will identify both the contract and episode horizon.

The open-ended campaign retains the ten-day payment schedule and overdue costs but has no terminal game deadline. It continues until successful ownership, death, or a collision. The car balance can remain overdue after day 10. This tests whether removing forced timeout improves eventual repayment while contractual pressure remains.

The debt order is a decision. An agent may preserve liquidity, prioritize expensive personal debt, or reduce the car balance to meet a milestone. All-in repayment is one available operation, not an automatic default.

## 4. Physical resources, work, and map

### Driving, fuel, hunger, and sleep

Fuel and hunger change during actual operations and travel. Their values and immediate consequences are observed.

- Fuel consumption depends on distance, acceleration, and active work. A parked idle vehicle uses little or no fuel according to a fixed documented rule.
- A fuel station sells a small refill or a full tank at the displayed quote. The charge, delivered volume, and elapsed service time are logged together.
- The visible food gauge is satiety, with `1 = fully fed` and `0 = death`. Time and work consume satiety. Eating restores it. Logs may additionally report `hunger = 1 - satiety`, with the conversion stated explicitly.
- Fatigue retains the work, travel, and sleep cycle. High fatigue and hunger can reduce steering responsiveness and work efficiency through bounded, deterministic coefficients.
- Work and gambling stay available at full fatigue when their ordinary eligibility conditions hold. A living driver may choose them even with low satiety. Reaching zero satiety produces an explicit death terminal state.
- At zero fuel, throttle requests slow pushing movement, provisionally capped at 0.2 map units per simulation-time unit. The agent still steers and chooses its destination. Pushing increases fatigue and consumes satiety more quickly than ordinary driving.

There is no automatic borrowing at cash or fuel depletion. The driver can push to the bank for another microloan, push to any other destination, work if resources and location permit, or wait. Food consumption continues while waiting, travelling, and sleeping. Failure to obtain food eventually causes death. The baseline implements the requested pushing mechanic rather than a roadside rescue service.

### Three restaurants

Food restoration is drawn once per completed purchase from a bounded uniform distribution. The provisional replacement menu is:

| Venue | Base daily price distribution, USD | Satiety restoration | Mean restoration | Service ticks | Location |
|---|---|---|---:|---:|---|
| Budget diner | Uniform(4, 6) | Uniform(0.04, 0.06) | 0.05 | 2 | Near the fuel station |
| Cafe | Uniform(12, 18) | Uniform(0.12, 0.18) | 0.15 | 5 | Distant food district |
| Restaurant | Uniform(30, 40) | Uniform(0.55, 0.75) | 0.65 | 10 | Same distant food district |

The cafe restores three times the diner's amount on average. Even two maximum cafe portions restore 0.36, below the restaurant's minimum portion of 0.55. At mean prices, four cafe visits cost $60 and restore 0.60 before consumption during service, compared with $35 and 0.65 for one restaurant visit. A small purchase remains useful when the agent cannot afford the larger meal or risks starving en route. Repeated visits remain legal and their actual cost and time enter the evaluation.

Actual restoration is capped by remaining satiety capacity. A restaurant portion adds at most 0.75, so it cannot take a nearly empty gauge to full. Satiety continues to decrease while travelling and eating. The public menu shows price, portion range, and service time. The exact portion draw is revealed on the receipt after service.

Prices are sampled at a daily quote update and stay fixed within that quote period. Re-entering the venue does not resample a cheaper offer. Portions are sampled independently by purchase ID in the reference task. Mean-only and wider matched-mean portion distributions are separate experiment arms. Resource consumption and distance are calibrated before fixing the final menu.

The first release uses food quantity, service time, and travel cost. Satiety consumption continues during service, so the driver must reach a restaurant and complete a purchase before depletion. Meal-linked work bonuses, impaired sleep after a cheap meal, and hidden illness are optional experimental treatments. A restaurant benefit would have a logged duration and expiration. It would not directly grant reward or stack indefinitely.

### Work without a job center

Replace employment and rehiring with independent available shifts. Missing a shift loses that opportunity and income. The next work window opens normally. This retains the cost of staying at a casino or sleeping through work while removing the job-center trip and employment state.

The historical baseline did contain job losses. In the saved 128-episode final evaluations, 11 episodes of the original upper-tail actor and 120 episodes of mixed PPO seed 29 included a firing, with zero rehires in both groups. The current project change reflects the requested simplification.

The calibrated base shift pays $175 before the day's public wage multiplier. Starting a shift accepts its displayed wage, fuel budget, and service terms. Settlement uses those accepted terms even if midnight occurs during service. Satiety and fatigue effects operate through documented duration or net income. Either mechanism must be measured independently before combining them.

### Destinations

The implemented map is 40 × 30 and contains home, work, bank, garage, slots, blackjack, a fuel station, and three food entrances. The diner is near the fuel station and serves as a small emergency purchase. The cafe and restaurant share a distant district, reached through the same main approach with comparable final branches. Their geodesic route lengths from home and work should differ by at most 5%, subject to feasibility checks. The diner is intentionally a different distance.

Venue locations are fixed across training seeds and ordinary evaluations. A separate diagnostic can exchange cafe and restaurant identities between their two sites. The 40 × 30 coordinates are stored in src/env/city.py. Cafe and restaurant route balance is checked automatically. 

Every entrance must have a collision-free approach and parking area. Service destinations must not overlap traffic lanes. Reachability checks cover arrival and departure. Route length, expected fuel consumption, travel time, and satiety loss are recorded alongside the meal comparison.

The physics uses the course-compatible continuous controls and lidar. Shared-controller diagnostics will also log intended destination, steering assistance, lack of movement, and recovery attempts to detect the deadlocks observed in the historical baseline.

## 5. Uncertainty and casino design

### Core variation

Add bounded, persistent daily market conditions rather than unrelated fresh noise in every input each tick.

| Variable | Initial design | Information available to the actor |
|---|---|---|
| Shift demand | Daily multiplier in {0.75, 1.0, 1.35} with provisional probabilities {0.3, 0.4, 0.3} | Today's quote and remaining work window |
| Food prices | Venue-specific uniform distributions in the menu above, sampled once per day | Current quotes, quote age, and availability |
| Food restoration | One independent bounded portion draw per completed meal | Portion range beforehand and actual receipt afterward |
| Fuel prices | Provisional daily Uniform(0.85, 1.25) multiplier of the base price | Current quote and availability |
| Traffic delay | A temporary slow segment in one route | Lidar and a public congestion indicator |
| Service disruption | Rare, bounded work delay or fuel-service surcharge | Event notice and actual receipt |

Daily regimes persist long enough for planning. Randomized starts vary cash, fuel, hunger, fatigue, and debt within recorded ranges. A distinct crisis cohort starts with low fuel and cash. Normal and crisis cohorts remain separately reported.

Choose transition parameters and frequency so the reference controller can complete normal scenarios with ordinary income. A harder tail can contain genuinely bad sequences and is reported as such. A policy's failure is interpreted relative to the available opportunities.

### Slots and blackjack

Keep both games and actual blackjack card resolution. The actor chooses hit or stand itself in the main learned policy. Stakes are explicit small, medium, and large presets with visible payout rules and a shared cooldown.

A core slot model remains stationary with negative expectation. A later experimental model has a persistent hidden regime with a noisy public indicator. For every allowed regime, `probability × gross_payout < stake` by a configured margin. Hidden regime transitions are independent of how often the agent chooses to bet. Blackjack commission or table limits may vary by published daily rules. Card outcomes still come from the card model.

Observed wins can inform a belief only in the model that actually contains temporal dependence. Independent outcomes provide the stationary comparison. The proposal will test whether memory helps infer persistent conditions, rather than assume a short winning streak predicts another win.

### Screenshot proposals retained for controlled extensions

| Suggestion | Proposed disposition |
|---|---|
| Automatic overdraft | Superseded by the confirmed voluntary bank-microloan rule |
| Fuel-dependent vehicle mass | Small optional physics ablation after the basic fuel model |
| Large random wages and prices | Stress-test distributions after ordinary solvability is established |
| Surprise tips or repair bills | Capped event treatment using reproducible event streams |
| Hidden meal quality | Optional partial-observation task with observable downstream effects |
| Drifting casino probabilities | Optional persistent-regime test with an independent-outcome control |
| Novelty reward | Exploration ablation only after the external objective and baseline work |
| More bodily needs | Deferred. Hunger, fuel, fatigue, debt, and time already produce competing requirements. |

## 6. State, observations, actions, and text

The public observation includes lidar, speed, heading, destinations and route distances, elapsed time, accepted service status, cash and the two debt totals, active microloan count and a permutation-invariant summary of their rates, fuel, satiety, fatigue, current quotes, work windows, loan terms, milestones, recent event receipts, and legal operations. The complete loan list remains visible in the panel and event log. If later actions select individual loans, their per-loan observations and padding masks will also be provided. Hidden variables include future events and, in the optional regime task, the latent market state. Public action masks must not reveal hidden variables through their eligibility.

The main action has two continuous vehicle controls and a categorical operation. Operation presets cover meal purchase at the current restaurant, small/full fuel purchase, new microloan sizes, payment sizes by debt type, game/stake initiation, hit, stand, sleep, wait, and garage completion. The driving branch also controls pushing when fuel is empty. Presets avoid an extra continuous financial-amount head at the outset. Financial amounts can later become an experimental action parameter if presets prove limiting.

Remove anxiety, mood, tilt, habit-driven control changes, and automatic impulse wagers from the current project core. Voluntary repeated gambling can still emerge through policy decisions under a deadline. Any claim about that behavior must come from action traces and matched comparisons.

The historical baseline dialogue is generated in `baseline/src/replay.py`, function `dialogue`. It picks an event-specific opening, formats a balance, and chooses an ending using cash, fatigue, tilt, and debt. A seed controls those choices during video rendering.

The project renders deterministic English event commentary from the saved ledger, such as “Paid $50 toward personal debt. Balance remaining: $28.” It can combine short event-specific clauses, but cannot invent an intention or claim to read the policy's thoughts. The dialogue renderer does not alter state or reward and runs after simulation.

## 7. Reward and ledger checks

All principal algorithms share the external reward: +1 for ownership plus a cash reserve bonus capped at 0.05, -0.1 on any unsuccessful terminal, and -0.1(1-gamma) on each nonterminal step. Terminal steps have no additional running charge. For an unsuccessful episode of T physical steps, the discounted base return is exactly -0.1, independent of failure time. For ownership it is -0.1 + gamma^(T-1)(1.1 + reserve_bonus), so earlier otherwise identical success is strictly preferred. The separate financial increment is added below. There is no additional debt-running or timeout-shortfall penalty; actual interest and fees reduce net worth. This replaces an earlier calibration reward whose larger terminal failure penalty made delayed starvation preferable to an early crash. The complete discounted running cost is bounded by 0.1. This reward is shared across principal agents and horizons; T09 separately removes the secondary financial increment.

Potential shaping adds `gamma * Phi(next_state) - Phi(state)` using the full observed financial position and relevant physical resources with zero terminal potential. A purchase exchanges cash for a usable resource, such as fuel, food in progress, or a reserved wager. The potential must account for that exchange consistently. The transition tests will check the discounted telescoping sum and terminal handling. Changing potential can improve the learning signal but cannot substitute for a valid external objective.

Every resource event follows a conservation check. Borrowing changes cash and debt together. Loan principal repayment changes both together. Charges are booked once. A wager is reserved once and released once. A meal or refill is charged when ordered, consumes service time, and is delivered only if the agent survives until completion. No refund is granted on death or collision. An interruption cannot both refund and deliver the same purchase.

Evaluation reports the external objective, event counts, and physical outcomes separately from shaped training return. There is no direct visiting bonus, gambling bonus, cash-from-credit bonus, or per-payment bonus. Repeated payment/refund loops must not produce external reward.

RND, if tested, uses its own predictor and normalization and decays to zero before final evaluation. Its input excludes unbounded clock noise and random event IDs that could create permanently novel observations. The final frozen policy is evaluated without novelty reward.

## 8. Agent design and learning schedule

Hybrid DSAC remains a main current project algorithm. The original Gaussian driving head, categorical service head, and quantile critics will be adapted to the new observation and action definitions. Its historical weights and results are preserved in the baseline. Main current project runs train from fresh initializations.

The three primary agents are Hybrid DSAC with an MLP, hybrid PPO with an MLP, and hybrid PPO with a GRU. This compares algorithms within the feedforward architecture and memory within PPO. A recurrent Hybrid DSAC is an additional registered arm. Every main agent learns both continuous driving and categorical operations.

All main arms share the external task reward, observation and action semantics, curriculum, and scenario banks. Algorithm, architecture, entropy settings, sample budget, source revision, and checkpoint identity are recorded separately. Economic behavioral cloning is disabled in the main comparison. RND is a named exploration treatment. Scenario drivers supply feasibility controls.

For the PPO driving branch, a Gaussian controls throttle and steering with a tanh correction. Its joint ratio includes the selected categorical log probability and the continuous probability only when a driving command is active. Recurrent policies preserve sequence order, previous-action and event inputs, episode reset masks, and consistent hidden-state reconstruction. Rollout boundaries bootstrap continuing tasks. Actual death, collision, and finite game deadlines terminate episodes.

The shared curriculum retains the financial completion goal in every stage. Accessible starts and lower principal occupy the first 20% of physical transitions, normal services and both liabilities the next 30%, and the full assigned task the final 50%. There is no casino-specific teaching stage. A separate navigation-only pilot would have its own identifier and any transfer recorded.

Scenario IDs 1–10 are reserved for reusable development checks, including cases 6 and 7, and excluded from every training data source. Training scenarios begin at 11, with N = 10,000 giving IDs 11–10010. N is configurable. At N = 100,000 the range becomes 11–100010. Validation uses 1000001–1000064 and final standard testing uses 2000001–2000256. Reserved-bank overlap checks run before training. Repetitions A–E use separate model-initialization seeds 201–205, with calibration initializations 101/102 kept separate from main repetitions. Common scenario histories are generated independently of model initialization and policy-sampling RNG. Different actions can lead to different trajectories and exposure. [Exact split and evaluation rules](EXPERIMENTS.md#randomness-and-common-scenarios)

The initial registered budget is 2,097,152 physical transitions per run with 64 parallel environments. A complete resume checkpoint includes optimizer, RNG, collector state, and the DSAC replay buffer or recurrent state where applicable. Timing, convergence, and storage requirements will be measured on the pilot. [Full training and randomness protocol](EXPERIMENTS.md)

## 9. Experiments and policy catalog

The main matrix contains three agents, three horizons, and five training seeds, for 45 independent runs. The first block is the ten-day comparison. Five-day and open-ended runs follow on the same final environment and payment calendar. The open-ended game is observed to a prespecified 30-day administrative cutoff, with surviving unpaid trials recorded as censored.

Eleven additional training arms test the curriculum, market randomization, RND, payment milestones, casino availability, recurrent Hybrid DSAC, mean-only meals, wider meal distributions, financial progress, navigation cues, and the discount curriculum. Each uses five repetitions, adding 55 registered runs. These are subsequent comparison blocks, whose parameters will be frozen before execution. Casino-regime, reduced-quote, and accelerated-contract designs X01–X03 add 25 registered runs, for 125 total.

Frozen-policy tests separately change the horizon, driving effectiveness, actuation noise, a driving guard, and financial or resource shocks. Those tests hold learned weights fixed. Retraining under a changed condition receives a different run ID. Every evaluation reports whether the driver was learned or assisted.

The 125 enumerated training runs appear in [registry.csv](experiments/registry.csv) with status planned. Their shared design is recorded in [protocol.json](experiments/protocol.json). A run ID such as dsac_mlp_10d_curriculum_dynamic_repA resolves to its algorithm, architecture, initialization seed, task, and source/configuration hashes when executed. Frozen-policy tests and additional designs have their own IDs in [the master experiment list](EXPERIMENTS.md#all-comparisons-at-a-glance).

For each run save the initial, curriculum-boundary, periodic, validation-selected, and final checkpoints. Log every training, validation, and test episode in compressed chunks with physical trajectories, requested and executed controls, resources, public quotes, financial receipts, and individual microloans. Post-simulation rendering uses those records. Case selection follows a fixed ordering of test seeds.

The result catalog includes first completion, first ownership after one or several microloans, first voluntary repeat bet after a loss, each restaurant, pushing, starvation, missed payments, and route deadlock. Absent cases remain explicitly absent. All-seed statistics accompany examples.

Report ownership, external return, debt-clearance times, personal-loan count and interest, cash reserve, resource consumption, restaurant choices, sleep, missed shifts, game participation, starvation, collisions, and stalled time. Independent training repetitions and paired scenario variability have separate summaries. A matched outcome probe must hold the stated finances, time, and resource state fixed and record any recurrent-history intervention.

## 10. Implementation order and acceptance conditions

| Stage | Deliverable and completion check |
|---|---|
| Ledger and resources | Car liability, individual microloan ledger, one cash account, validated fees and interest, fuel/pushing and satiety/death transitions, service cancellation and episode-boundary checks |
| Map and controls | Every venue can be reached, parked at, and departed from. Gas, steering, drag, and moving obstacles remain active. |
| Feasibility | Reference strategies complete normal scenarios; failure cases identify whether the budget, resources, or routing caused them |
| Learning | A full-state MLP pilot first, then the sequence-trained GRU. All reported successes identify the controller and observation type. |
| Experiments | Frozen settings and seed banks, selected and last checkpoints, per-episode results, control comparisons, and measured uncertainty |
| Replay and reports | Video generated from numeric logs, saved policy replay, two signed assignment reports, research report, and tested README commands |

The implementation will use a new Gambler’s Taxi source tree outside the baseline archive. Old weights have incompatible observations and action definitions and will be identified as the historical baseline artifacts. Gambler’s Taxi files will record a configuration/schema version so a loader cannot silently treat them as compatible.

## 11. Confirmed choices and provisional parameters

Configuration recorded on 18 November 2026

1. Start with exactly $100 freely spendable living money. Later balances are unrestricted.
2. Car payments have their own due dates. Finishing the car loan is the principal objective, and any outstanding personal loan blocks ownership.
3. Every bank microloan appears as an individual entry, with aggregate outstanding count and balance.
4. Empty fuel allows slow, tiring pushing in any direction. Borrowing still requires a voluntary trip to the bank.
5. Zero satiety causes death. Waiting without money can therefore end in starvation.
6. Compare five-day, ten-day, and open-ended horizons.
7. Retain Hybrid DSAC as a main algorithm and compare it with PPO. Distinguish algorithm from memory architecture.
8. Train without economic demonstration targets in the main arms and let the actor choose among services and games under the common task objective.
9. Use aligned scenario banks, multiple independent training seeds, explicit scenario variants, and complete logs with an unambiguous policy registry.
10. Reserve scenarios 1–10 for development checks. Training starts at 11 and its count is configurable without changing those ten cases.
11. Food portions use bounded distributions. The diner restores approximately 0.05 near the fuel station, the cafe approximately 0.15, and the restaurant approximately 0.65.
12. Cafe and restaurant locations remain fixed in one distant district with comparable routes. The diner has its own deliberately convenient location.

Provisional values to calibrate before the experiment freeze include loan rates and sizes, payment targets and fees, fuel capacity and consumption, pushing fatigue, food depletion, venue prices, and event frequencies. The proposal treats loan completion as the purchase price and uses the garage for physical ownership confirmation. The optional screenshot mechanisms are separately configurable. The financial environment, four learner implementations, logging, offline renderer and report builder are implemented. Engineering verification and execution statuses are tracked in reports/results.md.

## References

- [Course starter repository](https://github.com/InnoRL/IAEML-assignment)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Hybrid continuous and discrete actions](https://arxiv.org/abs/1912.11077)
- [Recurrent control under partial observation](https://arxiv.org/abs/2110.12628)
- [Implementation details in policy gradients](https://arxiv.org/abs/2005.12729)
- [Random Network Distillation](https://arxiv.org/abs/1810.12894)

## Pre-freeze learning calibration: protocol 0.5

A full-budget engineering pilot failed on validation. Before any final test, DSAC entropy temperatures were rescaled to 0.000006/0.000012, bounded navigation shaping was strengthened without casino targets, and checkpoint selection was aligned with mean discounted external return. Primary evaluation samples the trained policy using scenario/tick-specific policy keys; deterministic extraction is a separate case. The failed pilot is retained and does not count as a main experiment. This bundle of changes is engineering calibration, not an isolated causal ablation.

## Physics and accessible-start correction: protocol 0.6

Free Wait now coasts under friction instead of erasing momentum. The first 20% curriculum balances reachable near-ownership states, a bank start with $25 car debt, and an easy-stage work start. Cash is $100 in each. Later stages and every validation/test use the standard home reset. Actions and the ownership goal remain unchanged, with no demonstrations or economic action override. Earlier pilots are retained under their source digests and excluded from final comparisons.

## Learning support and bounded financial progress: protocol 0.7

The first 20% curriculum now has four distance substages: 1.56, 3, 4.5 and 6 units from ownership, introducing bank/work cases after 10%. A public-state-derived navigation cue exposes a nearby resource/contract route without casino targets or action overrides. The secondary financial reward is U(W_next)-U(W_current), U(W)=0.1 clip(W/2000,-1,1), whose discounted contribution is bounded by 0.2. This is an explicit objective term, not policy-invariant shaping. T09 removes that term; T10 removes the derived cue. These add ten registered runs; the subsequent discount ablation brings the registry to 125. Earlier failed pilots remain separate.

## Shared driving and learned economic decisions

The primary policy selects 26 explicit intentions: wait, work, sleep, three food venues, two refueling quantities, three loan amounts, six repayment amounts, three slot stakes, three blackjack stakes, hit, stand and ownership. A common route controller converts the selected destination and two Gaussian style parameters into real acceleration and steering. Speed factors range from 0.8 to 1.0 and steering factors from 0.9 to 1.1. Physics, fuel, hunger, fatigue, traffic, collisions and time remain active. The controller does not select a financial strategy, a restaurant, a casino or a loan. No economic demonstrations are used.

An intention runs for at most 32 physical ticks and can execute its requested operation once. Service completion or game feedback returns control immediately. Wait lasts at most eight ticks and ends earlier when an ongoing service finishes. Blackjack hit/stand decisions return control after one card action. Padded storage slots freeze the simulation and do not count as experience. Each decision reward sums its physically discounted rewards; bootstrapping uses gamma raised to its actual duration. Physical gamma is 0.999 in the final task. The policy sees 153 public environment features plus the previous-intention one-hot vector, for 179 features. A persistent PPO GRU updates per decision; finite-history DSAC retains 32 decisions, spanning at most 1024 physical ticks.

Every run receives 4,194,304 active physical transitions, with final overshoot below 16,384 because collection uses 64 environments and eight decisions. DSAC makes 32 updates of 256 examples per collection; PPO makes three epochs in environment minibatches of eight. The initial 20% of the economic curriculum starts at home with $200 principal, slower resource consumption and fixed prices. Its four equal substages use physical discounts 0.995/0.997/0.998/0.999, except T11. The next 30% uses $500 debt and mixed home/service starts. The final 50% uses the complete $1000 task at home. Validation always uses the complete task. Raw-steering driving-school pilots are preserved as calibration history and are not the primary method.

The controller's development benchmark covers 900 directed service-pair routes on scenarios 1–10. The selected 24-tick safety forecast with a four-tick held command achieved 779 arrivals, 23 collisions and 98 unfinished routes. Ninety unfinished routes were departures from the garage, where successful campaigns already terminate. This is a route diagnostic, not an ownership result or a claim of collision-free driving. The longer-forecast frozen intervention is reported separately.

The 125 registered rows remain the experiment matrix. Main initializations are 201–205, calibration uses 101/102, and execution checks use 9091/9092. Development scenarios 1–10, training 11–10010, validation 1000001–1000064 and final scenarios 2000001–2000256 are disjoint. Test-bank evaluation begins after the final source/configuration freeze. All training and evaluation are headless; videos are reconstructed afterward from saved physical trajectories and decisions.
