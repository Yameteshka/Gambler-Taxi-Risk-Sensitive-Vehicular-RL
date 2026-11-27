# Gambler’s Taxi: Experiment protocol

Iuliia Kurnaeva  

## All comparisons at a glance

Training comparisons vary the specified factor. Frozen-policy comparisons keep weights and normalization fixed and use held-out scenarios.

| ID | Comparison | Training or frozen evaluation | What is measured |
|---|---|---|---|
| D00 | Fixed diagnostics 1–10 and feasible reference strategies | Development checks | Accounting, survival, map access, and existence of a successful route |
| M01 | Hybrid DSAC–MLP vs. PPO–MLP | Train both under the same task and experience budget | Algorithm, ownership, survival, debt, time, and cash |
| M02 | PPO–MLP vs. PPO–GRU | Train both under the same task and experience budget | Contribution of memory |
| H01 | Five-day vs. ten-day vs. open-ended | Train each main agent separately for each horizon | Adaptation to available time |
| H02 | The same ten-day policy under all three horizons | Frozen evaluation | Immediate response to a changed deadline |
| T01 | Curriculum vs. full task from the first transition | Train on the ten-day DSAC–MLP reference | Value of gradual difficulty |
| T02 | Dynamic vs. fixed market during training | Retrain, then evaluate on the same dynamic suite | Robustness to changing quotes and wages |
| T03 | Standard exploration vs. decaying RND | Retrain with the same external objective | Exploration and final task performance |
| T04 | Scheduled installments vs. final payment only | Retrain with the changed contract | Payment pressure and borrowing |
| T05 | Games available vs. unavailable | Retrain with the changed action availability | Contribution of gambling opportunities |
| T06 | Hybrid DSAC–MLP vs. Hybrid DSAC–GRU | Train both on the ten-day reference | Memory within the original algorithm |
| T07 | Random meal restoration vs. deterministic means | Retrain with prices and locations held fixed | Contribution of food-outcome uncertainty |
| T08 | Reference vs. wider meal distributions with the same means | Retrain with prices and locations held fixed | Sensitivity to food variance and survival risk |
| T09 | Bounded wealth-progress reward enabled vs. disabled | Retrain | Contribution and side effects of the secondary financial objective |
| T10 | Derived public-state route cue enabled vs. disabled | Retrain | Dependence on feature engineering |
| T11 | Temporal discount curriculum vs. constant 0.999 | Retrain | Credit assignment during early skill acquisition |
| F01 | Standard control vs. weaker steering/braking | Frozen evaluation | Physical-control robustness |
| F02 | Standard control vs. noisy actuation | Frozen evaluation | Robustness to control errors |
| F03 | Standard control vs. a driving guard | Frozen evaluation | Collision reduction, stalling, and financial completion |
| F04 | Normal fuel vs. low fuel | Frozen evaluation | Refueling, pushing, and survival |
| F05 | Normal satiety vs. low satiety | Frozen evaluation | Emergency food choice and survival |
| F06 | $100 standard cash vs. low cash | Frozen evaluation | Liquidity and microloans |
| F07 | Normal vs. lower income | Frozen evaluation | Work, payments, and game participation |
| F08 | Normal fuel price vs. a price shock | Frozen evaluation | Reserve and routing decisions |
| F09 | No opening microloans vs. active microloans | Frozen evaluation | Debt ordering and ownership blocking |
| F10 | Standard conditions vs. a fixed combination of adverse conditions | Frozen evaluation | Joint robustness and failure cause |
| F11 | Reference food draws vs. means vs. wider matched-mean draws | Frozen evaluation | Response to changed meal reliability |
| F12 | Exchange cafe and restaurant labels between their two fixed sites | Frozen evaluation, independent diagnostic map | Residual site preference and use of public venue information |
| P01 | Matched win/loss and changes in cash, satiety, fuel, and time | Policy-output and rollout probes | Voluntary continuation and decision sensitivity |
| P02 | Recurrent history preserved vs. reset at a declared decision | Frozen recurrent-policy probe | Dependence on stored history |
| P03 | Critic predictions for physically equivalent stationary brake actions | Frozen learner diagnostic | Exact-transition check and a lower bound on approximation error |
| X01 | Independent casino outcomes vs. a persistent hidden regime | Additional environment and matched retraining | Whether memory responds to predictive information |
| X02 | Public quotes vs. reduced information | Additional observation condition and matched retraining | Value of available information |
| X03 | Normal vs. accelerated five-day installment calendar | Additional contract and matched retraining | Effect of payment dates at a fixed game horizon |

M01, M02, and H01 are covered by the 45 main runs. T01–T11 add 55 ten-day runs. H02, F01–F12, and P01–P03 use the saved policies and have their own evaluation records. X01–X03 now have 25 additional registered training arms and corresponding environment flags. Their result status remains planned until execution. The complete registry contains 125 runs.

## Agents and the question each comparison answers

Hybrid DSAC remains a primary algorithm in Gambler’s Taxi. The original implementation combines a Gaussian driving head, a categorical operation head, and twin quantile critics. We will adapt those interfaces to the new resources and individual loan ledger. Its historical weights remain in the baseline archive. The standard current project comparisons start each model from a new initialization, so previous demonstrations and incompatible features cannot silently influence one arm.

| Agent ID | Learning algorithm | Architecture | Continuous driving and operations | Main comparison |
|---|---|---|---|---|
| `dsac_mlp` | Hybrid DSAC, expected-return actor | Feedforward MLP | Learned by the actor | Preserve the original method under current project rules |
| `ppo_mlp` | Hybrid PPO | Feedforward MLP | Learned by the actor | Compare algorithms at the same observation and action interface |
| `ppo_gru` | Hybrid PPO | Recurrent GRU | Learned by the actor | Compare memory within PPO |
| `dsac_gru` | Recurrent Hybrid DSAC | Recurrent GRU | Learned by the actor | Extension for an algorithm-by-memory interaction |

PPO and DSAC are update algorithms. MLP and GRU define how observations are represented by the network. A checkpoint stores a particular trained policy. A scenario defines external conditions. Each appears in a separate registry field.

The main agents receive the same public observation, previous action, and previous public event information. Recurrent agents additionally maintain history through hidden state. A full-information diagnostic can expose latent regimes only in its own labeled condition. Record parameter counts, features, action schemas, update ratios, entropy settings, gradient steps, and compute time. Equal environment experience is the primary sample-budget comparison. The different algorithms retain their appropriate update rules.

## Objective and freedom of choice

Every main run optimizes the same external task reward. The aim is to stay alive, clear the car loan and all microloans, and complete ownership. Earlier completion and a remaining operating reserve are valued through the same formula in each horizon. The policy chooses its route, whether to work, whether to gamble, which venue or game to use, how much to spend or borrow, and which debt category to repay.

There are no economic demonstration labels or game-specific rewards in the primary runs. Casino quotes and public stakes are observations. The casino is one available action among the services in the world. Scripted references run only as feasibility controls and separate evaluations.

The proposed external reward is:

```text
completion = +1.0 + 0.05 * clip(remaining_cash / 500, 0, 1)
any_unsuccessful_terminal = -0.1
nonterminal_tick = -0.1 * (1 - gamma)
financial_increment = U(net_worth_next) - U(net_worth_current)
U(W) = 0.1 * clip(W / 2000, -1, 1)
training_shaping = gamma * Phi(next) - Phi(current)
```

There is no running charge on the terminal step. The discounted base return of every unsuccessful finite trajectory is exactly -0.1, independently of its duration. Ownership at physical step T gives -0.1 + gamma^(T-1)(1.1 + reserve_bonus). Actual interest, fees and purchases affect the separate bounded financial objective. Thus delaying an otherwise identical failure cannot improve the base objective, while earlier identical success is preferred. The reference gamma is 0.999; early curriculum discounts are specified in protocol.json and T11 removes that schedule.

The financial increment is an explicit secondary objective, not policy-invariant shaping. Its total discounted signed contribution has absolute value at most 0.2; this does not bound the sum of absolute increments. T09 removes it. Potential shaping uses zero terminal potential and telescopes, so it cannot manufacture terminal return from a repeated visit or payment loop. Its nearest-service cue excludes casinos; this aid is disclosed and removed by T10. Evaluations exclude shaping and novelty reward.

## Randomness and common scenarios

Two different kinds of seeds are required.

**Initialization seeds** control model initialization, policy sampling, data order, and optimization. Repetitions A–E use 201, 202, 203, 204, and 205. Calibration pilots use separate initializations 101/102. The public run name uses the repetition letter, and the registry records its numeric initialization seed. Different architectures receive valid, architecture-specific initializations.

**Scenario seeds** define initial vehicle and account conditions, daily market history, traffic initialization, and independent event streams. The following allocation keeps the first ten diagnostic scenarios fixed when training grows.

| Purpose | Scenario IDs | Use |
|---|---|---|
| Reusable development checks | 1–10 | Playback and diagnostics, including cases 6 and 7 |
| Training | 11 through 10 + N | N = 10,000 initially, giving 11–10010 |
| Checkpoint validation | 1000001–1000064 | Selection and tuning within the protocol |
| Final standard test | 2000001–2000256 | Held-out comparison after selection |
| Additional test cohorts | Separate blocks starting at 2100001 | Held-out adverse-condition cohorts |

N is a configuration value. At N = 100,000 training uses 11–100010, while diagnostics and final evaluation IDs remain unchanged. A split validator must reject any requested training range that overlaps a reserved evaluation block. Scenario IDs 1–10 are excluded from training resets, replay buffers, curriculum starts, and demonstrations. Development feedback on those cases can still influence design, so final performance is reported on the untouched final bank. The numeric initialization seeds are a separate RNG namespace and do not select an episode.

The training scenario order for each environment lane is reproducible and shared across agents. A reset advances that lane's episode counter. An agent that survives longer encounters fewer resets per physical-transition budget, so the protocol reports episodes and exposure alongside transition counts.

Use named random streams for markets, traffic, service disruption, tips if enabled, and game outcomes. Exogenous events are keyed by scenario ID, absolute day/tick, and event type. Casino settlements use named game streams and a shared resolved-round ordinal; the optional latent regime evolves on a daily stream. A bet size does not select another random outcome. Different actions cannot consume the random number intended for tomorrow's fuel price. Actions still determine where the driver is, which events are encountered, and when a wager is placed. A common seed aligns the scenario generation, not the resulting trajectories.

Training episodes sample many conditions. A hundred or a thousand network initializations are unnecessary for the initial study. Five independent trainings and hundreds of held-out episodes expose both optimization variation and environmental variation. The number of repetitions and their uncertainty must be reported with every comparative result.

## Food distributions and fixed geography

The reference menu uses independent uniform portion draws. The budget diner restores Uniform(0.04, 0.06), the cafe restores Uniform(0.12, 0.18), and the restaurant restores Uniform(0.55, 0.75). Their means are 0.05, 0.15, and 0.65. Reference daily price quotes use Uniform(4, 6), Uniform(12, 18), and Uniform(30, 40) USD, respectively. Each day's price is known before purchase and remains fixed during the quote period. The portion is sampled once per purchase and delivered at completion if the driver survives and appears in its receipt. Gains are clipped to remaining satiety capacity.

In T07, each portion is exactly its reference mean. In T08, the distributions become Uniform(0.02, 0.08), Uniform(0.06, 0.24), and Uniform(0.35, 0.95), preserving all three raw portion means. Price and other service rules stay aligned. Saturation clipping can change the expected usable gain, so the analysis also calculates E[min(portion, remaining_capacity)] at the decision state. F11 measures the immediate sensitivity of a fixed actor to those distribution changes. Each distribution version is logged.

The diner sits near the fuel station. The cafe and restaurant have fixed neighboring entrances in a distant district, with a common approach and comparable final branches. Target geodesic distances from home and work differ by at most 5% between cafe and restaurant. Distances and travel costs are measured on the final map. The layout is fixed in the main experiments. F12 is an explicit identity-swap diagnostic, with its own map ID.

For a uniform draw, E[X] = (lower + upper) / 2. These means describe the resources a meal supplies. A decision also depends on the actual quote, clipping at the satiety ceiling, hunger consumption during service and travel, fuel costs, missed work, remaining cash, and future debt payments. Meal selection optimizes expected future task return under survival constraints. A different choice from the best mean-portion-per-dollar ratio can therefore be rational. A paired action intervention or rollout comparison is needed before attributing it to a learned mistake.

The analysis will compare observed choices against cash- and resource-feasible alternatives, using expected effective restoration, conservative restoration, route costs, and an estimated probability of surviving to service completion. Expected values alone cannot establish the policy's internal reasoning.

## Main matrix and budgets

The primary grid contains the three main agents and three training horizons.

| Axis | Values |
|---|---|
| Agent | `dsac_mlp`, `ppo_mlp`, `ppo_gru` |
| Horizon | `5d`, `10d`, `open` |
| Training repetition | A–E, initialization seeds 201–205 |
| Curriculum | Shared schedule |
| Market | Dynamic reference distribution |
| Driving | Shared route controller; real continuous dynamics |
| External reward | `ownership_financial_progress` |
| Initial state, standard cohort | $100 cash, $1,000 car principal, no microloans |
| Training experience per run | 4,194,304 physical transitions |

This is 45 planned main runs and 188,743,680 physical transitions. First validate the world and train ten-day pilots. The ten-day main block has 15 runs. Other horizons follow once the training and logging pipeline is verified.  Any increase must apply across the affected comparison block and receive a new protocol revision.

All curriculum stages retain the ownership objective. Early stages use lower principal, favorable resources, and shorter routes to the same financial completion. There is no separate “go to the casino” stage. A navigation-only task can be used as a separately named pilot, with its transition budget and any transferred weights recorded.

Use the first 20% of training for accessible starts, the next 30% for normal services and both liabilities, and the final 50% for the complete assigned task. Stage boundaries use physical-transition counts and fixed interpolation rules. The same observation and action schema exists throughout. The source and frozen configuration determine which prices and resource rates change.

The open-ended game keeps its payment dates and overdue consequences. Its validation and test observation window is 30 days. A surviving unfinished trial is right-censored at that window. It is kept distinct from death, collision, or a genuine finite deadline. The saved state permits later continuation.

The main horizon arms retain the same day-3, day-6, and day-10 payment calendar and change the episode deadline. A separately registered accelerated five-day contract can move installments to days 2, 4, and 5. That arm would change the contract as well as the available time.

## Training ablations

Use the ten-day `dsac_mlp` condition as the initial reference for external training changes. Every arm uses the same five seeds and physical-transition budget.

| Variant | Single planned change |
|---|---|
| `full_task_from_start` | Remove curriculum while retaining final conditions |
| `fixed_market_training` | Fix training quotes and demand, then test on the same dynamic suite |
| `rnd_exploration` | Add a capped, normalized RND bonus and decay it to zero |
| `final_payment_only` | Remove intermediate payment milestones |
| `no_casino_training` | Disable wagering and retrain to measure the contribution of its availability |
| `dsac_gru` | Add recurrent sequence learning to the original algorithm |
| `meal_mean_only` | Replace random meal restoration with its mean, keeping quotes and geometry unchanged |
| `meal_high_variance` | Widen restoration distributions while preserving their means |

Together with T09 financial progress, T10 derived navigation cues and T11 temporal discount, these eleven extensions add 55 planned runs. A GRU version of a training intervention can be added as a crossed comparison if the result requires it, with its additional runs registered before execution.

The fixed reward and aligned seeds make each contrast interpretable within its stated axis. A scheduled-payment comparison changes the task's contract. A no-casino comparison changes the available actions. Those are explicitly labeled changes, rather than attributed to the network architecture.

## Frozen-policy evaluations

A frozen policy means its weights, observation normalization, and action-selection rule are fixed. Each evaluation records its own scenario setting and RNG mode. Learned recurrent state can evolve during an episode and resets between episodes.

Evaluate the selected main policies under:

- Five days, ten days, and the open-ended window, with the active deadline visible in the observation.
- Standard full-information public quotes and a separately labeled reduced-information condition.
- Standard driving, weakened brakes/steering response, mild actuation noise, and a driving-guard diagnostic.
- Fuel-price shock, low initial fuel, low satiety, low liquidity, lower income, and an active personal-debt start.
- A prespecified combined adverse cohort.
- Stationary independent casino outcomes and the separately defined persistent-regime task, if implemented.

The “unlicensed driver” idea is operationalized as a measurable change to control effectiveness or input noise. It is an environmental driving treatment. A separately trained poor driver would introduce another learned component and require a distinct comparison.

These evaluations identify robustness to changed conditions. The matching retrained variants answer adaptation questions. A frozen ten-day policy on five days and a policy trained for five days have separate result labels.

## Registry, checkpoints, and traces

The registry has one row per independent training run. Its full configuration records algorithm, architecture, reward version, horizon, curriculum, market distribution, controller, observation/action schema, exploration, training seed, scenario banks, source hashes, dependencies, and model/optimizer settings. The accompanying JSON is a design specification, and the CSV rows remain `planned` until execution.

A readable example run ID is `dsac_mlp_10d_curriculum_dynamic_repA`. A replay identifies that run, an immutable checkpoint hash, evaluation variant, scenario seed, and stochastic-action seed. “Best” means selected by the recorded validation rule.

Save:

- Initial configuration and the random initialization.
- Curriculum boundary checkpoints.
- Periodic checkpoints at fixed physical-transition counts.
- The validation-selected checkpoint and the final checkpoint.
- Optimizer, RNG, replay state for DSAC, and recurrent/collector state for resume.
- Every training episode's action, resource, ledger, market, and physical trajectory records in compressed chunks.
- Every validation and test episode with its terminal cause or administrative truncation.

Measure log size in the pilot to choose chunking and compression before the full campaign. Periodic full state plus recorded trajectory and event data enables visual reconstruction without model inference. A second replay mode reruns the saved policy and checks it against the recorded trajectory. Exact equality is measured and reported on the reference backend.

Policy outputs and action probabilities are evidence of decision sensitivity. Counterfactual financial and resource changes, and recurrent-state interventions, can probe that sensitivity. Event commentary is generated afterward from receipts. It cannot establish an agent's internal thoughts.

Select the main checkpoint by validation completion rate, then survival, then remaining debt, then deadline-adjusted completion time. Report external return, success, both debt-clearance rates, the cash reserve, interest, loan count, food, fuel, sleep, game participation, and failure cause. Independent training seeds are the main replication unit. Paired scenario summaries and intervals describe environmental variability within those runs.

## Implementation status and references

This protocol updates the Gambler’s Taxi design. The source adaptation and trainers are implemented; engineering verification and registered runs are tracked separately. The baseline retains the historical Hybrid DSAC artifacts.

- [Implementation plan](PLAN.md)
- [Planned run registry](experiments/registry.csv)
- [Structured protocol](experiments/protocol.json)
- [Hybrid SAC](https://arxiv.org/abs/1912.11077)
- [Distributional SAC](https://arxiv.org/abs/2004.14547)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Evaluation across training runs](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

## Pre-freeze learning calibration: protocol 0.5

A full-budget engineering pilot failed on validation. Before any final test, DSAC entropy temperatures were rescaled to 0.000006/0.000012, bounded navigation shaping was strengthened without casino targets, and checkpoint selection was aligned with mean discounted external return. Primary evaluation samples the trained policy using scenario/tick-specific policy keys; deterministic extraction is a separate case. The failed pilot is retained and does not count as a main experiment. This bundle of changes is engineering calibration, not an isolated causal ablation.

## Physics and accessible-start correction: protocol 0.6

Free Wait now coasts under friction instead of erasing momentum. The first 20% curriculum balances reachable near-ownership states, a bank start with $25 car debt, and an easy-stage work start. Cash is $100 in each. Later stages and every validation/test use the standard home reset. Actions and the ownership goal remain unchanged, with no demonstrations or economic action override. Earlier pilots are retained under their source digests and excluded from final comparisons.

## Learning support and bounded financial progress: protocol 0.7

The first 20% curriculum now has four distance substages: 1.56, 3, 4.5 and 6 units from ownership, introducing bank/work cases after 10%. A public-state-derived navigation cue exposes a nearby resource/contract route without casino targets or action overrides. The secondary financial reward is U(W_next)-U(W_current), U(W)=0.1 clip(W/2000,-1,1), whose discounted contribution is bounded by 0.2. This is an explicit objective term, not policy-invariant shaping. T09 removes that term; T10 removes the derived cue. These added ten registered runs; the subsequent discount ablation brought the registry to 125. Earlier failed pilots remain separate.

## Shared driving and learned economic decisions

The primary policy selects 26 explicit intentions: wait, work, sleep, three food venues, two refueling quantities, three loan amounts, six repayment amounts, three slot stakes, three blackjack stakes, hit, stand and ownership. A common route controller converts the selected destination and two Gaussian style parameters into real acceleration and steering. Speed factors range from 0.8 to 1.0 and steering factors from 0.9 to 1.1. Physics, fuel, hunger, fatigue, traffic, collisions and time remain active. The controller does not select a financial strategy, a restaurant, a casino or a loan. No economic demonstrations are used.

An intention runs for at most 32 physical ticks and can execute its requested operation once. Service completion or game feedback returns control immediately. Wait lasts at most eight ticks and ends earlier when an ongoing service finishes. Blackjack hit/stand decisions return control after one card action. Padded storage slots freeze the simulation and do not count as experience. Each decision reward sums its physically discounted rewards; bootstrapping uses gamma raised to its actual duration. Physical gamma is 0.999 in the final task. The policy sees 153 public environment features plus the previous-intention one-hot vector, for 179 features. A persistent PPO GRU updates per decision; finite-history DSAC retains 32 decisions, spanning at most 1024 physical ticks.

Every run receives 4,194,304 active physical transitions, with final overshoot below 16,384 because collection uses 64 environments and eight decisions. DSAC makes 32 updates of 256 examples per collection; PPO makes three epochs in environment minibatches of eight. The initial 20% of the economic curriculum starts at home with $200 principal, slower resource consumption and fixed prices. Its four equal substages use physical discounts 0.995/0.997/0.998/0.999, except T11. The next 30% uses $500 debt and mixed home/service starts. The final 50% uses the complete $1000 task at home. Validation always uses the complete task. Raw-steering driving-school pilots are preserved as calibration history and are not the primary method.

The controller's development benchmark covers 900 directed service-pair routes on scenarios 1–10. The selected 24-tick safety forecast with a four-tick held command achieved 779 arrivals, 23 collisions and 98 unfinished routes. Ninety unfinished routes were departures from the garage, where successful campaigns already terminate. This is a route diagnostic, not an ownership result or a claim of collision-free driving. The longer-forecast frozen intervention is reported separately.

The 125 registered rows remain the experiment matrix. Main initializations are 201–205, calibration uses 101/102, and execution checks use 9091/9092. Development scenarios 1–10, training 11–10010, validation 1000001–1000064 and final scenarios 2000001–2000256 are disjoint. Test-bank evaluation begins after the final source/configuration freeze. All training and evaluation are headless; videos are reconstructed afterward from saved physical trajectories and decisions.
