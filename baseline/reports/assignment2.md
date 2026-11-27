# Assignment 2: Hybrid Distributional SAC for Gambler’s Taxi

Iuliia Kurnaeva  

## 1. Experimental setup and reinforcement-learning model

### Task and experiment design

The first experiment asked whether one learned policy could drive, complete shifts, and buy out the taxi within 545 ticks. We then introduced late starts to test decisions under a time constraint. On late starts, only two shifts remain and their wages are insufficient under that version's rules. Subsequent versions expanded the campaign to eight days and introduced daily expenses. A separate bank transaction then made repayment a prerequisite for buyout. The current ten-day campaign starts with $1,000 of debt and values earlier repayment. These experiments have separate results because the available routes and the meaning of success changed with the rules. [Environment and reward](assignment1.md)

Each policy observes the vehicle, its surroundings, financial state, time, and an action mask. It chooses continuous steering and acceleration when driving and a categorical operation when interacting. The critic estimates discounted return from rewards for success, collision penalties, and other terms in the respective task version. Scripted work and casino policies provide feasibility checks and reference points.

### Hybrid DSAC

The actor has two 128-unit hidden layers. Its continuous head outputs parameters of a diagonal Gaussian for gas and steering, and the categorical head scores the legal operations. The `tanh` transform bounds the controls. Its Jacobian enters the log density. The continuous entropy term applies to driving. The categorical branch uses a masked softmax, and the update sums over legal operations. Separate learned temperatures regulate the two exploration terms. [Hybrid SAC](https://arxiv.org/abs/1912.11077) and [SAC with automatic temperature adjustment](https://arxiv.org/abs/1812.05905)

Two critics estimate 24 return quantiles per state and action. In the target, the critic with the lower mean supplies its full distribution for each next action. The categorical policy weights those distributions, the entropy correction adjusts the target, and quantile Huber regression updates the critics. Bootstrapping stops at terminal transitions. Target weights track the learned critics through a moving average. This design combines hybrid action selection with risk-sensitive valuation. [DSAC](https://arxiv.org/abs/2004.14547) and [Implementation](../src/algorithms/hybrid_dsac.py)

The neutral actor uses the mean return. The CVaR variant assigns 35% of its action score to the mean of the bottom 20% return quantiles and 65% to the overall mean. The later upside variant assigns 55% to the top 20%. The evaluation also reports the mean final net worth of the worst 20% of episodes. This is a financial outcome measured from rollouts, separate from the critic's soft return.

### Training and evaluation

Initial imitation helps the actor see successful routes before they are common in its own experience. The scenario driver subsequently labels states visited by the learned actor, supplying corrective actions after deviations. Those labels contribute to the imitation loss. The critic's replay mini-batches combine approximately three quarters online experience with one quarter recorded scenario transitions. The start curriculum gradually removes starts near intermediate goals. [Training code](../src/train.py)

The reported comparisons use 48 validation starts to select the checkpoint and 128 separate test starts to evaluate it. The later debt campaign selects by buyouts, debt clearance, time to clearance, crashes, and net worth in that order. Test reports include buyout, crashes, and timeouts. Later experiments also record bank repayments, games, and final debt. Scenario comparisons within the same setup use the same seed set. Most historical model comparisons use one training seed. The extended debt study adds fresh training seeds and held-out starts.

## 2. Computational optimizations

Training and evaluation run headlessly in JAX. On the local RTX 3080 Ti Laptop GPU under Ubuntu WSL2, `vmap` batches 64 independent environments, `lax.scan` collects 12 steps per call and loops through updates, and JIT compiles those loops with fixed array shapes. The replay buffer stays on the device. Terminal transitions enter replay before reset. Precomputed route-distance fields avoid running a path search at every step. Pygame and video encoding run separately from learning.

A standalone benchmark with 4,480 transitions measured 478 steps/s for individual submission and 1,350 steps/s with batches of 64. Two mixed-start training logs report approximately 66.7k and 69.8k environment transitions/s after compilation. Those rates describe different loops and amounts of work. [Step benchmark](../artifacts/benchmark.txt), [Neutral training log](../artifacts/mixed_neutral_training.jsonl) and [CVaR training log](../artifacts/mixed_risk_training.jsonl)

## 3. Difficulties, resolutions, and results

| Difficulty | Change | What the result showed |
|---|---|---|
| Successful episodes were rare in initial online collection. | Added imitation, corrective labels, and a curriculum that approaches normal starts. | On early validation starts, the neutral actor improved from 68.75% after imitation to 100% at the selected Hybrid DSAC checkpoint. |
| Small driving errors sent the actor into states it had rarely seen. | Added labels for visited states and transferred an early-start actor to the mixed-start task. | Overall buyouts increased. Learning a profitable response to late starts remained difficult. |
| A short horizon rewarded a few favorable gambling outcomes. | Introduced daily expenses, a longer horizon, and tighter casino scheduling. | Learned neutral and CVaR policies in the eight-day test chose work and rest. |
| The original buyout could settle an outstanding debt automatically. | Introduced a required visit to the bank and added demonstrations of casino-to-bank-to-garage routes. | A scripted driver completed the entire chain. The selected learned actors still failed to do so on the test set. |
| The initial $1,000 campaign combined navigation and financial failures. | Tested a driving guard on the same Hybrid DSAC actors and learned destination choice with a shared driver in the extended study. | The guard sharply reduced crashes while the old actors still reached zero repayment opportunities. Destination policies learned to clear the debt and buy out. |

### Historical comparisons

| Configuration | Neutral Hybrid DSAC | CVaR Hybrid DSAC | Interpretation |
|---|---:|---:|---|
| Early starts | 125/128 buyouts, 0 crashes | 124/128 buyouts, 1 crash | Both worked and slept without wagering. |
| Early and late starts, initial | 26/128 buyouts, 42 crashes | 23/128 buyouts, 25 crashes | Both had 0/43 late buyouts. CVaR's worst-20% net worth was $39.80 versus $8.91. |
| Early and late starts, after transfer | 77/128 buyouts, 23 crashes | 81/128 buyouts, 37 crashes | Both achieved 1/43 late buyouts. Two scripted alternatives got 7/43 and 8/43 on late starts. |
| Eight-day economy | 87/128 buyouts, 20 crashes | 103/128 buyouts, 9 crashes | Both avoided gambling. The scripted work policy achieved 128/128 buyouts. |

On the mixed-start transfer test, the neutral policy averaged 3.37 slot wagers and 1.12 loans on late starts. CVaR averaged 1.23 wagers and 0.05 loans. One neutral replay bought out after borrowing and placing three slot bets under the historical settlement rules. When the physical effects of emotion were disabled for the eight-day CVaR actor, the result changed from 103 to 104 buyouts and from 9 to 10 crashes. The comparison evaluates a fixed actor trained with one seed. [Historical comparisons](../artifacts/evaluation.json), [Eight-day neutral](../artifacts/hard_world_evaluation.json) and [Eight-day CVaR](../artifacts/hard_world_risk_evaluation.json)

### Bank, habit, and the debt campaign

Two subsequent models with mandatory repayment achieved 76 and 74 buyouts out of 128, with zero buyouts after repayment on the test set. The second model saw additional repayment demonstrations. Intermediate checkpoints had individual successes after repayment on validation. The scripted bank-and-casino control achieved 12 test buyouts after repayment in that version. [First bank experiment](../artifacts/bank_rule_evaluation.json) and [Additional repayment demonstrations](../artifacts/bank_settlement_evaluation.json)

In the paired habit experiment, turning the mechanic on yielded two forced wagers for the evaluated actor and 20 for the bank-and-casino script across 256 trials per condition. Actor buyouts changed from 167 to 166 with 47 crashes in both conditions. A separate habit-enabled retraining recorded 74/128 buyouts and one forced bet in batch evaluation. The saved video of an override uses the scenario trace whose event was verified in playback. [Paired evaluation](../artifacts/dependence.json) and [Habit retraining](../artifacts/dependence_training_evaluation.json)

At $1,000 opening debt, the neutral and upside Hybrid DSAC actors each used a budget of 499,200 online transitions. Both got 0/128 buyouts. The upside actor averaged 0.56 blackjack hands per episode versus 0.09 for the neutral actor in that comparison. Its failure to clear debt was confirmed on two additional training seeds. In the extension, a six-tick driving guard reduced crashes from 128 to 27 and 28 for the two original actors on fresh starts. Neither reached a legal repayment. With a shared driver, six PPO policies trained with work demonstrations bought out in 75–104 out of 128 episodes each. Three policies trained with mixed demonstrations reached 5–29. The PPO result concerns learned destination choice under the shared controller. [Historical debt evaluation](../artifacts/debt_upside_evaluation.json) and [Research report](debt_research.md)

## 4. Videos, model checkpoints, and local reproduction

| Result | Video | Model checkpoint | Numeric log |
|---|---|---|---|
| Primary submission result. An end-to-end Hybrid DSAC buys out the taxi at tick 386. | [Assignment 2](../artifacts/assignment2.mp4) | [Early neutral](../artifacts/early_neutral.npz) | [Seed 20000](../artifacts/policy_episode.npz) |
| Eight-day campaign. An end-to-end Hybrid DSAC completes seven shifts and buys out. | [Eight-day buyout](../artifacts/hard_world_policy.mp4) | [Eight-day neutral](../artifacts/hard_world_actor.npz) | [Seed 20000](../artifacts/hard_world_policy_episode.npz) |
| $1,000 campaign. An end-to-end Hybrid DSAC wagers and crashes. | [Debt campaign failure](../artifacts/debt_upside.mp4) | [Upside actor](../artifacts/debt_upside_actor.npz) | [Seed 20000](../artifacts/debt_upside_episode.npz) |
| $1,000 campaign. The destination PPO with the shared driver buys out after a voluntary repeat wager. | [Risky buyout](../artifacts/debt_research/cases/seed_91038.mp4) | [Mixed PPO](../artifacts/debt_research/models/mixed_demos_urgency_1_seed_29/selected.npz) | [Seed 91038](../artifacts/debt_research/cases/seed_91038_render.npz) |

Install the [README dependencies](../readme.md), then run the commands below from the repository root. Rendering a saved log reproduces the trajectory used in the submitted clip. Recording from a checkpoint runs a fresh inference episode using its stored configuration. The loop and backend can change exact trajectories near collision boundaries.

```bash
python src/replay.py render --log artifacts/policy_episode.npz --video runs/replay/assignment2.mp4
python src/replay.py record --checkpoint artifacts/early_neutral.npz --seed 20000 --log runs/replay/early_actor.npz
python src/replay.py render --log runs/replay/early_actor.npz --video runs/replay/early_actor.mp4
python src/replay.py render --log artifacts/debt_research/cases/seed_91038_render.npz --video runs/replay/debt-win.mp4
```

The snapshots contain policy weights and configuration for inference. The README gives commands to train and evaluate the current campaign. The older economic versions have saved checkpoints, logs, and aggregate results. Exact fresh training on an older version also requires its corresponding source revision. [Training and artifact index](../readme.md)
