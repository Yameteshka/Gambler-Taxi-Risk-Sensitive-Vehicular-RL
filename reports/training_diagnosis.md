# Training failure diagnosis

## Training differences

The earlier successful destination learner did not learn the economic task entirely from scratch. `baseline/src/debt_learning.py` runs 1200 supervised minibatch updates before PPO, and retains an imitation loss weighted by 0.04 during PPO. Its shared controller executes nine categorical destination choices, and automatically handles blackjack card decisions. Its feature network has two 96-unit layers. The default 60 PPO iterations collect 64 environments x 24 decisions x 12 physical ticks x 60 = 1,105,920 scheduled physical steps, including terminal padding, plus demonstration collection and supervised updates.

The current run budget is 4,194,304 active physical steps per model, excluding padding. There are 125 registered runs, versus nine earlier destination-policy comparisons in the cited research table. Current actors have 26 intentions, explicit card decisions, Gaussian driving styles and 128-unit networks; Hybrid DSAC also updates twin quantile critics and entropy temperatures. Current PPO and DSAC are not computationally equivalent. The measured completed-run median training times were approximately 10.3 minutes for PPO MLP, 11.6 minutes for PPO GRU and 32.2 minutes for DSAC MLP under concurrent execution. These are observed timings, not standalone hardware benchmarks.

The environment now requires food and fuel management. The older model had no satiety-depletion death condition. In the current environment, initial satiety 0.85 and baseline consumption 0.0018 per physical step imply fewer than 473 ticks without food, even before extra hunger from work and movement. Many saved standard-test endpoint counters show roughly two or three paid shifts and zero delivered meals, refills or sleep. The resulting deaths occur before the multi-day contract can be completed. Other policies repeat borrowing and repayment. 

At the inspection cutoff, 18 models had completed training and standard tests, with all 18 standard trajectory replays passing. Their combined 4608 standard episodes contained zero ownership completions, 4155 starvation deaths and 453 collisions. Sixteen runs had completed their entire assigned evaluation suite. Execution checks establish reproducibility and specified mechanics, not successful learning. Launching the full replication matrix before demonstrating reliable learning of the complete resource-management chain was premature.

## Recovery requirements

Preserve the frozen unsuccessful study as a separate result. Do not overwrite checkpoints or tune against its final test bank while calling that bank untouched. First demonstrate a feasible complete work/food/fuel/sleep/payment route on development conditions. Then compare a clearly labeled demonstration-initialized learner against the existing from-scratch learner under the same new environment. Demonstrations must cover survival and contract completion; an economic demonstration is strategy supervision and must not be described as autonomous discovery. Casino-directed examples must not silently determine the claimed voluntary risk result. Start with one PPO feasibility run and inspect complete trajectories before repeating a large matrix. A source/design change requires a separated protocol revision and evaluation provenance.

## Inspectable sources

- [Earlier destination training](../baseline/src/debt_learning.py)
- [Earlier nine-choice controller](../baseline/src/debt_strategy.py)
- [Earlier reported destination-policy results](../baseline/reports/debt_research.md)
- [Current environment and resource rates](../src/env/resource_taxi.py)
- [Current service interface](../src/strategy.py)
- [Current training implementation](../src/strategy_training.py)
- [Preserved registered results](results.md)

The earlier report also contains unsuccessful end-to-end agents. Its successful destination policies achieved 75–104 completions per 128 work-demonstration episodes and 5–29 per 128 mixed-demonstration episodes; it did not establish that every historical method solved every scenario.

## Food interaction diagnosis

A subsequent development-only execution test covered 60 cases: Home/Work to each of three food services on scenario IDs 1-10. No case delivered a meal within one 32-tick intention. Keeping the same requested destination across successive intentions delivered a meal in all 60 cases. These were unit-test commands, never demonstration data or weight updates. In a saved PPO MLP repetition D, 73 food intentions produced no delivered meals; all 64 observed next decisions after a full 32-tick food intention switched destination. The short decision window interrupted food trips before purchase.

[Food execution tests](../experiments/food_execution_diagnosis.json) and [Recorded policy choices](../experiments/food_policy_diagnosis.json)
