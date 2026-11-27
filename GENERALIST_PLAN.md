# Four learned agents under one training distribution

## Training

Train PPO MLP, PPO GRU, Hybrid DSAC MLP and Hybrid DSAC GRU sequentially, with fresh initialization seed 301 and 4,194,304 active physical transitions each. Every architecture receives the same ten-day training task, scenario pool 11–10010, public observations, action interface, external objective and curriculum. Episode lengths determine how many resets each run sees. The design uses one initialization per architecture.

The four equal experience stages use car principal $50/$300/$700/$1000, home-start probabilities 25%/50%/75%/100%, and baseline hunger multipliers 0.65/0.8/1/1. The first stage has fixed markets; subsequent stages use dynamic quotes. Non-home starts balance work, food, fuel, bank and casino locations. Food starts have low satiety; fuel starts have little fuel; casino starts are at opening. All actions remain policy choices. The curriculum changes the initial state distribution. The last quarter uses the complete normal home-start task. Physical discount remains 0.999. Replay resets at genuine curriculum boundaries; learned weights persist.

There are no economic demonstrations, forced casino visits or task-specific action labels. A single choice of destination executes for up to 128 physical ticks, ending on service completion, game feedback, a terminal state, or arrival at an unavailable service. Wait is at most eight ticks. Actual physical costs, motion and traffic remain active. Returns and continuation discount use actual active duration. Arrival at a closed or unaffordable service returns control rather than repeatedly circling.

The generic food-execution regression test covers 60 development cases. Before the interface repair, none completed a meal in one intention; after repair, all 60 did. This checks mechanics and does not teach the policy any action.

## Readiness before final tests

After each training run, replay and audit the selected policy on the 64 validation scenarios. Save actual completed work, sleep, meals, refills, wagers, loans, payments and ownership counts. These show observed behavior, not verbal knowledge. A policy that never plays cannot be claimed to exhibit gambling or loss chasing. Casino avoidance can nevertheless be consistent with its objective; do not force a bet to make a video interesting.

Before consuming the new final banks, require every architecture to show at least one validation ownership and completed work, meals, sleep and refilling. These are necessary readiness checks, not a sufficient claim of convergence. Record missing behaviors and stop final testing if readiness fails. Do not silently call a completed training budget a successful learner.

## Frozen comparisons

Reuse each selected policy with no updates across the scenario interventions in the existing experiment matrix: deadlines, control effectiveness/noise, income, fuel prices, initial cash/food/fuel, personal debt, meal distributions and service identity. Use 256 common episodes per condition. The new standard bank is 3000001–3000256; adverse banks begin at 3100001 in disjoint blocks of 100000, each with 256 episodes and matched controls. Previously exposed test banks remain historical evidence.

Training uses a ten-day task. Five-day and continuing cases measure horizon transfer, not separate horizon-optimal training. Frozen scenario interventions measure robustness; they are not retraining ablations. No-curriculum, exploration-bonus and other training-method ablations require separately budgeted trainings and are deferred until these four agents are usable.

## Artifacts and execution

- `experiments/generalist_study.py`: sequential training and evaluation driver.
- `artifacts/generalist_study/freeze.json`: immutable source/design identity.
- `artifacts/generalist_study/status.json`: current model and phase.
- `artifacts/generalist_study/training/<kind>/`: selected/final/initial policies, full resume checkpoints, source snapshots, learning history and physical trajectories.
- `artifacts/generalist_study/readiness/<kind>/`: validation trajectories and exact replay audits.
- `reports/generalist_study/readiness.json`: actual skill and ownership counts.
- `artifacts/generalist_study/evaluation/<kind>/<case>/`: new final evaluations, only after readiness.
- `experiments/archive_interrupted_study/`: original source, design and registry for the previous run; its large artifacts remain in their original locations.

Run on the local CUDA environment with `PYTHONPATH=src python experiments/generalist_study.py`.
