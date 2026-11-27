# Gradual curriculum pilot

## Rationale

The historical transition at 25% changed principal, exploration, start distribution, hunger and market together; DSAC replay was also cleared. The completed DSAC MLP had no paid shifts after that transition. This motivates separating changes, but does not identify a single proven cause. PPO ownership confirms that the full world is feasible.

## Sequence

The exact schedule is executable in [src/progressive_curriculum.py](src/progressive_curriculum.py).

1. Balanced service starts, $50 debt, fixed market, hunger rate 0.65 of reference and exploratory action mixture 0.8.
2. Lower the mixture to 0.5, then 0.2, in separate lessons. Keep the world unchanged.
3. Introduce mixed starts with 25% home probability, keeping all economic parameters unchanged.
4. Raise principal through $100, $150, $200, $300, $400 and $500. Keep the exploration mixture at 0.2, market fixed and hunger unchanged throughout these six lessons.
5. Raise home-start probability to 50%, 75% and 100% in separate lessons.
6. Raise hunger rate to 0.8, then 1.0 of reference.
7. Enable the dynamic market.
8. Lower exploration to 0.1, 0.05 and zero in separate lessons.

There are 19 lessons. Adjacent lessons change exactly one declared factor; the reset sampler and its home probability are one start-distribution factor. Start cash stays $100. Action availability, the reward, food distributions, loan mechanics and shared driving remain unchanged. Economic decisions are always sampled; there are no economic action labels or casino-specific bonuses.

Replay is retained when only the behavior exploration mixture changes. For other curriculum changes it is cleared conservatively; weights and optimizer state persist. This separates the debt increases from exploration removal and avoids treating experience from changed environments as automatically compatible. More lessons incur extra compilation and validation costs; no wall-time speedup is assumed.

## Advancement checks

After at least 131,072 active physical training steps in a lesson, check the learned actor every 65,536 steps. Checks use the first 16 existing validation IDs, 1000001–1000016, with additional exploration disabled. The curriculum uses these development scores for advancement. Actor sampling remains stochastic, with the evaluator's fixed policy RNG.

- At least four ownership episodes out of 16 are required to advance.
- Once principal exceeds the initial $100, at least two successful episodes must themselves include completed paid work. Work observed only in losing episodes does not satisfy this condition.
- The first lesson additionally requires actor-completed work, meals, sleep and refueling, plus broad actual-event exposure during training, including games, borrowing, repayment and loan closures.
- Advancement begins with garage starts that have no remaining debt. Later gates test earning and settlement.
- If a lesson fails for 524,288 active physical steps, save its checkpoint and diagnosis and stop. Do not silently move to a harder world or extend the budget.
- If the total budget expires before every lesson gate passes, record failure and do not mark the run complete.

Training events, actor-only gate outcomes and final-world validation remain separate. All phases use the existing training IDs 11–10010; final test banks are untouched. Adaptive lesson durations can differ across architectures, so this compares the same advancement rule and ordered lessons under an equal total budget, rather than an identical realized number of steps per lesson. Report the actual exposures before making architecture comparisons.

## Run and artifacts

From the project directory in the existing JAX environment:

```bash
python experiments/check_progressive_curriculum.py
XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/check_progressive_collection.py
python experiments/progressive_pilot.py --model dsac_mlp
```

The pilot defaults to 4,194,304 active physical steps and initialization seed 301. `--stop-after` saves a resumable checkpoint; use the same command with `--resume` to continue without changing sources or treatment. Results go only to `artifacts/progressive_curriculum_pilot/training/<model>/`:

- `curriculum_gates.jsonl`: actor-only checks and advancement decisions.
- `curriculum_validation/`: physical trajectories for those checks.
- `lesson_<index>_passed.npz`: successful lesson checkpoints.
- `cold_start_coverage.json`: counts of training actions and service outcomes.
- `curriculum_failure.json`: a lesson or total-budget failure.
- Existing training traces, final-world validation, selection and resume artifacts.

## Pilot results

The learner reached the lesson-0 cap at 614,296 physical steps. Last easy-context check: 14/16 ownership but no work, meals, sleep or refueling. Last full $500 validation: 0/64 ownership, 39 starvation deaths and 25 collisions. No later lesson was reached, so this run cannot establish whether gradual debt increases help. See the [diagnosis and evidence](reports/progressive_curriculum_pilot/report.md); this candidate has not fixed the learning problem.
