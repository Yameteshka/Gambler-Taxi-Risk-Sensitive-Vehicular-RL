# Food planning recovery: implementation and diagnostic status

## New evidence changes the diagnosis

The frozen-checkpoint experiment in `food_retention/results.json` evaluated eleven checkpoints (176 actor-only development episodes) on exactly the original food-approach context. Both PPO variants scored 16/16 at each of four checkpoints: after the food lesson, after the personal-debt lesson, after the $350 lesson, and at the final failed checkpoint. They still ate and completed ownership in that context. The food-approach skill persisted through the last checkpoint. The observed performance gap is in the full campaign.

DSAC MLP scored 0/16 with no meals both before and after the failed food lesson. DSAC GRU's selected reference scored 13/16.

## Recovery curriculum and gates

- A separate `food_budget_retention` curriculum preserves the original curriculum. Eighteen lessons add food planning from work, bank and home, randomized starting liquidity ($60-$120) and satiety, plus a $425 bridge between $350 and $500.
- Rehearsal probability is 35%; half of rehearsal draws preferentially sample introduced food contexts. The final standard-game lesson uses the original home reset. Standard starting money remains $100.
- A current lesson must produce at least 12 ownership completions per 16 scenarios on each of two development banks. Food lessons additionally need at least 12 successful episodes that actually completed a meal. This prevents aggregate meals on unsuccessful episodes from certifying mastery.
- Earlier earning, food, fuel and personal-debt contexts are checked on both banks before advancement. Selected-policy full-campaign readiness additionally requires 48/64 ownership. These are development acceptance thresholds, not claims of statistical optimality or final-test performance.
- An opt-in unavailable-service feedback duration supports eight physical ticks instead of one. Historical behavior remains the default. Time, hunger and discounted reward still accrue for every active physical tick. Choices remain voluntary, including choosing an unavailable service.
- The separate development launcher `experiments/budget_recovery.py` defaults to registration only. Its default treatment changes the curriculum while keeping historical entropy and one-tick feedback. `--wait-ticks 8` and `--alpha-d` register distinct treatments; they are not silently combined. No final banks are called by this launcher.

## Verification completed

`recovery_checks/checks.json` records successful checks of all eighteen JIT-compiled, vectorized reset stages, accounting conservation, finite 194-feature observations, randomized remote-start distributions, strict gate rejection cases, and real one/eight-tick refused-work execution with hunger and reward reconciliation. Source hashes are recorded; the launcher refuses stale engineering checks. Python syntax checks passed for the changed modules.

The checks cover the implementation. The next training comparison measures food-context acquisition and meal planning in the full campaign while varying curriculum and DSAC temperature separately.

## Reproduction

Use the project JAX environment with `XLA_PYTHON_CLIENT_PREALLOCATE=false`:

```text
python -B experiments/check_food_retention.py
python -B experiments/check_budget_recovery.py
python -B experiments/budget_recovery.py
```

The last command records the plan. Add `--train` to run the development recovery. The registered baseline plan is `reports/food_budget_wait1_alpha3e-06/plan.json`. Its decision ceiling remains 262,144 per model, so increased training duration is not confounded with the initial curriculum intervention. One initialization remains exploratory.
