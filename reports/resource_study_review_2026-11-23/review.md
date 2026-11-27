# Ownership study review: one qualified policy, three failed learners

## Selected checkpoints on standard development scenarios

| Model | Selected ownership | Starvation | Passed lessons | Actual decisions | Training completed | Readiness |
|---|---:|---:|---:|---:|---|---|
| dsac_mlp | 0/64 | 64 | 3/14 | 79,872 | No | Failed |
| ppo_mlp | 8/64 | 56 | 12/14 | 151,552 | No | Failed |
| dsac_gru | 60/64 | 4 | 14/14 | 262,144 | Yes | Passed |
| ppo_gru | 3/64 | 61 | 12/14 | 139,264 | No | Failed |

The development bank contains 64 scenarios. Each architecture used one initialization and a ceiling of 262,144 decisions.

## What DSAC GRU demonstrated

The selected checkpoint is listed in `policy_registry.json`. Replay matched the saved policy and controller trajectories with zero logged float error.

| Retained development context | Ownership |
|---|---:|
| earning | 16/16 |
| food_approach | 13/16 |
| fuel | 16/16 |
| personal_debt | 16/16 |

Across 64 standard episodes, DSAC GRU completed 223 paid shifts, 102 meals, 108 refills and 375 car payments. It opened one loan and completed zero wagers and zero sleeps.

## Failure patterns and interpretation

- **dsac_mlp** stopped at `meal_then_ownership`. Its last lesson check had 0/16 ownership and 16 starvation outcomes. In the selected standard validation, 64 episodes starved without a completed meal, 58 starved with at least $40 remaining, and 0 had a prolonged stopped-DRIVE flag.
- **ppo_mlp** stopped at `home_debt_500`. Its last lesson check had 1/16 ownership and 15 starvation outcomes. In the selected standard validation, 56 episodes starved without a completed meal, 7 starved with at least $40 remaining, and 0 had a prolonged stopped-DRIVE flag.
- **ppo_gru** stopped at `home_debt_500`. Its last lesson check had 1/16 ownership and 15 starvation outcomes. In the selected standard validation, 60 episodes starved without a completed meal, 22 starved with at least $40 remaining, and 0 had a prolonged stopped-DRIVE flag.

Food-context success and full-campaign food use differed. Some drivers still had cash at starvation. The stopped-DRIVE flag measures prolonged low-speed driving requests.

## Data and reporting integrity

This review ran 214 recorded checks; 0 failed. Checks include checkpoint finiteness, source checksums, decision accounting, sequential curriculum advancement, unique scenario IDs, endpoint/report reconciliation, zero debts at ownership, nonnegative balances, microloan totals and saved replay hashes.

Two original reporting issues matter: the auto-report reads obsolete decision/update field names, and failed readiness entries use a zero-ownership placeholder. Neither should be treated as a measured performance result. `training_summary.csv` uses the actual raw fields and `review.json` retains the recorded checks.

## Sources and reproduction

- [Study status](../study/data/cohort/status.json) and [design](../study/data/cohort/freeze.json).
- [Training and gate records](../study/data/cohort/) and [checkpoint metadata](../study/checkpoints/manifest.json).
- [Selected episodes and audit results](review.json).
- [Local reproduction](../study/reproduce.md).
