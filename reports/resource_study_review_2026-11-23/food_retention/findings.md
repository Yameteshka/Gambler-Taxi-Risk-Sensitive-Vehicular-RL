# Matched food-skill retention check

| Model | Saved checkpoint | Ownership / 16 | Mean completed meals |
|---|---|---:|---:|
| ppo_mlp | lesson_06_passed | 16 | 1.25 |
| ppo_mlp | lesson_09_passed | 16 | 1 |
| ppo_mlp | lesson_11_passed | 16 | 1.062 |
| ppo_mlp | last | 16 | 1 |
| ppo_gru | lesson_06_passed | 16 | 1 |
| ppo_gru | lesson_09_passed | 16 | 1 |
| ppo_gru | lesson_11_passed | 16 | 1 |
| ppo_gru | last | 16 | 1 |
| dsac_mlp | lesson_02_passed | 0 | 0 |
| dsac_mlp | last | 0 | 0 |
| dsac_gru | selected | 13 | 1 |

**Both PPO actors retained the original food-approach skill through their final failed checkpoint: every tested PPO checkpoint achieved 16/16 ownership with meals in this context.** The same actors rarely ate in the full campaign. The result isolates a gap between the food-approach context and campaign-level planning.

DSAC MLP achieved 0/16 before and after its failed food lesson, without a completed meal. Its problem includes initial acquisition of this skill, unlike PPO. DSAC GRU reference achieved 13/16.

The same 16 development scenario IDs and food-approach reset were used for all checkpoints. Reusing development IDs controls this diagnostic; it does not supply independent final-test evidence. These checkpoint samples locate persistence of a narrow skill, not the exact update where full-game planning failed.

Evidence: results.json, per-checkpoint numeric endpoints. Reproduction: experiments/check_food_retention.py. Original full-campaign results remain in the parent review.
