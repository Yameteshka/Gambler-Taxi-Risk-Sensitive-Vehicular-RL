# Training pipeline review

The DSAC MLP forward-KL recovery used 16,384 decisions and 628,781 physical ticks. It continued one saved actor, critic and optimizer state with fresh replay. Evaluation used the same 16 development scenarios before and after the change, with no external exploration.

| Context | Ownership before | Ownership after |
|---|---:|---:|
| At the restaurant with low satiety | 0/16 | 16/16 |
| Food approach | 0/16 | 11/16 |
| Food planning from home | 0/16 | 11/16 |
| Standard $500 campaign | 0/16 | 0/16 |

The final campaign completed 93 meals and no refills. The replay matched the original aggregate outcomes. Thirteen agents ended with zero fuel. Fifteen starved and one crashed. Five cleared the car loan and none had both debts cleared. The mean amount borrowed was $407.81. These traces show acquisition of food behavior and a subsequent budgeting problem involving both fuel and loans. They do not measure performance of from-scratch forward-KL training.

The recovery trial used food phases and the full campaign. It skipped the explicit fuel and personal-debt lessons, which limits conclusions about the full curriculum. The fixed curriculum now requires a completed refuel in at least 12 successful episodes per 16-episode fuel bank. The same check applies when revisiting that skill later in training.

The previous selection path could qualify one snapshot and then export a different snapshot that ranked higher on mean return. The recovery curriculum now writes the actual policy that passes its final actor and retained-skill gates and wins at least 48 of 64 standard development episodes. Reward-ranked snapshots remain available as diagnostic checkpoints. The trainer also handles short decision budgets when computing validation cadence.

PPO analysis is in [PPO_DIAGNOSIS.md](PPO_DIAGNOSIS.md). The optional native exploration mode samples directly from the policy whose likelihood the PPO update uses. During exploratory curriculum stages the loss includes a uniform-to-policy cross entropy over legal actions. The term is off when the curriculum exploration parameter is zero. The initial coefficient is 0.002 at exploration strength 0.2. This changes the training objective for the PPO variant and needs a learning comparison. It does not change action availability or add environment rewards.

Engineering verification passed for all 18 reset stages, the refuel rejection gate and the refused-service timing checks. A separate GPU check collected 512 real transitions with PPO MLP, reproduced the saved behavior log probabilities from the native policy within 0.00002, and completed an optimizer step with finite loss and changed parameters. The updated four-policy recovery has yet to demonstrate campaign performance. The final scenario banks are reserved.
