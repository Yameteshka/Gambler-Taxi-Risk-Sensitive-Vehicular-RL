# Tuned training cohort

The three new policy configurations use seed 301, training scenario IDs 11–10010 and a ceiling of 262,144 decisions. From curriculum stage 5 onward, each reset independently has a 25% chance of using the complete standard home state with $100 cash and $500 car debt. The final stage uses the standard home reset. The first 13 curriculum stages must qualify before the last 102,400 decisions reserved for the final five stages. Intermediate checks already used a 12/16 threshold per bank.

| Policy | Training change |
|---|---|
| DSAC MLP | Forward-KL in stages 1–12, reverse-KL from stage 13. Retain the critic, replay and optimizer state at the switch. The two-frame MLP observation includes the previous action. |
| PPO MLP | Two-frame observation and previous action. Up to 12 epochs, clip range 0.15 and minibatch KL early stopping at 0.02. |
| PPO GRU | The same PPO policy update with an explicit resource observation and a threefold step size for the value-only output weights. Shared layers use the common optimizer. |
| DSAC GRU | The historically trained, hash-verified reference, with its original observation inputs. |

PPO scales value residuals by the current rollout return standard deviation with a floor of 0.25. It retains the critic prediction in return units. This is a normalized value loss, without PopArt or a change to the reward. Its exploration prior coefficient starts at 0.002 and reaches zero at stage 14. The minibatch KL threshold stops subsequent updates in the rollout after the threshold is observed. It does not impose a hard bound on the last applied update.

The enriched observations expose a fuel range estimate and its margin above road distance to the station. The estimate uses fuel divided by consumption per unit distance. Travel and time spent on a shift can consume fuel, so this is not a prediction of arrival. Car and personal debt totals, time-of-day and the horizon already appeared separately in the observation. The enriched inputs add a critical-satiety indicator and an explicit time-to-deadline value. Equal frame widths are checked with the real environment.

Readiness and headline scenario evaluation use the highest-probability discrete choice and mean continuous control. The original deterministic ablation becomes a separately labeled sampling sensitivity condition, preserving 30 distinct conditions. Only policies that satisfy the readiness checks enter the final banks.

## What the earlier evidence supports

The paired inference check on 16 fixed development scenarios gave sample/argmax ownership counts of 0/2 for PPO MLP, 1/0 for PPO GRU, 15/12 for the frozen DSAC GRU, and 0/0 for the forward-KL DSAC MLP pilot. The variable response supports keeping the sampling sensitivity result visible. On all 64 development scenarios the frozen DSAC GRU achieved 59 completions under argmax, above the 48/64 gate. Local preflight also verified frame stacking in the real environment and a real rollout and optimizer update for each PPO architecture. See `tuned_preflight.json` for the measured checks.

On the earlier forward-KL pilot, 13 of 16 episodes exhausted the tank. The terminal outcomes were 15 starvation events and one collision. The pilot also completed 93 meals. Food frequency by itself does not identify a causal effect of KL direction or prove diffuse logits. The new stage-based switch is a treatment, and several other pipeline changes are bundled with it. An improvement would support the package as a whole. [Chan et al., JMLR](https://www.jmlr.org/beta/papers/v23/21-054.html)

The stage-based switch adapts a research idea to a hybrid actor. The cited Bidirectional SAC paper studies Gaussian policies and continuous-control benchmarks. The hybrid setting uses the schedule as a separate empirical treatment. [Zhang et al., arXiv](https://arxiv.org/abs/2506.01639)
