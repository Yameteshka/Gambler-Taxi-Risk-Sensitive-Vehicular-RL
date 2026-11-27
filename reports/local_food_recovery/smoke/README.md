# Local training smoke results

| Test | Result | Main evidence |
|---|---|---|
| 1. Observation history | Pass | A complete random episode matched the current-frame, previous-frame and action representation exactly. The reset action is zero for the stacked input. The labeled batch included a meal and refuel. The MLP fit 32 of 32 actions in 200 steps. |
| 4. Standard resets and evaluation | Pass | 28 of 100 resets were standard home starts. The actual collector ran a 512-decision batch. Two 64-episode GRU evaluations scored 59/64 with bitwise identical endpoints and returns. |
| 2. DSAC, original input | Fail | 5,000 gradient updates. Final entropy was 0.79 at the station and 1.61 at the bank. The standard campaign completed one refill per episode and no meals. All 16 episodes starved with fuel remaining. |
| 2. DSAC, stacked follow-up | Fail | 16/16 in the ownership-only lesson and 0/16 in the standard campaign. All 16 standard episodes starved. Entropy was 0.051 at the station and 1.144 at the bank. Median maximum probabilities were 0.992 and 0.644. Discrete temperature fell from 0.05 to 0.02076. |
| 3. PPO MLP | Fail | 0/16 before and after 16,384 decisions. Maximum explained variance at 8k–10k was 0.401. Maximum estimated KL was 0.0492. The guard stopped 29 of 32 rollouts. |
| 3. PPO GRU | Fail | 0/16 before and after 16,384 decisions. Maximum explained variance at 8k–10k was 0.521. Maximum estimated KL was 0.0295. The guard stopped 26 of 32 rollouts. |

The short PPO runs start from fresh weights directly in the complete campaign. They test the specified stress condition. They do not measure performance after a full progressive curriculum. The DSAC trial on the old inputs used a fixed discrete temperature of 0.05. Its observed entropy cannot be attributed to an automatic target entropy. Satiety was present in the original input.

The stacked DSAC follow-up uses an adapted warm checkpoint, 370 input features and a fixed replay sample from four real-game contexts. It has a discrete target entropy of 0.2 times the log of the legal action count. Discrete temperature adapts while the continuous temperature stays fixed. It uses 2,500 forward-KL updates followed by 2,500 reverse-KL updates, with the same fixed replay. These choices make it a combined diagnostic. They do not isolate the effect of stacking from temperature.

The JSON files contain measurements and, for PPO, minibatch update counts and explained variance by rollout. Reproduce the first round with `python -B experiments/smoke_training.py all`. Reproduce the stacked follow-up with `python -B experiments/smoke_training.py dsac_stacked`. Both use development scenarios only.
