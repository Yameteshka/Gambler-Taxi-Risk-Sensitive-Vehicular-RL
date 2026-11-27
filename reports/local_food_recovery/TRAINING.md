# Training and comparison

The cohort has three fresh training runs and one frozen reference. Fresh runs use the same seed (301), training scenario IDs (11–10010), 18-stage curriculum, 10-day standard campaign with $500 car debt, and a ceiling of 262,144 policy decisions per run. Each stage checks two 16-episode development banks. It also checks previously acquired skills on both banks before advancing. Food and fuel stages count the corresponding completed service in successful episodes. The exported policy must pass the final skill checks and complete at least 48 of 64 standard development episodes.

| Policy | Training treatment | Previous standard development result |
|---|---|---:|
| DSAC MLP | Train from scratch with forward-KL for the discrete actor. Its target comes from its own twin quantile critics. Continuous controls use the reparameterized SAC objective. | 0/64 |
| PPO MLP | Train from scratch using native categorical sampling, with the same probabilities used by the collector and the PPO ratio. Add a uniform-over-legal-actions cross-entropy term on exploratory curriculum stages. | 8/64 |
| PPO GRU | Same PPO treatment and curriculum, with a recurrent actor and value representation. | 3/64 |
| DSAC GRU | Reuse the hash-verified checkpoint from the original ownership curriculum with standard reverse-KL. Replay it on the development bank before the comparison. | 60/64 |

The curriculum changes starting location, satiety, cash and liabilities over its stages, then returns to the standard home start. The final environment still requires clearing both debts before ownership. Meals, work, sleep, gambling, refuelling and borrowing are the actor's decisions. The new PPO exploration term is symmetric over legal actions and has coefficient 0.002 at exploration strength 0.2.

The local forward-KL DSAC MLP warm-fork won 16/16 in the first food lesson and 11/16 in each of two food transfer contexts. In the full campaign it completed 93 meals across 16 episodes but won 0/16. Thirteen episodes ended with zero fuel and none cleared both debts. The new PPO path passed a real 512-transition collection and optimizer integration check with matching likelihoods. Its new learning performance remains to be measured.

The evaluation runs all 30 registered conditions with 256 episodes per policy and condition if all four policies meet their readiness criteria. This is 120 policy-condition records and 30,720 evaluation episodes, with paired scenario IDs across policies and replay audits of all recorded results. The test banks are separate from training and development. If readiness fails, the job preserves the diagnostics and keeps the final banks unused.

The comparison measures these four policies under the same test conditions. The frozen reference has a different training history and each architecture has one initialization, so the results cannot isolate the effect of architecture or estimate between-seed variability.
