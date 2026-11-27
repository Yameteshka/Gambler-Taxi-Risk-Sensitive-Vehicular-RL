# Learning recovery: primary sources and testable implications

## Sources

| Primary source | What is relevant here | Scope or difference from this study |
|---|---|---|
| [Ma et al., DSAC, JAIR 2025 / original 2020 preprint](https://arxiv.org/html/2004.14547v3), [author code](https://github.com/xtma/dsac) | Quantile distributional SAC, continuous-control benchmarks including BipedalWalkerHardcore and risk-sensitive navigation. Appendix B reports a replay warmup, actor learning rate 3e-4, fixed temperatures and normalization. | It is not a published solution of our life-and-debt environment. Our finite-quantile hybrid model is an adaptation, not an exact reproduction. |
| [Delalleau et al., Hybrid SAC, AAAI RL in Games 2020](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2020/02/11174405/AAAI20-RLG_paper_29.pdf) | Mixed discrete/continuous policies, separate entropy terms, and a Watch Dogs 2 driving application. | Two policy heads do not guarantee learning a sparse multi-stage economic task. |
| [OpenAI Spinning Up, SAC equations and implementation guidance](https://spinningup.openai.com/en/latest/algorithms/sac.html) | Entropy enters future critic targets; random initial exploration and replay warmup are explicit. Its reference implementation uses one gradient update per environment interaction. | Physical simulator ticks are not automatically network interactions. Its settings are not universal tuning guarantees. |
| [Florensa et al., Reverse Curriculum Generation, CoRL 2017](https://arxiv.org/abs/1707.05300), [authors' explanation](https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/) | Keep the goal and reward fixed while expanding the distribution of initial states; focus on starts of intermediate difficulty. | Our structured reset lessons are an engineering adaptation, not the original automatic Brownian-start algorithm. |
| [Duan et al., DSAC-T](https://arxiv.org/abs/2310.05858), [author implementation](https://github.com/Jingliang-Duan/DSAC-v2) | A distinct Gaussian-distribution DSAC family with refinements for critic estimation and reward-scale robustness. | It is not interchangeable with Ma et al.'s quantile DSAC. We do not claim to implement DSAC-T. |

## Confirmed local problems

1. The previous curriculum asked for optional resource actions in a task solved from initial cash. Its actor-only check improved to 14/16 easy ownership while the full-game ownership remained 0/64. It never reached the gradual debt changes.
2. The preserved 614,296-step pilot contained only 9,216 network decisions and 576 optimizer updates. We must report physical steps, decisions, update counts and wall time separately. Calling those physical ticks an equivalent number of training interactions is misleading.
3. In the latest full validation, collisions clustered on travel back toward work. Therefore navigation remains a measurable source of failure; it cannot simply be assumed solved because a controller exists.

## Hypotheses requiring experiments

- A bonus per decision has different physical-time consequences for a three-tick bank operation and a long journey. With the initial temperatures and our reward scale, a conservative entropy-return upper bound is about 4.74, compared with an ownership reward near 1. A fixed-temperature comparison tests whether this incentive changes the learned policy. A temperature of 3e-6 for each branch bounds the same quantity below 0.014.
- Actor learning rate, replay warmup and update/decision ratio differ from the cited DSAC configuration. They must be calibrated on small known tasks before another expensive full-game study.
- The hand-derived route cue and potential can dominate early approximate values even though properly terminalized potential shaping preserves the exact optimal policy. A reward-only training treatment removes this finite-learning complication while preserving the stated external objective. This is a changed training treatment and must be reported.

## Candidate procedure

Use one ownership objective and one transition/reward kernel throughout training. Change reset contexts rather than repeatedly changing interest, hunger, debt-contract parameters and markets. Start near ownership, then settlement, then at work with a genuine cash shortfall, then food- and fuel-constrained states, personal debt, and standard home starts. Retain replay across these reset-only changes and mix earlier contexts into training to reduce forgetting. Every economic action is still chosen by the network.

Do not require discretionary food or sleep in contexts already solvable without them. Check resource skills where ownership actually requires or benefits from them. Casino play is not rewarded in final-task training. Negative-expected-value games may rationally remain unused in some conditions.

The local acceptance sequence is: distribution/target regression checks; fast synthetic multi-step economic learning with the production update code for both algorithms; matched real-driver routes; short real-game learning cases. 

No method here can guarantee winning every adverse stochastic scenario. A valid final result requires competent learning plus honest uncertainty and failure rates on untouched banks; testing is not an action script the policy must reproduce.
