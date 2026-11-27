# DSAC curriculum and exploration diagnosis

The completed DSAC MLP selected policy has zero ownership, 43 starvation deaths and 21 collisions on 64 validation episodes. All 17 validation checkpoints had zero ownership. Actor weights changed (L2 distance approximately 12.77), and numerical losses were finite. 

Training counters identify the collapse point. The first curriculum stage after cold start ($50 car principal, $100 cash, exploration mixture 0.2) produced 2653 ownership events and 55 paid shifts. At the next stage, principal rises to $150, exploration mixture becomes zero and replay is cleared. The $150 stage records zero paid shifts, zero ownership and 4752 new loans. The $300 stage records zero paid shifts and 170239 loans. The final $500 stage records zero paid shifts and 141434 loans.

The $50 task largely permits repaying from starting cash. Increasing debt above starting cash introduces the need to earn, exactly when extra exploration is removed and earlier experience is discarded. Debt, exploration and replay changed at the same transition. Their individual effects can be tested by varying one at a time. Clearing replay can be necessary when reward/dynamics change; blindly retaining incompatible transitions would create another problem.

The next useful diagnostic is a controlled earning-required subtask, with a known external objective and retained exploration, before applying a larger curriculum. Separately audit entropy per variable-duration decision and test its annealing/normalization. Validate successful online earnings and repayment, rather than checking only whether random cold-start behavior once exercised the mechanics. No expert economic action labels are required.

An exact-equivalence probe also found nonzero critic approximation error for physically identical bank actions; the certified absolute mean-error lower bounds were 0.00128 and 0.00058. This is a local approximation finding, not a proven explanation for global task failure.

[Per-phase counters](dsac_schedule.json) and [Critic equivalence evidence](dsac_critic_equivalence.json)
