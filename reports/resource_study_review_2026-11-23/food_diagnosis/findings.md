# Why the failed policies do not eat

## Observed behavior

| Selected checkpoint, 64 development episodes | Food intentions | Completed meals | Ownership |
|---|---:|---:|---:|
| DSAC MLP | 2 | 0 | 0 |
| PPO MLP | 1 | 0 | 8 |
| PPO GRU | 2 | 1 | 3 |
| DSAC GRU | 104 | 102 | 60 |

DSAC GRU requested food in 60 episodes; PPO GRU made both food requests in one episode. The service-intention mask permits choosing food globally; physical eligibility on arrival separately requires sufficient cash and parking. All four saved evaluation configurations have the public menu enabled, with location/distance, satiety, prices and portion distributions available as inputs. 

### PPO: paying first without preserving a food budget

For selected PPO MLP on scenario 1000001, the physical trace shows:

1. At tick 0 it chooses work, completes a shift, and reaches tick 60 with $231.25 and satiety 0.684.
2. After three unsuccessful work requests, it chooses full car payment at tick 63; on completion it has $0 and satiety 0.501.
3. Another shift raises cash to $131.25. At tick 227 it chooses full payment again, leaving $0 and satiety 0.167 at tick 278.
4. It chooses work again and dies at tick 345 before the shift finishes. It never requests food in this episode.

Selected PPO GRU on the same scenario repeatedly chooses $100 car payments, including a partial payment of its last $31.25 at tick 282. It then heads to work with $0 and satiety 0.146 and dies at tick 347. It could afford the quoted restaurant immediately before that last payment ($30.26); travel/service feasibility at that late point has not been tested here.

PPO MLP's only food request is a cafe trip at tick 338 with $0 and satiety 0.023. It dies during the trip. PPO GRU's one successful meal occurs at the beginning of one episode with satiety 0.85; its later diner request occurs with $0 and satiety 0.062 and ends in starvation before purchase.

Thus these failures involve omitted or late food decisions and poor liquidity planning, not merely a meal that the simulator refused to restore after successful completion. This conclusion describes these recorded paths; it is not a universal diagnosis of every possible trajectory.

### DSAC MLP: different failed checkpoints have different loops

The selected checkpoint makes 4,152 work requests; 4,032 do not start a shift, including 3,904 requests lasting one physical tick. `strategy.advance` returns immediately on arrival when the requested operation is unavailable; `strategy.legal` still permits the global Work intention after a shift has been claimed or while its window is closed. Repeating that choice consumes hunger and network decisions without earning wages. The immediate return allows the actor to repeat the same unavailable action on the next tick.

At its last actual food-lesson gate, all 16 episodes begin near the restaurant with $100, no car debt and satiety 0.04. In every episode the actor immediately chooses ownership completion instead of eating and starves on the trip. On scenario 1000001 it dies after 18 ticks. Food is therefore neither hidden nor unaffordable at that lesson's reset.

Its latest saved standard-game validation differs from the selected checkpoint: it makes no food requests, and its low-satiety decisions include 1,175 Borrow100 and 1,192 MicroAll choices. This is a repeated borrowing and repayment sequence. The final training journal records discrete entropy approximately 0.022 with fixed discrete temperature approximately 0.000003. The low entropy indicates a concentrated action distribution.

## What was insufficient in the training protocol

Both PPO variants passed the four food-specific lessons before failing the full-$500 lesson. Their passing $350 gates had only 4/16 ownership episodes and zero completed meals. The gate allows progression after 4/16 ownership on intermediate lessons; food-specific lessons additionally require only a positive aggregate meal count. Retained food tests are run after training qualification, rather than required for each stage transition. Training includes 20% rehearsal resets, but this alone did not ensure retention.

These facts support a transfer/retention concern and show a weakness in the acceptance protocol. They do not distinguish catastrophic forgetting from dependence on restaurant-local reset states, insufficient economic exploration, or inaccurate long-horizon values. A matched checkpoint-by-context evaluation and controlled changes are needed for that distinction.

The financial reward is a scaled change in net worth. Food immediately reduces net worth, while its survival benefit is delayed. Repaying principal transfers cash against debt and is not itself a net-worth gain. A principal payment by itself contributes zero to the net-worth term. The delayed value of eating may be learned poorly; that remains a hypothesis requiring a controlled check.

## Proposed recovery, with changes separated

1. Preserve the qualified DSAC GRU and all failed checkpoints. First evaluate saved food-lesson, $350 and failed final checkpoints on the same development-only food contexts and full campaign. This locates the loss of food behavior without retraining or consuming final banks.
2. Strengthen curriculum acceptance: require performance across multiple development batches and repeated retention checks of earlier food/fuel/debt contexts at every transition. Choose thresholds before recovery runs. 
3. Train budgeting and timely food planning from varied home/work/bank starts and varied satiety/cash, gradually expanding route length and debt. Continue mixed rehearsal. All actions remain voluntary; do not reserve money automatically, force a meal or reward casino attendance.
4. Test unavailable-service execution separately: preserve voluntary choices, but use an explicitly documented, consistent feedback/wait duration rather than thousands of one-tick failed requests. Verify physical time, discounting and hunger accounting. This changes the action interface and requires a new protocol label if adopted.
5. For DSAC MLP, separately inspect actor action probabilities and critic targets at the near-restaurant starvation states; compare exploration/entropy settings in short matched local runs. For PPO, inspect the same checkpoints before and after the $250/$350 lessons. Change one component per diagnostic experiment, then apply the accepted common training distribution to all four architectures.

## Reproduction

Run `python experiments/diagnose_food_logs.py` with NumPy. The per-decision records are in [decisions.csv](decisions.csv), and [diagnosis.json](diagnosis.json) records the source trails.
