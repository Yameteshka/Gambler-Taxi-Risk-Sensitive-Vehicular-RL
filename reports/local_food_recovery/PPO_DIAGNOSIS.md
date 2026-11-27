# PPO food choice in the campaign

The original saved policies assign food high probability in the food lesson and very low probability in the campaign. Among recorded decisions with 12–20% satiety and at least $40, the median probability of choosing any of the three food actions was:

| Policy | Food lesson | Standard campaign |
|---|---:|---:|
| PPO MLP | 99.43% (16 decisions) | 0.00208% (5 decisions) |
| PPO GRU | 98.98% (16 decisions) | 0.0446% (21 decisions) |

These are decisions from different checkpoints and contexts, so the numbers describe observed behavior without identifying the causal effect of any single observation feature. Actor probabilities were reconstructed from saved weights and observations, including GRU memory across decision blocks. The reconstructed categorical entropies matched the recorded values within 0.00053. The detailed table is in `ppo_food_probabilities.json`.

The collector and update both use the same categorical mixture during early training. For native policy probability p(a), epsilon e and K legal actions, the sampled probability is m(a) = (1-e)p(a) + e/K. The gradient of log m(a) is the gradient of log p(a) multiplied by (1-e)p(a)/m(a). When the native policy has almost excluded an action, uniform exploration can still execute it while the PPO likelihood gradient for that sample is extremely weak. The entropy bonus also operates on the mixture. This is a property of the implemented objective, with consistent behavior log probabilities.

At epsilon 0.2, the median share of the mixed probability contributed by the native policy across the three food actions in the selected campaign slice was 0.053% for MLP and 1.18% for GRU. These are aggregate shares rather than per-action gradient multipliers. This explains why visible food coverage during exploration can coexist with negligible food selection by the frozen policy.

The next PPO intervention should target the policy distribution used by the actor. Replace the external uniform mixture in early training with policy-native exploration, keeping collection and the PPO ratio on the same distribution. A training-only forward constraint from a broad distribution over legal actions to the actor can give excluded actions a usable gradient. Anneal it and evaluate the ordinary frozen actor with no exploratory override. Before changing the advantage estimator, log the signed advantages of actually completed meals in campaign-like starts. If useful meals still receive weak or negative advantages, investigate value bootstrapping across the eight-decision collector boundary separately. No action-specific rewards or fixed food decisions are needed for this diagnosis.
