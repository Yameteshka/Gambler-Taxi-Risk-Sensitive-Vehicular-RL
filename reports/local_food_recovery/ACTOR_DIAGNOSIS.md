# Actor saturation: source-backed diagnosis and isolated implementation

## Measured failure on recorded states

The control DSAC MLP critic ranks Restaurant highest among **all legal actions in all 16 recorded near-restaurant hunger states**. At the actor's own mean driving style, mean Q(Restaurant) is 0.83343 and Q(Buy) is -0.07746. Nevertheless, the actor gives Restaurant probability approximately 8.06e-10.

`actor_projection_check.json` evaluates the actual new production objective on these fixed inputs. The original reverse-KL objective has mean Restaurant logit gradient -4.59e-11; the optional forward-KL categorical projection has -0.0625 (the batch has 16 states), about 1.36 billion times larger in magnitude. Both point in the correct direction, but the original update is effectively starved of signal at this saturated policy. The legal-action mask is respected and the original objective matches its pre-refactor expression numerically.

The gradient comparison holds the critic and recorded states fixed.

## Implemented intervention

The original SAC objective remains the default `categorical_actor_update='reverse_kl'`. A distinctly named `forward_kl` mode fits the categorical actor to a detached legal-action Boltzmann target derived from its own critic. The Gaussian branch retains its reparameterized objective; the quantile critic and environment rules are unchanged. The helper is used consistently by MLP and recurrent actor paths. There are no hand-labeled economic actions.

The categorical cross-entropy is intentionally unscaled: alpha_d sets the target distribution's entropy, rather than suppressing its learning gradient. Continuous and categorical objectives therefore have a different relative weighting from vanilla SAC. This is an explicit algorithm variant, not an exact reproduction or algebraic correction of standard Hybrid SAC. `experiments/budget_recovery.py --actor-update forward_kl` registers it separately. Existing source-freeze checks must be renewed before any training; the old acceptance files are intentionally not relabeled as valid for the new implementation.

## Literature and limits

- Garg et al., AISTATS 2022, [An Alternate Policy Gradient Estimator for Softmax Policies](https://proceedings.mlr.press/v151/garg22b.html), directly studies poor adaptation when policies concentrate probability on suboptimal actions.
- Christodoulou, 2019, [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/html/1910.07207v2), equation 12, gives the probability-weighted categorical SAC objective retained as our baseline.
- Chan et al., JMLR 2022, [Greedification Operators for Policy Optimization: Investigating Forward and Reverse KL Divergences](https://www.jmlr.org/beta/papers/v23/21-054.html), compares the two projections and cautions that forward-KL reduction does not inherit every reverse-KL policy-improvement guarantee.

Reproduce the fixed-input check with `python -B experiments/check_actor_projection.py` in the project JAX environment. It forces CPU execution and performs no simulator rollouts or training runs.
