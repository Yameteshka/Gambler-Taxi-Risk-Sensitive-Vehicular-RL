# Reward audit: delayed failure and return bounds

## Why delayed starvation was a learning trap

An earlier calibration combined a roughly 0.0001 per-step running cost with a -1 terminal death or collision penalty and discount 0.999. Ignoring the separately reported financial term, while the debt fraction remained saturated, a failure at physical step T had return

`G_old(T) = -0.1(1 - gamma^T) - gamma^(T-1)`.

This increases as failure is delayed. A policy that stays still until starvation can therefore beat an exploratory policy that crashes earlier.

## Current base objective

Let kappa=0.1. Every nonterminal step costs `kappa(1-gamma)`, an unsuccessful terminal pays `-kappa`, and ownership pays `1+b`, where `0 <= b <= 0.05` rewards a remaining cash reserve. There is no additional running cost on a terminal step.

For any finite unsuccessful trajectory of T physical steps,

`G_failure = -kappa(1-gamma) sum(t=0..T-2, gamma^t) - kappa gamma^(T-1) = -kappa`.

For ownership,

`G_success(T,b) = -kappa + gamma^(T-1)(1+kappa+b)`.

Thus delaying an identical failure cannot improve the base return, and earlier otherwise identical ownership is strictly preferred. The verifier checks naturally occurring starvation at distinct times and a separate deadline against the identity. The financial increment remains a separate, explicitly declared secondary objective.

## Bound for the secondary financial term

The task adds `U(W_next)-U(W_current)`, where `U(W)=0.1 clip(W/2000,-1,1)` and W includes cash, reserved stakes and all liabilities. Loan proceeds and principal repayments are wealth-neutral; interest, fees, purchases, wages and gambling settlements have their actual financial effects.

For a trajectory of length T, its discounted financial contribution is

`F = -U(W0) + (1-gamma) sum(t=1..T-1, gamma^(t-1) U(Wt)) + gamma^(T-1) U(WT)`.

The coefficients of the future utility terms are nonnegative and sum to one. Consequently `F` lies in `[-U(W0)-0.1, -U(W0)+0.1]`. Its absolute value is at most 0.2. For two trajectories from the same financial start, their financial contributions differ by at most 0.2. The bound applies to the signed total contribution.

For a fixed initial state and a finite horizon H, the external-return advantage of any successful trajectory over any failed trajectory is therefore at least

`(1+kappa) gamma^(H-1) - 0.2`.

At gamma=0.999 this is positive for both the five-day (775-step) and ten-day (1550-step) principal tasks. In those finite tasks, even the worst allowed financial contribution cannot reverse the pathwise preference for ownership.

The same positive lower bound does not hold for arbitrarily late ownership in the continuing task, or even over the entire 30-day administrative observation window. The continuing arm optimizes discounted composite return. Episodes still active at the administrative horizon are right-censored. Potential shaping has a separate terminal telescoping identity and does not change these complete-episode comparisons from an identical start. The training-only novelty treatment is excluded from final evaluation.

## Execution checks

Independent intended-rule checks cover cash/debt conservation, borrowing, repayment, charges, service timing, starvation, pushing, movement, collision geometry, action eligibility and terminal behavior. Evaluation replay audits reconstruct every active action, policy choice, control, resource field and final account.

Suspicious stationary policies are inspected through requested and applied controls, action legality, learned action probabilities, critic values, resource changes and relevant development counterfactuals. A pointwise critic certificate compares controls that give exactly the same immediate reward and next state. If their predicted means differ by g, at least one mean error is at least g/2. The analogous Wasserstein bound follows by the triangle inequality.

Fixed held-out scenarios support reproducible matched estimates and confidence intervals. Approximately thirteen observations determine the lowest 5% of a 256-scenario policy evaluation, and training variability must also be reported across the five initializations.

## Continuing-task return truncation

At a 30-day administrative window K=4650, a remaining base-reward suffix has absolute value at most 1.05 (ownership and reserve dominate the 0.1 failure/running bound). The remaining discounted financial contribution, measured from the censored state, has absolute value at most 0.2. Therefore the absolute difference between the recorded partial discounted external return and any future complete return is at most `1.25 * gamma^K`, about 0.012 at gamma=0.999.
