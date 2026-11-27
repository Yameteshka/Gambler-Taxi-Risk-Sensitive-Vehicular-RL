# Assignment 1: Gambler's Taxi Environment

Iuliia Kurnaeva  

## Scope and development

The initial idea was a choice between stable employment and uncertain casino payouts. The course requires a vehicle environment, so services became destinations on a road map. The current task studies financial planning under survival costs: the driver starts at home with $100, a $1000 car loan, fuel and food reserves. The goal is to repay both car debt and every personal loan, then reach the garage to complete ownership. Hunger, fuel, fatigue and travel time create competing uses of cash and time. Mood, anxiety, imposed gambling impulses, dismissal and the job center were removed. Work remains available at full fatigue; running out of fuel allows slow pushing, and running out of satiety causes death.

The environment is implemented in a new file, [resource_taxi.py](../src/env/resource_taxi.py), as a subclass of the course `BaseEnv`, with `init_params`, `reset`, `step` and `get_observation`. The original `base.py` is preserved. The map is 40 × 30, with ten service zones, static circular obstacles and three moving circular vehicles. Both the agent and obstacle circles have radius one; swept collisions use the sum of radii. This explicitly resolves an ambiguity in the starter collision helper, whose default center-distance threshold is only one. The course's lidar intersection implementation is reused. Static route fields are precomputed once for all services and supplied as distances and local directions. They account for static geometry, while moving traffic is sensed locally. They do not select the agent's economic destination.

## Dynamics, state and observations

A kinematic bicycle model offers a turning radius and braking distance at a low simulation cost. The continuous actions are acceleration/braking and target steering, each normalized to [-1,1]. A categorical operation branch chooses drive, wait, work, sleep, eat, two refueling quantities, three borrowing amounts, six repayment choices, three slot stakes, three blackjack stakes, hit, stand or ownership completion. There are 25 physical categorical choices. The learned service interface maps 26 intentions onto these physical operations and a shared route controller, with 179 public input features. Operations require parking within 1.55 map units at speed below 0.2. Eligibility depends on location, service state, cash, opening hours and debt; it does not ban work or gambling at full fatigue.

With dt=0.16, wheelbase=0.8 and steering limit=0.58 radians, steering approaches its target at a fatigue-dependent rate. Traction is 2.9 times positive acceleration; braking is 5.1 times the negative command, reduced by fatigue. Positive-speed drag is 0.22+0.055 v². Speed is clipped to [0,3.8]. Translation and yaw use the mean of old and new speed and a midpoint heading. This is a kinematic model, not a tire-slip model or a claim of realistic passenger-car dynamics.

```text
steer' = steer + (0.72 - 0.36 fatigue) × control_strength × (command × 0.58 - steer)
v' = clip(v + dt × (traction_or_braking - drag), 0, vmax)
yaw_increment = dt × mean(v,v') × tan(steer') / wheelbase
position' = position + dt × mean(v,v') × direction_at_midpoint
```

No fuel means voluntary low-speed pushing, capped at 0.2, with greater hunger and fatigue cost. The vehicle can be pushed anywhere, rather than being teleported to a station. Free Wait uses zero controls and coasts under friction; it does not stop momentum instantly. Timed services hold the parked car stationary while traffic, hunger and interest continue. Sleep lasts 24 ticks. A work shift lasts 28 ticks and may start during the first 70 ticks of each 155-tick day, once per day. It consumes fuel. Daily wages are $175 times 0.75, 1 or 1.35, with probabilities 0.3, 0.4 and 0.3. The provisional wage was $145; development-only calibration showed insufficient cash for the diagnostic strategy, so it was increased before final testing.

The numerical state includes position, heading, speed, steering, clock, service progress, cash, outstanding car principal and charges, satiety, fuel, fatigue, cards, daily shift status, counters and individual loan arrays. The public observation contains 16 lidar lengths, service-relative bearings and route gradients, resources, debts, rates, public quotes, clock and horizon, service status, previous public action/event, and blackjack information. Hidden cards and future random draws are excluded. The recurrent extensions may retain history. State and observations retain the base renderer field names such as `agent_pos`, `goal_pos`, `path_array` and `collision_rays`.

## Accounts, food and games

Cash has no $100 cap after reset. Car installments require cumulative principal repayment of 30%, 60% and 100% at days 3, 6 and 10; a missed checkpoint adds a $10 charge once. The five-day campaign uses the same calendar in the main comparison. Open-ended campaigns retain the contract calendar but have no game deadline; evaluation is administratively censored at 30 days. The accelerated five-day calendar is a separate extension.

Voluntary bank loans of $25/$50/$100 create distinct IDs and terms. The car rate is 0.5% daily and personal loans charge 3% daily, accrued per tick using (1+r)^(1/155)-1. Principal repayment and fees are accounted for separately. Repayment of personal loans follows highest rate, then oldest ID. New borrowing increases cash and liabilities equally. Personal debt prevents ownership even when the car balance is zero. Loan storage can grow during continuing training; evaluation storage is allocated from its observation window. These are implementation allocations, not a credit limit.

| Food service | Price draw per day | Portion draw per purchase | Service time |
|---|---|---|---:|
| Budget diner near fuel | U($4,$6) | U(0.04,0.06) | 2 ticks |
| Distant cafe | U($12,$18) | U(0.12,0.18) | 5 ticks |
| Distant restaurant | U($30,$40) | U(0.55,0.75) | 10 ticks |

Daily prices stay fixed within a day. Portions are sampled once per purchase and delivered after service, provided the driver survives. Satiety is clipped at one, so a large meal can be partly wasted. No venue guarantees full restoration. Cafe and restaurant sites have nearly equal static route distances from home and work; the verification tolerance is 5%. The cheap diner is deliberately closer. Dining does not directly remove fatigue. Deterministic-mean and higher-variance matched-mean food conditions are registered separately.

Slots have 12% win probability, gross payout multiplier 325/55 and stakes $10/$25/$55. Expected net return is -29.09% of stake. Blackjack uses replacement cards, soft aces, a dealer standing on 17, natural payouts and explicit hit/stand. A 10% commission applies to winning profit. Exact dynamic programming for this implemented model gives optimal expected net return -6.98% of stake. The independent-outcome core does not make a table due to win after losses. Opening hours, service time, shared cooldown and travel cost constrain gambling. Casino visits receive no special reward, and no action override forces a bet.

## Reward and safeguards

All principal algorithms share the external reward: +1 for ownership plus a cash reserve bonus capped at 0.05, -0.1 on any unsuccessful terminal, and -0.1(1-gamma) on each nonterminal step. Terminal steps have no additional running charge. For an unsuccessful episode of T physical steps, the discounted base return is exactly -0.1, independent of failure time. For ownership it is -0.1 + gamma^(T-1)(1.1 + reserve_bonus), so earlier otherwise identical success is strictly preferred. The separate financial increment is added below. There is no additional debt-running or timeout-shortfall penalty; actual interest and fees reduce net worth. This replaces an earlier calibration reward whose larger terminal failure penalty made delayed starvation preferable to an early crash.

An explicit bounded secondary financial objective adds U(W_next)-U(W_current), where U(W)=0.1 clip(W/2000,-1,1). Borrowing and repayment are wealth-neutral apart from charges. Its entire discounted contribution is bounded in absolute value by 0.2, so it cannot create an unbounded debt penalty or an unlimited wage-farming objective. It is linear in wealth within +/- $2000 and saturates outside; saturation is recorded and limits risk interpretations. This financial term changes the objective and is explicitly ablated in T09. It is distinct from policy-invariant potential shaping.

Dense feedback has no casino-directed target or visit bonus and is potential-based: F=gamma Phi(next)-Phi(current). The bounded potential combines net-worth progress, food, fuel, fatigue and distance to useful services. It is zero at all true terminals. Its discounted telescoping sum is -Phi(start) for a terminal episode, so cycling through sleep, borrowing and repayment cannot create extra terminal return through shaping alone. Loans are not income, debt payments are not a second income event, and there is no repeated visit bonus. The verifier checks the shaping identity and cash/debt conservation. Administrative censoring of continuing evaluation is not an environment terminal and is reported separately.

## Reflection and verification

The central difficulty was separating economic failure from navigation and accounting errors. Development checks found unsafe stops near traffic, late food planning, excessive bank trips, and a batch-axis error in the loan-count log; each was corrected before freezing the main study. All final evaluation actions and endpoint accounts are independently replayed by [audit_results.py](../src/audit_results.py); this complements intended-rule checks rather than proving optimality. Correct circle geometry is checked independently of a controller's apparent success. Development calibrations are retained under `runs/` and do not count as final policy comparisons. [verify.py](../src/verify.py) covers driving/coasting, pushing, starvation, absorbing terminals, individual loans, repayments, ownership blocking, checkpoint charges, meal timing, scenario splits, route balance and shaping. [analytic_checks.py](../src/analytic_checks.py) computes game expectations separately from learned outcomes.

## Results, video and reproduction

Eighteen of 125 registered runs completed training and received standard evaluations. The assignment does not require an optimal agent. Final outcomes are reported in [results.md](results.md), including unsuccessful policies. The numeric log is the source of truth: training and evaluation are headless; the renderer reads saved trajectories afterward. Commentary consists of deterministic English event receipts generated from those records, not the agent's internal reasoning. The replay shows individual microloans and total liabilities.

Use the [README](../README.md) setup instructions, run `python src/verify.py`, then `python src/evaluate.py --reference reserve --scenario-start 1 --episodes 10 --output runs/reference`. Render with `python src/render_replay.py runs/reference --scenario-id 1 --output runs/reference.mp4`. Learned-policy evaluation and replay commands are documented there too. [Course repository](https://github.com/InnoRL/IAEML-assignment).
