# Assignment 1: Gambler’s Taxi Environment

Iuliia Kurnaeva  

## 1. Vehicle dynamics, state, actions, and observations

The project began with an economic choice: earn a small steady wage, gamble for a larger immediate payout, sleep, or borrow. The vehicular assignment turned each choice into a trip. Time spent driving, parking, working, or sleeping determines which opportunities are still available. A late casino visit can leave the driver tired on the way home and short of time before the next shift.

The final task starts with $1,000 of debt. The driver has ten game days to clear the debt at the bank and buy out the taxi for $115 at the garage. The JAX environment implements `init_params`, `reset`, `step`, and `get_observation` in a new class derived from the course's `BaseEnv`. The 30 × 20 map has seven destinations, stationary obstacles, and three moving vehicles. The environment reuses the base lidar geometry. Precomputed route distances to all seven destinations account for static obstacles. Moving traffic is handled by the lidar and the collision check on each step. [Environment](../src/env/gamblers_taxi.py), [Map and distances](../src/env/city.py) and [Course starter](https://github.com/InnoRL/IAEML-assignment)

### Vehicle dynamics and justification

The continuous state contains position, heading, speed, and actual steering angle. The driving action supplies two normalized values in `[-1, 1]`. The longitudinal command sets acceleration or braking. The second sets a target steering angle. The actual steering angle approaches its target over time. The discrete branch selects one of 12 operations, including drive, wait, work, sleep, borrow, repay, start either game, hit, stand, and buy out.

A kinematic bicycle model gives the car a turning radius, inertia, and a braking distance with an inexpensive fixed-size JAX update. The model uses a 0.16 time step, a wheelbase of 0.8 map units, a maximum speed of 3.8, and a nominal steering limit of 0.58 radians. Rolling resistance and speed-dependent drag subtract `0.22 + 0.055 v²` at positive speed. Traction, braking, and drag first update speed. The average old and new speeds then determine heading change and translation.

```text
v_next = clip(v + dt × (traction_or_braking − drag), 0, vmax)
heading_next = heading + dt × mean(v, v_next) × tan(steering) / wheelbase
position_next = position + dt × mean(v, v_next) × direction_at_midpoint
```

Fatigue reduces steering response and braking strength. Low mood can weaken traction, tilt can raise the steering limit, and anxiety can slow steering response further. At full fatigue the nominal steering-response factor falls from 0.72 to 0.36 before the anxiety adjustment. The braking coefficient falls from 5.1 to 3.366. Work and casino operations remain available at full fatigue when their ordinary conditions are met. A swept collision test computes the minimum relative separation during the step, so a contact between two sampled positions can still terminate the episode. [Vehicle and collision code](../src/env/gamblers_taxi.py)

### Observations and interaction rules

The agent sees 16 lidar readings, distances, directions and route gradients to destinations, vehicle speed and steering, cash and debt, shift status, time, fatigue, mood, tilt, anxiety, and habit. Blackjack observations include the player's total, whether the hand has a soft ace, and the dealer's exposed card. A boolean action mask enforces each operation's current eligibility. A building interaction requires a speed below 0.2 within 1.55 map units of its center.

| Operation | Rule and rationale |
|---|---|
| Work and sleep | A completed 28-tick shift pays $145. Sleep takes 24 ticks and reduces fatigue. Missing two consecutive shifts can lead to dismissal. The job center provides a route back to employment. |
| Borrow and repay | A loan increases cash and principal by $60 and adds 0.35 to anxiety before clipping. Total debt is capped for lending at $1,300. Interest accrues at 0.012% per tick. The bank applies available cash to the balance and repayment reduces anxiety. |
| Slots | A $55 wager returns $325 with probability 0.12. The net outcomes are +$270 and −$55, with an expectation of −$16 per spin. |
| Blackjack | The $90 game uses cards drawn with replacement and a dealer that stands at 17. A regular winning hand makes $81 after commission, a natural can make $121.50, and a push returns the stake. Hit and stand are explicit actions. |
| Schedule and buyout | Casino initiation is allowed on daily ticks 90–144. Both venues share a 24-tick cooldown after settlement. Daily expenses are $8. The garage requires zero debt and sufficient cash for the $115 buyout. |

Habit grows after resolved games, particularly after wins. Losses also raise tilt. Sleep accelerates habit decay. If the driver tries to leave while another bet is legal, the optional impulse mechanism can replace that driving command with a wager. The log records the requested action, executed action, and impulse counter. This instrumentation supports a direct check of whether the environment or the policy initiated the wager. [Implementation](../src/env/gamblers_taxi.py) and [Illustrative impulse video](../artifacts/dependence.mp4)

The bankroll and casino schedule give a bound on shortcuts. At most three rounds can start on day one. Even with the highest opening cash, a full shift, and three maximum slot wins, the driver has $28 + $145 + 3 × $270 = $983 before any extra principal. Clearing the opening debt and buying out requires at least $1,115. Borrowing adds an equal amount to cash and debt before interest.

## 2. Reward design and justification

The base reward is +1 for buyout and −1 for a crash. At the deadline the penalty is up to −0.25, scaled by the shortfall in net worth. Each step also subtracts `(0.65 / 1550) × min(debt / 1000, 1.25)`. The debt term values early partial and complete repayment. The buyout requirement still values keeping enough cash to finish after clearing the bank balance.

To improve the learning signal between terminal events, the step reward adds `γ Φ(s_next) − Φ(s)`, with `γ = 0.9985`. The bounded potential combines net-worth progress and distance to a useful destination. Once net worth is sufficient, it directs progress toward the bank while a debt remains and the garage after repayment. The potential is zero at all terminal states, so its discounted sum along a completed episode is `−Φ(s_start)`. Income and gamble outcomes affect intermediate progress through cash and debt. Visit counts and emotion variables receive no direct reward. The debt cost is an explicit part of the objective. [Reward implementation](../src/env/gamblers_taxi.py) and [Ng, Harada, and Russell](https://aima.cs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

## 3. Difficulties, solutions, and validation

| Difficulty | Response and rationale |
|---|---|
| The original economic concept gave travel no cost. | Required the driver to reach and park at a destination. Driving, services, and fatigue now share a single clock. |
| A scenario driver clipped an obstacle on the way to slots. | Revised the approach and used distance fields around static obstacles. Swept collisions still register contact during a step. |
| A big casino payout could end short early campaigns before the work, credit, and sleep mechanics mattered. | Added a shared cooldown, day windows, a longer campaign, and explicit bank settlement. The final rules yield an algebraic first-day bound. |
| A replay could make an impulse look like a voluntary choice. | Recorded the policy request and the executed action separately. The study can disable the override while retaining habit dynamics. |
| Rendering inside the simulation would burden training. | Implemented interactive Pygame playback and an offline renderer that consumes numeric logs. |

The verification script checks driving acceleration and coasting, parking eligibility, one wage per shift, loan and repayment accounting, fatigue, missed shifts and rehiring, both games, buyout gating, collisions, and the discounted shaping sum. JIT is used in transition checks. In the current verification run the scripted work strategy got 12 buyouts from 24 starts. The destination remained reachable while some runs still failed in traffic. [Verification script](../src/verify.py)

## 4. Results video and local reproduction

The main video follows a scripted driver from a $1,000 debt to a buyout at tick 1381. It completed nine shifts, slept nine times, made two payments, and retained $29.51 after purchase. The comparison clip follows a credit-funded gambling route. It completed five shifts, four sleeps, nine slot spins, and nine blackjack hands before timing out with $1,265.49 of debt. The comparison clip was recorded before a scenario-controller speed adjustment. [Main video](../artifacts/debt_work.mp4), [Main log](../artifacts/debt_work_episode.npz), [Gambling comparison](../artifacts/debt_chase.mp4) and [Comparison log](../artifacts/debt_chase_episode.npz)

From the repository root after installing the [README dependencies](../readme.md):

```bash
python src/verify.py
python src/play.py --dependence
python src/replay.py render --log artifacts/debt_work_episode.npz --video runs/replay/assignment1.mp4
python src/replay.py render --log artifacts/debt_chase_episode.npz --video runs/replay/assignment1-risk.mp4
```

To generate a fresh trajectory with the current scripted controller, run `python src/replay.py record --strategy work_sleep --seed 71000 --log runs/replay/new-environment.npz`. The saved logs preserve the exact trajectories used in the report videos. The interactive frontend extends the course Pygame renderer. [Course repository](https://github.com/InnoRL/IAEML-assignment)
