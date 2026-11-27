"""Scripted drivers for repeatable environment checks and policy baselines."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from algorithms.hybrid_dsac import Transition
from env import city
from env import gamblers_taxi as taxi


ZONE_ACTIONS = jnp.array((taxi.SHIFT, taxi.SLEEP, taxi.APPLY, taxi.BORROW, taxi.SPIN, taxi.DEAL, taxi.BUY))
MOVES = jnp.array(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)))
CHASE_LOSSES = 6
FAST_CASH = 7


def _drive_to(state, params, zone):
    config = params.config
    pos = state.agent_pos
    target = params.zone_centers[zone]
    nearby = jnp.clip(jnp.rint(pos).astype(jnp.int32) + MOVES, jnp.array((1, 1)),
                      jnp.array((city.WIDTH - 2, city.HEIGHT - 2)))
    targets = jnp.concatenate((target[None, :], nearby), axis=0)
    displacement = targets - pos
    obstacles = jnp.concatenate((state.static_obstacles, state.kinematic_obstacles), axis=0)
    offset = obstacles[None, :, :] - pos
    fraction = jnp.clip(jnp.sum(offset * displacement[:, None, :], axis=-1) /
                        jnp.maximum(jnp.sum(displacement**2, axis=-1, keepdims=True), 1e-5), 0., 1.)
    clearance = jnp.min(jnp.linalg.norm(offset - fraction[..., None]*displacement[:, None, :], axis=-1), axis=-1)
    values = params.fields[zone, nearby[:, 1], nearby[:, 0]]
    values = jnp.concatenate((jnp.array((-1.,)), values))
    cost = values - 0.08*jnp.linalg.norm(displacement, axis=-1)
    costs = jnp.where(clearance >= 1.04, cost, cost + 100. + 10.*(1.04-clearance))
    best = jnp.argmin(costs)
    vector = displacement[best]
    f = state.agent_forward_dir
    angle = jnp.arctan2(f[0]*vector[1] - f[1]*vector[0], jnp.dot(f, vector))
    turn = jnp.clip(1.8 * angle, -1., 1.)
    distance = jnp.linalg.norm(target - pos)
    stop = distance < config.zone_radius - 0.20
    # Turn around slowly enough to fit the streets and brake before a zone.
    desired = jnp.minimum(3.2 - 2.05*jnp.minimum(jnp.abs(angle), 1.),
                          jnp.maximum(0.70, 2.1*(distance - config.zone_radius + 0.25)))
    if config.starting_debt:
        turn_speed = 2.7 - 1.9*jnp.minimum(jnp.abs(angle), 1.)
        near_obstacle = jnp.min(jnp.linalg.norm(obstacles - pos, axis=-1))
        desired = jnp.minimum(desired, jnp.minimum(turn_speed, jnp.clip(near_obstacle - 0.35, 0.8, 2.7)))
    desired = jnp.where(stop, 0., desired)
    desired = jnp.where(clearance[best] < 0.90, jnp.minimum(desired, 0.65), desired)
    throttle = jnp.clip((desired - state.speed)*1.8 + jnp.where(desired > 0., 0.23, 0.), -1., 1.)
    return jnp.array((throttle, turn), dtype=jnp.float32)


def scripted_policy(state, obs, params, strategy=0):
    if isinstance(strategy, int) and strategy == CHASE_LOSSES:
        return chase_losses(state, obs, params)
    if isinstance(strategy, int) and strategy == FAST_CASH:
        return fast_cash(state, obs, params)
    c = params.config
    net = taxi._balance(state)
    ready = net >= c.buyout
    next_shift = c.day - (state.time % c.day)
    slots_open = state.cash >= c.slot_stake
    blackjack_open = state.cash >= c.blackjack_stake
    time_left = c.horizon - state.time
    work_possible = time_left > 65
    can_risk = (slots_open | blackjack_open) & (time_left > 45)
    shortfall = c.buyout - net
    income_left = c.salary * (
        jnp.maximum(time_left - 60 - next_shift, 0) // c.day
        + (time_left > next_shift + 60)
        + (~state.shift_claimed & (state.time % c.day < c.shift_window) & state.employed)
    )
    risk = ((strategy == 1) & can_risk & (shortfall > income_left))
    risk |= (strategy >= 2) & (can_risk | (strategy == 4))
    casino = jnp.where(
        (strategy == 3) | ~slots_open, city.BLACKJACK, city.SLOTS
    )
    casino = jnp.where((strategy == 4) & ~(slots_open | blackjack_open), city.BANK, casino)
    need_sleep = (state.fatigue > 0.57) & (time_left > 95)
    can_sleep = state.shift_claimed | ~state.employed | (state.fatigue >= 0.88)
    # After a shift, recover at home and wait there until the next commute.
    go_home = can_sleep & (need_sleep | (state.shift_claimed & ~risk))
    work_goal = jnp.where(state.employed | ~work_possible, city.WORK, city.JOBS)
    settlement_goal = jnp.where(state.debt > 0., city.BANK, city.GARAGE)
    if c.starting_debt:
        phase = state.time % c.day
        bank_payment = (state.debt > 0.) & (state.cash >= 3*c.salary) & (
            (phase >= c.shift_window) | state.shift_claimed
        )
        work_goal = jnp.where(bank_payment, city.BANK, work_goal)
        go_home &= ~bank_payment
    goal = jnp.where(ready, settlement_goal, jnp.where(
        go_home & ~((strategy >= 2) & can_risk), city.HOME, jnp.where(risk, casino, work_goal)
    ))
    return scenario_action(state, obs, params, goal)


def chase_losses(state, obs, params):
    """Spend available pay at alternating tables and rest when fatigue takes over."""
    c = params.config
    paid = state.counters[taxi.PAID]
    slots = paid % 2 == 0
    casino = jnp.where(slots, city.SLOTS, city.BLACKJACK)
    stake = jnp.where(slots, c.slot_stake, c.blackjack_stake)
    phase = state.time % c.day
    shift_open = (phase < c.shift_window) & ~state.shift_claimed & state.employed
    casino_hours = (phase >= c.casino_open) & (phase < c.casino_close)
    earn = (state.cash < stake) | shift_open | (~state.employed & ~casino_hours)
    work = jnp.where(state.employed, city.WORK, city.JOBS)
    rest = (state.fatigue >= 0.78) | ((state.fatigue >= 0.42) & ~shift_open & ~casino_hours)
    goal = jnp.where(rest, city.HOME, jnp.where(earn, work, casino))
    return scenario_action(state, obs, params, goal)


def mixed_expert(state, obs, params):
    if params.config.starting_debt:
        return fast_cash(state, obs, params)
    # Use the conditional strategy on starts where wages alone cannot cover the buyout.
    c = params.config
    late_shortfall = (state.time >= 228) & (taxi._balance(state) < c.buyout) & (
        state.cash < 30.
    ) & (state.counters[taxi.PAID] == 0)
    started_risk = state.counters[taxi.SPINS] + state.counters[taxi.HANDS] + state.counters[taxi.LOANS] > 0
    risk = (late_shortfall | started_risk) & (taxi._balance(state) < c.buyout)
    choice = jnp.where(risk, 4, 0)
    return scripted_policy(state, obs, params, choice)


def fast_cash(state, obs, params):
    """Try the two fast payouts after a shift, with a small loan after a loss."""
    c = params.config
    phase = state.time % c.day
    payday = state.shift_claimed
    stake = jnp.where(state.counters[taxi.PAID] % 2 == 1, c.slot_stake, c.blackjack_stake)
    casino = jnp.where(state.counters[taxi.PAID] % 2 == 1, city.SLOTS, city.BLACKJACK)
    # Keep a new loan separate from earnings. Interest has already increased the
    # opening balance, so check the real lending limit before choosing the bank.
    may_borrow = state.debt + c.loan_unit <= c.starting_debt + c.loan_limit
    seek_cash = (state.cash < stake) & (state.cash + c.loan_unit >= stake) & may_borrow
    funded = (state.cash >= stake) | seek_cash
    after_shift = payday | (phase >= c.shift_window)
    chase = after_shift & (phase < c.casino_close) & funded & (state.fatigue < 0.98)
    ready = taxi._balance(state) >= c.buyout
    baseline = scripted_policy(state, obs, params, 0)
    play = scenario_action(state, obs, params, jnp.where(seek_cash, city.BANK, casino), repay=False)
    return jax.tree.map(lambda a, b: jnp.where(chase & ~ready, a, b), play, baseline)


def scenario_action(state, obs, params, zone, repay=None):
    choice = ZONE_ACTIONS[zone]
    if repay is None:
        repay = ((taxi._balance(state) >= params.config.buyout) | (
            (params.config.starting_debt > 0.) & (state.cash >= params.config.salary - params.config.operating_cost)
        ))
    can_settle = (state.debt > 0.) & repay
    choice = jnp.where((zone == city.BANK) & can_settle, taxi.REPAY, choice)
    total, soft = taxi._hand_total(state.hand)
    dealer = jnp.minimum(state.dealer[0], 10)
    hit = (total < 12) | ((total < 17) & ((dealer >= 7) | (dealer == 1))) | (soft & (total < 18))
    choice = jnp.where(state.activity == taxi.DEAL, jnp.where(hit, taxi.HIT, taxi.STAND), choice)
    fallback = jnp.where(obs.action_mask[taxi.DRIVE], taxi.DRIVE, jnp.argmax(obs.action_mask))
    choice = jnp.where(obs.action_mask[choice], choice, fallback).astype(jnp.int32)
    control = _drive_to(state, params, zone)
    return taxi.TaxiAction(jnp.where(choice == taxi.DRIVE, control, 0.), choice)


def scripted_trace(keys, params, strategy):
    env = taxi.GamblersTaxiEnv

    def reset(key):
        return env.reset(key, params)

    def single(key, obs, state):
        active = state.outcome == taxi.ACTIVE
        action = jax.lax.switch(
            jnp.where(strategy >= 5, strategy - 4, 0),
            (lambda: scripted_policy(state, obs, params, strategy),
             lambda: mixed_expert(state, obs, params),
             lambda: chase_losses(state, obs, params),
             lambda: fast_cash(state, obs, params)),
        )
        next_obs, new, reward, done, _ = env.step(key, state, action, params)
        transition = Transition(obs.features, obs.action_mask, action.control, action.choice,
                                reward, done, next_obs.features, next_obs.action_mask)
        select = lambda new, old: jnp.where(active, new, old)
        return jax.tree.map(select, next_obs, obs), jax.tree.map(select, new, state), (transition, state, active)

    obs, states = jax.vmap(reset)(keys)

    def body(carry, _):
        keys, obs, states = carry
        keys, step_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0))
        next_obs, next_states, trace = jax.vmap(single)(step_keys, obs, states)
        return (keys, next_obs, next_states), trace

    (_, _, states), trace = jax.lax.scan(
        body, (keys, obs, states), None, length=params.config.horizon
    )
    return states, trace


def rollout(params, seed, strategy=0, actions=None):
    """Collect one real transition per tick, including the terminal transition."""
    env = taxi.GamblersTaxiEnv
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)
    step = eqx.filter_jit(env.step)
    driver = eqx.filter_jit(partial(scripted_policy, strategy=strategy))
    for _ in range(params.config.horizon):
        key, action_key, step_key = jax.random.split(key, 3)
        action = driver(state, obs, params) if actions is None else actions(action_key, obs)
        next_obs, new, reward, done, info = step(step_key, state, action, params)
        yield obs, state, action, next_obs, new, reward, done, info
        obs, state = next_obs, new
        if bool(done):
            break
