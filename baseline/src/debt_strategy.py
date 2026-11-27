"""A fixed-duration interface for learning destinations under shared driving."""

import jax
import jax.numpy as jnp

from env import city
from env import gamblers_taxi as taxi
from debt_navigation import guarded_control, to_zone
import scenarios

WORK, SLEEP, APPLY, REPAY, BORROW, SLOTS, BLACKJACK, BUY, WAIT = range(9)
NAMES = ("work", "sleep", "apply", "repay", "borrow", "slots", "blackjack", "buy", "wait")
ZONES = jnp.array((city.WORK, city.HOME, city.JOBS, city.BANK, city.BANK,
                   city.SLOTS, city.BLACKJACK, city.GARAGE, city.HOME))
OPERATIONS = jnp.array((taxi.SHIFT, taxi.SLEEP, taxi.APPLY, taxi.REPAY, taxi.BORROW,
                        taxi.SPIN, taxi.DEAL, taxi.BUY, taxi.DRIVE))
STEPS = 12


def legal(state, params):
    c = params.config
    return jnp.array((
        state.employed, state.fatigue > 0.08, ~state.employed,
        (state.debt > 0) & (state.cash > 0), state.debt+c.loan_unit <= c.starting_debt+c.loan_limit,
        state.cash >= c.slot_stake, state.cash >= c.blackjack_stake,
        (state.debt == 0) & (state.cash >= c.buyout), True,
    ))


def features(state, params):
    c = params.config
    phase = state.time % c.day
    return jnp.concatenate((
        jnp.array([state.cash/c.starting_debt, state.debt/c.starting_debt,
                   state.fatigue, state.anxiety, state.tilt, state.mood, state.habit,
                   taxi._urge(state, params), state.time/c.horizon, phase/c.day,
                   state.employed, state.misses/2, state.shift_claimed,
                   jnp.clip((state.next_round-state.time)/c.day, 0, 1),
                   state.remaining/c.day], dtype=jnp.float32),
        taxi._distance(state.agent_pos, params)/30.,
        jax.nn.one_hot(state.activity, taxi.NUM_ACTIONS),
        legal(state, params).astype(jnp.float32),
    ))


def action(state, obs, params, choice, used):
    zone = ZONES[choice]
    operation = OPERATIONS[choice]
    open_hand = state.activity == taxi.DEAL
    direct = scenarios.scenario_action(state, obs, params, zone)
    ready = obs.action_mask[operation] & (operation != taxi.DRIVE) & ~used
    fallback = jnp.argmax(obs.action_mask).astype(jnp.int32)
    requested = jnp.where(open_hand, direct.choice, jnp.where(ready, operation, fallback))
    driving = (requested == taxi.DRIVE) & (choice != WAIT) & ~used
    control = jax.lax.cond(driving, lambda: to_zone(state, params, zone), lambda: jnp.array([-1., 0.]))
    return taxi.TaxiAction(control, requested), used | ready


def advance(key, state, choice, params, record=None):
    env = taxi.GamblersTaxiEnv
    c = params.config

    def tick(carry, index):
        current, total, used = carry
        obs = env.get_observation(current, params)
        cmd, used = action(current, obs, params, choice, used)
        _, next_state, reward, _, info = env.step(jax.random.fold_in(key, index), current, cmd, params)
        active = current.outcome == taxi.ACTIVE
        next_state = jax.tree.map(lambda a, b: jnp.where(active, a, b), next_state, current)
        row = record(current, next_state, obs, info) if record is not None else None
        return (next_state, total + jnp.where(active, c.gamma**index * reward, 0.), used), row

    (next_state, reward, _), rows = jax.lax.scan(
        tick, (state, jnp.float32(0), jnp.bool_(False)), jnp.arange(STEPS)
    )
    return (next_state, reward, rows) if record is not None else (next_state, reward)


def work_plan(state, params):
    c = params.config
    phase = state.time % c.day
    open_shift = (phase < c.shift_window) & ~state.shift_claimed & state.employed
    enough = taxi._balance(state) >= c.buyout
    bank = (state.debt > 0) & ((state.cash >= 3*c.salary) | enough)
    return jnp.where(enough & (state.debt == 0), BUY,
                     jnp.where(enough | (bank & ~open_shift), REPAY,
                               jnp.where(open_shift, WORK, jnp.where(
                                   state.fatigue > 0.22, SLEEP, jnp.where(state.employed, WAIT, APPLY)))))


def mixed_plan(state, params):
    c = params.config
    phase = state.time % c.day
    alternate = state.counters[taxi.PAID] % 2 == 0
    bet = jnp.where(alternate, BLACKJACK, SLOTS)
    play = (phase >= c.shift_window) & (phase < c.casino_close) & (state.fatigue < 0.65)
    funded = legal(state, params)[bet]
    need_loan = ~funded & legal(state, params)[BORROW] & (state.habit > 0.45)
    enough = taxi._balance(state) >= c.buyout
    return jnp.where(play & ~enough & (funded | need_loan), jnp.where(funded, bet, BORROW),
                     work_plan(state, params))
