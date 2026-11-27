"""JAX vehicle dynamics and the fixed-horizon taxi economy."""

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState
from env import city


(
    DRIVE, SHIFT, SLEEP, APPLY, BORROW, REPAY, SPIN, DEAL, HIT, STAND, BUY, WAIT,
) = range(12)
ACTION_NAMES = (
    "Drive", "Work shift", "Sleep", "Apply for work", "Borrow", "Repay", "Spin",
    "Deal", "Hit", "Stand", "Buy out taxi", "Wait",
)
NUM_ACTIONS = len(ACTION_NAMES)
(
    HIRED, FIRED, MISSED, PAID, LOANS, REPAYMENTS, SPINS, HANDS, WINS, LOSSES, PUSHES, SLEEPS,
    IMPULSES,
) = range(13)
COUNTER_NAMES = (
    "hired", "fired", "missed", "paid", "loans", "repayments", "spins", "hands",
    "wins", "losses", "pushes", "sleeps", "impulses",
)
ACTIVE, SUCCESS, CRASH, TIMEOUT = range(4)
OUTCOME_NAMES = ("active", "success", "crash", "timeout")
MAX_CARDS = 22
DEPENDENCE_FEATURES = 3
ANXIETY_FEATURES = 1
ECONOMY_FEATURES = 2
DEBT_CAMPAIGN_FEATURES = 6


@dataclass(frozen=True)
class TaxiConfig:
    horizon: int = 1550
    day: int = 155
    shift_window: int = 70
    shift_duration: int = 28
    sleep_duration: int = 24
    job_duration: int = 22
    spin_duration: int = 9
    loan_duration: int = 5
    starting_cash: float = 18.
    late_start_probability: float = 0.
    salary: float = 145.
    buyout: float = 115.
    loan_unit: float = 60.
    loan_limit: float = 300.
    interest_per_tick: float = 0.00012
    slot_stake: float = 55.
    slot_payout: float = 325.
    slot_probability: float = 0.12
    blackjack_stake: float = 90.
    blackjack_win_commission: float = 0.10
    zone_radius: float = 1.55
    parking_speed: float = 0.2
    dt: float = 0.16
    max_speed: float = 3.8
    wheelbase: float = 0.8
    max_steer: float = 0.58
    gamma: float = 0.9985
    enable_emotion_effects: bool = True
    mood_traction: float = 0.18
    tilt_steering: float = 0.22
    enable_dependence: bool = True
    anxiety_steering: float = 0.42
    operating_cost: float = 8.
    casino_open: int = 90
    casino_close: int = 145
    casino_round_interval: int = 24
    timeout_shortfall_penalty: float = 0.25
    starting_debt: float = 1000.
    debt_time_weight: float = 0.65
    enable_impulse_override: bool = True


class TaxiParams(BaseEnvParams):
    config: TaxiConfig = eqx.field(static=True)
    fields: jax.Array
    zone_centers: jax.Array


class TaxiState(BaseEnvState):
    speed: jax.Array
    steering: jax.Array
    cash: jax.Array
    debt: jax.Array
    stake: jax.Array
    fatigue: jax.Array
    mood: jax.Array
    tilt: jax.Array
    anxiety: jax.Array
    habit: jax.Array
    abstinence: jax.Array
    impulse_checked: jax.Array
    next_round: jax.Array
    debt_free_at: jax.Array
    employed: jax.Array
    misses: jax.Array
    shift_claimed: jax.Array
    activity: jax.Array
    remaining: jax.Array
    hand: jax.Array
    dealer: jax.Array
    outcome: jax.Array
    counters: jax.Array


class TaxiObservation(BaseEnvObservation):
    features: jax.Array
    action_mask: jax.Array

    def as_vector(self):
        return self.features


class TaxiAction(NamedTuple):
    control: jax.Array
    choice: jax.Array


def _hand_total(hand):
    hard = jnp.minimum(hand, 10).sum(axis=-1)
    soft = jnp.any(hand == 1, axis=-1) & (hard <= 11)
    return hard + 10 * soft, soft


def _draw(key):
    return jax.random.randint(key, (MAX_CARDS,), 1, 14, dtype=jnp.int32)


def _dealer_hand(hand, ranks):
    def add_card(i, cards):
        total, _ = _hand_total(cards)
        return cards.at[i].set(jnp.where(total < 17, ranks[i], 0))

    return jax.lax.fori_loop(2, MAX_CARDS, add_card, hand)


def _balance(state):
    return state.cash + state.stake - state.debt


def _phase(state, config):
    return state.time % config.day


def _near_zones(state, params):
    return jnp.linalg.norm(params.zone_centers - state.agent_pos, axis=-1) <= params.config.zone_radius


def _legal_actions(state, params):
    c = params.config
    parked = (jnp.abs(state.speed) < c.parking_speed) & _near_zones(state, params)
    free = (state.activity == DRIVE) & (state.outcome == ACTIVE)
    blackjack = (state.activity == DEAL) & (state.remaining == 0) & (state.outcome == ACTIVE)
    can_work = state.employed & ~state.shift_claimed & (_phase(state, c) < c.shift_window)
    phase = _phase(state, c)
    casino_open = (phase >= c.casino_open) & (phase < c.casino_close)
    can_gamble = casino_open & (state.time >= state.next_round)
    due = c.operating_cost * ((state.time + 1) % c.day == 0)
    return jnp.array((
        free,
        free & parked[city.WORK] & can_work,
        free & parked[city.HOME] & (state.fatigue > 0.08),
        free & parked[city.JOBS] & ~state.employed,
        free & parked[city.BANK] & (state.debt + c.loan_unit <= c.starting_debt + c.loan_limit),
        free & parked[city.BANK] & (state.debt > 0.) & (state.cash > 0.),
        free & parked[city.SLOTS] & can_gamble & (state.cash >= c.slot_stake),
        free & parked[city.BLACKJACK] & can_gamble & (state.cash >= c.blackjack_stake),
        blackjack,
        blackjack,
        free & parked[city.GARAGE] & (state.debt == 0.) & (state.cash - due >= c.buyout),
        ~(free | blackjack),
    ))


def _urge(state, params):
    near_casino = jnp.any(_near_zones(state, params)[city.SLOTS:city.BLACKJACK + 1])
    withdrawal = jnp.clip((state.abstinence - 12.) / 75., 0., 1.)
    return state.habit * (0.18 + 0.35*withdrawal + 0.30*near_casino + 0.17*state.tilt)


def _impulse(key, state, choice, control, mask, params):
    leaving = (choice == DRIVE) & (control[0] > 0.10) & (mask[SPIN] | mask[DEAL])
    attempt = leaving & ~state.impulse_checked
    probability = 0.65*jnp.clip((_urge(state, params) - 0.15)/0.65, 0., 1.)
    # A separate subkey preserves the payout RNG for paired comparisons.
    slip = attempt & jax.random.bernoulli(jax.random.fold_in(key, 83), probability)
    gamble = jnp.where(mask[SPIN], SPIN, DEAL)
    return jnp.where(slip, gamble, choice), attempt, slip


def _distance(pos, params):
    pos = jnp.clip(pos, 0., jnp.array((city.WIDTH - 1.001, city.HEIGHT - 1.001)))
    x, y = pos.astype(jnp.int32)
    u, v = pos - jnp.array((x, y))
    a, b = params.fields[:, y, x], params.fields[:, y, x + 1]
    d, e = params.fields[:, y + 1, x], params.fields[:, y + 1, x + 1]
    return (1-v)*((1-u)*a + u*b) + v*((1-u)*d + u*e)


def _potential(state, params, distances):
    config = params.config
    money = jnp.clip((_balance(state) + config.starting_debt)
                     / (config.starting_debt + config.buyout), -0.5, 1.)
    shift_open = (_phase(state, config) < config.shift_window) & ~state.shift_claimed
    useful = jnp.array((
        state.employed,
        state.fatigue >= 0.32,
        ~state.employed,
        True,
        True,
        True,
        False,
    ))
    # A closed shift is farther away in time, even if the car is at its door.
    costs = distances.at[city.WORK].add(jnp.where(shift_open, 0., 4.))
    settlement_distance = jnp.where(state.debt > 0., distances[city.BANK], distances[city.GARAGE])
    distance = jnp.where(money >= 1., settlement_distance,
                         jnp.min(jnp.where(useful, costs, city.WIDTH + city.HEIGHT)))
    return jnp.where(state.outcome == ACTIVE, 0.45*money - 0.20*jnp.clip(distance/18., 0., 1.), 0.)


def _drive(state, control, params):
    config = params.config
    throttle, turn = jnp.clip(control, -1., 1.)
    mood, tilt = (state.mood, state.tilt) if config.enable_emotion_effects else (0., 0.)
    anxiety = state.anxiety if config.enable_emotion_effects else 0.
    steer_limit = config.max_steer * (1. + config.tilt_steering * tilt)
    desired = turn * steer_limit
    response = (0.72 - 0.36*state.fatigue) * (1. - config.anxiety_steering*anxiety)
    steering = state.steering + response*(desired - state.steering)
    traction = 2.9 * (1. + config.mood_traction * jnp.minimum(mood, 0.))
    brake = 5.1 * (1. - 0.34*state.fatigue)
    resistance = jnp.where(state.speed > 0., 0.22 + 0.055*state.speed**2, 0.)
    acceleration = jnp.where(throttle >= 0., throttle*traction, throttle*brake) - resistance
    speed = jnp.clip(state.speed + config.dt*acceleration, 0., config.max_speed)
    mean_speed = 0.5*(state.speed + speed)
    yaw = config.dt * mean_speed * jnp.tan(steering) / config.wheelbase
    heading = jnp.arctan2(state.agent_forward_dir[1], state.agent_forward_dir[0])
    forward = jnp.array((jnp.cos(heading + yaw), jnp.sin(heading + yaw)))
    midway = jnp.array((jnp.cos(heading + yaw/2), jnp.sin(heading + yaw/2)))
    return state.agent_pos + config.dt*mean_speed*midway, forward, speed, steering


def _traffic_step(state, params):
    limits = jnp.asarray(city.TRAFFIC_LIMITS)
    velocity = state.kinematic_obst_velocities
    positions = state.kinematic_obstacles + params.config.dt * velocity
    x = positions[:, 0]
    low, high = limits[:, 0], limits[:, 1]
    rebound = (x < low) | (x > high)
    reflected = jnp.where(x < low, 2*low - x, jnp.where(x > high, 2*high - x, x))
    return positions.at[:, 0].set(reflected), jnp.where(rebound[:, None], -velocity, velocity)


def _collision(state, pos, traffic):
    def intersects(old_obstacle, new_obstacle):
        start = state.agent_pos - old_obstacle
        end = pos - new_obstacle
        delta = end - start
        along = jnp.clip(-jnp.sum(start * delta, axis=1) /
                         jnp.maximum(jnp.sum(delta**2, axis=1), 1e-8), 0., 1.)
        return jnp.any(jnp.sum((start + along[:, None]*delta)**2, axis=1) < city.COLLISION_RADIUS**2)

    margin = city.BOUNDARY_MARGIN
    outside = jnp.any((pos < margin) | (pos > jnp.array((city.WIDTH - 1, city.HEIGHT - 1)) - margin))
    return outside | intersects(state.static_obstacles, state.static_obstacles) | intersects(
        state.kinematic_obstacles, traffic
    )


def _begin(state, choice, params):
    c = params.config
    durations = jnp.array((0, c.shift_duration, c.sleep_duration, c.job_duration, c.loan_duration,
                          c.loan_duration, c.spin_duration, 0, 0, 0, 0, 0), dtype=jnp.int32)
    starts = (choice >= SHIFT) & (choice <= DEAL)
    deposit = jnp.where(choice == SPIN, c.slot_stake, jnp.where(choice == DEAL, c.blackjack_stake, 0.))
    repayment = jnp.minimum(state.debt, state.cash)
    principal = jnp.where(choice == BORROW, c.loan_unit, jnp.where(choice == REPAY, -repayment, 0.))
    return eqx.tree_at(
        lambda s: (s.activity, s.remaining, s.cash, s.debt, s.stake, s.shift_claimed, s.counters),
        state,
        (jnp.where(starts, choice, state.activity), jnp.where(starts, durations[choice], state.remaining),
         state.cash + principal - deposit, state.debt + principal, state.stake + deposit,
         state.shift_claimed | (choice == SHIFT),
         state.counters.at[LOANS].add(choice == BORROW).at[REPAYMENTS].add(choice == REPAY)
         .at[SPINS].add(choice == SPIN).at[HANDS].add(choice == DEAL)),
    )


def _resolve_cards(key, state, choice, commission):
    key, dealer_key = jax.random.split(key)
    ranks, dealer_ranks = _draw(key), _draw(dealer_key)
    hand = jnp.where(choice == DEAL, jnp.zeros_like(state.hand).at[:2].set(ranks[:2]), state.hand)
    dealer = jnp.where(choice == DEAL, jnp.zeros_like(state.dealer).at[:2].set(dealer_ranks[:2]),
                       state.dealer)
    count = jnp.minimum(jnp.count_nonzero(hand), MAX_CARDS - 1)
    hand = hand.at[count].set(jnp.where(choice == HIT, ranks[count], hand[count]))
    player, _ = _hand_total(hand)
    natural = (choice == DEAL) & (player == 21)
    dealer_natural = (choice == DEAL) & (_hand_total(dealer)[0] == 21)
    finish = (choice == STAND) | ((choice == HIT) & (player >= 21)) | natural | dealer_natural
    dealer_total, _ = _hand_total(_dealer_hand(dealer, dealer_ranks))
    win = (player <= 21) & ((dealer_total > 21) | (player > dealer_total) | (natural & ~dealer_natural))
    win &= ~dealer_natural
    push = (player <= 21) & ((dealer_total == player) | (natural & dealer_natural)) & (
        ~dealer_natural | natural
    )
    profit = jnp.where(natural, 1.5, 1.) * (1. - commission)
    multiplier = jnp.where(win, 1. + profit, jnp.where(push, 1., 0.))
    return hand, dealer, finish, multiplier * state.stake


class GamblersTaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, config=TaxiConfig()):
        c = config
        fields = city.distance_fields(c.zone_radius)
        params = TaxiParams(
            max_steps_in_episode=c.horizon, fps=round(1/c.dt), step_size=c.dt,
            map_height_width=(city.HEIGHT, city.WIDTH),
            nav_grid_shape_dtype=((city.HEIGHT, city.WIDTH), jnp.float32),
            path_array_shape_dtype=((0, 2), jnp.float32), discretization_scale=1,
            num_ray_sensors=16, agent_FOV=float(np.pi), perception_radius=6.,
            config=c, fields=jnp.asarray(fields), zone_centers=jnp.asarray(city.ZONES, dtype=jnp.float32),
        )
        _, init_state = GamblersTaxiEnv.reset(key, params)
        return params, init_state

    @staticmethod
    def reset(key, env_params, init_state=None):
        params = env_params
        c = params.config
        del init_state
        k_traffic, k_cash, k_time = jax.random.split(key, 3)
        traffic = jnp.asarray(city.TRAFFIC, dtype=jnp.float32)
        traffic = traffic.at[:, 0].add(jax.random.uniform(k_traffic, (len(city.TRAFFIC),), minval=-0.6, maxval=0.6))
        cash = c.starting_cash + jax.random.uniform(k_cash, (), minval=-10., maxval=10.)
        tick = jax.random.randint(k_time, (), 0, 95, dtype=jnp.int32)
        if c.late_start_probability > 0:
            k_late, k_offset = jax.random.split(jax.random.fold_in(key, 51))
            late = jax.random.bernoulli(k_late, c.late_start_probability)
            tick = jnp.where(late, jax.random.randint(k_offset, (), 228, 268, dtype=jnp.int32), tick)
            cash = jnp.where(late, jnp.maximum(cash, c.slot_stake), cash)
        zero = jnp.float32(0.)
        state = TaxiState(
            time=tick, goal_pos=params.zone_centers[city.GARAGE], agent_pos=jnp.asarray(city.START, dtype=jnp.float32),
            agent_forward_dir=jnp.array((1., 0.)), static_obstacles=jnp.asarray(city.WALLS, dtype=jnp.float32),
            kinematic_obstacles=traffic, kinematic_obst_velocities=jnp.asarray(city.TRAFFIC_VELOCITY, dtype=jnp.float32),
            path_array=jnp.empty((0, 2), dtype=jnp.float32), speed=zero, steering=zero,
            cash=jnp.maximum(cash, 0.), debt=jnp.float32(c.starting_debt), stake=zero,
            fatigue=jnp.float32(0.16), mood=zero, tilt=zero,
            anxiety=jnp.float32(0.35 if c.starting_debt else 0.), habit=zero, abstinence=jnp.int32(0),
            impulse_checked=jnp.bool_(False), next_round=jnp.int32(0), debt_free_at=jnp.int32(-1),
            employed=jnp.bool_(True), misses=jnp.int32(0),
            shift_claimed=jnp.bool_(False), activity=jnp.int32(DRIVE), remaining=jnp.int32(0),
            hand=jnp.zeros((MAX_CARDS,), dtype=jnp.int32), dealer=jnp.zeros((MAX_CARDS,), dtype=jnp.int32),
            outcome=jnp.int32(ACTIVE), counters=jnp.zeros((len(COUNTER_NAMES),), dtype=jnp.int32),
        )
        return GamblersTaxiEnv.get_observation(state, params), state

    @staticmethod
    def get_observation(env_state, env_params):
        state, params = env_state, env_params
        c = params.config
        pos = state.agent_pos
        direction = state.agent_forward_dir
        left = jnp.array((-direction[1], direction[0]))
        obstacles = jnp.concatenate((state.static_obstacles, state.kinematic_obstacles))
        rays, lengths = BaseEnv._collision_ray_intersections(pos, direction, obstacles, params)
        boundary = jnp.where(rays > 0, jnp.array((city.WIDTH - 1, city.HEIGHT - 1)) - pos, -pos)
        wall_distances = jnp.min(jnp.where(jnp.abs(rays) > 1e-6, boundary / jnp.where(
            jnp.abs(rays) > 1e-6, rays, 1.), jnp.inf), axis=1)
        lengths = jnp.clip(jnp.minimum(lengths, wall_distances), 0., params.perception_radius)
        offsets = params.zone_centers - pos
        ego = jnp.stack((offsets @ direction, offsets @ left), axis=1)
        distances = _distance(pos, params)
        axes = jnp.array(((0.65, 0.), (0., 0.65)))
        slopes = (jax.vmap(lambda delta: _distance(pos-delta, params))(axes)
                  - jax.vmap(lambda delta: _distance(pos+delta, params))(axes)).T
        bearings = jnp.stack((slopes @ direction, slopes @ left), axis=1) / 3.
        total, soft = _hand_total(state.hand)
        dealing = state.activity == DEAL
        features = jnp.concatenate((
            lengths / params.perception_radius, ego.ravel()/25., distances/30., bearings.ravel(),
            jnp.array((
                state.speed/c.max_speed, state.steering/c.max_steer,
                state.cash/max(c.buyout, c.starting_debt), state.debt/max(c.buyout, c.starting_debt),
                state.stake/max(c.buyout, c.starting_debt),
                state.fatigue, state.mood, state.tilt, state.employed, state.misses/2.,
                state.shift_claimed, _phase(state, c)/c.day,
                jnp.maximum(c.shift_window - _phase(state, c), 0)/c.day,
                (c.horizon-state.time)/c.horizon, state.remaining/c.day,
                jnp.where(dealing, total/21., 0.), dealing & soft,
                jnp.where(dealing, jnp.minimum(state.dealer[0], 10)/10., 0.),
            )),
            jax.nn.one_hot(state.activity, NUM_ACTIONS),
            jnp.array((state.habit, _urge(state, params), state.impulse_checked), dtype=jnp.float32)
            if c.enable_dependence else jnp.zeros((DEPENDENCE_FEATURES,), dtype=jnp.float32),
            state.anxiety[None],
            jnp.array((jnp.clip((state.next_round-state.time)/c.day, 0., 1.),
                       (_phase(state, c) >= c.casino_open) & (_phase(state, c) < c.casino_close)),
                      dtype=jnp.float32),
            jnp.array((
                state.debt / max(c.starting_debt, c.loan_limit),
                c.salary / max(c.starting_debt, c.buyout), c.shift_duration / c.day,
                (c.slot_payout-c.slot_stake) / max(c.starting_debt, c.buyout), c.slot_probability,
                1.5*(1.-c.blackjack_win_commission)*c.blackjack_stake / max(c.starting_debt, c.buyout),
            ), dtype=jnp.float32),
        ), dtype=jnp.float32)
        return TaxiObservation(distances[city.GARAGE], slopes[city.GARAGE],
                               rays * lengths[:, None], features, _legal_actions(state, params))

    @staticmethod
    def step(key, env_state, action, env_params):
        old, params = env_state, env_params
        c = params.config
        mask = _legal_actions(old, params)
        requested = jnp.clip(action.choice, 0, NUM_ACTIONS - 1)
        choice = jnp.where(mask[requested], requested, jnp.argmax(mask).astype(jnp.int32))
        if c.enable_dependence and c.enable_impulse_override:
            choice, attempt, slip = _impulse(key, old, choice, action.control, mask, params)
        else:
            attempt = slip = jnp.bool_(False)
        state = _begin(old, choice, params)
        pos, forward, speed, steer = _drive(state, action.control, params)
        driving = choice == DRIVE
        pos = jnp.where(driving, pos, state.agent_pos)
        forward = jnp.where(driving, forward, state.agent_forward_dir)
        traffic, velocities = _traffic_step(state, params)
        crash = _collision(old, pos, traffic)
        tick = state.time + 1
        remaining = jnp.maximum(state.remaining - 1, 0)
        finish = (state.remaining > 0) & (remaining == 0)
        earned = finish & (state.activity == SHIFT) & ~crash
        hired = finish & (state.activity == APPLY) & ~crash
        slept = finish & (state.activity == SLEEP) & ~crash
        missed = (tick % c.day == c.shift_window) & state.employed & ~state.shift_claimed
        misses = jnp.where(earned | hired, 0, state.misses + missed)
        fired = missed & (misses >= 2)
        employed = (state.employed & ~fired) | hired
        spin_key, cards_key = jax.random.split(key)
        spin_finish = finish & (state.activity == SPIN)
        spin_payout = jnp.where(jax.random.bernoulli(spin_key, c.slot_probability), c.slot_payout, 0.)
        cards = (choice == DEAL) | (choice == HIT) | (choice == STAND)
        hand, dealer, card_finish, card_payout = jax.lax.cond(
            cards, lambda: _resolve_cards(cards_key, state, choice, c.blackjack_win_commission),
            lambda: (state.hand, state.dealer, jnp.bool_(False), jnp.float32(0.)),
        )
        gamble_finish = (spin_finish | card_finish) & ~crash
        payout = jnp.where(gamble_finish, jnp.where(spin_finish, spin_payout, card_payout), 0.)
        win = gamble_finish & (payout > state.stake)
        lose = gamble_finish & (payout < state.stake)
        push = gamble_finish & ~(win | lose)
        daily_cost = c.operating_cost * (tick % c.day == 0)
        cash = state.cash + c.salary*earned + payout - daily_cost
        debt = state.debt * (1. + c.interest_per_tick)
        stake = jnp.where(gamble_finish, 0., state.stake)
        working = state.activity == SHIFT
        sleeping = state.activity == SLEEP
        gambling = (state.activity == SPIN) | (state.activity == DEAL)
        night = tick % c.day >= c.shift_window
        fatigue = jnp.clip(state.fatigue + 0.0009 + 0.001*driving + 0.0035*working
                           + 0.005*gambling + 0.004*gambling*night - 0.032*sleeping, 0., 1.)
        mood = jnp.clip(0.995*state.mood + 0.19*win - 0.23*lose - 0.24*fired
                        + 0.12*slept - 0.0015*working - 0.0008*(debt > 0.), -1., 1.)
        tilt = jnp.clip(0.985*state.tilt + 0.12*lose + 0.06*win - 0.20*slept, 0., 1.)
        debt_ratio = jnp.clip(debt / max(c.starting_debt, c.loan_limit), 0., 1.)
        pressure = jnp.where(debt > 0., 0.18 + 0.57*debt_ratio, 0.)
        relief = jnp.where(choice == REPAY, 0.26*(old.debt-state.debt)/jnp.maximum(old.debt, 1e-6), 0.)
        anxiety = jnp.clip(0.975*state.anxiety + 0.025*pressure
                           + 0.35*(choice == BORROW) + 0.08*(lose & (debt > 0.))
                           - relief, 0., 1.)
        if c.enable_dependence:
            increment = 0.14*gamble_finish + 0.10*win + 0.04*lose
            decay = jnp.where(sleeping, 0.992, 0.9991)
            habit = jnp.clip(decay*state.habit + (1.-state.habit)*increment, 0., 1.)
            abstinence = jnp.where(gambling, 0, state.abstinence + 1)
            near = jnp.any(_near_zones(state, params)[city.SLOTS:city.BLACKJACK + 1])
            impulse_checked = (state.impulse_checked | attempt) & near & ~gamble_finish
        else:
            habit, abstinence, impulse_checked = state.habit, state.abstinence, state.impulse_checked
        success = (choice == BUY) & ~crash & (cash >= c.buyout)
        outcome = jnp.where(crash, CRASH, jnp.where(success, SUCCESS, jnp.where(tick >= c.horizon, TIMEOUT, ACTIVE)))
        cash = jnp.where(success, cash - c.buyout, cash)
        counters = state.counters.at[HIRED].add(hired).at[FIRED].add(fired).at[MISSED].add(missed)
        counters = counters.at[PAID].add(earned).at[WINS].add(win).at[LOSSES].add(lose)
        counters = counters.at[PUSHES].add(push).at[SLEEPS].add(slept)
        counters = counters.at[IMPULSES].add(slip)
        new = TaxiState(
            time=tick, goal_pos=state.goal_pos, agent_pos=pos, agent_forward_dir=forward,
            static_obstacles=state.static_obstacles, kinematic_obstacles=traffic,
            kinematic_obst_velocities=velocities, path_array=state.path_array,
            speed=jnp.where(driving, speed, 0.), steering=jnp.where(driving, steer, 0.),
            cash=cash, debt=debt, stake=stake, fatigue=fatigue, mood=mood, tilt=tilt, anxiety=anxiety,
            habit=habit, abstinence=abstinence, impulse_checked=impulse_checked,
            next_round=jnp.where(gamble_finish, tick + c.casino_round_interval, state.next_round),
            debt_free_at=jnp.where(debt > 0., -1, jnp.where(
                (old.debt > 0.) & (choice == REPAY), tick, state.debt_free_at
            )),
            employed=employed, misses=misses, shift_claimed=state.shift_claimed & (tick % c.day != 0),
            activity=jnp.where(finish | card_finish, DRIVE, state.activity), remaining=remaining,
            hand=hand, dealer=dealer, outcome=outcome, counters=counters,
        )
        done = new.outcome != ACTIVE
        shortfall = jnp.clip((c.buyout - _balance(new))/(c.buyout+c.starting_debt), 0., 1.)
        debt_time_cost = (c.debt_time_weight/c.horizon) * jnp.minimum(
            debt / max(c.starting_debt, c.loan_limit), 1.25
        )
        base_reward = (jnp.float32(success) - jnp.float32(crash)
                       - c.timeout_shortfall_penalty*(outcome == TIMEOUT)*shortfall - debt_time_cost)
        shaping = c.gamma*_potential(new, params, _distance(pos, params)) - _potential(
            old, params, _distance(old.agent_pos, params)
        )
        obs = GamblersTaxiEnv.get_observation(new, params)
        info = {"outcome": outcome, "base_reward": base_reward, "shaping": shaping,
                "choice": choice, "requested_choice": requested, "net_worth": _balance(new) + c.buyout*success,
                "events": counters - old.counters, "daily_cost": daily_cost, "debt_time_cost": debt_time_cost,
                "gambling_net": jnp.where(gamble_finish, payout-state.stake, 0.),
                "game_resolved": jnp.where(gamble_finish, jnp.where(spin_finish, SPIN, DEAL), 0)}
        return obs, new, base_reward + shaping, done, info
