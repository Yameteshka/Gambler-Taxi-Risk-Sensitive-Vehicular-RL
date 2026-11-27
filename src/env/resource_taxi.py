"""Driving, survival resources, individual loans, and reproducible market events."""

from dataclasses import dataclass, replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState
from env import cards, city


(DRIVE, WAIT, WORK, SLEEP, EAT, REFUEL_SMALL, REFUEL_FULL,
 BORROW_25, BORROW_50, BORROW_100, CAR_25, CAR_100, CAR_ALL,
 MICRO_25, MICRO_100, MICRO_ALL, SPIN_SMALL, SPIN_MEDIUM, SPIN_LARGE,
 DEAL_SMALL, DEAL_MEDIUM, DEAL_LARGE, HIT, STAND, BUY) = range(25)
ACTION_NAMES = ("Drive", "Wait", "Work", "Sleep", "Eat", "Refill 5", "Fill tank",
                "Borrow 25", "Borrow 50", "Borrow 100", "Car payment 25",
                "Car payment 100", "Pay car in full", "Microloan payment 25",
                "Microloan payment 100", "Pay microloans in full", "Slots 10",
                "Slots 25", "Slots 55", "Blackjack 10", "Blackjack 30",
                "Blackjack 90", "Hit", "Stand", "Complete ownership")
NUM_ACTIONS = len(ACTION_NAMES)
(PAID, MISSED, LOANS, CAR_PAYMENTS, MICRO_PAYMENTS, SPINS, HANDS, WINS,
 LOSSES, PUSHES, SLEEPS, MEALS, REFILLS, FEES) = range(14)
COUNTER_NAMES = ("paid", "missed", "loans", "car_payments", "micro_payments",
                 "spins", "hands", "wins", "losses", "pushes", "sleeps",
                 "meals", "refills", "fees")
ACTIVE, SUCCESS, CRASH, TIMEOUT, DEATH = range(5)
OUTCOME_NAMES = ("active", "ownership", "collision", "deadline", "starvation")
FOOD_LOW = (.04, .12, .55)
FOOD_HIGH = (.06, .18, .75)


@dataclass(frozen=True)
class Config:
    days: int = 10  # Zero denotes a continuing campaign.
    day: int = 155
    dt: float = .16
    gamma: float = .999
    failure_cost: float = .1
    starting_cash: float = 100.
    starting_debt: float = 1000.
    starting_fuel: float = 12.
    starting_satiety: float = .85
    loan_capacity: int = 528
    car_daily_rate: float = .005
    micro_daily_rate: float = .03
    tank: float = 25.
    fuel_distance: float = .035
    fuel_work: float = .09
    hunger_tick: float = .0018
    hunger_work: float = .0015
    hunger_drive: float = .0005
    hunger_push: float = .003
    push_speed: float = .2
    max_speed: float = 3.8
    wheelbase: float = .8
    max_steer: float = .58
    zone_radius: float = 1.55
    parking_speed: float = .2
    salary: float = 175.
    shift_duration: int = 28
    shift_window: int = 70
    sleep_duration: int = 24
    loan_duration: int = 3
    spin_duration: int = 9
    casino_open: int = 90
    casino_close: int = 145
    cooldown: int = 24
    slot_probability: float = .12
    slot_multiplier: float = 325./55.
    blackjack_commission: float = .10
    installments: bool = True
    market: str = "dynamic"
    meals: str = "reference_uniform"
    casino: bool = True
    control_strength: float = 1.
    actuation_noise: float = 0.
    food_swap: bool = False
    congestion: bool = True
    starting_microloan: float = 0.
    fuel_price_multiplier: float = 1.
    accessible_start: bool = False
    hidden_casino_regime: bool = False
    reduced_quotes: bool = False
    accelerated_schedule: bool = False
    school_level: int = 0
    financial_progress: bool = True
    navigation_cue: bool = True
    intermediate_starts: bool = False
    intermediate_home_fraction: float = 1.
    public_food_menu: bool = False  # Opt-in observation schema; old policies retain their inputs.
    exploration_epsilon: float = 0.
    ppo_native_exploration: bool = False
    exploration_starts: bool = False
    training_reset_level: int = -1  # Reset-only curriculum; -1 is the standard game.
    training_rehearsal: float = .2
    reward_shaping: bool = True
    predictive_driver: bool = False
    predictive_escape: bool = False
    ownership_reset_level: int = -1
    budget_reset_level: int = -1  # Opt-in food-planning curriculum; standard reset remains unchanged.
    standard_start_rehearsal: float = 0.
    resource_observation: bool = False
    observation_history: bool = False
    ppo_epochs: int = 3
    ppo_target_kl: float = 0.
    ppo_clip: float = .2
    ppo_value_head_lr_multiplier: float = 1.
    ppo_normalize_value_loss: bool = False
    ppo_exploration_decay: bool = False
    unavailable_wait_ticks: int = 1  # Historical interface; registered recovery can use longer feedback.

    @property
    def horizon(self):
        return self.days*self.day if self.days else 0


class Params(BaseEnvParams):
    config: Config = eqx.field(static=True)
    fields: jax.Array
    zone_centers: jax.Array


class State(BaseEnvState):
    casino_regime: jax.Array
    scenario_id: jax.Array
    speed: jax.Array
    steering: jax.Array
    cash: jax.Array
    car_debt: jax.Array
    car_paid: jax.Array
    fuel: jax.Array
    satiety: jax.Array
    fatigue: jax.Array
    activity: jax.Array
    remaining: jax.Array
    shift_claimed: jax.Array
    escrow: jax.Array
    service_price: jax.Array
    service_value: jax.Array
    service_venue: jax.Array
    next_round: jax.Array
    round_count: jax.Array
    meal_count: jax.Array
    hand: jax.Array
    dealer: jax.Array
    loan_ids: jax.Array
    loan_principal: jax.Array
    loan_balance: jax.Array
    loan_rates: jax.Array
    loan_opened: jax.Array
    interest: jax.Array
    fees: jax.Array
    borrowed: jax.Array
    fuel_spent: jax.Array
    food_spent: jax.Array
    wages: jax.Array
    gambling_profit: jax.Array
    push_ticks: jax.Array
    car_cleared_at: jax.Array
    all_cleared_at: jax.Array
    previous_action: jax.Array
    previous_event: jax.Array
    counters: jax.Array
    outcome: jax.Array


class Observation(BaseEnvObservation):
    features: jax.Array
    action_mask: jax.Array

    def as_vector(self):
        return self.features


class Action(NamedTuple):
    control: jax.Array
    choice: jax.Array


def event_key(state, stream, index):
    key = jax.random.fold_in(jax.random.key(37411), state.scenario_id.astype(jnp.uint32))
    return jax.random.fold_in(jax.random.fold_in(key, stream), jnp.asarray(index, dtype=jnp.uint32))


def micro_debt(state):
    return state.loan_balance.sum(axis=-1)


def net_worth(state):
    return state.cash+state.escrow-state.car_debt-micro_debt(state)


def public_food_menu(state, params):
    """Known terms by physical venue, excluding the next random portion draw.

    Columns: current price, raw portion lower/mean/upper, service ticks.
    Portions describe a distribution, not guaranteed usable satiety on arrival.
    """
    c = params.config
    low, high = jnp.array(FOOD_LOW), jnp.array(FOOD_HIGH)
    if c.meals == "meal_high_variance":
        low, high = jnp.array((.02,.06,.35)), jnp.array((.08,.24,.95))
    mean = (low+high)/2
    if c.meals == "meal_mean_only":
        low = high = mean = jnp.array((.05,.15,.65))
    duration = jnp.array((2.,5.,10.))
    if c.food_swap:
        permutation = jnp.array((0,2,1))
        low, mean, high, duration = (x[permutation] for x in (low,mean,high,duration))
    prices = jnp.where(c.reduced_quotes,0.,quotes(state,params)[0])
    return jnp.stack((prices,low,mean,high,duration),axis=-1)


def quotes(state, params):
    c = params.config
    u = jax.random.uniform(event_key(state, 1, state.time//c.day), (5,))
    food = jnp.array((4., 12., 30.))+u[:3]*jnp.array((2., 6., 10.))
    wage = c.salary*jnp.where(u[3] < .3, .75, jnp.where(u[3] < .7, 1., 1.35))
    fuel = 1.2*(.85+.40*u[4])
    if c.market == "fixed":
        food, wage, fuel = jnp.array((5., 15., 35.)), jnp.float32(c.salary*1.03), jnp.float32(1.26)
    if c.food_swap:
        food = food.at[1:].set(food[1:][::-1])
    return food, wage, fuel*c.fuel_price_multiplier


def distances(pos, params):
    pos = jnp.clip(pos, 0., jnp.array((city.WIDTH-1.001, city.HEIGHT-1.001)))
    x, y = pos.astype(jnp.int32)
    u, v = pos-jnp.array((x, y))
    f = params.fields
    return (1-v)*((1-u)*f[:, y, x]+u*f[:, y, x+1]) + v*((1-u)*f[:, y+1, x]+u*f[:, y+1, x+1])


def legal(state, params):
    c = params.config
    near = jnp.linalg.norm(params.zone_centers-state.agent_pos, axis=-1) <= c.zone_radius
    parked = near & (state.speed < c.parking_speed)
    free = (state.activity == DRIVE) & (state.outcome == ACTIVE)
    playing = (state.activity == DEAL_SMALL) & (state.remaining == 0) & (state.outcome == ACTIVE)
    busy = ~(free | playing) & (state.outcome == ACTIVE)
    food, _, fuel_price = quotes(state, params)
    food_here = jnp.any(parked[city.DINER:] & (state.cash >= food))
    casino = c.casino & (state.time % c.day >= c.casino_open) & (state.time % c.day < c.casino_close)
    casino &= state.time >= state.next_round
    return jnp.concatenate((
        jnp.array((free, free | busy | playing, free & parked[city.WORK] & ~state.shift_claimed &
                   (state.time % c.day < c.shift_window),
                   free & parked[city.HOME] & (state.fatigue > .01),
                   free & food_here & (state.satiety < 1.),
                   free & parked[city.FUEL] & (state.fuel < c.tank) &
                   (state.cash >= jnp.minimum(5., c.tank-state.fuel)*fuel_price),
                   free & parked[city.FUEL] & (state.fuel < c.tank) &
                   (state.cash >= (c.tank-state.fuel)*fuel_price))),
        jnp.repeat((free & parked[city.BANK])[None], 3),
        jnp.repeat((free & parked[city.BANK] & (state.cash > 0) & (state.car_debt > 0))[None], 3),
        jnp.repeat((free & parked[city.BANK] & (state.cash > 0) & (micro_debt(state) > 0))[None], 3),
        (free & parked[city.SLOTS] & casino) & (state.cash >= jnp.array((10., 25., 55.))),
        (free & parked[city.BLACKJACK] & casino) & (state.cash >= jnp.array((10., 30., 90.))),
        jnp.array((playing, playing, free & parked[city.GARAGE] &
                   (state.car_debt == 0) & (micro_debt(state) == 0))),
    ))


def drive(state, control, params):
    c = params.config
    throttle, turn = jnp.clip(control, -1., 1.)
    steering = state.steering+(.72-.36*state.fatigue)*c.control_strength*(turn*c.max_steer-state.steering)
    drag = jnp.where(state.speed > 0, .22+.055*state.speed**2, 0.)
    force = jnp.where(throttle >= 0, 2.9*throttle, 5.1*(1-.34*state.fatigue)*c.control_strength*throttle)
    powered = state.fuel > 0
    speed = jnp.clip(state.speed+c.dt*(force-drag), 0., c.max_speed)
    push = jnp.clip(state.speed+c.dt*(throttle*.8-drag), 0., c.push_speed)
    speed = jnp.where(powered, speed, push)
    # A signed public daily congestion notice corresponds to the actual slow region.
    congested = (state.time//c.day % 3 == 1) & (state.agent_pos[0] > 23) & (state.agent_pos[0] < 27)
    speed = jnp.where(c.congestion & congested, jnp.minimum(speed, 1.1), speed)
    mean = .5*(state.speed+speed)
    heading = jnp.arctan2(state.agent_forward_dir[1], state.agent_forward_dir[0])
    yaw = c.dt*mean*jnp.tan(steering)/c.wheelbase
    direction = jnp.array((jnp.cos(heading+yaw), jnp.sin(heading+yaw)))
    midpoint = jnp.array((jnp.cos(heading+yaw/2), jnp.sin(heading+yaw/2)))
    return state.agent_pos+c.dt*mean*midpoint, direction, speed, steering


def traffic_step(state, params):
    positions = state.kinematic_obstacles+params.config.dt*state.kinematic_obst_velocities
    low, high = jnp.asarray(city.TRAFFIC_LIMITS).T
    x = positions[:, 0]
    rebound = (x < low) | (x > high)
    x = jnp.where(x < low, 2*low-x, jnp.where(x > high, 2*high-x, x))
    return positions.at[:, 0].set(x), jnp.where(rebound[:, None], -state.kinematic_obst_velocities,
                                               state.kinematic_obst_velocities)


def collision(state, pos, traffic, traffic_margin=0.):
    def hits(old, new, radius=city.COLLISION_RADIUS):
        start, end = state.agent_pos-old, pos-new
        delta = end-start
        fraction = jnp.clip(-(start*delta).sum(axis=1)/jnp.maximum((delta**2).sum(axis=1), 1e-8), 0., 1.)
        return jnp.any(((start+fraction[:, None]*delta)**2).sum(axis=1) < radius**2)
    return (jnp.any(pos < city.BOUNDARY_MARGIN) |
            jnp.any(pos > jnp.array((city.WIDTH-1, city.HEIGHT-1))-city.BOUNDARY_MARGIN) |
            hits(state.static_obstacles, state.static_obstacles) |
            hits(state.kinematic_obstacles, traffic, city.COLLISION_RADIUS+traffic_margin))


def useful_targets(state, params):
    work_open = (state.time % params.config.day < params.config.shift_window) & ~state.shift_claimed
    debt = state.car_debt+micro_debt(state)
    due = (2, 4) if params.config.accelerated_schedule else (3, 6)
    next_target = params.config.starting_debt*jnp.where(state.time < due[0]*params.config.day, .3,
                   jnp.where(state.time < due[1]*params.config.day, .6, 1.))
    payment_gap = jnp.maximum(next_target-state.car_paid, 0.)
    bank_useful = (state.cash < 20.) | (params.config.installments & (payment_gap > 0) & (state.cash >= payment_gap+25.)) | (
        (micro_debt(state) > 0) & (state.cash >= micro_debt(state)+20.))
    useful = jnp.array((state.fatigue > .6, work_open, bank_useful, debt == 0,
                        False, False, state.fuel < 8., state.satiety < .5,
                        state.satiety < .5, state.satiety < .5))
    # Once settlement is affordable, supply dense progress toward the task's
    # financial and physical endpoint. The policy still chooses every action.
    settlement = jnp.zeros(10, dtype=jnp.bool_).at[
        jnp.where(debt == 0, city.GARAGE, city.BANK)].set(True)
    useful = jnp.where((debt == 0) | (state.cash >= debt+20.), settlement, useful)
    return useful


def wealth_utility(state):
    return .1*jnp.clip(net_worth(state)/2000., -1., 1.)


def potential(state, params):
    progress = jnp.clip((net_worth(state)+params.config.starting_debt) /
                        jnp.maximum(params.config.starting_debt, 100.), -.5, 1.5)
    distance = distances(state.agent_pos, params)
    useful = useful_targets(state, params)
    nearby = jnp.where(jnp.any(useful), jnp.min(jnp.where(useful, distance, 80.)), 0.)
    value = .4*progress+.2*state.satiety+.08*state.fuel/params.config.tank-.1*state.fatigue-.5*jnp.minimum(nearby, 40.)/40.
    return jnp.where(state.outcome == ACTIVE, value, 0.)


class ResourceTaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, config=Config()):
        params = Params(config.horizon or 2**30, round(1/config.dt), config.dt,
                        (city.HEIGHT, city.WIDTH), ((city.HEIGHT, city.WIDTH), jnp.float32),
                        ((0, 2), jnp.float32), 1, 16, float(np.pi), 6., config,
                        jnp.asarray(city.distance_fields(config.zone_radius)),
                        jnp.asarray(city.ZONES, dtype=jnp.float32))
        scenario = jax.random.randint(key, (), 11, 10011)
        return params, ResourceTaxiEnv.reset(scenario, params)[1]

    @staticmethod
    def reset(scenario_id, params, init_state=None):
        del init_state
        c = params.config
        zero, integer = jnp.float32(0.), jnp.int32(0)
        state = State(
            casino_regime=jnp.int32(0),
            time=integer, goal_pos=params.zone_centers[city.GARAGE],
            agent_pos=jnp.asarray(city.START, dtype=jnp.float32),
            agent_forward_dir=jnp.array((1., 0.)), static_obstacles=jnp.asarray(city.WALLS, dtype=jnp.float32),
            kinematic_obstacles=jnp.asarray(city.TRAFFIC, dtype=jnp.float32),
            kinematic_obst_velocities=jnp.asarray(city.TRAFFIC_VELOCITY, dtype=jnp.float32),
            path_array=jnp.empty((0, 2), dtype=jnp.float32), scenario_id=jnp.int32(scenario_id),
            speed=zero, steering=zero, cash=jnp.float32(c.starting_cash), car_debt=jnp.float32(c.starting_debt),
            car_paid=zero, fuel=jnp.float32(c.starting_fuel), satiety=jnp.float32(c.starting_satiety),
            fatigue=jnp.float32(.16), activity=jnp.int32(DRIVE), remaining=zero,
            shift_claimed=jnp.bool_(False), escrow=zero, service_price=zero, service_value=zero,
            service_venue=integer, next_round=integer, round_count=integer, meal_count=integer,
            hand=jnp.zeros(cards.MAX_CARDS, dtype=jnp.int32), dealer=jnp.zeros(cards.MAX_CARDS, dtype=jnp.int32),
            loan_ids=jnp.zeros(c.loan_capacity, dtype=jnp.int32),
            loan_principal=jnp.zeros(c.loan_capacity), loan_balance=jnp.zeros(c.loan_capacity),
            loan_rates=jnp.zeros(c.loan_capacity), loan_opened=jnp.zeros(c.loan_capacity, dtype=jnp.int32),
            interest=zero, fees=zero, borrowed=zero, fuel_spent=zero, food_spent=zero,
            wages=zero, gambling_profit=zero,
            push_ticks=integer, car_cleared_at=jnp.int32(-1), all_cleared_at=jnp.int32(-1),
            previous_action=jnp.int32(WAIT), previous_event=zero,
            counters=jnp.zeros(len(COUNTER_NAMES), dtype=jnp.int32), outcome=jnp.int32(ACTIVE),
        )
        jitter = jax.random.uniform(event_key(state, 2, integer), (len(city.TRAFFIC),), minval=-.5, maxval=.5)
        start_zone = jnp.asarray((city.HOME, city.WORK, city.BANK, city.GARAGE, city.SLOTS,
                                 city.FUEL, city.DINER, city.CAFE, city.RESTAURANT))[scenario_id % 9]
        state = replace(state, kinematic_obstacles=state.kinematic_obstacles.at[:, 0].add(jitter),
            casino_regime=jax.random.bernoulli(event_key(state, 40, 0)).astype(jnp.int32),
            agent_pos=jnp.where(c.accessible_start, params.zone_centers[start_zone], state.agent_pos),
            loan_ids=state.loan_ids.at[0].set(jnp.int32(c.starting_microloan > 0)),
            loan_principal=state.loan_principal.at[0].set(c.starting_microloan),
            loan_balance=state.loan_balance.at[0].set(c.starting_microloan),
            loan_rates=state.loan_rates.at[0].set(c.micro_daily_rate),
            borrowed=jnp.float32(c.starting_microloan),
            counters=state.counters.at[LOANS].set(jnp.int32(c.starting_microloan > 0)))
        if c.accessible_start:
            # Reachable near-goal states teach the same ownership task before
            # longer economic chains. No action labels or forced decisions.
            case = jnp.where(c.school_level < 2, 0, scenario_id % 3)
            offset = jax.random.uniform(event_key(state, 3, 0), (3,))
            # Retain short successful routes while expanding the support. A hard
            # jump to only longer routes caused catastrophic loss of the skill.
            upper = jnp.array((1.60, 3., 4.5, 6.))[c.school_level]
            radius = 1.56+(upper-1.56)*offset[0]**2
            angle_range = jnp.array((.15, .6, jnp.pi, jnp.pi))[c.school_level]
            angles = jax.random.uniform(event_key(state, 5, 0), (32,), minval=-angle_range, maxval=angle_range)
            approaches = params.zone_centers[city.GARAGE]-radius*jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
            obstacles = jnp.concatenate((state.static_obstacles, state.kinematic_obstacles))
            clear = jnp.linalg.norm(approaches[:, None]-obstacles[None], axis=-1).min(axis=1) > 2.2
            clear &= jnp.all((approaches > 1.2) & (approaches < jnp.array((city.WIDTH-2.2, city.HEIGHT-2.2))), axis=1)
            index = jnp.argmax(clear)
            angle = jnp.where(jnp.any(clear), angles[index], 0.)
            approach = params.zone_centers[city.GARAGE]-radius*jnp.array((jnp.cos(angle), jnp.sin(angle)))
            heading_error = jnp.array((.35, .5, 1., jnp.pi))[c.school_level]*(2*offset[2]-1)
            position = jnp.where(case == 0, approach,
                jnp.where(case == 1, params.zone_centers[city.BANK], params.zone_centers[city.WORK]))
            debt = jnp.where(case == 0, 0., jnp.where(case == 1, 25., c.starting_debt))
            state = replace(state, agent_pos=position, car_debt=debt,
                            car_paid=jnp.float32(c.starting_debt)-debt,
                            agent_forward_dir=jnp.where((case == 0),
                                jnp.array((jnp.cos(angle+heading_error), jnp.sin(angle+heading_error))),
                                state.agent_forward_dir))
        if c.intermediate_starts:
            # Gradually expand resource/financial starts back to the normal reset.
            draw = jax.random.uniform(event_key(state, 4, 0))
            service = jnp.asarray((city.WORK, city.DINER, city.CAFE, city.RESTAURANT,
                                   city.FUEL, city.BANK, city.SLOTS, city.BLACKJACK))[scenario_id % 8]
            at_home = draw < c.intermediate_home_fraction
            state = replace(state, agent_pos=jnp.where(at_home, jnp.asarray(city.START, dtype=jnp.float32),
                                                       params.zone_centers[service]),
                satiety=jnp.where(~at_home & (service >= city.DINER), .2, state.satiety),
                fuel=jnp.where(~at_home & (service == city.FUEL), 1., state.fuel),
                time=jnp.where(~at_home & ((service == city.SLOTS) | (service == city.BLACKJACK)),
                               jnp.int32(c.casino_open), state.time))
        if c.exploration_starts:
            # Balanced contexts expose mechanics, never prescribe an action.
            zone = scenario_id % 10
            settled = zone == city.GARAGE
            initial_micro = jnp.where((zone == city.BANK) & (scenario_id % 20 < 10),25.,0.)
            state = replace(state,agent_pos=params.zone_centers[zone],fatigue=jnp.float32(.8),
                satiety=jnp.where(zone>=city.DINER,.2,state.satiety),
                fuel=jnp.where(zone==city.FUEL,1.,state.fuel),
                time=jnp.where((zone==city.SLOTS)|(zone==city.BLACKJACK),jnp.int32(c.casino_open),state.time),
                car_debt=jnp.where(settled,0.,state.car_debt),car_paid=jnp.where(settled,c.starting_debt,0.),
                loan_ids=state.loan_ids.at[0].set(jnp.int32(initial_micro>0)),
                loan_balance=state.loan_balance.at[0].set(initial_micro),
                loan_principal=state.loan_principal.at[0].set(initial_micro),
                counters=state.counters.at[LOANS].set(jnp.int32(initial_micro>0)))
        if c.training_reset_level >= 0:
            from resource_curriculum import reset_context
            state = reset_context(state, params)
        if c.ownership_reset_level >= 0:
            from ownership_curriculum import reset_context
            state = reset_context(state, params)
        if c.budget_reset_level >= 0:
            from budget_curriculum import reset_context
            state = reset_context(state, params)
        return ResourceTaxiEnv.get_observation(state, params), state

    @staticmethod
    def get_observation(state, params):
        c = params.config
        pos, direction = state.agent_pos, state.agent_forward_dir
        left = jnp.array((-direction[1], direction[0]))
        obstacles = jnp.concatenate((state.static_obstacles, state.kinematic_obstacles))
        rays, lengths = BaseEnv._collision_ray_intersections(pos, direction, obstacles, params)
        boundary = jnp.where(rays > 0, jnp.array((city.WIDTH-1, city.HEIGHT-1))-pos, -pos)
        wall = jnp.min(jnp.where(jnp.abs(rays) > 1e-6, boundary/jnp.where(jnp.abs(rays) > 1e-6, rays, 1.), jnp.inf), axis=1)
        lengths = jnp.clip(jnp.minimum(lengths, wall), 0., params.perception_radius)
        distance = distances(pos, params)
        axes = jnp.array(((.65, 0), (0, .65)))
        slopes = (jax.vmap(lambda d: distances(pos-d, params))(axes) -
                  jax.vmap(lambda d: distances(pos+d, params))(axes)).T
        ego = params.zone_centers-pos
        bearings = jnp.stack((slopes@direction, slopes@left), axis=1)/3.
        useful = useful_targets(state, params)
        nearest = jnp.argmin(jnp.where(useful, distance, 80.))
        route_cue = jnp.concatenate((bearings[nearest], distance[nearest, None]/50.))
        route_cue = jnp.where(c.navigation_cue & jnp.any(useful), route_cue, 0.)
        food, wage, fuel = quotes(state, params)
        if c.reduced_quotes:
            food, wage, fuel = jnp.zeros_like(food), jnp.float32(0), jnp.float32(0)
        player, soft = cards.total(state.hand)
        playing = state.activity == DEAL_SMALL
        micro = micro_debt(state)
        active_loans = state.loan_balance > 0
        mean_rate = jnp.sum(jnp.where(active_loans, state.loan_rates, 0.))/jnp.maximum(active_loans.sum(), 1)
        features = jnp.concatenate((
            lengths/params.perception_radius,
            jnp.stack((ego@direction, ego@left), axis=1).ravel()/40.,
            distance/50., bearings.ravel(),
            jnp.array((state.speed/c.max_speed, state.steering/c.max_steer,
                       state.cash/1000., state.car_debt/1000., micro/1000.,
                       active_loans.sum()/10., mean_rate/.03, state.fuel/c.tank,
                       state.satiety, state.fatigue, state.time/(10*c.day),
                       state.time % c.day/c.day, float(c.days)/30.,
                       jnp.maximum(c.horizon-state.time, 0)/(10*c.day) if c.days else 1.,
                       state.remaining/c.day, state.shift_claimed,
                       jnp.clip((state.next_round-state.time)/c.day, 0, 1),
                       state.escrow/100., wage/200., fuel/2.,
                       jnp.where(playing, player/21., 0.), playing & soft,
                       jnp.where(playing, jnp.minimum(state.dealer[0], 10)/10., 0.),
                       state.car_paid/1000., c.car_daily_rate, c.micro_daily_rate,
                       (state.time//c.day % 3 == 1) & c.congestion,
                       state.previous_event/200., c.slot_probability, c.slot_multiplier/10.,
                       jnp.where(c.hidden_casino_regime & ~c.reduced_quotes,
                           jnp.where(jax.random.bernoulli(event_key(state, 41, state.time//c.day), .7),
                                     state.casino_regime, 1-state.casino_regime), .5))),
            food/50., jax.nn.one_hot(state.activity, NUM_ACTIONS),
            jax.nn.one_hot(state.previous_action, NUM_ACTIONS),
            route_cue,
        ), dtype=jnp.float32)
        if c.public_food_menu:
            menu = public_food_menu(state,params)
            menu = menu/jnp.array((50.,1.,1.,1.,10.))
            features = jnp.concatenate((features,menu.ravel()))
        return Observation(distance[city.GARAGE], slopes[city.GARAGE], rays*lengths[:, None],
                           features, legal(state, params))

    @staticmethod
    def step(key, state, action, params):
        """All random outcomes use scenario streams; key is reserved for policy-independent extensions."""
        del key
        c, old = params.config, state
        mask = legal(old, params)
        requested = jnp.clip(action.choice, 0, NUM_ACTIONS-1)
        choice = jnp.where(mask[requested], requested, jnp.where(mask[WAIT], WAIT, DRIVE))
        free = old.activity == DRIVE
        starts = free & (choice != DRIVE) & (choice != WAIT) & (choice != BUY)
        borrowing = (choice >= BORROW_25) & (choice <= BORROW_100)
        paying_car = (choice >= CAR_25) & (choice <= CAR_ALL)
        paying_micro = (choice >= MICRO_25) & (choice <= MICRO_ALL)
        spinning = (choice >= SPIN_SMALL) & (choice <= SPIN_LARGE)
        dealing = (choice >= DEAL_SMALL) & (choice <= DEAL_LARGE)
        eating = choice == EAT
        refilling = (choice == REFUEL_SMALL) | (choice == REFUEL_FULL)
        food_prices, wage, fuel_price = quotes(old, params)
        venue = jnp.argmin(jnp.linalg.norm(params.zone_centers[city.DINER:]-old.agent_pos, axis=1))
        food_venue = jnp.where(c.food_swap & (venue > 0), 3-venue, venue)
        meal_low, meal_high = jnp.array(FOOD_LOW), jnp.array(FOOD_HIGH)
        if c.meals == "meal_high_variance":
            meal_low, meal_high = jnp.array((.02, .06, .35)), jnp.array((.08, .24, .95))
        meal_draw = jax.random.uniform(event_key(old, 10+food_venue, old.meal_count))
        meal_value = meal_low[food_venue]+meal_draw*(meal_high[food_venue]-meal_low[food_venue])
        if c.meals == "meal_mean_only":
            meal_value = jnp.array((.05, .15, .65))[food_venue]
        volume = jnp.minimum(c.tank-old.fuel, jnp.where(choice == REFUEL_SMALL, 5., c.tank))
        service_price = jnp.where(eating, food_prices[venue], jnp.where(refilling, volume*fuel_price, 0.))
        service_value = jnp.where(eating, meal_value, jnp.where(refilling, volume, jnp.where(choice == WORK, wage, 0.)))
        loan_amount = jnp.array((25., 50., 100.))[jnp.clip(choice-BORROW_25, 0, 2)]
        car_amount = jnp.minimum(jnp.minimum(old.cash, old.car_debt),
                                jnp.array((25., 100., jnp.inf))[jnp.clip(choice-CAR_25, 0, 2)])
        micro_amount = jnp.minimum(jnp.minimum(old.cash, micro_debt(old)),
                                  jnp.array((25., 100., jnp.inf))[jnp.clip(choice-MICRO_25, 0, 2)])
        car_amount = jnp.where(paying_car, car_amount, 0.)
        micro_amount = jnp.where(paying_micro, micro_amount, 0.)
        free_slot = jnp.argmin(jnp.where(old.loan_balance == 0, jnp.arange(c.loan_capacity), c.loan_capacity))
        capacity_needed = borrowing & ~jnp.any(old.loan_balance == 0)
        order = jnp.lexsort((old.loan_ids, -old.loan_rates))
        balances = old.loan_balance[order]
        available = micro_amount-(jnp.cumsum(balances)-balances)
        repayments = jnp.zeros_like(balances).at[order].set(jnp.clip(available, 0., balances))
        repayments = jnp.where((micro_amount >= micro_debt(old)) & paying_micro, old.loan_balance, repayments)
        loan_balance = old.loan_balance-repayments
        loan_balance = loan_balance.at[free_slot].add(jnp.where(borrowing, loan_amount, 0.))
        ids = old.loan_ids.at[free_slot].set(jnp.where(borrowing, old.counters[LOANS]+1, old.loan_ids[free_slot]))
        principals = old.loan_principal.at[free_slot].set(jnp.where(borrowing, loan_amount, old.loan_principal[free_slot]))
        rates = old.loan_rates.at[free_slot].set(jnp.where(borrowing, c.micro_daily_rate, old.loan_rates[free_slot]))
        opened = old.loan_opened.at[free_slot].set(jnp.where(borrowing, old.time, old.loan_opened[free_slot]))
        stake = jnp.where(spinning, jnp.array((10., 25., 55.))[jnp.clip(choice-SPIN_SMALL, 0, 2)],
                          jnp.where(dealing, jnp.array((10., 30., 90.))[jnp.clip(choice-DEAL_SMALL, 0, 2)], 0.))
        duration = jnp.where(choice == WORK, c.shift_duration,
                    jnp.where(choice == SLEEP, c.sleep_duration,
                    jnp.where(eating, jnp.array((2, 5, 10))[food_venue],
                    jnp.where(refilling, 4, jnp.where(spinning, c.spin_duration,
                    jnp.where(borrowing | paying_car | paying_micro, c.loan_duration, 0))))))
        nonprincipal = jnp.maximum(old.car_debt-(c.starting_debt-old.car_paid), 0.)
        principal_paid = jnp.maximum(car_amount-nonprincipal, 0.)
        state = replace(old, cash=old.cash+jnp.where(borrowing, loan_amount, 0.)-car_amount-micro_amount-service_price-stake,
                        car_debt=old.car_debt-car_amount, car_paid=old.car_paid+principal_paid,
                        loan_ids=ids, loan_principal=principals, loan_balance=loan_balance,
                        loan_rates=rates, loan_opened=opened,
                        activity=jnp.where(starts, jnp.where(spinning, SPIN_SMALL,
                                            jnp.where(dealing, DEAL_SMALL, choice)), old.activity),
                        remaining=jnp.where(starts, duration.astype(jnp.float32), old.remaining),
                        escrow=old.escrow+stake,
                        service_price=jnp.where(starts, service_price, old.service_price),
                        service_value=jnp.where(starts, service_value, old.service_value),
                        service_venue=jnp.where(eating, venue, old.service_venue),
                        shift_claimed=old.shift_claimed | (choice == WORK))
        state = replace(state,car_paid=jnp.where(paying_car & (state.car_debt == 0),c.starting_debt,state.car_paid))
        driving = choice == DRIVE
        coasting = free & (choice == WAIT)
        moving = driving | coasting
        noise = c.actuation_noise*jax.random.normal(event_key(old, 24, old.time), (2,))
        pos, direction, speed, steering = drive(state, jnp.where(driving, action.control+noise, 0.), params)
        pos = jnp.where(moving, pos, old.agent_pos)
        direction = jnp.where(moving, direction, old.agent_forward_dir)
        motion = moving & (jnp.linalg.norm(pos-old.agent_pos) > 1e-6)
        traffic, velocity = traffic_step(state, params)
        crashed = collision(old, pos, traffic)
        pushing = driving & (old.fuel <= 0) & (speed > 0)
        working, sleeping = state.activity == WORK, state.activity == SLEEP
        progress = jnp.where(working & (old.fuel <= 0), .20, 1.)
        remaining = jnp.maximum(state.remaining-progress, 0.)
        finish = (state.remaining > 0) & (remaining == 0)
        hunger = c.hunger_tick+c.hunger_work*working+c.hunger_drive*motion+c.hunger_push*pushing
        depleted = state.satiety-hunger <= 0
        completed = finish & ~depleted & ~crashed
        earned, ate, filled = completed & working, completed & (state.activity == EAT), completed & (
            (state.activity == REFUEL_SMALL) | (state.activity == REFUEL_FULL))
        slept = completed & sleeping
        spin_finish = completed & (state.activity == SPIN_SMALL)
        probability = jnp.where(c.hidden_casino_regime, jnp.where(old.casino_regime == 1, .16, .08), c.slot_probability)
        spin_win = jax.random.bernoulli(event_key(old, 30, old.round_count), probability)
        spin_payout = jnp.where(spin_win, state.escrow*c.slot_multiplier, 0.)
        hand, dealer, card_finish, card_payout = cards.resolve(
            event_key(old, 31, old.round_count*cards.MAX_CARDS+jnp.count_nonzero(old.hand)),
            state.hand, state.dealer, dealing, choice == HIT, choice == STAND,
            state.escrow, c.blackjack_commission)
        card_action = dealing | (choice == HIT) | (choice == STAND)
        card_finish &= card_action & ~depleted & ~crashed
        hand, dealer = jnp.where(card_action, hand, old.hand), jnp.where(card_action, dealer, old.dealer)
        game_finish = spin_finish | card_finish
        payout = jnp.where(spin_finish, spin_payout, jnp.where(card_finish, card_payout, 0.))
        won, lost = game_finish & (payout > state.escrow), game_finish & (payout < state.escrow)
        tick = state.time+1
        due_days, targets = jnp.array((3, 6, 10)), c.starting_debt*jnp.array((.3, .6, 1.))
        if c.accelerated_schedule:
            due_days = jnp.array((2, 4, 5))
        fee = 10.*jnp.any((tick == due_days*c.day) & (state.car_paid+1e-4 < targets)) * c.installments*(state.car_debt > 0)
        car_interest = state.car_debt*((1+c.car_daily_rate)**(1/c.day)-1)
        loan_interest = state.loan_balance*((1+state.loan_rates)**(1/c.day)-1)
        fuel_used = c.fuel_distance*jnp.linalg.norm(pos-old.agent_pos)*~pushing+c.fuel_work*working
        satiety = jnp.clip(state.satiety-hunger+jnp.where(ate, state.service_value, 0.), 0., 1.)
        fatigue = jnp.clip(state.fatigue+.0008+.001*motion+.0035*working+.015*pushing-.032*sleeping, 0., 1.)
        success = (choice == BUY) & ~crashed & ~depleted
        deadline = (tick >= c.horizon) & (c.days > 0)
        outcome = jnp.where(crashed, CRASH, jnp.where(depleted, DEATH,
                  jnp.where(success, SUCCESS, jnp.where(deadline, TIMEOUT, ACTIVE))))
        counters = state.counters.at[PAID].add(earned).at[MISSED].add(
            (tick % c.day == c.shift_window) & ~state.shift_claimed)
        counters = counters.at[LOANS].add(borrowing).at[CAR_PAYMENTS].add(paying_car).at[MICRO_PAYMENTS].add(paying_micro)
        counters = counters.at[SPINS].add(spinning).at[HANDS].add(dealing).at[WINS].add(won).at[LOSSES].add(lost)
        counters = counters.at[PUSHES].add(game_finish & ~(won | lost)).at[SLEEPS].add(slept)
        counters = counters.at[MEALS].add(ate).at[REFILLS].add(filled).at[FEES].add(fee > 0)
        car_debt = state.car_debt+car_interest+fee
        loan_balance = state.loan_balance+loan_interest
        clear_car, clear_all = car_debt == 0, (car_debt+loan_balance.sum()) == 0
        new = replace(state, time=tick, agent_pos=pos, agent_forward_dir=direction,
                      casino_regime=jnp.where((tick % c.day == 0) & c.hidden_casino_regime &
                          jax.random.bernoulli(event_key(old, 42, tick//c.day), .1),
                          1-state.casino_regime, state.casino_regime),
                      speed=jnp.where(moving, speed, 0.), steering=jnp.where(moving, steering, 0.),
                      kinematic_obstacles=traffic, kinematic_obst_velocities=velocity,
                      cash=state.cash+jnp.where(earned, state.service_value, 0.)+payout,
                      car_debt=car_debt, loan_balance=loan_balance, fuel=jnp.clip(
                          state.fuel-fuel_used+jnp.where(filled, state.service_value, 0.), 0., c.tank),
                      satiety=satiety, fatigue=fatigue, remaining=remaining,
                      activity=jnp.where(finish | card_finish, DRIVE, state.activity),
                      shift_claimed=state.shift_claimed & (tick % c.day != 0),
                      escrow=jnp.where(game_finish | (outcome != ACTIVE), 0., state.escrow), hand=hand, dealer=dealer,
                      next_round=jnp.where(game_finish, tick+c.cooldown, state.next_round),
                      round_count=state.round_count+game_finish, meal_count=state.meal_count+eating,
                      interest=state.interest+car_interest+loan_interest.sum(), fees=state.fees+fee,
                      borrowed=state.borrowed+jnp.where(borrowing, loan_amount, 0.),
                      fuel_spent=state.fuel_spent+jnp.where(refilling, service_price, 0.),
                      food_spent=state.food_spent+jnp.where(eating, service_price, 0.),
                      wages=state.wages+jnp.where(earned, state.service_value, 0.),
                      gambling_profit=state.gambling_profit+jnp.where(game_finish, payout-state.escrow, 0.),
                      push_ticks=state.push_ticks+pushing, counters=counters, outcome=outcome,
                      car_cleared_at=jnp.where(clear_car & (state.car_cleared_at < 0), tick, state.car_cleared_at),
                      all_cleared_at=jnp.where(clear_all & (state.all_cleared_at < 0), tick, state.all_cleared_at),
                      previous_action=choice, previous_event=payout-state.escrow*game_finish+
                      jnp.where(earned, state.service_value, 0.))
        # An unsuccessful trajectory has base discounted return -failure_cost,
        # independently of how long it delays failure. Success still benefits
        # from earlier completion. Interest remains a real economic cost.
        failed = (outcome != ACTIVE) & (outcome != SUCCESS)
        running = c.failure_cost*(1-c.gamma)*(outcome == ACTIVE)
        external = success*(1+.05*jnp.clip(new.cash/500., 0., 1.))-c.failure_cost*failed-running
        financial = (wealth_utility(new)-wealth_utility(old))*c.financial_progress
        external += financial
        shaping = (c.gamma*potential(new, params)-potential(old, params)
                   if c.reward_shaping else jnp.float32(0.))
        execute = (old.outcome == ACTIVE) & ~capacity_needed
        new = jax.tree.map(lambda a, b: jnp.where(execute, a, b), new, old)
        info = {"requested": requested, "executed": choice,
                "slots_legal": mask[SPIN_SMALL], "blackjack_legal": mask[DEAL_SMALL],
                "external_reward": jnp.where(execute, external, 0.),
                "financial_reward": jnp.where(execute, financial, 0.),
                "applied_control": jnp.where(driving & execute, jnp.clip(action.control+noise, -1., 1.), 0.),
                "shaping": jnp.where(execute, shaping, 0.), "capacity_needed": capacity_needed,
                "events": jnp.where(execute, new.counters-old.counters, 0),
                "gambling_net": jnp.where(game_finish & execute, payout-state.escrow, 0.),
                "game_resolved": jnp.where(game_finish & execute, jnp.where(spin_finish, SPIN_SMALL, DEAL_SMALL), 0),
                "car_payment": jnp.where(execute, car_amount, 0.),
                "micro_payment": jnp.where(execute, micro_amount, 0.),
                "loan_opened": jnp.where(borrowing & execute, old.counters[LOANS]+1, 0),
                "loan_repayments": jnp.where(execute, repayments, 0.)}
        return ResourceTaxiEnv.get_observation(new, params), new, jnp.where(execute, external+shaping, 0.), new.outcome != ACTIVE, info
