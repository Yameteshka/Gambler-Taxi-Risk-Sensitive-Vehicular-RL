"""Learned service intentions executed by a shared physical driving controller.

No work/casino/loan strategy is prescribed here. A policy explicitly selects
every intention and wager. At most one requested operation executes per decision.
"""
from dataclasses import fields, replace
import jax
import jax.numpy as jnp
from env import resource_taxi as env, city
import navigation
from telemetry import frame

STEPS = 128
WAIT_STEPS = 8
(WAIT, WORK, SLEEP, DINER, CAFE, RESTAURANT, FUEL_SMALL, FUEL_FULL,
 BORROW_25, BORROW_50, BORROW_100, CAR_25, CAR_100, CAR_ALL,
 MICRO_25, MICRO_100, MICRO_ALL, SLOTS_SMALL, SLOTS_MEDIUM, SLOTS_LARGE,
 BLACKJACK_SMALL, BLACKJACK_MEDIUM, BLACKJACK_LARGE, HIT, STAND, BUY) = range(26)
NAMES = ("Wait", "Work", "Sleep", "Budget diner", "Cafe", "Restaurant", "Refill 5", "Fill tank",
         "Borrow 25", "Borrow 50", "Borrow 100", "Car payment 25", "Car payment 100", "Pay car in full",
         "Microloan payment 25", "Microloan payment 100", "Pay microloans in full", "Slots 10", "Slots 25", "Slots 55",
         "Blackjack 10", "Blackjack 30", "Blackjack 90", "Hit", "Stand", "Complete ownership")
COUNT = len(NAMES)
CONTINUOUS = tuple(i for i in range(COUNT) if i not in (WAIT, HIT, STAND))
ZONES = jnp.array((city.HOME, city.WORK, city.HOME, city.DINER, city.CAFE, city.RESTAURANT,
                   city.FUEL, city.FUEL, city.BANK, city.BANK, city.BANK, city.BANK, city.BANK, city.BANK,
                   city.BANK, city.BANK, city.BANK, city.SLOTS, city.SLOTS, city.SLOTS,
                   city.BLACKJACK, city.BLACKJACK, city.BLACKJACK, city.BLACKJACK, city.BLACKJACK, city.GARAGE))
OPERATIONS = jnp.array((env.WAIT, env.WORK, env.SLEEP, env.EAT, env.EAT, env.EAT,
                        env.REFUEL_SMALL, env.REFUEL_FULL, env.BORROW_25, env.BORROW_50, env.BORROW_100,
                        env.CAR_25, env.CAR_100, env.CAR_ALL, env.MICRO_25, env.MICRO_100, env.MICRO_ALL,
                        env.SPIN_SMALL, env.SPIN_MEDIUM, env.SPIN_LARGE, env.DEAL_SMALL, env.DEAL_MEDIUM,
                        env.DEAL_LARGE, env.HIT, env.STAND, env.BUY))


class State(env.State):
    previous_option: jax.Array
    previous_features: jax.Array


def reset(scenario, params):
    obs, state = env.ResourceTaxiEnv.reset(scenario, params)
    history = decision_features(obs.features, state, params) if params.config.observation_history else jnp.zeros(0)
    state = State(**{f.name: getattr(state, f.name) for f in fields(state)},
                  previous_option=jnp.int32(-1 if params.config.observation_history else WAIT),
                  previous_features=history)
    return get_observation(state, params), state


def legal(state, params):
    # Financial eligibility does not impose work/sleep/gambling preferences.
    allowed = jnp.ones(COUNT, dtype=jnp.bool_)
    allowed = allowed.at[CAR_25:CAR_ALL+1].set((state.car_debt > 0) & (state.cash > 0))
    allowed = allowed.at[MICRO_25:MICRO_ALL+1].set((env.micro_debt(state) > 0) & (state.cash > 0))
    allowed = allowed.at[SLOTS_SMALL:SLOTS_LARGE+1].set(params.config.casino & (state.cash >= jnp.array((10.,25.,55.))))
    allowed = allowed.at[BLACKJACK_SMALL:BLACKJACK_LARGE+1].set(params.config.casino & (state.cash >= jnp.array((10.,30.,90.))))
    allowed = allowed.at[HIT:STAND+1].set(False)
    allowed = allowed.at[BUY].set((state.car_debt == 0) & (env.micro_debt(state) == 0))
    playing = (state.activity == env.DEAL_SMALL) & (state.remaining == 0)
    cards = jnp.zeros(COUNT, dtype=jnp.bool_).at[jnp.array((WAIT,HIT,STAND))].set(True)
    allowed = jnp.where(playing, cards, allowed)
    busy = (state.activity != env.DRIVE) & ~playing
    allowed = jnp.where(busy, jax.nn.one_hot(WAIT,COUNT,dtype=jnp.bool_), allowed)
    return jnp.where(state.outcome == env.ACTIVE, allowed, jax.nn.one_hot(WAIT,COUNT,dtype=jnp.bool_))


def decision_features(features, state, params):
    if not params.config.resource_observation:
        return features
    c=params.config
    station_distance=env.distances(state.agent_pos,params)[city.FUEL]
    fuel_range=state.fuel/jnp.maximum(c.fuel_distance,1e-6)
    remaining_ticks=jnp.maximum(0,c.horizon-state.time)
    signals=jnp.array((fuel_range/100.,(fuel_range-station_distance)/100.,
        remaining_ticks/(10*c.day) if c.days else 1., state.satiety<.2))
    return jnp.concatenate((features,signals))


def get_observation(state, params):
    obs = env.ResourceTaxiEnv.get_observation(state, params)
    current=decision_features(obs.features,state,params)
    # Keep the historical 194 inputs in their original order when disabled.
    return replace(obs, features=jnp.concatenate((current, state.previous_features, jax.nn.one_hot(state.previous_option, COUNT))),
                   action_mask=legal(state, params))


def advance(state, action, params, controller="shared", stop_tick=2**30):
    mask = legal(state, params)
    requested = jnp.clip(action.choice, 0, COUNT-1)
    option = jnp.where(mask[requested], requested, WAIT)
    operation, zone = OPERATIONS[option], ZONES[option]
    previous=state.previous_features
    if params.config.observation_history:
        observation=env.ResourceTaxiEnv.get_observation(state,params)
        previous=decision_features(observation.features,state,params)

    def tick(carry, index):
        current, total, used, receipt, finished, refused_ticks = carry
        obs = env.ResourceTaxiEnv.get_observation(current, params)
        free = current.activity == env.DRIVE
        direct = (option == HIT) | (option == STAND)
        at_destination = jnp.linalg.norm(current.agent_pos-params.zone_centers[zone]) <= params.config.zone_radius
        ready = obs.action_mask[operation] & at_destination & ~used & (option != WAIT)
        moving = free & ~used & ~ready & ~direct & (option != WAIT)
        command = jnp.where(ready, operation, jnp.where(moving, env.DRIVE, env.WAIT))

        def drive():
            # Continuous parameters adjust driving style around the same shared
            # controller; they cannot turn a travel intention into permanent braking.
            style = jnp.where(controller == "fixed", jnp.zeros(2), jnp.clip(action.control,-1.,1.))
            if params.config.predictive_driver:
                return navigation.predictive_guard(current,params,zone,.9+.1*style[0],1.+.1*style[1],
                    horizon=60 if controller=='guard' else 40)
            nominal = navigation.nominal_to_zone(current, params, zone, .9+.1*style[0], 1.+.1*style[1])
            return navigation.guard(current, params, nominal, 40 if controller == "guard" else 24)
        controls = jax.lax.cond(moving, drive, lambda: jnp.zeros(2))
        physical = env.Action(controls, command)
        _, new, reward, _, info = env.ResourceTaxiEnv.step(jax.random.key(0), current, physical, params)
        within = (current.time < stop_tick) & ~finished & (current.outcome == env.ACTIVE)
        new = jax.tree.map(lambda a,b:jnp.where(within,a,b),new,current)
        reward = jnp.where(within,reward,0.)
        info = jax.tree.map(lambda x:jnp.where(within,x,jnp.zeros_like(x)),info)
        row = frame(current, new, physical, reward, info)
        row["active"] &= within
        row["macro_requested"] = requested
        row["macro_executed"] = option
        row["macro_control"] = action.control
        row["decision_start"] = index == 0
        receipt += jnp.where((current.outcome == env.ACTIVE) & within,
                             info["gambling_net"]+new.wages-current.wages, 0.)
        # Return control after feedback: no queued second operation can execute
        # before the policy observes the wage, meal or game outcome.
        service_finished = ((current.remaining > 0) | ready) & (new.remaining == 0)
        requested_finished = (used | ready) & (new.remaining == 0)
        wait_expired = (option == WAIT) & (index+1 >= WAIT_STEPS)
        # A closed/unaffordable service reports failure on arrival rather than
        # keeping the car circling until an arbitrary travel timer expires.
        unavailable = at_destination & (current.speed < params.config.parking_speed) & free & ~obs.action_mask[operation] & (option != WAIT)
        refused_ticks += unavailable & within
        refusal_finished = unavailable & (refused_ticks >= params.config.unavailable_wait_ticks)
        finished |= service_finished | requested_finished | wait_expired | refusal_finished | (new.outcome != env.ACTIVE)
        return (new, total+params.config.gamma**index*reward, used | ready, receipt, finished, refused_ticks), row

    (new, reward, _, receipt, _, _), rows = jax.lax.scan(tick,
        (state, jnp.float32(0), jnp.bool_(False), jnp.float32(0), jnp.bool_(False),jnp.int32(0)), jnp.arange(STEPS))
    new = replace(new, previous_option=option, previous_event=receipt, previous_features=previous)
    return get_observation(new, params), new, reward, new.outcome != env.ACTIVE, rows
