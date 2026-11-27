"""Ownership-only reset curriculum with rehearsal under one transition kernel.

All decisions remain policy decisions. Reset contexts make a skill useful for
ownership; they never label an action or replace the ownership objective.
"""
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp

NAME = 'resource_frontier'
MIN_PHYSICAL = 65536
MIN_DECISIONS = 2048
MAX_PHYSICAL = 786432
GATE_INTERVAL = 65536
VALIDATION_IDS = tuple(range(1000001,1000017))
REQUIRE_COLD_COVERAGE = False


@dataclass(frozen=True)
class Lesson:
    name: str


LESSONS = tuple(Lesson(name) for name in (
    'ownership_endpoint','bank_settlement','earning_required','food_required',
    'fuel_required','fatigued_home','personal_debt','home_debt_250',
    'home_debt_350','home_debt_500','standard_home_actor'))


def configuration(final,index):
    if final.starting_debt != 500.:
        raise ValueError('Resource frontier is registered for $500 original principal')
    return replace(final,training_reset_level=index if index<10 else -1,
                   exploration_epsilon=.2 if index<10 else 0.)


def evaluation_configuration(config):
    return replace(config,training_rehearsal=0.,exploration_epsilon=0.)


def retain_replay(previous,current):
    # Only resets / behavior exploration change. The transition and reward
    # functions are identical, including the original $500 contract.
    return True


def reset_context(state,params):
    from env import resource_taxi as env,city
    c=params.config
    mix=jax.random.uniform(env.event_key(state,51,0)) < c.training_rehearsal
    previous=jax.random.randint(env.event_key(state,52,0),(),0,max(c.training_reset_level,1))
    level=jnp.where(mix & (c.training_reset_level>0),previous,c.training_reset_level)
    locations=jnp.array((city.GARAGE,city.BANK,city.WORK,city.RESTAURANT,
                         city.FUEL,city.HOME,city.BANK,city.HOME,city.HOME,city.HOME))
    debt=jnp.array((0.,50.,150.,150.,150.,150.,150.,250.,350.,500.))[level]
    micro=jnp.where(level==6,50.,0.)
    state=replace(state,agent_pos=params.zone_centers[locations[level]],
        car_debt=debt,car_paid=c.starting_debt-debt,
        car_cleared_at=jnp.where(debt==0,jnp.int32(0),jnp.int32(-1)),
        all_cleared_at=jnp.where((debt==0)&(micro==0),jnp.int32(0),jnp.int32(-1)),
        satiety=jnp.where(level==3,.12,state.satiety),
        fuel=jnp.where(level==4,0.,state.fuel),fatigue=jnp.where(level==5,.95,state.fatigue),
        loan_ids=state.loan_ids.at[0].set(jnp.int32(micro>0)),
        loan_principal=state.loan_principal.at[0].set(micro),
        loan_balance=state.loan_balance.at[0].set(micro),
        loan_rates=state.loan_rates.at[0].set(jnp.where(micro>0,c.micro_daily_rate,0.)),
        loan_opened=state.loan_opened.at[0].set(0),
        counters=state.counters.at[env.LOANS].set(jnp.int32(micro>0)))
    return state


def gate(index,summary,successful_earning_episodes,exposure_passed):
    if summary['episodes'] != len(VALIDATION_IDS):
        raise ValueError('Unexpected curriculum evaluation bank size')
    failures=[]
    if summary['ownership']<4:failures.append('fewer_than_four_ownership_episodes')
    if index>=2 and successful_earning_episodes<2:
        failures.append('fewer_than_two_earning_ownership_episodes')
    # Resource use is only required in a context designed to require it;
    # discretionary visits in an already-solvable task are not a mastery gate.
    if index==3 and summary['meals']==0:failures.append('no_meal_in_food_required_context')
    if index==4 and summary['refills']==0:failures.append('no_refill_in_fuel_required_context')
    return {'passed':not failures,'failures':failures,'actor_only':True,
        'exploration_epsilon':0.,'successful_earning_episodes':int(successful_earning_episodes),
        'scope':'Development ownership readiness.'}
