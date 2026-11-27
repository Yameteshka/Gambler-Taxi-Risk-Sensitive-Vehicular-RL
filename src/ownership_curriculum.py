"""Reset-only ownership frontier with isolated food prerequisites.

No action labels, demonstrations, extra rewards, or economic action injection.
The earlier resource frontier remains available for historical replay.
"""
from dataclasses import dataclass,replace
import jax
import jax.numpy as jnp

NAME='ownership_frontier'
MIN_PHYSICAL=65536
MIN_DECISIONS=2048
MAX_PHYSICAL=786432
MAX_DECISIONS=65536
GATE_INTERVAL=65536
VALIDATION_IDS=tuple(range(1000001,1000017))
REQUIRE_COLD_COVERAGE=False

@dataclass(frozen=True)
class Lesson:
    name:str

LESSONS=tuple(Lesson(name) for name in (
    'ownership_endpoint','bank_settlement','earning_required','meal_then_ownership',
    'meal_then_settlement','meal_then_earning','food_approach','fuel_required',
    'fatigued_home','personal_debt','home_debt_250','home_debt_350',
    'home_debt_500','standard_home_actor'))
RETAINED_CONTEXTS=((2,'earning'),(6,'food_approach'),(7,'fuel'),(9,'personal_debt'))

def configuration(final,index):
    if final.starting_debt!=500.:raise ValueError('Ownership frontier requires original principal $500')
    return replace(final,training_reset_level=-1,
        ownership_reset_level=index if index<len(LESSONS)-1 else -1,
        exploration_epsilon=.2 if index<len(LESSONS)-1 else 0.)

def evaluation_configuration(config):return replace(config,training_rehearsal=0.,exploration_epsilon=0.)
def retain_replay(previous,current):return True

def reset_context(state,params):
    from env import resource_taxi as env,city
    c=params.config
    mix=jax.random.uniform(env.event_key(state,51,0))<c.training_rehearsal
    previous=jax.random.randint(env.event_key(state,52,0),(),0,max(c.ownership_reset_level,1))
    level=jnp.where(mix&(c.ownership_reset_level>0),previous,c.ownership_reset_level)
    locations=jnp.array((city.GARAGE,city.BANK,city.WORK,city.RESTAURANT,
        city.RESTAURANT,city.RESTAURANT,city.RESTAURANT,city.FUEL,city.HOME,
        city.BANK,city.HOME,city.HOME,city.HOME))
    debt=jnp.array((0.,50.,150.,0.,50.,150.,50.,150.,150.,150.,250.,350.,500.))[level]
    micro=jnp.where(level==9,50.,0.)
    time=jnp.where(level==5,jnp.int32(100),jnp.int32(0))
    # Advance traffic to the chosen reset clock phase without moving the car.
    def traffic_tick(i,s):
        pos,velocity=env.traffic_step(s,params)
        return replace(s,kinematic_obstacles=pos,kinematic_obst_velocities=velocity)
    state=jax.lax.fori_loop(0,time,traffic_tick,state)
    position=jnp.where(level==6,jnp.array((31.,17.)),params.zone_centers[locations[level]])
    return replace(state,time=time,agent_pos=position,
        car_debt=debt,car_paid=c.starting_debt-debt,
        car_cleared_at=jnp.where(debt==0,time,jnp.int32(-1)),
        all_cleared_at=jnp.where((debt==0)&(micro==0),time,jnp.int32(-1)),
        satiety=jnp.where((level==3)|(level==4),.04,jnp.where((level==5)|(level==6),.12,state.satiety)),
        fuel=jnp.where(level==7,0.,state.fuel),fatigue=jnp.where(level==8,.95,state.fatigue),
        loan_ids=state.loan_ids.at[0].set(jnp.int32(micro>0)),
        loan_principal=state.loan_principal.at[0].set(micro),
        loan_balance=state.loan_balance.at[0].set(micro),
        loan_rates=state.loan_rates.at[0].set(jnp.where(micro>0,c.micro_daily_rate,0.)),
        loan_opened=state.loan_opened.at[0].set(time),
        counters=state.counters.at[env.LOANS].set(jnp.int32(micro>0)))

def gate(index,summary,successful_earning_episodes,exposure_passed):
    if summary['episodes']!=len(VALIDATION_IDS):raise ValueError('Wrong development bank size')
    failures=[]
    if summary['ownership']<(8 if index==len(LESSONS)-1 else 4):failures.append('insufficient_ownership')
    if index in (2,5,7,8,9,10,11,12,13) and successful_earning_episodes<2:
        failures.append('fewer_than_two_earning_ownership_episodes')
    if index in (3,4,5,6) and summary['meals']==0:failures.append('no_meal_in_food_context')
    if index==7 and summary['refills']==0:failures.append('no_refill_in_empty_tank_context')
    return {'passed':not failures,'failures':failures,'actor_only':True,
        'exploration_epsilon':0.,'successful_earning_episodes':int(successful_earning_episodes),
        'scope':'Development readiness only; reset contexts are not standard-game wins.'}
