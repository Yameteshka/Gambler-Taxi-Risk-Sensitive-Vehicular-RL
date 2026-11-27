"""Reset-only food budgeting progression with mandatory actor skill retention.

The original ownership curriculum is preserved. No rewards, demonstrations,
financial reserves, economic action masks or automatic meals are introduced.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import ownership_curriculum as original

NAME='food_budget_retention'
LESSONS=tuple(original.Lesson(name) for name in (
    *(x.name for x in original.LESSONS[:7]),
    'food_from_work','food_from_bank','food_from_home',
    'fuel_required','fatigued_home','personal_debt','home_debt_250',
    'home_debt_350','home_debt_425','home_debt_500','standard_home_actor'))
MIN_PHYSICAL=original.MIN_PHYSICAL
MIN_DECISIONS=original.MIN_DECISIONS
MAX_PHYSICAL=original.MAX_PHYSICAL
MAX_DECISIONS=original.MAX_DECISIONS
GATE_INTERVAL=original.GATE_INTERVAL
VALIDATION_IDS=original.VALIDATION_IDS
SECOND_BANK=tuple(range(1000017,1000033))
REQUIRE_COLD_COVERAGE=False
EXTRA_GATE_METRICS=True
SELECT_QUALIFIED_CHECKPOINT=True
STANDARD_OWNERSHIP_MIN=48
RETAINED_CONTEXTS=((2,'earning'),(6,'food_approach'),(7,'food_from_work'),
                   (8,'food_from_bank'),(9,'food_from_home'),(10,'fuel'),(12,'personal_debt'))
FOOD_STAGES=(3,4,5,6,7,8,9)
EARNING_STAGES=(2,5,7,8,9,10,11,12,13,14,15,16,17)
REQUIRED_SUCCESSES=12  # Registered 75% development threshold per bank.

def configuration(final,index):
    if final.starting_debt!=500.:raise ValueError('Food budgeting requires $500 original principal')
    if not 0<=index<len(LESSONS):raise ValueError('Unknown budgeting lesson')
    epsilon=.2 if index<len(LESSONS)-1 else 0.
    if final.ppo_exploration_decay:
        epsilon*=max(0.,1-index/13.)
    return replace(final,training_reset_level=-1,ownership_reset_level=-1,
        budget_reset_level=index if index<len(LESSONS)-1 else -1,
        training_rehearsal=0. if final.standard_start_rehearsal and index>=4 else .35,
        exploration_epsilon=epsilon)

def evaluation_configuration(config):
    return replace(config,training_rehearsal=0.,standard_start_rehearsal=0.,exploration_epsilon=0.)

def retain_replay(previous,current):return True

def reset_context(state,params):
    from env import resource_taxi as env,city
    c=params.config;index=c.budget_reset_level
    standard=state
    rng=env.event_key(state,61,0)
    mix=jax.random.uniform(rng)<c.training_rehearsal
    previous=jax.random.randint(env.event_key(state,62,0),(),0,max(index,1))
    # Half of rehearsals deliberately sample previously introduced food starts.
    food_previous=jax.random.randint(env.event_key(state,63,0),(),3,max(min(index,10),4))
    prefer_food=(index>3)&jax.random.bernoulli(env.event_key(state,64,0),.5)
    previous=jnp.where(prefer_food,food_previous,previous)
    level=jnp.where(mix&(index>0),previous,index)
    mapping=jnp.array((0,1,2,3,4,5,6,6,6,6,7,8,9,10,11,12,12))
    mapped=mapping[level]
    locations=jnp.array((city.GARAGE,city.BANK,city.WORK,city.RESTAURANT,
        city.RESTAURANT,city.RESTAURANT,city.RESTAURANT,city.FUEL,city.HOME,
        city.BANK,city.HOME,city.HOME,city.HOME))
    debt=jnp.array((0.,50.,150.,0.,50.,150.,50.,150.,150.,150.,250.,350.,500.))[mapped]
    micro=jnp.where(mapped==9,50.,0.)
    clock=jnp.where(mapped==5,jnp.int32(100),jnp.int32(0))
    def traffic_tick(i,s):
        pos,velocity=env.traffic_step(s,params)
        return replace(s,kinematic_obstacles=pos,kinematic_obst_velocities=velocity)
    state=jax.lax.fori_loop(0,clock,traffic_tick,state)
    pos=jnp.where(mapped==6,jnp.array((31.,17.)),params.zone_centers[locations[mapped]])
    state=replace(state,time=clock,agent_pos=pos,car_debt=debt,car_paid=c.starting_debt-debt,
        car_cleared_at=jnp.where(debt==0,clock,jnp.int32(-1)),
        all_cleared_at=jnp.where((debt==0)&(micro==0),clock,jnp.int32(-1)),
        satiety=jnp.where((mapped==3)|(mapped==4),.04,jnp.where((mapped==5)|(mapped==6),.12,state.satiety)),
        fuel=jnp.where(mapped==7,0.,state.fuel),fatigue=jnp.where(mapped==8,.95,state.fatigue),
        loan_ids=state.loan_ids.at[0].set(jnp.int32(micro>0)),
        loan_principal=state.loan_principal.at[0].set(micro),loan_balance=state.loan_balance.at[0].set(micro),
        loan_rates=state.loan_rates.at[0].set(jnp.where(micro>0,c.micro_daily_rate,0.)),
        loan_opened=state.loan_opened.at[0].set(clock),counters=state.counters.at[env.LOANS].set(jnp.int32(micro>0)))
    remote=(level>=7)&(level<=9)
    place=jnp.array((city.WORK,city.BANK,city.HOME))[jnp.clip(level-7,0,2)]
    u=jax.random.uniform(env.event_key(state,65,0),(2,))
    low=jnp.array((.30,.22,.32))[jnp.clip(level-7,0,2)]
    high=jnp.array((.48,.38,.50))[jnp.clip(level-7,0,2)]
    debt=jnp.where(remote,150.,jnp.where(level==15,425.,state.car_debt))
    contextual=replace(state,
        agent_pos=jnp.where(remote,params.zone_centers[place],state.agent_pos),
        cash=jnp.where(remote,60.+60.*u[0],state.cash),
        satiety=jnp.where(remote,low+(high-low)*u[1],state.satiety),
        car_debt=debt,car_paid=c.starting_debt-debt)
    if index<4 or not c.standard_start_rehearsal:
        return contextual
    use_standard=jax.random.bernoulli(env.event_key(standard,66,0),c.standard_start_rehearsal)
    return jax.tree.map(lambda base,lesson:jnp.where(use_standard,base,lesson),standard,contextual)

def gate(index,summary,successful_earning_episodes,exposure_passed,
         successful_meal_episodes=0,successful_refuel_episodes=0):
    if summary['episodes']!=16:raise ValueError('Expected one 16-episode development bank')
    failures=[]
    if summary['ownership']<REQUIRED_SUCCESSES:failures.append('ownership_below_12_of_16')
    if index in FOOD_STAGES and successful_meal_episodes<REQUIRED_SUCCESSES:
        failures.append('fewer_than_12_ownership_episodes_with_meal')
    if index in EARNING_STAGES and successful_earning_episodes<REQUIRED_SUCCESSES:
        failures.append('fewer_than_12_earning_ownership_episodes')
    if index==10 and successful_refuel_episodes<REQUIRED_SUCCESSES:
        failures.append('fewer_than_12_ownership_episodes_with_refuel')
    return {'passed':not failures,'failures':failures,'actor_only':True,
        'exploration_epsilon':0.,'successful_meal_episodes':int(successful_meal_episodes),
        'successful_earning_episodes':int(successful_earning_episodes),
        'successful_refuel_episodes':int(successful_refuel_episodes),
        'scope':'Development readiness on each validation bank.'}

def additional_checks(final,index):
    # A candidate must pass another current-context bank and each earlier skill
    # on both banks. Rehearsal is off in every acceptance evaluation.
    yield 'current_second_bank',index,evaluation_configuration(configuration(final,index)),SECOND_BANK
    for stage,name in RETAINED_CONTEXTS:
        if stage>=index:continue
        for bank,ids in enumerate((VALIDATION_IDS,SECOND_BANK)):
            yield f'retained_{name}_bank_{bank}',stage,evaluation_configuration(configuration(final,stage)),ids
