"""Reference driving and explicit driving-guard diagnostics."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from env import cards, city
from env import resource_taxi as t


MOVES = jnp.array(((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)))


def guard(state, params, nominal, horizon=24, hold=4, traffic_margin=0.):
    controls = jnp.concatenate((nominal[None], jnp.array([
        [-1., -1.], [-1., 0.], [-1., 1.], [0., -1.], [0., 0.], [0., 1.],
        [1., -1.], [1., 0.], [1., 1.]])))

    def predict(control):
        def tick(carry, i):
            s, safe = carry
            cmd = jnp.where(i < hold, control, jnp.array((-1., control[1])))
            pos, direction, speed, steering = t.drive(s, cmd, params)
            traffic, velocity = t.traffic_step(s, params)
            crash = t.collision(s, pos, traffic, traffic_margin=traffic_margin)
            s = replace(s, agent_pos=pos, agent_forward_dir=direction, speed=speed,
                        steering=steering, kinematic_obstacles=traffic, kinematic_obst_velocities=velocity)
            return (s, safe & ~crash), crash
        (_, safe), crashes = jax.lax.scan(tick, (state, jnp.bool_(True)), jnp.arange(horizon))
        first_collision = jnp.min(jnp.where(crashes, jnp.arange(horizon), horizon))
        return safe, first_collision

    safe, first_collision = jax.vmap(predict)(controls)
    difference = ((controls-nominal)**2*jnp.array((1., .34))).sum(axis=-1)
    return controls[jnp.argmin(difference+jnp.where(safe, 0., 100.+15.*(horizon-first_collision)))]


def nominal_to_zone(state, params, zone, speed_factor=1., steering_factor=1.):
    target = params.zone_centers[zone]
    nearby = jnp.clip(jnp.rint(state.agent_pos).astype(jnp.int32)+MOVES*2, 2,
                       jnp.array((city.WIDTH-3, city.HEIGHT-3)))
    candidates = jnp.concatenate((target[None], nearby.astype(jnp.float32)))
    vectors = candidates-state.agent_pos
    obstacles = state.static_obstacles
    offset = obstacles[None]-state.agent_pos
    fraction = jnp.clip((offset*vectors[:, None]).sum(axis=-1)/
                        jnp.maximum((vectors**2).sum(axis=-1, keepdims=True), 1e-5), 0., 1.)
    clearance = jnp.linalg.norm(offset-fraction[..., None]*vectors[:, None], axis=-1).min(axis=-1)
    route = jnp.concatenate((jnp.array((-1.,)), params.fields[zone, nearby[:, 1], nearby[:, 0]]))
    route += jnp.where(clearance >= city.COLLISION_RADIUS+.08, 0., 100.+20*(city.COLLISION_RADIUS+.1-clearance))
    vector = vectors[jnp.argmin(route)]
    direction = state.agent_forward_dir
    angle = jnp.arctan2(direction[0]*vector[1]-direction[1]*vector[0], jnp.dot(direction, vector))
    distance = jnp.linalg.norm(target-state.agent_pos)
    desired = jnp.minimum(speed_factor*(2.2-1.4*jnp.minimum(jnp.abs(angle), 1.)),
                          jnp.maximum(.2, 1.8*(distance-.3)))
    desired = jnp.where(distance < .35, 0., desired)
    control = jnp.array((jnp.clip(1.8*(desired-state.speed)+jnp.where(desired > 0, .23, 0.), -1., 1.),
                         jnp.clip(steering_factor*1.8*angle, -1., 1.)))
    return control


def to_zone(state, params, zone):
    return guard(state, params, nominal_to_zone(state, params, zone))


def predictive_guard(state, params, zone, speed_factor=1., steering_factor=1., horizon=40):
    """Compare controls using a replanned route, rather than a parked future.

    This controller receives the already selected destination only. It neither
    chooses economic actions nor modifies collisions, resource costs or traffic.
    """
    nominal = nominal_to_zone(state, params, zone, speed_factor, steering_factor)
    grid = jnp.array([(a,b) for a in (-1.,0.,1.) for b in (-1.,-.5,0.,.5,1.)])
    controls = jnp.concatenate((nominal[None], grid))

    def predict(control):
        def tick(carry, index):
            current, safe, clearance, first = carry
            planned = nominal_to_zone(current, params, zone, speed_factor, steering_factor)
            command = jnp.where(index < 4, control, planned)
            pos, direction, speed, steering = t.drive(current, command, params)
            traffic, velocity = t.traffic_step(current, params)
            crash = t.collision(current, pos, traffic)
            obstacles = jnp.concatenate((current.static_obstacles, traffic))
            gap = jnp.linalg.norm(obstacles-pos, axis=-1).min()-city.COLLISION_RADIUS
            current = replace(current, agent_pos=pos, agent_forward_dir=direction, speed=speed,
                steering=steering, kinematic_obstacles=traffic, kinematic_obst_velocities=velocity)
            return (current, safe & ~crash, jnp.minimum(clearance,gap),
                    jnp.minimum(first,jnp.where(crash,index,horizon))), None
        (end, safe, clearance, first), _ = jax.lax.scan(tick,
            (state,jnp.bool_(True),jnp.float32(100.),jnp.int32(horizon)),jnp.arange(horizon))
        distance = t.distances(end.agent_pos, params)[zone]
        difference = ((control-nominal)**2*jnp.array((1.,.34))).sum()
        score = .3*difference+.25*distance+.25/(jnp.maximum(clearance,0.)+.2)
        return score+jnp.where(safe,0.,1000.+15.*(horizon-first)),safe

    scores,safe = jax.vmap(predict)(controls)
    chosen=controls[jnp.argmin(scores)]
    if params.config.predictive_escape:
        # A collision in the *nominal continuation* does not prove that every
        # future control is unsafe. When all such continuations fail, use the
        # short stopping-distance guard and replan on the next physical tick.
        # Otherwise the long forecast can repeatedly choose zero acceleration
        # to postpone an avoidable distant collision and never leave the spot.
        chosen=jax.lax.cond(jnp.any(safe),lambda:chosen,
            lambda:guard(state,params,nominal,horizon=24))
    return chosen


def reference_policy(state, obs, params, style="reserve"):
    c = params.config
    all_paid = state.car_debt+t.micro_debt(state) == 0
    food = city.DINER
    at_diner = jnp.linalg.norm(state.agent_pos-params.zone_centers[city.DINER]) < .65
    near_death = (state.satiety < .5) | (at_diner & (state.satiety < .85))
    needs_fuel = state.fuel < 4.
    rested = state.fatigue < .45
    shift_open = (state.time % c.day < c.shift_window) & ~state.shift_claimed
    pay = state.cash > state.car_debt+t.micro_debt(state)+40.
    zone = jnp.where(all_paid, city.GARAGE, jnp.where(near_death, food,
                     jnp.where(needs_fuel, city.FUEL, jnp.where(shift_open, city.WORK,
                     jnp.where(state.fatigue > .68, city.HOME,
                     jnp.where(pay, city.BANK, city.WORK))))))
    if style == "opportunistic":
        play = (state.time % c.day >= c.casino_open) & (state.time % c.day < c.casino_close) & (state.cash > 80.)
        zone = jnp.where(play & ~near_death & ~all_paid & ~needs_fuel, city.SLOTS, zone)
    micro = t.micro_debt(state)
    payment = jnp.where(micro > 0, t.MICRO_100, t.CAR_100)
    payment = jnp.where(state.cash > state.car_debt+micro+40,
                        jnp.where(micro > 0, t.MICRO_ALL, t.CAR_ALL), payment)
    op = jnp.array((t.SLEEP, t.WORK, t.CAR_100, t.BUY, t.SPIN_MEDIUM,
                    t.DEAL_SMALL, t.REFUEL_SMALL, t.EAT, t.EAT, t.EAT))[zone]
    op = jnp.where(zone == city.BANK, payment, op)
    op = jnp.where((zone == city.HOME) & rested, t.WAIT, op)
    op = jnp.where((zone == city.WORK) & ~shift_open, t.WAIT, op)
    # With very little cash, obtain a voluntary loan before attempting a purchase.
    broke = (needs_fuel | near_death) & (state.cash < 6)
    zone = jnp.where(broke, city.BANK, zone)
    op = jnp.where(broke, t.BORROW_50, op)
    total, soft = cards.total(state.hand)
    hand_action = jnp.where((total < 12) | ((total < 17) & (jnp.minimum(state.dealer[0], 10) >= 7)) |
                           (soft & (total < 18)), t.HIT, t.STAND)
    op = jnp.where(state.activity == t.DEAL_SMALL, hand_action, op)
    close = jnp.linalg.norm(state.agent_pos-params.zone_centers[zone]) < .65
    op = jnp.where((state.activity == t.DRIVE) & (~close | (state.speed >= c.parking_speed)), t.DRIVE, op)
    choice = jnp.where(obs.action_mask[op], op, jnp.where(obs.action_mask[t.DRIVE], t.DRIVE, t.WAIT))
    control = jax.lax.cond(choice == t.DRIVE, lambda: to_zone(state, params, zone),
                           lambda: jnp.array((-1., 0.)))
    return t.Action(control, choice.astype(jnp.int32))
