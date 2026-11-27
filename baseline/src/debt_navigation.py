"""Shared controller for the financial-policy experiment."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from env import gamblers_taxi as taxi
import scenarios


def guarded_control(state, params, nominal):
    """Choose the closest command whose braking trajectory clears obstacles."""
    alternatives = jnp.array([
        [0., -1.], [0., 0.], [0., 1.],
        [0.30, -1.], [0.30, 0.], [0.30, 1.],
        [-1., -1.], [-1., 0.], [-1., 1.],
    ])
    controls = jnp.concatenate((nominal[None], alternatives))
    checks = 6

    def predict(control):
        def step(carry, index):
            current, safe = carry
            command = jnp.where(index < 2, control, jnp.array([-1., control[1]]))
            pos, forward, speed, steering = taxi._drive(current, command, params)
            traffic, velocities = taxi._traffic_step(current, params)
            collided = taxi._collision(current, pos, traffic)
            new = replace(current, agent_pos=pos, agent_forward_dir=forward, speed=speed,
                          steering=steering, kinematic_obstacles=traffic, kinematic_obst_velocities=velocities)
            return (new, safe & ~collided), collided
        (_, safe), collisions = jax.lax.scan(step, (state, jnp.bool_(True)), jnp.arange(checks))
        first_collision = jnp.min(jnp.where(collisions, jnp.arange(checks), checks))
        return safe, first_collision

    def avoid():
        safe, first_collision = jax.vmap(predict)(controls)
        difference = jnp.sum((controls - nominal)**2 * jnp.array([1.0, 0.34]), axis=1)
        score = difference + jnp.where(safe, 0., 100. + 15.*(checks-first_collision))
        return controls[jnp.argmin(score)]

    safe, _ = predict(nominal)
    return jax.lax.cond(safe, lambda: nominal, avoid)


def to_zone(state, params, zone):
    return guarded_control(state, params, scenarios._drive_to(state, params, zone))
