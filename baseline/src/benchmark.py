"""Measure headless dynamics throughput with and without batched dispatch."""

from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

from env import gamblers_taxi as taxi


def benchmark():
    env = taxi.GamblersTaxiEnv
    key = jax.random.key(17)
    params, state = env.init_params(key)
    count, ticks = 64, 70
    keys = jax.random.split(key, count)
    scalar = eqx.filter_jit(env.step)
    batched = eqx.filter_jit(jax.vmap(env.step, in_axes=(0, 0, None, None)))
    action = taxi.TaxiAction(jnp.array((0., 0.)), jnp.int32(taxi.DRIVE))
    starts = jax.vmap(lambda k: env.reset(k, params)[1])(keys)
    singles = [jax.tree.map(lambda x: x[i], starts) for i in range(count)]
    jax.block_until_ready(singles)
    jax.block_until_ready(scalar(key, state, action, params))
    jax.block_until_ready(batched(keys, starts, action, params))

    # Run the same number of simulated transitions with separate or batched dispatch.
    time = perf_counter()
    for episode_key, state in zip(keys, singles):
        for _ in range(ticks):
            _, state, _, _, _ = scalar(episode_key, state, action, params)
        jax.block_until_ready(state)
    serial = perf_counter() - time
    time = perf_counter()
    state = starts
    for _ in range(ticks):
        _, state, _, _, _ = batched(keys, state, action, params)
    jax.block_until_ready(state)
    batch = perf_counter() - time
    print(f"device={jax.devices()[0]} transitions={count*ticks} "
          f"serial_steps_per_second={count*ticks/serial:.1f} "
          f"batch_steps_per_second={count*ticks/batch:.1f} speedup={serial/batch:.1f}x")


if __name__ == "__main__":
    benchmark()
