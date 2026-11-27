"""Deterministic checks for transition accounting, episode boundaries and reward shaping."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env import city
from env import gamblers_taxi as t
from train import evaluate_policy, keys, summary


def verify():
    env = t.GamblersTaxiEnv
    params, initial = env.init_params(jax.random.key(11))
    step = eqx.filter_jit(env.step)
    c = params.config

    def park(zone, **changes):
        return replace(initial, agent_pos=params.zone_centers[zone], **changes)

    def execute(state, action, control=(0., 0.)):
        return step(jax.random.fold_in(jax.random.key(31), state.time), state,
                    t.TaxiAction(jnp.array(control, dtype=jnp.float32), jnp.int32(action)), params)

    def do(state, action):
        return execute(state, action)[1]

    def wait(state, n):
        for _ in range(n):
            state = do(state, t.WAIT)
        return state

    moving = execute(initial, t.DRIVE, (1., 0.))[1]
    assert float(moving.speed) > 0
    coasting = execute(moving, t.DRIVE)[1]
    assert 0 <= float(coasting.speed) < float(moving.speed)
    parked = park(city.WORK, time=jnp.int32(0))
    working = do(parked, t.SHIFT)
    assert not bool(env.get_observation(working, params).action_mask[t.SHIFT])
    paid = wait(working, c.shift_duration - 1)
    np.testing.assert_allclose(paid.cash - parked.cash, c.salary, rtol=1e-6)
    assert int(paid.counters[t.PAID]) == 1 and int(do(paid, t.SHIFT).counters[t.PAID]) == 1
    speeding = replace(parked, speed=jnp.float32(c.parking_speed + 0.05))
    assert not bool(env.get_observation(speeding, params).action_mask[t.SHIFT])

    bank = park(city.BANK)
    loan = do(bank, t.BORROW)
    np.testing.assert_allclose(loan.cash-bank.cash, c.loan_unit, rtol=1e-6)
    assert float(t._balance(loan)) < float(t._balance(bank))
    assert float(loan.anxiety) > float(bank.anxiety)
    anxiety_index = -1-t.ECONOMY_FEATURES-t.DEBT_CAMPAIGN_FEATURES
    np.testing.assert_allclose(env.get_observation(loan, params).features[anxiety_index], loan.anxiety)
    loan = wait(loan, c.loan_duration - 1)
    assert float(loan.anxiety) > float(bank.anxiety)
    anxious_turn = execute(loan, t.DRIVE, (1., 1.))[1].steering
    calm_turn = execute(replace(loan, anxiety=jnp.float32(0.)), t.DRIVE, (1., 1.))[1].steering
    assert 0 < float(anxious_turn) < float(calm_turn)
    funded_loan = replace(loan, cash=loan.debt + 10.)
    paid_back = wait(do(funded_loan, t.REPAY), c.loan_duration - 1)
    assert float(paid_back.debt) == 0.
    assert float(paid_back.anxiety) < float(loan.anxiety)
    np.testing.assert_allclose(t._balance(paid_back), t._balance(funded_loan), atol=2e-6)
    partial = do(replace(loan, cash=jnp.float32(2.)), t.REPAY)
    assert float(partial.cash) == 0. and float(partial.debt) > 0.

    bedtime = park(city.HOME, time=jnp.int32(c.shift_window - 3), misses=jnp.int32(1),
                   fatigue=jnp.float32(0.96))
    rested = wait(do(bedtime, t.SLEEP), c.sleep_duration-1)
    assert not bool(rested.employed) and int(rested.counters[t.FIRED]) == 1
    assert float(rested.fatigue) < float(bedtime.fatigue)
    rehired = wait(do(replace(rested, agent_pos=params.zone_centers[city.JOBS]), t.APPLY), c.job_duration-1)
    assert bool(rehired.employed) and int(rehired.counters[t.HIRED]) == 1

    slots = park(city.SLOTS, cash=jnp.float32(2*c.slot_stake), time=jnp.int32(c.casino_open))
    too_early = replace(slots, time=jnp.int32(c.casino_open - 1))
    assert not bool(env.get_observation(too_early, params).action_mask[t.SPIN])
    assert not bool(env.get_observation(replace(slots, time=jnp.int32(c.casino_close)), params).action_mask[t.SPIN])
    reserved = do(slots, t.SPIN)
    np.testing.assert_allclose(reserved.cash, slots.cash-c.slot_stake)
    np.testing.assert_allclose(reserved.cash + reserved.stake, slots.cash)
    resolved = wait(reserved, c.spin_duration-1)
    assert int(resolved.counters[t.SPINS]) == 1 and float(resolved.stake) == 0
    assert int(resolved.counters[t.WINS] + resolved.counters[t.LOSSES]) == 1
    assert not bool(env.get_observation(resolved, params).action_mask[t.SPIN])
    different_table = replace(resolved, agent_pos=params.zone_centers[city.BLACKJACK])
    assert not bool(env.get_observation(different_table, params).action_mask[t.DEAL])
    assert bool(env.get_observation(replace(slots, fatigue=jnp.float32(1.)), params).action_mask[t.SPIN])
    assert bool(env.get_observation(replace(parked, fatigue=jnp.float32(1.)), params).action_mask[t.SHIFT])
    aces = jnp.array([[1, 1, 9], [1, 10, 10], [13, 12, 4]])
    np.testing.assert_array_equal(t._hand_total(aces)[0], [21, 21, 24])
    table = park(city.BLACKJACK, cash=jnp.float32(c.blackjack_stake), time=jnp.int32(c.casino_open))
    hand = do(table, t.DEAL)
    if int(hand.activity) == t.DEAL:
        hand = do(hand, t.STAND)
    assert float(hand.stake) == 0 and int(hand.counters[t.HANDS]) == 1

    garage = park(city.GARAGE, cash=jnp.float32(c.buyout + 8.), debt=jnp.float32(4.))
    assert not bool(env.get_observation(garage, params).action_mask[t.BUY])
    _, blocked, _, done, _ = execute(garage, t.BUY)
    assert not bool(done) and float(blocked.debt) > 0.
    at_bank = replace(blocked, agent_pos=params.zone_centers[city.BANK])
    settled = wait(do(at_bank, t.REPAY), c.loan_duration - 1)
    assert not bool(env.get_observation(settled, params).action_mask[t.BUY])
    back_at_garage = replace(settled, agent_pos=params.zone_centers[city.GARAGE])
    _, result, _, done, info = execute(back_at_garage, t.BUY)
    assert bool(done) and int(result.outcome) == t.SUCCESS and float(info["base_reward"]) == 1.
    np.testing.assert_allclose(result.cash, 8. - float(blocked.debt), atol=5e-6)
    np.testing.assert_allclose(result.debt, 0., atol=1e-6)
    _, crash, _, done, info = execute(replace(initial, agent_pos=jnp.asarray(city.WALLS[0], dtype=jnp.float32)), t.DRIVE)
    assert bool(done) and int(crash.outcome) == t.CRASH
    np.testing.assert_allclose(info["base_reward"] + info["debt_time_cost"], -1., atol=1e-6)

    last = replace(initial, time=jnp.int32(c.horizon - 3))
    phi = float(t._potential(last, params, t._distance(last.agent_pos, params)))
    shaping = 0.
    for i in range(3):
        _, last, _, done, info = execute(last, t.DRIVE)
        shaping += c.gamma**i*float(info["shaping"])
    assert bool(done) and int(last.outcome) == t.TIMEOUT
    np.testing.assert_allclose(shaping, -phi, atol=1e-7)
    assert float(info["base_reward"]) < 0.
    midnight = replace(initial, time=jnp.int32(c.day - 1))
    _, billed, _, _, charges = execute(midnight, t.DRIVE)
    np.testing.assert_allclose(billed.cash, midnight.cash - c.operating_cost)
    assert float(charges["daily_cost"]) == c.operating_cost
    max_rounds = (c.casino_close - 1 - c.casino_open) // (1 + c.casino_round_interval) + 1
    max_gain = max(c.slot_payout - c.slot_stake, 1.5*(1-c.blackjack_win_commission)*c.blackjack_stake)
    assert c.starting_cash + 10 + c.salary + max_rounds*max_gain < c.buyout + c.starting_debt
    assert float(initial.debt) == 1000.
    assert float(charges["debt_time_cost"]) > 0.
    dependence = replace(params, config=replace(c, enable_dependence=True))
    persistent = replace(slots, habit=jnp.float32(0.80), abstinence=jnp.int32(100))
    impulse = eqx.filter_jit(jax.vmap(
        lambda key, state: t._impulse(key, state, jnp.int32(t.DRIVE), jnp.array((1., 0.)),
                                     t._legal_actions(state, dependence), dependence),
        in_axes=(0, None),
    ))
    many = keys(67000, 96)
    _, _, weak = impulse(many, slots)
    _, _, strong = impulse(many, persistent)
    _, _, checked = impulse(many, replace(persistent, impulse_checked=jnp.bool_(True)))
    assert not bool(weak.any()) and bool(strong.any()) and not bool(checked.any())
    bet = t.TaxiAction(jnp.zeros(2), jnp.int32(t.SPIN))
    wait_action = t.TaxiAction(jnp.zeros(2), jnp.int32(t.WAIT))
    _, habit, _, _, _ = step(jax.random.key(0), slots, bet, dependence)
    for tick in range(c.spin_duration - 1):
        _, habit, _, _, _ = step(jax.random.key(tick), habit, wait_action, dependence)
    assert float(habit.habit) > 0. and int(habit.abstinence) == 0
    _, next_habit, _, _, _ = step(
        jax.random.key(99), habit, t.TaxiAction(jnp.zeros(2), jnp.int32(t.DRIVE)), dependence
    )
    assert 0. < float(next_habit.habit) < float(habit.habit) and int(next_habit.abstinence) == 1
    print("Transition checks passed")
    test = eqx.filter_jit(evaluate_policy)
    for strategy in range(5):
        result = summary(test(None, params, keys(50000, 24), strategy=strategy), c)
        print(f"scenario {strategy}: {result}")
        if strategy == 0:
            assert result["success"] > 0


if __name__ == "__main__":
    verify()
