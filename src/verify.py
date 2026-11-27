"""Meaningful accounting, survival, randomness, split, and route checks."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env import city
from env import resource_taxi as t
from protocol import check_splits, load
from pathlib import Path
import json
from protocol import ROOT, source_hashes


def verify():
    params, _ = t.ResourceTaxiEnv.init_params(jax.random.key(0))
    _, initial = t.ResourceTaxiEnv.reset(6, params)
    step = eqx.filter_jit(t.ResourceTaxiEnv.step)
    c = params.config

    def command(s, op, control=(0., 0.)):
        return step(jax.random.key(0), s, t.Action(jnp.array(control), jnp.int32(op)), params)

    def park(zone, **values):
        return replace(initial, agent_pos=params.zone_centers[zone], **values)

    assert float(initial.cash) == 100 and float(initial.car_debt) == 1000
    assert initial.loan_balance.sum() == 0
    moving = command(initial, t.DRIVE, (1., 0.))[1]
    assert moving.speed > 0 and command(moving, t.DRIVE)[1].speed < moving.speed
    fast = replace(initial, speed=jnp.float32(2.))
    coast = command(fast, t.WAIT)[1]
    assert 0 < coast.speed < fast.speed
    assert float(jnp.linalg.norm(coast.agent_pos-fast.agent_pos)) > 0
    pushed = command(replace(initial, fuel=jnp.float32(0)), t.DRIVE, (1., 0.))[1]
    assert 0 < pushed.speed <= c.push_speed and pushed.push_ticks == 1
    assert pushed.fatigue > moving.fatigue and pushed.satiety < moving.satiety
    dead = command(replace(initial, satiety=jnp.float32(.0001)), t.WAIT)[1]
    assert dead.outcome == t.DEATH
    assert command(dead, t.DRIVE, (1., 0.))[1].time == dead.time
    bank = park(city.BANK)
    first = command(bank, t.BORROW_25)[1]
    assert float(command(bank, t.BORROW_25)[4]["financial_reward"]) <= 0
    assert abs(float(t.wealth_utility(replace(bank, cash=jnp.float32(1e8))))) <= .100001
    first = replace(first, activity=jnp.int32(t.DRIVE), remaining=jnp.float32(0))
    second = command(first, t.BORROW_50)[1]
    assert np.count_nonzero(np.asarray(second.loan_balance)) == 2
    np.testing.assert_allclose(second.cash, bank.cash+75)
    assert second.borrowed == 75 and second.counters[t.LOANS] == 2
    # Borrowing transfers equal cash and liabilities; interest makes net worth worse.
    before_worth = t.net_worth(bank)
    assert t.net_worth(first) < before_worth
    assert abs(float(t.net_worth(first)-before_worth)) < .1
    assert set(np.asarray(second.loan_ids[second.loan_balance > 0])) == {1, 2}
    before = replace(second, activity=jnp.int32(t.DRIVE), remaining=jnp.float32(0))
    after = command(before, t.MICRO_25)[1]
    np.testing.assert_allclose(before.cash-after.cash, 25, atol=1e-5)
    assert t.micro_debt(after) < t.micro_debt(before)
    paid = command(replace(before, cash=jnp.float32(2000)), t.MICRO_ALL)[1]
    assert t.micro_debt(paid) == 0
    many = jnp.zeros_like(initial.loan_balance).at[:128].set(jnp.arange(1,129)*1.234567)
    many_state = park(city.BANK,cash=jnp.float32(50000),loan_balance=many)
    assert t.micro_debt(command(many_state,t.MICRO_ALL)[1]) == 0
    # A settled car obligation cannot be resurrected by a rounded principal total.
    settled = replace(initial,car_debt=jnp.float32(0),car_paid=jnp.float32(999.99),time=jnp.int32(3*c.day-1))
    assert command(settled,t.WAIT)[1].car_debt == 0
    assert not t.legal(park(city.GARAGE, car_debt=jnp.float32(0),
                          loan_balance=second.loan_balance), params)[t.BUY]
    assert command(park(city.GARAGE, car_debt=jnp.float32(0)), t.BUY)[1].outcome == t.SUCCESS
    due = replace(initial, time=jnp.int32(3*c.day-1))
    charged = command(due, t.WAIT)[1]
    assert charged.fees == 10 and command(charged, t.WAIT)[1].fees == 10
    meal = park(city.RESTAURANT, cash=jnp.float32(100), satiety=jnp.float32(.1))
    old = meal
    meal = command(meal, t.EAT)[1]
    for _ in range(9):
        meal = command(meal, t.WAIT)[1]
    assert .55-.02 < float(meal.satiety-old.satiety) < .75
    assert meal.counters[t.MEALS] == 1 and 30 <= meal.food_spent <= 40
    # Discounted potential shaping cannot manufacture wealth from a loop.
    trajectory = [initial]
    shaped = []
    for _ in range(8):
        _, nxt, _, _, info = command(trajectory[-1], t.WAIT)
        trajectory.append(nxt)
        shaped.append(float(info["shaping"]))
    actual = sum(c.gamma**i*x for i, x in enumerate(shaped))
    expected = c.gamma**8*float(t.potential(trajectory[-1], params))-float(t.potential(initial, params))
    assert abs(actual-expected) < 1e-5
    # Delaying the same failure cannot improve the base task reward. Test
    # naturally occurring deaths at different times and a separate deadline.
    base_params = replace(params, config=replace(c, financial_progress=False))
    def failure_return(satiety, p):
        _, s = t.ResourceTaxiEnv.reset(6, p)
        s = replace(s, satiety=jnp.float32(satiety))
        def tick(carry, _):
            s, value = carry
            _, new, _, _, information = t.ResourceTaxiEnv.step(jax.random.key(0), s,
                t.Action(jnp.zeros(2), jnp.int32(t.WAIT)), p)
            value += p.config.gamma**s.time*information["external_reward"]
            return (new, value), None
        return jax.lax.scan(tick, (s, jnp.float32(0)), jnp.arange(512))[0]
    for satiety in (.1, .8):
        failed_state, total = eqx.filter_jit(failure_return)(satiety, base_params)
        assert failed_state.outcome == t.DEATH
        np.testing.assert_allclose(total, -c.failure_cost, atol=2e-6)
    deadline_params = replace(base_params, config=replace(base_params.config, days=1))
    failed_state, total = eqx.filter_jit(failure_return)(1., deadline_params)
    assert failed_state.outcome == t.TIMEOUT
    np.testing.assert_allclose(total, -c.failure_cost, atol=2e-6)
    np.testing.assert_allclose(t.ResourceTaxiEnv.get_observation(initial, params).features,
        t.ResourceTaxiEnv.reset(6, params)[0].features)
    np.testing.assert_allclose(t.quotes(old, params)[0], t.quotes(
        replace(old, time=jnp.int32(20)), params)[0])
    for origin in (city.HOME, city.WORK):
        x, y = city.ZONES[origin]
        a, b = np.asarray(params.fields)[[city.CAFE, city.RESTAURANT], y, x]
        assert abs(a-b)/max(a, b) <= .05, (origin, a, b)
    spec = load()
    assert check_splits(spec) == (11, 10010)
    check_splits({**spec, "training_scenario_count": 100000})
    try:
        check_splits({**spec, "training_scenario_count": 1000000})
    except ValueError:
        pass
    else:
        raise AssertionError("An overlapping training pool was accepted")
    (ROOT/"experiments/environment_verification.json").write_text(json.dumps({
        "passed": True, "source_hashes": source_hashes(),
        "checks": "Physics, survival, individual loans, cash/debt conservation, ownership, fees, food, RNG, splits, routes and shaping"}, indent=2), encoding="utf-8")
    print("Accounting, survival, random-stream, split, and route checks passed", flush=True)


if __name__ == "__main__":
    verify()
