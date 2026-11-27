"""Exact model calculations, separate from empirical policy evaluation."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from env.resource_taxi import Config


CARD = tuple((rank, (4 if rank == 10 else 1)/13) for rank in range(1, 11))


def score(hard, ace):
    return hard+10 if ace and hard <= 11 else hard


@lru_cache(None)
def dealer_distribution(hard, ace):
    total = score(hard, ace)
    if total >= 17:
        out = np.zeros(23)
        out[min(total, 22)] = 1.
        return out
    return sum(p*dealer_distribution(hard+rank, ace or rank == 1) for rank, p in CARD)


def blackjack_optimal_ev(commission=.1):
    """Infinite replacement shoe, no split/double, dealer S17, natural peek."""
    expectation = 0.
    for up, p_up in CARD:
        distribution = np.zeros(23)
        natural_probability = 0.
        for hole, probability in CARD:
            if (up == 1 and hole == 10) or (up == 10 and hole == 1):
                natural_probability += probability
            else:
                distribution += probability*dealer_distribution(up+hole, up == 1 or hole == 1)
        distribution /= 1-natural_probability

        @lru_cache(None)
        def value(hard, ace):
            total = score(hard, ace)
            if total > 21:
                return -1.
            stand = sum(p*((1-commission) if (dealer == 22 or dealer < total)
                           else (0. if dealer == total else -1.)) for dealer, p in enumerate(distribution))
            if total == 21:
                return stand
            hit = sum(p*value(hard+rank, ace or rank == 1) for rank, p in CARD)
            return max(stand, hit)

        for a, p_a in CARD:
            for b, p_b in CARD:
                if (a == 1 and b == 10) or (a == 10 and b == 1):
                    payoff = (1-natural_probability)*1.5*(1-commission)
                else:
                    payoff = -natural_probability+(1-natural_probability)*value(a+b, a == 1 or b == 1)
                expectation += p_up*p_a*p_b*payoff
    return expectation


def main():
    c = Config()
    ev = blackjack_optimal_ev(c.blackjack_commission)
    assert ev < 0
    slot_ev = c.slot_probability*c.slot_multiplier-1
    assert .16*c.slot_multiplier < 1
    result = {"slots_expected_net_per_dollar": slot_ev,
        "blackjack_optimal_expected_net_per_dollar": ev,
        "blackjack_scope": "Exact dynamic programming for the implemented replacement-card model, not casino blackjack in general.",
        "hidden_regime_best_slot_expected_net_per_dollar": .16*c.slot_multiplier-1,
        "discounted_running_cost_upper_bound": c.failure_cost,
        "base_failure_return": "-kappa for every finite unsuccessful episode, independently of failure time; kappa=0.1. Financial increments are separate.",
        "base_success_return": "-kappa + gamma**(T-1)*(1+kappa+cash_bonus), for T physical steps. Earlier identical success is strictly preferred.",
        "failure_time_identity": {str(n): -c.failure_cost*(1-c.gamma**(n-1))-c.failure_cost*c.gamma**(n-1) for n in (1, 56, 155, 473, 1550)},
        "absolute_discounted_total_financial_reward_bound": .2,
        "discounted_training_rnd_bonus_bound_per_fixed_task": .05,
        "financial_utility": "U(W)=0.1*clip(W/2000,-1,1); external increment U(W_next)-U(W_current). Linear within +/- $2000; saturation is explicitly reported.",
        "shaping_identity": "sum(gamma**t * F_t) = -Phi(s0) + gamma**T*Phi(sT); Phi(terminal)=0",
        "max_cash_without_gambling_or_borrowing_before_costs": {str(days): c.starting_cash+days*c.salary*1.35 for days in (5, 10)},
        "food_expected_price_per_raw_satiety": {"diner": 5/.05, "cafe": 15/.15, "restaurant": 35/.65},
        "food_warning": "Ratios of means are aggregate planning benchmarks. Clipping, travel, service time, survival and liquidity change realized value.",
        "inference_warning": "Ownership and return estimates use 256 held-out scenarios."}
    path = Path(__file__).resolve().parents[1]/"experiments/analytic_checks.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
