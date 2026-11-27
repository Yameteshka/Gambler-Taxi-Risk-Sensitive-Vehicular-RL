"""One-factor curriculum transitions with actor-only advancement checks.

This is an opt-in treatment. Historical cold_debt500 runs keep their schedule.
The gate is a development criterion, not a statistical performance guarantee.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Lesson:
    name: str
    debt: float = 50.
    epsilon: float = .8
    start_distribution: str = 'services'
    home_probability: float = 0.
    hunger_multiplier: float = .65
    market: str = 'fixed'


def lessons():
    result = [Lesson('service_exposure')]

    def add(name, **changes):
        result.append(replace(result[-1], name=name, **changes))

    add('exploration_50', epsilon=.5)
    add('exploration_20', epsilon=.2)
    add('mixed_starts', start_distribution='mixed', home_probability=.25)
    for debt in (100., 150., 200., 300., 400., 500.):
        add(f'debt_{int(debt)}', debt=debt)
    for probability in (.5, .75, 1.):
        add(f'home_{int(probability*100)}', home_probability=probability)
    add('hunger_80', hunger_multiplier=.8)
    add('hunger_100', hunger_multiplier=1.)
    add('dynamic_market', market='dynamic')
    for epsilon in (.1, .05, 0.):
        add(f'exploration_{int(epsilon*100):02}', epsilon=epsilon)
    return tuple(result)


LESSONS = lessons()
NAME = 'progressive_debt500'
MIN_PHYSICAL = 131072
MAX_PHYSICAL = 524288
GATE_INTERVAL = 65536
VALIDATION_IDS = tuple(range(1000001, 1000017))


def configuration(final, index):
    lesson = LESSONS[index]
    if final.starting_debt != 500.:
        raise ValueError('This registered curriculum targets $500 principal')
    return replace(final, starting_debt=lesson.debt,
        exploration_epsilon=lesson.epsilon,
        exploration_starts=lesson.start_distribution == 'services',
        intermediate_starts=lesson.start_distribution == 'mixed' and lesson.home_probability < 1.,
        intermediate_home_fraction=lesson.home_probability,
        hunger_tick=final.hunger_tick*lesson.hunger_multiplier, market=lesson.market)


def changed_factors(previous, current):
    a, b = LESSONS[previous], LESSONS[current]
    # The reset sampler and its home probability define one start-distribution factor.
    factors = ('debt', 'epsilon', 'hunger_multiplier', 'market')
    changed = [key for key in factors if getattr(a, key) != getattr(b, key)]
    if (a.start_distribution, a.home_probability) != (b.start_distribution, b.home_probability):
        changed.append('start_distribution')
    return changed


def retain_replay(previous, current):
    # A behavior-mixture change does not alter environment rewards or transitions.
    return changed_factors(previous, current) == ['epsilon']


def gate(index, summary, successful_earning_episodes, exposure_passed):
    """Evaluate the actor with epsilon=0, without action labels or forced choices."""
    failures = []
    if summary['episodes'] != len(VALIDATION_IDS):
        raise ValueError('Advancement uses exactly the registered 16 validation scenarios')
    if summary['ownership'] < 4:
        failures.append('fewer_than_four_ownership_episodes')
    if LESSONS[index].debt > 100. and successful_earning_episodes < 2:
        failures.append('fewer_than_two_successful_episodes_with_paid_work')
    if index == 0:
        if not exposure_passed:
            failures.append('cold_start_event_coverage_incomplete')
        for skill in ('paid', 'meals', 'sleeps', 'refills'):
            if summary[skill] <= 0:
                failures.append(f'actor_did_not_complete_{skill}')
    return {'passed': not failures, 'failures': failures,
        'successful_earning_episodes': int(successful_earning_episodes),
        'actor_only': True, 'exploration_epsilon': 0.,
        'scope': 'Development curriculum advancement criterion.'}
