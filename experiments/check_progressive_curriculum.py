"""Verify isolation of factors, replay policy, advancement and final configuration."""
from pathlib import Path
from dataclasses import asdict, fields
import ast
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import progressive_curriculum as curriculum


def check(final):
    lessons = curriculum.LESSONS
    assert lessons[0].debt == 50. and lessons[-1].debt == 500.
    transitions = []
    for index in range(1, len(lessons)):
        changed = curriculum.changed_factors(index-1, index)
        assert len(changed) == 1, (index, changed)
        assert curriculum.retain_replay(index-1, index) == (changed == ['epsilon'])
        if changed == ['debt']:
            assert 0 < lessons[index].debt-lessons[index-1].debt <= 100.
            assert lessons[index].epsilon == .2
        transitions.append({'lesson': lessons[index].name, 'changed': changed[0],
            'retain_replay': curriculum.retain_replay(index-1, index)})
    configurations = [curriculum.configuration(final, index) for index in range(len(lessons))]
    last = configurations[-1]
    for field in fields(final):
        assert getattr(last, field.name) == getattr(final, field.name), field.name
    fake = {'episodes': 16, 'ownership': 4, 'paid': 1., 'meals': 1., 'sleeps': 1., 'refills': 1.}
    assert not curriculum.gate(0, fake, 0, False)['passed']
    assert not curriculum.gate(0, {**fake, 'meals': 0}, 0, True)['passed']
    earning = next(i for i, item in enumerate(lessons) if item.debt == 150.)
    assert not curriculum.gate(earning, fake, 0, True)['passed']
    assert curriculum.gate(earning, fake, 2, True)['passed']
    assert not curriculum.gate(earning, {**fake, 'ownership': 0}, 2, True)['passed']
    for file in ('src/progressive_curriculum.py', 'src/strategy_training.py', 'experiments/progressive_pilot.py'):
        ast.parse((ROOT/file).read_text(encoding='utf-8'))
    result = {'passed': True, 'lessons': [asdict(item) for item in lessons],
        'transitions': transitions, 'gate_rejects_exposure_only': True,
        'gate_rejects_success_without_work_above_starting_cash': True,
        'validation_ids': list(curriculum.VALIDATION_IDS),
        'scope': 'Schedule and gate regression checks; not a learning result.'}
    return result


if __name__ == '__main__':
    from env.resource_taxi import Config
    result = check(Config(starting_debt=500., public_food_menu=True, loan_capacity=256))
    (ROOT/'experiments/progressive_curriculum_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': result['passed'], 'lesson_count': len(result['lessons'])}))
