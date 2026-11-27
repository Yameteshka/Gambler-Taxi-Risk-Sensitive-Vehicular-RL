"""Compare sample and argmax on identical development cases and weights."""
from pathlib import Path
import json
import sys

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import models
from env import resource_taxi as env
from evaluate import evaluate

review = json.loads((ROOT / 'reports/resource_study_review_2026-11-23/review.json').read_text())
paths = {model['model']: ROOT / model['checkpoints']['selected']['path']
         for model in review['models'] if model['model'] in ('ppo_mlp', 'ppo_gru', 'dsac_gru')}
paths['dsac_mlp'] = ROOT / 'artifacts/local_food_recovery/dsac_mlp_control_continued_forward_kl/final.npz'
output = ROOT / 'reports/local_food_recovery/inference_modes.json'
results = []
for kind, path in paths.items():
    actor, metadata = models.load_policy(path)
    config = env.Config(**metadata['evaluation_environment'])
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    for deterministic in (False, True):
        summary, _, _ = evaluate(actor, kind, params, range(1000001, 1000017),
                                 deterministic=deterministic, interface='strategy')
        row = {'model': kind, 'checkpoint': path.relative_to(ROOT).as_posix(),
               'mode': 'argmax_mean' if deterministic else 'sample',
               'scenario_ids': list(range(1000001, 1000017)), 'summary': summary}
        results.append(row)
        output.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(kind, row['mode'], {key: summary[key] for key in
              ('ownership', 'starvation', 'collision', 'meals', 'refills')}, flush=True)
