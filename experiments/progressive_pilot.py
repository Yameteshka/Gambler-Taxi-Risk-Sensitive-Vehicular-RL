"""Run one explicitly separate gradual-curriculum pilot before a four-agent study."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import protocol
import progressive_curriculum as curriculum
from strategy_training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('dsac_mlp', 'dsac_gru', 'ppo_mlp', 'ppo_gru'), default='dsac_mlp')
    parser.add_argument('--steps', type=int, default=4194304)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    algorithm, architecture = args.model.split('_')
    row = {**protocol.registry()[0], 'run_id': args.model,
        'algorithm': 'hybrid_dsac' if algorithm == 'dsac' else 'hybrid_ppo',
        'architecture': architecture, 'initialization_seed': '301',
        'curriculum': curriculum.NAME, 'starting_debt': '500', 'training_horizon': '10d',
        'exploration': 'standard', 'training_market': 'dynamic', 'casino': 'enabled',
        'physical_transition_budget': str(args.steps), 'public_food_menu': 'enabled',
        'protocol_revision': 'progressive_debt500_01', 'status': 'planned'}
    train(row, resume=args.resume, stop_after=args.stop_after,
          output_root='artifacts/progressive_curriculum_pilot/training')


if __name__ == '__main__':
    main()
