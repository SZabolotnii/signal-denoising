"""
Експеримент 2: порівняння типів базисних функцій DSGE.

Навчає окрему Hybrid U-Net для кожного типу базису та порівнює результати.
Визначає оптимальний basis_type для радіосигналів QPSK/FSK.

Базиси (аналогічно HAR-статті, Table 1):
  - 'fractional':    sign(x)|x|^p, p ∈ {0.5, 1.5, 2.0}  (найкращий у HAR)
  - 'polynomial':    x^2, x^3, x^4
  - 'trigonometric': sin(x), sin(2x), sin(3x)
  - 'robust':        tanh(x), sigmoid(x), atan(x)

Запуск:
    cd experiments/
    python experiment_2_basis.py --dataset non_gaussian --epochs 20
"""

import argparse
import os
import sys
import subprocess
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


BASIS_CONFIGS = {
    'fractional':    {'powers': [0.5, 1.5, 2.0]},
    'polynomial':    {'powers': [2.0, 3.0, 4.0]},
    'trigonometric': {'powers': [1.0, 2.0, 3.0]},  # freqs for sin()
    'robust':        {'powers': []},                 # fixed inside DSGEFeatureExtractor
}


def run_training(basis: str, powers: list, dataset: str, epochs: int) -> dict:
    """Запускає training_hybrid.py для одного basis_type і повертає метрики."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), '..', 'train', 'training_hybrid.py'),
        '--dataset',    dataset,
        '--epochs',     str(epochs),
        '--dsge-basis', basis,
        '--lr',         '5e-5',
        '--no-wandb',
    ]
    if powers:
        cmd.extend(['--dsge-powers'] + [str(p) for p in powers])
    print(f'\n{"="*60}')
    print(f'[Exp2] Training with basis: {basis}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'[Exp2] ERROR training {basis}')
        return {}
    return {}  # метрики беруться з inference_hybrid нижче


def collect_metrics_after_training(basis: str, dataset: str, dsge_order: int = 3) -> dict:
    """Запускає inference_hybrid.py і парсить stdout."""
    from inference.inference_hybrid import main as inf_main, DEFAULT
    cfg = {**DEFAULT, 'dataset_type': dataset, 'dsge_order': dsge_order, 'dsge_basis': basis}
    # Redirect: замість plt.show() — збереження у файл (inference_hybrid вже зберігає PNG)
    try:
        inf_main(cfg)
    except Exception as e:
        print(f'[Exp2] inference error for {basis}: {e}')
    return {}


def main(cfg: dict):
    results = {}
    for basis in ['fractional', 'polynomial', 'trigonometric', 'robust']:
        run_training(basis, BASIS_CONFIGS[basis]['powers'],
                     cfg['dataset'], cfg['epochs'])

    print('\n\n' + '═' * 60)
    print('EXPERIMENT 2 SUMMARY — BASIS COMPARISON')
    print('═' * 60)
    print('Run inference/inference_hybrid.py with "--dsge-basis <type>" for each basis.')
    print('Compare MSE and SNR values to find the best basis for your dataset.')
    print('\nRecommended order to test:')
    for i, basis in enumerate(['fractional', 'polynomial', 'trigonometric', 'robust'], 1):
        print(f"  {i}. python inference_hybrid.py --dataset {cfg['dataset']} --dsge-basis {basis}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Experiment 2: Basis type comparison')
    p.add_argument('--dataset', type=str, default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--epochs',  type=int, default=20)
    args = p.parse_args()
    main({'dataset': args.dataset, 'epochs': args.epochs})
