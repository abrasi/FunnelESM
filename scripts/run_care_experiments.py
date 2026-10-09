#!/usr/bin/env python3
"""Run a CARE LR convergence pilot or five-seed final comparison."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ARCHITECTURES = ('baseline', 'mean_layer_6', 'mean_layers_4_8', 'pool_query_layer_6', 'pool_query_layers_4_8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--test_csv')
    p.add_argument('--cluster_metadata_csv')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--phase', choices=['pilot', 'final'], required=True)
    p.add_argument('--learning_rates', nargs='+', type=float, default=[1e-5, 5e-5, 1e-4])
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--architectures', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--model_name', default='facebook/esm2_t12_35M_UR50D')
    p.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--validation_seed', type=int, default=0)
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = a.learning_rates if a.phase == 'pilot' else [a.learning_rate]
    seeds = [a.seeds[0]] if a.phase == 'pilot' else a.seeds
    manifest = {'phase': a.phase, 'config': vars(a), 'runs': []}
    for arch in a.architectures:
        for lr in grid:
            for seed in seeds:
                name = f'{arch}_lr_{lr:.2e}_seed_{seed}' if a.phase == 'pilot' else f'{arch}_seed_{seed}'
                destination = output / name
                cmd = [sys.executable, str(Path(__file__).with_name('train_care.py')),
                       '--train_csv', str(Path(a.train_csv).resolve()), '--output_dir', str(destination),
                       '--architecture', arch, '--model_name', a.model_name,
                       '--epochs', str(a.epochs), '--batch_size', str(a.batch_size),
                       '--learning_rate', str(lr), '--seed', str(seed),
                       '--validation_seed', str(a.validation_seed), '--mixed_precision', a.mixed_precision]
                if a.test_csv:
                    cmd += ['--test_csv', str(Path(a.test_csv).resolve())]
                if a.cluster_metadata_csv:
                    cmd += ['--cluster_metadata_csv', str(Path(a.cluster_metadata_csv).resolve())]
                if a.phase == 'pilot':
                    cmd += ['--convergence_pilot']
                manifest['runs'].append({'architecture': arch, 'lr': lr, 'seed': seed,
                                         'output_dir': str(destination), 'command': cmd})
    (output / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
    for run in manifest['runs']:
        print(' '.join(run['command']), flush=True)
        if not a.dry_run:
            subprocess.run(run['command'], check=True)
    if not a.dry_run:
        summarize(output, manifest)


def summarize(output, manifest):
    import numpy as np
    aggregates = {}
    for run in manifest['runs']:
        record = json.loads((Path(run['output_dir']) / 'run_result.json').read_text())
        curve = json.loads((Path(run['output_dir']) / 'epoch_history.json').read_text())
        evaluations = [r for r in curve if 'eval_loss' in r]
        key = run['architecture']
        if manifest['phase'] == 'pilot':
            aggregates.setdefault(key, []).append({'lr': run['lr'], 'seed': run['seed'],
                'best_val_loss': min(evaluations, key=lambda x: x['eval_loss'])['eval_loss'],
                'best_epoch': min(evaluations, key=lambda x: x['eval_loss']).get('epoch'),
                'epoch_history': str(Path(run['output_dir']) / 'epoch_history.json')})
        else:
            aggregates.setdefault(key, []).append(record)
    if manifest['phase'] == 'pilot':
        best = {name: min(records, key=lambda r: r['best_val_loss']) for name, records in aggregates.items()}
        results = {'per_architecture': aggregates, 'best_by_architecture': best,
                   'note': 'Select LR on validation only; use a common LR for final if desired.'}
        filename = 'pilot_selection.json'
    else:
        summary = {}
        for name, records in aggregates.items():
            flattened = [{**r.get('test_metrics', {}), **r.get('resources', {})} for r in records]
            keys = set.intersection(*(set(r) for r in flattened))
            summary[name] = {k: {'mean': float(np.mean([r[k] for r in flattened])),
                                   'std': float(np.std([r[k] for r in flattened]))}
                             for k in sorted(keys)}
        results = {'runs': aggregates, 'summary': summary}
        filename = 'care_final_results.json'
    (output / filename).write_text(json.dumps(results, indent=2))
    print(f'Saved {output / filename}')


if __name__ == '__main__':
    main()
