"""Fit a predeclared small L2 sweep on frozen features and export scene checkpoints."""

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
import torch

from .common import ROOT, git_state, load_config, repo_path, seed_everything, sha256, write_json
from .dataset import load_training_records
from .evaluate import save_validation
from .models import load_checkpoint
from .train import append_result


def check_cached_batch(batch, records, classes):
    """Reject mismatched IDs/labels rather than silently assigning features to rows."""
    features, targets, ids = batch
    if list(ids) != records['id'].tolist():
        raise ValueError('Cached feature IDs differ in content or order from shared split')
    mapping = {label: i for i, label in enumerate(classes)}
    expected = records['label'].map(mapping).to_numpy()
    if not np.array_equal(targets.numpy(), expected):
        raise ValueError('Cached labels disagree with the training manifest')
    if features.ndim != 2 or len(features) != len(records) or not torch.isfinite(features).all():
        raise ValueError('Invalid cached features')
    return features.numpy(), targets.numpy(), list(ids)


def paired_comparison(reference, candidate, resamples=2000, seed=42):
    """Align by ID and report descriptive paired accuracy differences."""
    if not reference['id'].is_unique or not candidate['id'].is_unique:
        raise ValueError('Duplicate comparison IDs')
    if set(reference['id']) != set(candidate['id']):
        raise ValueError('Different comparison ID sets')
    joined = reference[['id', 'true_label', 'pred_label']].merge(
        candidate[['id', 'true_label', 'pred_label']], on='id', validate='one_to_one',
        suffixes=('_reference', '_candidate'))
    if not joined.true_label_reference.equals(joined.true_label_candidate):
        raise ValueError('Comparison true labels differ')
    before = (joined.true_label_reference == joined.pred_label_reference).to_numpy()
    after = (joined.true_label_candidate == joined.pred_label_candidate).to_numpy()
    delta = after.astype(float) - before.astype(float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples)
    for start in range(0, resamples, 100):
        size = min(100, resamples - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        sampled[start:start + size] = delta[indices].mean(axis=1)
    return {
        'corrected': int((~before & after).sum()),
        'regressed': int((before & ~after).sum()),
        'net_correct': int(delta.sum()),
        'accuracy_difference': float(delta.mean()),
        'prediction_disagreements': int((joined.pred_label_reference != joined.pred_label_candidate).sum()),
        'descriptive_paired_bootstrap_95pct': np.quantile(sampled, [.025, .975]).tolist(),
        'uncertainty_note': 'Not corrected for repeated model selection; not an independent test estimate',
    }


def export_head(model, estimator, x, class_count):
    if not np.array_equal(estimator.classes_, np.arange(class_count)):
        raise ValueError('Estimator class indices differ from checkpoint class order')
    with torch.no_grad():
        model.head.weight.copy_(torch.as_tensor(estimator.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.as_tensor(estimator.intercept_, dtype=torch.float32))
        probabilities = model.head(torch.from_numpy(x)).softmax(1).numpy()
    original = estimator.predict_proba(x)
    if not np.array_equal(probabilities.argmax(1), original.argmax(1)):
        raise ValueError('Float32 exported classifier changes predicted labels')
    error = float(np.abs(probabilities - original).max())
    if error > 1e-5:
        raise ValueError(f'Export probability difference exceeds tolerance: {error}')
    return probabilities, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/regularization.yaml')
    parser.add_argument('--owner', default='unspecified (see run note)')
    args = parser.parse_args()
    config = load_config(args.config)
    experiment, training = config['experiment'], config['training']
    suite_id = experiment['suite_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', suite_id):
        parser.error('Invalid suite ID')
    values = training['C_values']
    if len(values) != len(set(values)) or not values or any(c <= 0 for c in values):
        parser.error('C values must be unique and positive')
    if (training['feature_scaling'] != 'none' or training['class_weight'] is not None or
        training['l1_ratio'] != 0.0 or training['solver'] != 'lbfgs'):
        parser.error('This experiment supports unscaled L2/lbfgs features only')
    if config['selection']['primary'] != 'validation_accuracy' or config['selection']['tiebreak'] != 'validation_log_loss_then_smaller_C':
        parser.error('Unsupported selection rule')
    note_path = repo_path(experiment['note'])
    if not note_path.is_file():
        parser.error('Create the pre-run experiment note before executing')
    suite = ROOT / 'outputs' / suite_id
    if suite.exists():
        parser.error('Suite output exists; use a fresh suite ID to repeat')
    run_ids = [f"{suite_id}-C{str(c).replace('.', 'p')}" for c in values]
    with (ROOT / 'experiments/results.csv').open(newline='') as stream:
        existing = {row['run_id'] for row in csv.DictReader(stream)}
    if existing.intersection(run_ids):
        parser.error('A run ID already exists in the experiment register')
    provenance = git_state()
    seed_everything(training['seed'])
    torch.set_num_threads(training['cpu_threads'])
    fit, val, classes, metadata = load_training_records(
        repo_path(config['data']['directory']), repo_path(config['data']['split_directory']))
    cache_path = repo_path(experiment['feature_cache'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    model, _, parent = load_checkpoint(repo_path(experiment['reference_checkpoint']), torch.device('cpu'))
    if (cache['classes'] != classes or parent['classes'] != classes or
        cache['split_metadata'] != metadata or parent['split_metadata'] != metadata or
        cache['source'] != parent['pretrained_source']):
        raise ValueError('Feature/checkpoint/split provenance mismatch')
    x_train, y_train, _ = check_cached_batch(cache['batches']['train'], fit, classes)
    x_val, y_val, val_ids = check_cached_batch(cache['batches']['validation'], val, classes)
    if x_train.shape[1] != model.head.in_features or x_val.shape[1] != model.head.in_features:
        raise ValueError('Feature dimensions mismatch')
    reference = pd.read_csv(repo_path(experiment['reference_predictions']))
    if reference['id'].tolist() != val_ids or reference['true_label'].tolist() != val['label'].tolist():
        raise ValueError('Reference validation predictions do not match the split')
    with torch.no_grad():
        ref_pred = model.head(torch.from_numpy(x_val)).argmax(1).numpy()
    if [classes[i] for i in ref_pred] != reference['pred_label'].tolist():
        raise ValueError('Reference checkpoint does not reproduce cached-feature baseline')
    suite.mkdir(parents=True)
    write_json(suite / 'config.json', config)
    write_json(suite / 'git.json', provenance)
    source = {'feature_cache_sha256': sha256(cache_path),
              'parent_checkpoint_sha256': sha256(repo_path(experiment['reference_checkpoint'])),
              'reference_predictions_sha256': sha256(repo_path(experiment['reference_predictions'])),
              'pretrained_source': cache['source']}
    write_json(suite / 'source.json', source)
    (suite / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    write_json(suite / 'environment.json', {
        'python': sys.version, 'platform': platform.platform(), 'sklearn': sklearn.__version__,
        'torch': str(torch.__version__), 'command': sys.argv, 'device': 'cpu',
    })
    date = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    summary = []
    for run_id, c in zip(run_ids, values):
        output = suite / run_id
        output.mkdir()
        resolved = copy.deepcopy(config)
        resolved['training']['C'] = c
        write_json(output / 'config.json', resolved)
        row = {
            'run_id': run_id, 'date': date, 'owner': args.owner, 'decision_id': 'D011',
            'hypothesis': 'L2 regularization and converged linear-head fitting improve frozen-feature generalization',
            'reference_run': experiment['reference_run'], 'status': 'failed',
            'code_commit': provenance['commit'], 'dirty_worktree': provenance['dirty'],
            'config_path': str((output / 'config.json').relative_to(ROOT)),
            'split_metadata_path': 'splits/metadata.json', 'seed': training['seed'],
            'architecture': 'resnet50', 'pretrained_source': parent['pretrained_weights'],
            'val_accuracy': '', 'val_examples': '', 'train_minutes': '',
            'artifacts_path': str(output.relative_to(ROOT)), 'notes_path': experiment['note'],
        }
        started = time.perf_counter()
        try:
            print(f'Fitting {run_id}: C={c}, L2, lbfgs; validation excluded from fit', flush=True)
            estimator = LogisticRegression(
                C=c, l1_ratio=0., solver='lbfgs', max_iter=training['max_iter'],
                tol=training['tolerance'], random_state=training['seed'], class_weight=None)
            with threadpool_limits(limits=training['cpu_threads']), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                estimator.fit(x_train, y_train)
                probabilities, export_error = export_head(model, estimator, x_val, len(classes))
                train_probabilities = estimator.predict_proba(x_train)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            metrics = save_validation(output, val_ids, y_val, probabilities, classes)
            paired = paired_comparison(reference, pd.read_csv(output / 'val_predictions.csv'),
                                       config['selection']['bootstrap_resamples'], training['seed'])
            checkpoint = copy.deepcopy(parent)
            checkpoint.update(state_dict=model.state_dict(), config=resolved, best_epoch=None,
                              provenance=provenance, parent_run=experiment['reference_run'])
            torch.save(checkpoint, output / 'best.pt')
            digest = sha256(output / 'best.pt')
            (output / 'best.pt.sha256').write_text(f'{digest}  best.pt\n')
            metrics.update({
                'run_id': run_id, 'C': float(c), 'converged': converged,
                'solver_iterations': int(estimator.n_iter_.max()),
                'train_accuracy': float((train_probabilities.argmax(1) == y_train).mean()),
                'train_log_loss': float(log_loss(y_train, train_probabilities, labels=np.arange(len(classes)))),
                'val_log_loss': float(log_loss(y_val,
                    probabilities.astype(np.float64) / probabilities.sum(axis=1, keepdims=True, dtype=np.float64),
                    labels=np.arange(len(classes)))),
                'export_max_probability_difference': export_error,
                'train_minutes': (time.perf_counter() - started) / 60,
                'checkpoint_sha256': digest, 'paired_vs_baseline': paired,
                'warnings': [str(w.message) for w in caught],
            })
            write_json(output / 'metrics.json', metrics)
            row.update(status='completed', val_accuracy=metrics['val_accuracy'],
                       val_examples=len(y_val), train_minutes=metrics['train_minutes'])
            summary.append(metrics)
            print(f"C={c}: val={metrics['val_accuracy']:.4%}, log_loss={metrics['val_log_loss']:.4f}, "
                  f"corrected={paired['corrected']}, regressed={paired['regressed']}, "
                  f"iterations={metrics['solver_iterations']}, converged={converged}", flush=True)
        except Exception:
            (output / 'failure.txt').write_text(traceback.format_exc())
            append_result(row)
            raise
        append_result(row)
    eligible = [run for run in summary if run['converged']]
    if not eligible:
        raise RuntimeError('No converged candidate; no winner selected')
    best = min(eligible, key=lambda run: (-run['val_accuracy'], run['val_log_loss'], run['C']))
    write_json(suite / 'summary.json', {
        'runs': summary, 'selected_run': best['run_id'], 'selection': config['selection'],
        'reference_run': experiment['reference_run'], 'source': source,
        'code_commit': provenance['commit'], 'dirty_worktree': provenance['dirty'],
    })
    pd.DataFrame([{k: v for k, v in run.items() if not isinstance(v, (dict, list))}
                  for run in summary]).to_csv(suite / 'comparison.csv', index=False)
    print(f"Selected {best['run_id']} by the predeclared rule. This needs confirmation on additional data.", flush=True)


if __name__ == '__main__':
    main()
