"""Spencer's RGB-thumbnail/histogram baseline on the team's verified shared split."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .common import ROOT, git_state, load_config, repo_path, sha256, write_json
from .dataset import load_training_records
from .train import append_result


def image_features(path, thumbnail_size=(16, 16), hist_bins=16):
    """Original Spencer feature recipe: 768 RGB thumbnail + 48 histogram values."""
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        pixels = np.asarray(rgb, dtype=np.uint8)
        thumbnail = np.asarray(
            rgb.resize(tuple(thumbnail_size), Image.Resampling.BILINEAR), dtype=np.float32
        ).ravel() / 255.0
    histograms = [np.histogram(pixels[:, :, channel], bins=hist_bins, range=(0, 256))[0]
                  / pixels.shape[0] / pixels.shape[1] for channel in range(3)]
    return np.concatenate([thumbnail, *histograms]).astype(np.float32)


def extract_features(records, image_root, thumbnail_size, hist_bins):
    return np.stack([image_features(Path(image_root) / path, thumbnail_size, hist_bins)
                     for path in tqdm(records['file_path'], desc='Classical features', mininterval=2)])


def fit_heads(x_train, y_train, settings):
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    svm = make_pipeline(StandardScaler(), LinearSVC(
        C=settings['C'], dual=False, max_iter=settings['max_iter'],
        random_state=settings['seed']))
    svm.fit(x_train, y_train)  # The scaler and classifier never see validation rows.
    return {'most_frequent': dummy, 'rgb_hist_linear_svm': svm}


def run_experiment(config_path, run_id, owner):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    config = load_config(config_path)
    note = ROOT / 'experiments/notes' / f'{run_id}.md'
    if not note.is_file():
        raise ValueError(f'Create a pre-run note first: {note}')
    with (ROOT / 'experiments/results.csv').open(newline='') as stream:
        used = {row['run_id'] for row in csv.DictReader(stream)}
    if any(f'{run_id}-{kind}' in used for kind in ('dummy', 'svm')):
        raise ValueError('Run IDs already recorded; choose a new suite ID')
    output = ROOT / 'outputs' / run_id
    if output.exists():
        raise ValueError('Output exists; choose a new suite ID')
    provenance = git_state()
    fit, val, classes, metadata = load_training_records(
        repo_path(config['data']['directory']), repo_path(config['data']['split_directory']))
    mapping = {label: i for i, label in enumerate(classes)}
    y_train, y_val = fit.label.map(mapping).to_numpy(), val.label.map(mapping).to_numpy()
    image_root = repo_path(config['data']['directory']) / 'images'
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    write_json(output / 'git.json', provenance)
    write_json(output / 'split_metadata.json', metadata)
    (output / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    start = time.perf_counter()
    feature_settings = config['features']
    x_train = extract_features(fit, image_root, **feature_settings)
    x_val = extract_features(val, image_root, **feature_settings)
    feature_seconds = time.perf_counter() - start
    fit_start = time.perf_counter()
    with threadpool_limits(limits=config['training']['cpu_threads']), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        models = fit_heads(x_train, y_train, config['training'])
    fit_seconds = time.perf_counter() - fit_start
    warnings_text = [str(w.message) for w in caught]
    metrics, checkpoint_hashes = [], {}
    predictions = val[['id', 'file_path', 'label']].copy()
    for name, model in models.items():
        predicted = model.predict(x_val)
        predictions[f'{name}_label'] = np.asarray(classes)[predicted]
        metric = {'model': name, 'accuracy': float(accuracy_score(y_val, predicted)),
                  'macro_f1': float(f1_score(y_val, predicted, labels=range(len(classes)),
                                           average='macro', zero_division=0)),
                  'correct': int((predicted == y_val).sum()), 'val_examples': len(val)}
        metrics.append(metric)
        pd.DataFrame(classification_report(y_val, predicted, labels=range(len(classes)),
                    target_names=classes, output_dict=True, zero_division=0)).T.to_csv(
                        output / f'{name}_classification_report.csv')
        pd.DataFrame(confusion_matrix(y_val, predicted, labels=range(len(classes))),
                     index=classes, columns=classes).rename_axis('true_label').to_csv(
                         output / f'{name}_confusion_matrix.csv')
        checkpoint_path = output / f'{name}.joblib'
        joblib.dump({'pipeline': model, 'classes': classes, 'features': feature_settings,
                     'split_metadata': metadata, 'config': config, 'provenance': provenance}, checkpoint_path)
        # Only reload the artifact just created locally; do not load untrusted joblib files.
        reloaded = joblib.load(checkpoint_path)
        if not np.array_equal(reloaded['pipeline'].predict(x_val), predicted):
            raise ValueError('Reloaded classical checkpoint changed predictions')
        checkpoint_hashes[name] = sha256(checkpoint_path)
    predictions.to_csv(output / 'val_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(output / 'metrics.csv', index=False)
    total_minutes = (time.perf_counter() - start) / 60
    summary = {
        'run_id': run_id, 'metrics': metrics, 'feature_seconds': feature_seconds,
        'fit_seconds': fit_seconds, 'total_minutes': total_minutes,
        'checkpoint_sha256': checkpoint_hashes, 'warnings': warnings_text,
        'code_commit': provenance['commit'], 'dirty_worktree': provenance['dirty'],
        'split_sha256': metadata['split_sha256'], 'feature_dimensions': x_train.shape[1],
        'source_method_commit': '859926a83d2b17bed0641fb9b5560038671a0a18',
        'source_method_author': 'Spencer Kolodny', 'execution_owner': owner,
        'checkpoint_reload_predictions_match': True,
    }
    write_json(output / 'summary.json', summary)
    for metric, suffix in zip(metrics, ('dummy', 'svm')):
        append_result({
            'run_id': f'{run_id}-{suffix}',
            'date': datetime.now(ZoneInfo('America/New_York')).date().isoformat(),
            'owner': owner, 'decision_id': 'D012',
            'hypothesis': 'Reproduce Spencer classical reference with shared loader and split',
            'reference_run': 'Spencer original 859926a', 'status': 'completed',
            'code_commit': provenance['commit'], 'dirty_worktree': provenance['dirty'],
            'config_path': str((output / 'config.json').relative_to(ROOT)),
            'split_metadata_path': 'splits/metadata.json', 'seed': config['training']['seed'],
            'architecture': metric['model'], 'pretrained_source': 'none',
            'val_accuracy': metric['accuracy'], 'val_examples': len(val),
            'train_minutes': total_minutes, 'artifacts_path': str(output.relative_to(ROOT)),
            'notes_path': str(note.relative_to(ROOT)),
        })
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/classical.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--owner', default='unspecified (see run note)')
    args = parser.parse_args()
    run_experiment(args.config, args.run_id, args.owner)


if __name__ == '__main__':
    main()
