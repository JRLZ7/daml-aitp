"""E001: extract frozen ResNet-50 features, then train a linear scene head."""

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
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from tqdm import tqdm

from .common import (ROOT, choose_device, git_state, load_config, repo_path,
                     seed_everything, sha256, write_json)
from .dataset import SceneDataset, load_training_records
from .evaluate import save_validation
from .models import SceneClassifier, baseline_weights


@torch.no_grad()
def extract_features(encoder, loader, device, description):
    encoder.eval()
    features, targets, ids = [], [], []
    for images, labels, batch_ids in tqdm(loader, desc=description, mininterval=2):
        features.append(encoder(images.to(device)).float().cpu())
        targets.append(labels)
        ids.extend(batch_ids)
    return torch.cat(features), torch.cat(targets), ids


def append_result(row):
    path = ROOT / 'experiments/results.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if any(existing['run_id'] == row['run_id'] for existing in reader):
            raise ValueError('Run ID already recorded; use a new run ID')
    with path.open('a', newline='') as stream:
        csv.DictWriter(stream, fieldnames=columns).writerow(row)


def run(config, output, provenance, device):
    seed = config['project']['seed']
    training = config['training']
    seed_everything(seed)
    torch.set_num_threads(training['cpu_threads'])
    if (config['model']['architecture'] != 'resnet50' or
        config['model']['freeze_encoder'] is not True or
        config['model']['preprocessing'] != 'pretrained_weights_transforms' or
        training['optimizer'] != 'AdamW' or training['loss'] != 'cross_entropy' or
        training['selection_metric'] != 'validation_accuracy'):
        raise ValueError('This trainer supports only the documented frozen baseline recipe')
    if training['epochs'] < 1 or training['head_batch_size'] < 1:
        raise ValueError('Epoch and head batch counts must be positive')
    data_dir = repo_path(config['data']['directory'])
    split_dir = repo_path(config['data']['split_directory'])
    fit, val, classes, metadata = load_training_records(data_dir, split_dir)
    write_json(output / 'class_order.json', classes)
    write_json(output / 'split_metadata.json', metadata)
    weights = baseline_weights(config['model']['pretrained_weights'])
    model = SceneClassifier(len(classes), pretrained=True)
    model.encoder.requires_grad_(False).eval().to(device)
    source = {
        'identifier': config['model']['pretrained_weights'], 'url': weights.url,
        'training_dataset': 'ImageNet-1K', 'recipe': weights.meta['recipe'],
        'documentation': 'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html',
        'transforms': repr(weights.transforms()),
    }
    downloaded = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
    source['sha256'] = sha256(downloaded)
    write_json(output / 'pretrained_source.json', source)
    batches = {}
    extraction_start = time.perf_counter()
    for name, records in (('train', fit), ('validation', val)):
        dataset = SceneDataset(records, data_dir / 'images', classes, weights.transforms())
        loader = DataLoader(dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])
        batches[name] = extract_features(model.encoder, loader, device, f'Extracting {name}')
    extraction_seconds = time.perf_counter() - extraction_start
    model.encoder.cpu()
    # Keep features for reproducible head-only comparisons, with their provenance.
    torch.save({'batches': batches, 'classes': classes, 'source': source,
                'split_metadata': metadata}, output / 'features.pt')
    x_train, y_train, _ = batches['train']
    x_val, y_val, val_ids = batches['validation']
    torch.manual_seed(seed)
    model.head.reset_parameters()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=training['learning_rate'],
                                 weight_decay=training['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=training['head_batch_size'], shuffle=True, generator=generator)
    history, best_accuracy, best_epoch, best_state = [], -1., None, None
    head_start = time.perf_counter()
    for epoch in range(1, training['epochs'] + 1):
        model.head.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.head(features), targets)
            loss.backward()
            optimizer.step()
        model.head.eval()
        with torch.no_grad():
            train_logits, val_logits = model.head(x_train), model.head(x_val)
            accuracy = float((val_logits.argmax(1) == y_val).float().mean())
            record = {
                'epoch': epoch, 'train_loss': float(criterion(train_logits, y_train)),
                'train_accuracy': float((train_logits.argmax(1) == y_train).float().mean()),
                'val_loss': float(criterion(val_logits, y_val)), 'val_accuracy': accuracy,
            }
        history.append(record)
        # Ties keep the earlier checkpoint; epoch count was specified before the run.
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = copy.deepcopy(model.head.state_dict())
        print(f"Epoch {epoch:02d}/{training['epochs']}: "
              f"train={record['train_accuracy']:.4f}, val={accuracy:.4f}", flush=True)
    head_seconds = time.perf_counter() - head_start
    model.head.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model.head(x_val).softmax(1).numpy()
    metrics = save_validation(output, val_ids, y_val.numpy(), probabilities, classes)
    checkpoint = {
        'format_version': 1, 'architecture': 'resnet50', 'classes': classes,
        'pretrained_weights': config['model']['pretrained_weights'],
        'state_dict': model.state_dict(), 'config': config, 'split_metadata': metadata,
        'pretrained_source': source, 'best_epoch': best_epoch, 'provenance': provenance,
    }
    torch.save(checkpoint, output / 'best.pt')
    digest = sha256(output / 'best.pt')
    (output / 'best.pt.sha256').write_text(f'{digest}  best.pt\n')
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    return {**metrics, 'best_epoch': best_epoch, 'epochs': training['epochs'],
            'feature_extraction_seconds': extraction_seconds,
            'head_training_seconds': head_seconds, 'checkpoint_sha256': digest,
            'pretrained_sha256': source['sha256'], 'feature_device': str(device),
            'head_device': 'cpu', 'pretrained_source': source['url']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Run ID may contain only letters, numbers, hyphens, and underscores')
    config = load_config(args.config)
    if args.device:
        config['training']['device'] = args.device
    with (ROOT / 'experiments/results.csv').open(newline='') as stream:
        if any(row['run_id'] == args.run_id for row in csv.DictReader(stream)):
            parser.error('Run ID already exists in results.csv')
    note = ROOT / 'experiments/notes' / f'{args.run_id}.md'
    if not note.is_file():
        parser.error(f'Create a pre-run note with hypothesis and ownership first: {note}')
    provenance = git_state()
    output = ROOT / 'outputs' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'git.json', provenance)
    (output / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    date = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    write_json(output / 'environment.json', {
        'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'command': sys.argv,
        'determinism': 'Seeded; deterministic algorithms warn on unsupported kernels; hardware may vary',
    })
    row = {
        'run_id': args.run_id, 'date': date, 'owner': 'Codex (AI-assisted local execution)',
        'decision_id': 'D005;D006;D009',
        'hypothesis': 'Frozen ImageNet ResNet-50 features support indoor scene classification',
        'reference_run': '', 'status': 'failed', 'code_commit': provenance['commit'],
        'dirty_worktree': provenance['dirty'],
        'config_path': f'outputs/{args.run_id}/config.json',
        'split_metadata_path': 'splits/metadata.json', 'seed': config['project']['seed'],
        'architecture': config['model']['architecture'],
        'pretrained_source': config['model']['pretrained_weights'],
        'val_accuracy': '', 'val_examples': '', 'train_minutes': '',
        'artifacts_path': f'outputs/{args.run_id}', 'notes_path': str(note.relative_to(ROOT)),
    }
    started = time.perf_counter()
    try:
        device = choose_device(config['training']['device'])
        print(f'Feature device: {device}; linear head device: cpu', flush=True)
        metrics = run(config, output, provenance, device)
        metrics['train_minutes'] = (time.perf_counter() - started) / 60
        write_json(output / 'metrics.json', metrics)
        row.update(status='completed', val_accuracy=metrics['val_accuracy'],
                   val_examples=metrics['val_examples'], train_minutes=metrics['train_minutes'])
    except Exception:
        (output / 'failure.txt').write_text(traceback.format_exc())
        append_result(row)
        raise
    append_result(row)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
