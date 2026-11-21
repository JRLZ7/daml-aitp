"""Evaluate a saved checkpoint on its recorded validation split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import choose_device, repo_path, sha256, write_json
from .dataset import SceneDataset, load_training_records
from .models import load_checkpoint


@torch.inference_mode()
def predict_probabilities(model, loader, device):
    model.eval()
    ids, targets, probabilities = [], [], []
    for images, labels, batch_ids in tqdm(loader, desc='Predicting', mininterval=2):
        logits = model(images.to(device)).float()
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        targets.extend(labels.tolist())
        ids.extend(batch_ids)
    return ids, np.asarray(targets), np.concatenate(probabilities)


def save_validation(output_dir, ids, targets, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        'id': ids, 'true_label': [classes[i] for i in targets],
        'pred_label': [classes[i] for i in predicted],
        'confidence': probabilities.max(axis=1),
    })
    for i, label in enumerate(classes):
        frame[f'p_{label}'] = probabilities[:, i]
    frame.to_csv(output_dir / 'val_predictions.csv', index=False)
    matrix = confusion_matrix(targets, predicted, labels=list(range(len(classes))))
    pd.DataFrame(matrix, index=classes, columns=classes).rename_axis('true_label').to_csv(
        output_dir / 'confusion_matrix.csv')
    per_class = pd.DataFrame({'label': classes, 'count': matrix.sum(axis=1),
                              'correct': matrix.diagonal()})
    per_class['accuracy'] = per_class['correct'] / per_class['count']
    per_class.to_csv(output_dir / 'per_class.csv', index=False)
    metrics = {'val_accuracy': float((predicted == targets).mean()),
               'correct': int((predicted == targets).sum()), 'val_examples': len(targets)}
    write_json(output_dir / 'validation_metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=repo_path('data/raw'))
    parser.add_argument('--split-dir', type=Path, default=repo_path('splits'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    device = choose_device(args.device)
    torch.set_num_threads(4)
    model, transform, checkpoint = load_checkpoint(args.checkpoint, device)
    _, val, classes, metadata = load_training_records(args.data_dir, args.split_dir)
    if classes != checkpoint['classes'] or metadata != checkpoint['split_metadata']:
        raise ValueError('Checkpoint class order or split metadata mismatch')
    dataset = SceneDataset(val, args.data_dir / 'images', classes, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ids, targets, probabilities = predict_probabilities(model, loader, device)
    metrics = save_validation(args.output_dir, ids, targets, probabilities, classes)
    write_json(args.output_dir / 'evaluation.json', {
        **metrics, 'checkpoint_sha256': sha256(args.checkpoint), 'device': str(device)})
    print(metrics)


if __name__ == '__main__':
    main()
