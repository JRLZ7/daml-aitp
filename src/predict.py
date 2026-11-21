"""Generate a competition CSV using only a saved scene checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import choose_device, repo_path
from .dataset import SceneDataset
from .evaluate import predict_probabilities
from .models import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=repo_path('data/raw'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a new path')
    device = choose_device(args.device)
    torch.set_num_threads(4)
    model, transform, checkpoint = load_checkpoint(args.checkpoint, device)
    classes = checkpoint['classes']
    provided = pd.read_csv(args.data_dir / 'classes.csv')['label'].tolist()
    if classes != provided:
        raise ValueError('Competition and checkpoint class orders differ')
    records = pd.read_csv(args.data_dir / 'test.csv', dtype=str)
    sample = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype=str)
    if list(sample.columns) != ['id', 'label']:
        raise ValueError('Sample submission must have exactly id,label columns')
    if (not records['id'].is_unique or not sample['id'].is_unique or
        set(records['id']) != set(sample['id'])):
        raise ValueError('Duplicate or mismatched test/sample IDs')
    # Only IDs and paths enter the inference dataset, even if an input has extra columns.
    dataset = SceneDataset(records[['id', 'file_path']], args.data_dir / 'images', classes, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    ids, _, probabilities = predict_probabilities(model, loader, device)
    if (probabilities.shape != (len(records), len(classes)) or
            not np.isfinite(probabilities).all() or len(ids) != len(set(ids)) or
            set(ids) != set(sample['id'])):
        raise ValueError('Incomplete or invalid model predictions')
    predicted = pd.Series([classes[i] for i in probabilities.argmax(1)], index=ids)
    result = pd.DataFrame({'id': sample['id'], 'label': sample['id'].map(predicted)})
    if result.isna().any().any() or not set(result['label']) <= set(provided):
        raise ValueError('Invalid prediction labels')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} predictions to {args.output}; nothing uploaded to Kaggle.')


if __name__ == '__main__':
    main()
