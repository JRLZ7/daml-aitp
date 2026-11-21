"""Manifest-based image loading and verified shared split assignments."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .common import sha256


def load_training_records(data_dir, split_dir):
    data_dir, split_dir = Path(data_dir), Path(split_dir)
    metadata = json.loads((split_dir / 'metadata.json').read_text())
    for name in ('train', 'classes'):
        if sha256(data_dir / f'{name}.csv') != metadata['manifest_sha256'][name]:
            raise ValueError(f'{name}.csv differs from the shared split manifest')
    classes = pd.read_csv(data_dir / 'classes.csv')['label'].tolist()
    if len(classes) != len(set(classes)):
        raise ValueError('Duplicate class names')
    records = pd.read_csv(data_dir / 'train.csv', dtype=str)
    if not records['id'].is_unique or not set(records['label']) <= set(classes):
        raise ValueError('Invalid training IDs or labels')
    frames = []
    for name in ('train_ids.csv', 'val_ids.csv'):
        path = split_dir / name
        if sha256(path) != metadata['split_sha256'][name]:
            raise ValueError(f'{name} differs from its recorded hash')
        ids = pd.read_csv(path, dtype=str)
        if list(ids.columns) != ['id'] or not ids['id'].is_unique:
            raise ValueError(f'Invalid split IDs: {name}')
        frame = ids.merge(records, on='id', how='left', validate='one_to_one', sort=False)
        if frame.isna().any().any():
            raise ValueError(f'Unknown or incomplete records: {name}')
        frames.append(frame)
    fit, val = frames
    if set(fit['id']) & set(val['id']):
        raise ValueError('Training and validation overlap')
    if set(fit['id']) | set(val['id']) != set(records['id']):
        raise ValueError('Split does not cover training records exactly')
    if len(fit) != metadata['train_rows'] or len(val) != metadata['validation_rows']:
        raise ValueError('Split counts differ from metadata')
    if set(fit['label']) != set(classes) or set(val['label']) != set(classes):
        raise ValueError('A split is missing classes')
    return fit, val, classes, metadata


class SceneDataset(Dataset):
    def __init__(self, records, image_root, classes, transform):
        self.records = records.reset_index(drop=True).copy()
        self.image_root = Path(image_root).resolve()
        self.class_to_idx = {label: i for i, label in enumerate(classes)}
        if len(classes) != len(self.class_to_idx):
            raise ValueError('Duplicate class names')
        self.transform = transform
        if 'label' in self.records and not set(self.records['label']) <= set(classes):
            raise ValueError('Unknown scene label')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        path = (self.image_root / row['file_path']).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError('Image path escapes image root')
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        target = self.class_to_idx[row['label']] if 'label' in row else -1
        return image, target, row['id']
