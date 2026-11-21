"""Validate course data and save a reproducible stratified split of training IDs."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import sklearn
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'splits')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = config['project']['seed']
    fraction = config['data']['validation_split']
    require(0 < fraction < 1, 'validation_split must be between zero and one')
    data_dir = args.data_dir.resolve()
    specs = {
        'train': (5327, ['id', 'file_path', 'label']),
        'test': (1340, ['id', 'file_path']),
        'classes': (67, ['label']),
        'sample_submission': (1340, ['id', 'label']),
    }
    tables = {}
    for name, (count, columns) in specs.items():
        frame = pd.read_csv(data_dir / f'{name}.csv', dtype=str)
        require(list(frame.columns) == columns, f'{name}: unexpected columns')
        require(len(frame) == count, f'{name}: expected {count} records')
        require(not frame.isna().any().any(), f'{name}: missing values')
        for column in columns:
            require(frame[column].str.strip().ne('').all(), f'{name}: blank {column}')
        if 'id' in columns:
            require(frame['id'].is_unique, f'{name}: duplicate IDs')
        tables[name] = frame

    train, test = tables['train'], tables['test']
    labels = tables['classes']['label']
    require(labels.is_unique, 'classes: duplicate labels')
    require(len(labels) == config['project']['num_classes'], 'Config class count mismatch')
    require(set(train['label']) == set(labels), 'Training classes do not match classes.csv')
    require(set(tables['sample_submission']['label']) <= set(labels), 'Invalid sample labels')
    require(set(test['id']) == set(tables['sample_submission']['id']), 'Test/sample ID mismatch')
    require(set(train['id']).isdisjoint(test['id']), 'Train/test IDs overlap')
    require(set(train['file_path']).isdisjoint(test['file_path']), 'Train/test paths overlap')

    # Decode images automatically; do not infer or recover withheld test labels.
    image_root = (data_dir / 'images').resolve()
    for name in ('train', 'test'):
        frame = tables[name]
        require(frame['file_path'].is_unique, f'{name}: duplicate image paths')
        for relative in tqdm(frame['file_path'], desc=f'Checking {name} images'):
            path = (image_root / relative).resolve()
            require(path.is_relative_to(image_root), f'Path escapes images directory: {relative}')
            require(path.is_file(), f'Missing image: {relative}')
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert('RGB').load()

    # Sort before splitting so CSV row order cannot alter the split assignments.
    train = train.sort_values('id').reset_index(drop=True)
    fit, val = train_test_split(
        train, test_size=fraction, stratify=train['label'], random_state=seed,
    )
    fit = fit.sort_values('id')
    val = val.sort_values('id')
    require(set(fit['id']).isdisjoint(val['id']), 'Training/validation overlap')
    require(set(fit['id']) | set(val['id']) == set(train['id']), 'Split loses records')
    require(set(fit['label']) == set(val['label']) == set(labels), 'Split loses classes')
    counts = pd.DataFrame({
        'total': train['label'].value_counts(),
        'train': fit['label'].value_counts(),
        'validation': val['label'].value_counts(),
    }).sort_index().rename_axis('label').reset_index()
    metadata = {
        'seed': seed,
        'validation_fraction': fraction,
        'train_rows': len(fit),
        'validation_rows': len(val),
        'test_rows': len(test),
        'classes': len(labels),
        'images_verified_and_decoded': len(train) + len(test),
        'split_method': 'Sort training IDs; sklearn train_test_split with label stratification',
        'sklearn_version': sklearn.__version__,
        'manifest_sha256': {name: sha256(data_dir / f'{name}.csv') for name in specs},
        'config_sha256': sha256(args.config),
    }
    outputs = {}
    for name, frame in (('train_ids.csv', fit[['id']]), ('val_ids.csv', val[['id']]),
                        ('class_counts.csv', counts)):
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False, lineterminator='\n')
        outputs[name] = buffer.getvalue()
    metadata['split_sha256'] = {
        name: hashlib.sha256(outputs[name].encode()).hexdigest()
        for name in ('train_ids.csv', 'val_ids.csv')
    }
    outputs['metadata.json'] = json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    # Keep existing team assignments intact if the data/config/environment changes.
    for name, content in outputs.items():
        path = args.output_dir / name
        require(not path.exists() or path.read_bytes() == content.encode(),
                f'{path} differs. Review changes and use a new --output-dir for a new split.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_bytes(content.encode())
    print(f'Validated all images. Saved {len(fit)} training and {len(val)} validation IDs to {args.output_dir}')


if __name__ == '__main__':
    main()
