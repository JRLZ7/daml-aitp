"""Independent checks of the real baseline artifacts and checkpoint re-evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import sha256, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--recheck-dir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run, recheck = args.run_dir, args.recheck_dir
checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
original_path = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-11ad3fa6.pth'
original = torch.load(original_path, map_location='cpu', weights_only=True)
assert sha256(original_path) == checkpoint['pretrained_source']['sha256']
checked = 0
for name, tensor in checkpoint['state_dict'].items():
    if name.startswith('encoder.'):
        assert torch.equal(tensor, original[name.removeprefix('encoder.')]), name
        checked += 1
before = pd.read_csv(run / 'val_predictions.csv')
after = pd.read_csv(recheck / 'val_predictions.csv')
classes = checkpoint['classes']
columns = [f'p_{label}' for label in classes]
assert before['id'].tolist() == after['id'].tolist()
assert before['pred_label'].tolist() == after['pred_label'].tolist()
assert before['true_label'].tolist() == after['true_label'].tolist()
assert np.isfinite(before[columns].to_numpy()).all()
assert np.allclose(before[columns].sum(axis=1), 1., atol=1e-6)
assert set(before['id']) == set(pd.read_csv(root/'splits/val_ids.csv')['id'])
truth = pd.read_csv(root/'data/raw/train.csv').set_index('id')['label']
assert before['true_label'].tolist() == before['id'].map(truth).tolist()
assert all(before['pred_label'] == [classes[i] for i in before[columns].to_numpy().argmax(1)])
metrics = json.loads((run/'metrics.json').read_text())
assert metrics['val_accuracy'] == float((before.true_label == before.pred_label).mean())
assert sha256(run/'best.pt') == metrics['checkpoint_sha256']
max_difference = float(np.abs(before[columns].to_numpy() - after[columns].to_numpy()).max())
assert max_difference < 1e-4, max_difference
report = {
    'encoder_tensors_identical_to_imagenet_checkpoint': checked,
    'checkpoint_hash_verified': True,
    'validation_ids_and_labels_verified': True,
    'all_reloaded_predictions_match': True,
    'max_probability_difference_after_reload': max_difference,
    'recheck_device': json.loads((recheck/'evaluation.json').read_text())['device'],
    'val_examples': len(before), 'correct': metrics['correct'],
}
write_json(run/'verification.json', report)
print(json.dumps(report, indent=2))
