import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from src.common import sha256
from src.dataset import SceneDataset, load_training_records
from src.train import extract_features, append_result
from src.models import SceneClassifier, WEIGHT_NAME
from src.predict import main as predict_main


class PipelineTests(unittest.TestCase):
    def test_result_register_preserves_rows_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experiments').mkdir()
            path = root / 'experiments/results.csv'
            path.write_bytes(b'run_id,status\n')
            with patch('src.train.ROOT', root):
                append_result({'run_id': 'one', 'status': 'completed'})
                self.assertNotIn(b'\r', path.read_bytes())
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'already recorded'):
                    append_result({'run_id': 'one', 'status': 'completed'})
                self.assertEqual(path.read_bytes(), before)

    def test_saved_model_prediction_preserves_sample_order_and_ignores_extra_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            for name in ('a', 'b'):
                Image.new('RGB', (16, 12), 'gray').save(root / 'images' / f'{name}.jpg')
            classes = ['bedroom', 'airport_inside']
            pd.DataFrame({'label': classes}).to_csv(root / 'classes.csv', index=False)
            pd.DataFrame({'id': ['a', 'b'], 'file_path': ['a.jpg', 'b.jpg'],
                          'label': ['INVALID', 'INVALID']}).to_csv(root / 'test.csv', index=False)
            pd.DataFrame({'id': ['b', 'a'], 'label': ['bedroom', 'bedroom']}).to_csv(
                root / 'sample_submission.csv', index=False)
            model = SceneClassifier(2, pretrained=False)
            with torch.no_grad():
                model.head.weight.zero_()
                model.head.bias.copy_(torch.tensor([0., 2.]))
            checkpoint = root / 'model.pt'
            torch.save({'architecture': 'resnet50', 'classes': classes,
                        'pretrained_weights': WEIGHT_NAME, 'state_dict': model.state_dict()}, checkpoint)
            output = root / 'prediction.csv'
            argv = ['predict', '--checkpoint', str(checkpoint), '--data-dir', str(root),
                    '--output', str(output), '--device', 'cpu']
            with patch('sys.argv', argv):
                predict_main()
            result = pd.read_csv(output)
            self.assertEqual(result.to_dict('list'),
                             {'id': ['b', 'a'], 'label': ['airport_inside', 'airport_inside']})

    def test_loader_preserves_manifest_order_and_explicit_class_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (12, 8), 'red').save(root / 'a.jpg')
            Image.new('L', (8, 12), 120).save(root / 'b.jpg')
            frame = pd.DataFrame({'id': ['second', 'first'], 'file_path': ['b.jpg', 'a.jpg'],
                                  'label': ['bedroom', 'airport_inside']})
            dataset = SceneDataset(frame, root, ['bedroom', 'airport_inside'], ToTensor())
            image, target, identifier = dataset[0]
            self.assertEqual((target, identifier), (0, 'second'))
            self.assertEqual(image.shape[0], 3)  # Grayscale converted to RGB.
            self.assertEqual(dataset[1][1:], (1, 'first'))

    def test_loader_rejects_path_outside_image_root(self):
        frame = pd.DataFrame({'id': ['x'], 'file_path': ['../secret.jpg']})
        with tempfile.TemporaryDirectory() as directory:
            dataset = SceneDataset(frame, directory, ['room'], ToTensor())
            with self.assertRaisesRegex(ValueError, 'escapes'):
                dataset[0]

    def test_overlap_is_rejected_even_with_consistent_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = pd.DataFrame({'id': ['a', 'b', 'c', 'd'],
                                    'file_path': ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
                                    'label': ['x', 'y', 'x', 'y']})
            records.to_csv(root / 'train.csv', index=False)
            pd.DataFrame({'label': ['x', 'y']}).to_csv(root / 'classes.csv', index=False)
            pd.DataFrame({'id': ['a', 'b']}).to_csv(root / 'train_ids.csv', index=False)
            pd.DataFrame({'id': ['a', 'd']}).to_csv(root / 'val_ids.csv', index=False)
            metadata = {
                'manifest_sha256': {name: sha256(root / f'{name}.csv') for name in ('train', 'classes')},
                'split_sha256': {name: sha256(root / name) for name in ('train_ids.csv', 'val_ids.csv')},
                'train_rows': 2, 'validation_rows': 2,
            }
            (root / 'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_training_records(root, root)

    def test_extraction_preserves_frozen_weights_and_batchnorm_statistics(self):
        torch.manual_seed(42)
        encoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten())
        encoder.requires_grad_(False)
        before = {key: tensor.clone() for key, tensor in encoder.state_dict().items()}
        images = torch.rand(5, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels, torch.arange(5)), batch_size=2)
        features, targets, ids = extract_features(encoder, loader, torch.device('cpu'), 'Test')
        self.assertEqual([int(i) for i in ids], list(range(5)))
        self.assertTrue(torch.equal(targets, labels))
        self.assertFalse(features.requires_grad)
        head = nn.Linear(2, 2)
        nn.CrossEntropyLoss()(head(features), targets).backward()
        self.assertIsNotNone(head.weight.grad)
        for key, tensor in encoder.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]), key)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
