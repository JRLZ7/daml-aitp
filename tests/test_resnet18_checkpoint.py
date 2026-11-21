import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models import load_checkpoint
from src.predict import main as predict_main


class ResNet18CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = resnet18(weights=None)
        cls.model.fc = nn.Linear(cls.model.fc.in_features, 2)
        cls.model.eval()
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        cls.checkpoint = {
            'architecture': 'resnet18', 'class_names': ['bedroom', 'airport_inside'],
            'class_to_idx': {'bedroom': 0, 'airport_inside': 1},
            'weights': 'IMAGENET1K_V1', 'model_state_dict': cls.model.state_dict(),
            'image_size': 224, 'resize_size': list(transform.resize_size),
            'mean': transform.mean, 'std': transform.std,
        }

    def test_historical_checkpoint_preserves_logits_and_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(self.checkpoint, path)
            restored, transform, metadata = load_checkpoint(path, torch.device('cpu'))
            image = transform(Image.new('RGB', (280, 190), 'gray')).unsqueeze(0)
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), self.model(image), rtol=0, atol=0)
            self.assertEqual(metadata['classes'], ['bedroom', 'airport_inside'])
            self.assertFalse(restored.training)

    def test_rejects_inconsistent_class_mapping_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            for changes, error in [
                ({'class_to_idx': {'bedroom': 1, 'airport_inside': 0}}, 'class mapping'),
                ({'resize_size': [232]}, 'preprocessing'),
                ({'weights': 'places365'}, 'ImageNet'),
            ]:
                with self.subTest(changes=changes):
                    torch.save({**self.checkpoint, **changes}, path)
                    with self.assertRaisesRegex(ValueError, error):
                        load_checkpoint(path, torch.device('cpu'))

    def test_submission_uses_historical_mapping_and_sample_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            for name in ('a', 'b'):
                Image.new('RGB', (40, 30), 'gray').save(root / 'images' / f'{name}.jpg')
            pd.DataFrame({'label': self.checkpoint['class_names']}).to_csv(root / 'classes.csv', index=False)
            pd.DataFrame({'id': ['a', 'b'], 'file_path': ['a.jpg', 'b.jpg']}).to_csv(root / 'test.csv', index=False)
            pd.DataFrame({'id': ['b', 'a'], 'label': ['bedroom', 'bedroom']}).to_csv(
                root / 'sample_submission.csv', index=False)
            state = {name: value.clone() for name, value in self.checkpoint['model_state_dict'].items()}
            state['fc.weight'].zero_()
            state['fc.bias'].copy_(torch.tensor([0., 2.]))
            path = root / 'model.pt'
            torch.save({**self.checkpoint, 'model_state_dict': state}, path)
            output = root / 'submission.csv'
            with patch('sys.argv', ['predict', '--checkpoint', str(path), '--data-dir', str(root),
                                    '--output', str(output), '--device', 'cpu']):
                predict_main()
            self.assertEqual(pd.read_csv(output).to_dict('list'),
                             {'id': ['b', 'a'], 'label': ['airport_inside', 'airport_inside']})
