import unittest

from PIL import Image
import torch
from torchvision.models import ResNet50_Weights

from src.preprocessing import Letterbox, make_preprocessing, preprocessing_from_checkpoint


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_retains_all_four_corners_and_aspect_ratio(self):
        image = Image.new('RGB', (400, 100), 'white')
        corners = [(0, 0, 'red'), (380, 0, 'green'), (0, 80, 'blue'), (380, 80, 'yellow')]
        for x, y, color in corners:
            image.paste(color, (x, y, x + 20, y + 20))
        result = Letterbox()(image)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((0, 0)), (124, 116, 104))
        self.assertEqual(result.getpixel((0, 83)), (124, 116, 104))
        self.assertEqual(result.getpixel((0, 84)), (255, 0, 0))
        self.assertEqual(result.getpixel((223, 84)), (0, 128, 0))
        self.assertEqual(result.getpixel((0, 139)), (0, 0, 255))
        self.assertEqual(result.getpixel((223, 139)), (255, 255, 0))
        self.assertEqual(result.getpixel((0, 140)), (124, 116, 104))

    def test_portrait_grayscale_is_rgb_and_padding_normalizes_near_zero(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        transform, _ = make_preprocessing('letterbox_224_imagenet_mean', weights)
        output = transform(Image.new('L', (100, 400), 255))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(output[:, 0, 0].abs().max().item(), .02)

    def test_legacy_checkpoints_keep_identical_pretrained_transform(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        image = Image.new('RGB', (320, 180), (17, 89, 203))
        legacy = preprocessing_from_checkpoint({}, weights)
        self.assertTrue(torch.equal(legacy(image), weights.transforms()(image)))

    def test_saved_policy_roundtrip_and_tampering_rejection(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        image = Image.new('RGB', (80, 160), 'blue')
        transform, spec = make_preprocessing('letterbox_224_imagenet_mean', weights)
        loaded = preprocessing_from_checkpoint({'preprocessing': spec}, weights)
        self.assertTrue(torch.equal(transform(image), loaded(image)))
        spec['size'] = 320
        with self.assertRaisesRegex(ValueError, 'specification'):
            preprocessing_from_checkpoint({'preprocessing': spec}, weights)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            preprocessing_from_checkpoint({'config': {'model': {
                'preprocessing': 'letterbox_224_imagenet_mean'}}}, weights)


if __name__ == '__main__':
    unittest.main()
