from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.classical import fit_heads, image_features


class ClassicalTests(unittest.TestCase):
    def test_feature_recipe_shape_histogram_normalization_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gray.png'
            Image.new('L', (13, 7), 128).save(path)
            features = image_features(path)
            self.assertEqual(features.shape, (816,))
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_allclose(features[:768], 128 / 255, atol=1e-7)
            np.testing.assert_allclose(features[768:].reshape(3, 16).sum(axis=1), 1.)

    def test_learned_scaling_is_fit_only_on_supplied_training_rows(self):
        x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        models = fit_heads(x, y, {'C': .01, 'max_iter': 5000, 'seed': 42})
        scaler = models['rgb_hist_linear_svm'].named_steps['standardscaler']
        np.testing.assert_array_equal(scaler.mean_, x.mean(axis=0))
        before = scaler.mean_.copy()
        models['rgb_hist_linear_svm'].predict(np.array([[1000., 2000.]]))
        np.testing.assert_array_equal(scaler.mean_, before)


if __name__ == '__main__':
    unittest.main()
