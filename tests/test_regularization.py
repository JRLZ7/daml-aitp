import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

from src.regularization import check_cached_batch, export_head, paired_comparison


class RegularizationTests(unittest.TestCase):
    def test_cache_rejects_wrong_id_order_or_target_mapping(self):
        records = pd.DataFrame({'id': ['a', 'b'], 'label': ['x', 'y']})
        features = torch.ones(2, 3)
        with self.assertRaisesRegex(ValueError, 'IDs'):
            check_cached_batch((features, torch.tensor([0, 1]), ['b', 'a']), records, ['x', 'y'])
        with self.assertRaisesRegex(ValueError, 'labels'):
            check_cached_batch((features, torch.tensor([1, 0]), ['a', 'b']), records, ['x', 'y'])

    def test_paired_comparison_aligns_ids_and_counts_both_directions(self):
        reference = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'true_label': ['x'] * 4,
                                  'pred_label': ['y', 'x', 'y', 'x']})
        candidate = pd.DataFrame({'id': ['d', 'c', 'b', 'a'], 'true_label': ['x'] * 4,
                                  'pred_label': ['y', 'x', 'x', 'x']})
        result = paired_comparison(reference, candidate, resamples=100, seed=42)
        self.assertEqual((result['corrected'], result['regressed'], result['net_correct']), (2, 1, 1))
        self.assertEqual(result['accuracy_difference'], .25)
        with self.assertRaisesRegex(ValueError, 'ID sets'):
            paired_comparison(reference, candidate.iloc[:3])

    def test_exported_multiclass_head_preserves_probabilities_and_class_order(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(36, 4)).astype('float32')
        y = np.tile([0, 1, 2], 12)
        estimator = LogisticRegression(C=.1, l1_ratio=0., max_iter=1000).fit(x, y)
        model = nn.Module()
        model.head = nn.Linear(4, 3)
        probabilities, error = export_head(model, estimator, x, 3)
        self.assertLess(error, 1e-5)
        np.testing.assert_array_equal(probabilities.argmax(1), estimator.predict(x))


if __name__ == '__main__':
    unittest.main()
