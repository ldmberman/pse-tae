import unittest
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.probe_confusion_pairs import exact_match, fast_auc, miou, rank_pairs


class ConfusionPairTests(unittest.TestCase):
    def test_oracle_gain_matches_direct_correction(self):
        cm = np.array([[90, 10, 0], [7, 2, 1], [1, 2, 7]])
        for r in rank_pairs(cm):
            true, pred, n = r['true'], r['predicted'], r['errors']
            fixed = cm.copy()
            fixed[true, pred] -= n
            fixed[true, true] += n
            expected = 100 * (miou(fixed) - miou(cm))
            self.assertAlmostEqual(expected, r['oracle_miou_gain_points'])
            self.assertGreater(expected, 0)
            np.testing.assert_array_equal(fixed.sum(1), cm.sum(1))

    def test_exact_matching_stays_in_split_and_balances_counts(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        sizes = np.array([6, 10, 10, 6, 10, 15, 10])
        allowed = np.arange(6)
        pairs = exact_match(allowed, labels, sizes, 0, 1, 31)
        self.assertEqual(pairs.shape, (2, 2))
        self.assertTrue(np.isin(pairs, allowed).all())
        self.assertEqual(np.unique(pairs).size, pairs.size)
        np.testing.assert_array_equal(sizes[pairs[:, 0]], sizes[pairs[:, 1]])
        np.testing.assert_array_equal(labels[pairs[:, 0]], 0)
        np.testing.assert_array_equal(labels[pairs[:, 1]], 1)

    def test_fast_auc_with_ties(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        score = np.array([.5, .5, .2, .1, .7, .9])
        self.assertAlmostEqual(fast_auc(y, score), roc_auc_score(y, score))


if __name__ == '__main__':
    unittest.main()
