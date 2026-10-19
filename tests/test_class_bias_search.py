import unittest
import numpy as np
from scripts.build_pr_explorer import metrics
from scripts.sweep_class_biases import sweep
from scripts.search_class_biases import grid_sweep, optimize


class ClassBiasSearchTests(unittest.TestCase):
    def test_fast_sweep_matches_full_argmax_including_ties(self):
        rng = np.random.RandomState(31)
        logits = rng.randint(-2, 3, (80, 20)).astype(float)
        labels = rng.randint(0, 20, 80)
        biases = rng.choice([-.5, 0, .5], 20)
        for row in sweep(logits, labels, biases, [-1, 0, 1]):
            adjusted = biases.copy()
            adjusted[row['class_id']] = row['bias']
            direct = metrics(labels, (logits + adjusted).argmax(1))
            for metric in ['miou', 'oa', 'mean_recall']:
                self.assertEqual(row[metric], direct[metric])

    def test_cumulative_sweep_matches_direct_for_ties_and_float_rounding(self):
        rng = np.random.RandomState(22)
        for logits in [rng.randint(-2, 3, (100, 20)).astype(float),
                       rng.normal(size=(100, 20)),
                       rng.choice([1e16, -1e16, .1, .2], (100, 20))]:
            labels = rng.randint(0, 20, len(logits))
            biases = rng.choice([-.1, 0, .1], 20)
            grid = [-3, -.1, 0, .1, 3]
            for row in grid_sweep(logits, labels, biases, grid):
                trial = biases.copy()
                trial[row['class_id']] = row['bias']
                predictions = (logits + trial).argmax(1)
                direct = metrics(labels, predictions)
                self.assertEqual(row['correct'], int(np.sum(labels == predictions)))
                for key in ['miou', 'oa', 'mean_recall']:
                    self.assertEqual(row[key], direct[key])

    def test_search_keeps_accuracy_constraint_and_monotonic_miou(self):
        rng = np.random.RandomState(42)
        logits = rng.normal(size=(80, 20))
        labels = rng.randint(0, 20, 80)
        minimum = int(np.sum(labels == logits.argmax(1)))
        result = optimize(logits, labels, np.zeros(20), minimum, max_updates=3)
        last = metrics(labels, logits.argmax(1))['miou']
        for row in result['trace']:
            self.assertGreaterEqual(row['correct'], minimum)
            self.assertGreater(row['miou'], last)
            last = row['miou']


if __name__ == '__main__':
    unittest.main()
