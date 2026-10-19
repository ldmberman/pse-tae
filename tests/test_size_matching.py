import copy
import unittest
from collections import Counter

from scripts.control_kurtosis_size import match, paired_metrics


def row(pid, label, pixels, fold=1):
    return dict(pid=pid, label=label, pixels=pixels, fold=fold, kurtosis=float(pixels))


class SizeMatchingTests(unittest.TestCase):
    def test_matching_uses_no_scores_and_preserves_folds(self):
        rows = [row('a', 0, 100), row('b', 0, 105), row('c', 18, 100),
                row('d', 18, 109), row('e', 18, 105, fold=2)]
        pairs = match(rows, 1.05)
        self.assertEqual(len(pairs), 2)
        used = []
        for p in pairs:
            a, b = p['meadow'], p['non_agricultural']
            self.assertEqual(a['fold'], b['fold'])
            self.assertLessEqual(max(a['pixels'], b['pixels']) / min(a['pixels'], b['pixels']), 1.05)
            used += [a['pid'], b['pid']]
        self.assertEqual(len(used), len(set(used)))
        altered = copy.deepcopy(rows)
        for i, r in enumerate(altered):
            r['kurtosis'] = (-1)**i * 1000
        ids = lambda ps: [(p['meadow']['pid'], p['non_agricultural']['pid']) for p in ps]
        self.assertEqual(ids(pairs), ids(match(altered, 1.05)))

    def test_exact_matching_balances_size_and_handles_ties(self):
        rows = [row('a', 0, 10), row('b', 0, 10), row('c', 0, 20),
                row('d', 18, 10), row('e', 18, 20), row('f', 18, 30)]
        pairs = match(rows, 1.0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(Counter(p['meadow']['pixels'] for p in pairs),
                         Counter(p['non_agricultural']['pixels'] for p in pairs))
        metrics = paired_metrics(pairs)
        self.assertEqual(metrics['pixel_count_roc_auc'], .5)
        self.assertEqual(metrics['kurtosis_roc_auc'], .5)
        self.assertEqual(metrics['within_pair_concordance'], .5)


if __name__ == '__main__':
    unittest.main()
