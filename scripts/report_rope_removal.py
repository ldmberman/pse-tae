"""Record the fixed nine-versus-eight comparison, with no retraining or reweighting search."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ensembles', type=Path, default=Path('outputs/ensembles'))
    p.add_argument('--difference', type=Path, default=Path('outputs/pair_review/rope_difference.json'))
    p.add_argument('--manifest', type=Path, default=Path('configs/ensembles.json'))
    p.add_argument('--output', type=Path, default=Path('records/rope_removal.json'))
    a = p.parse_args()
    config = json.loads(a.manifest.read_text())
    assert [n for n in config['ensembles']['nine'] if n != 'rope'] == config['ensembles']['without_rope']
    result = dict(fold=1, change='Remove only RoPE and renormalize the remaining equal weights to 1/8. '
                  'No retraining or weight selection. The distilled student retains its nine-model teacher.',
                  manifest_sha256=sha(a.manifest), splits={},
                  paired_test_miou_difference=json.loads(a.difference.read_text()))
    for split in ['val', 'test']:
        rows = {}
        reference = None
        for ensemble in ['nine', 'without_rope']:
            path = a.ensembles / ensemble / split
            report = json.loads((path / 'metrics.json').read_text())
            assert report['fold'] == 1 and report['split'] == split
            assert report['members'] == config['ensembles'][ensemble]
            with np.load(path / 'predictions.npz') as f:
                if reference is not None:
                    np.testing.assert_array_equal(reference[0], f['indices'])
                    np.testing.assert_array_equal(reference[1], f['labels'])
                reference = (f['indices'], f['labels'])
            rows[ensemble] = dict(metrics=report['metrics'], members=report['members'],
                                  weights=report['weights'], predictions_sha256=sha(path / 'predictions.npz'))
        rows['delta_points_eight_minus_nine'] = {
            k: 100 * (rows['without_rope']['metrics'][k] - rows['nine']['metrics'][k])
            for k in ['miou', 'oa', 'mean_recall']}
        result['splits'][split] = rows
    delta = result['splits']['test']['delta_points_eight_minus_nine']['miou']
    assert abs(delta - result['paired_test_miou_difference']['delta_points']) < 1e-10
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
