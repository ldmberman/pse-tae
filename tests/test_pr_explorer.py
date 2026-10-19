import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_pr_explorer import extract, sha, test_payload


class ExplorerSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.cache=self.root/'fold_1_features.npz'
        np.savez(self.cache,logits=np.random.RandomState(3).normal(size=(100,20)).astype('f4'),
                 labels=np.arange(100)%20)
        (self.root/'fold_1_extraction.json').write_text(json.dumps({'checkpoint_sha256':'baseline'}))
        self.data=self.root/'explorer'
        extract(argparse.Namespace(baseline_cache=self.cache,fold=1,data=self.data))
        self.validation=json.loads((self.data/'provenance.json').read_text())

    def test_matching_fold_exports_both_splits(self):
        payload=test_payload(self.data,self.validation)
        self.assertEqual(payload['provenance']['split'],'test')
        self.assertEqual(payload['provenance']['fold'],1)
        self.assertIn('logits',payload)

    def test_different_checkpoint_rejected(self):
        self.validation['baseline_checkpoint_sha256']='other'
        with self.assertRaises(AssertionError):test_payload(self.data,self.validation)

    def test_validation_parcels_cannot_be_used_as_test(self):
        path=self.data/'test_predictions.npz'
        with np.load(self.data/'validation_predictions.npz') as data:
            np.savez_compressed(path,logits=data['logits'],labels=data['labels'],
                                indices=data['indices'],fold=1,split='test')
        record_path=self.data/'test_provenance.json'
        record=json.loads(record_path.read_text())
        record['predictions_sha256']=sha(path)
        record_path.write_text(json.dumps(record))
        with self.assertRaises(AssertionError):test_payload(self.data,self.validation)

    def test_changed_prediction_file_rejected(self):
        path=self.data/'test_predictions.npz'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(AssertionError):test_payload(self.data,self.validation)


if __name__=='__main__':
    unittest.main()
