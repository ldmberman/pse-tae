"""Guard moment definitions and identity of the historical neural/tree partitions."""
import unittest
import numpy as np
from sklearn.model_selection import KFold
from learning.splits import partitions


class DiagnosticTests(unittest.TestCase):
    def test_partitions_match_original_loader_order(self):
        n=191412
        order=np.arange(n);np.random.RandomState(1).shuffle(order)
        original=list(KFold(5,shuffle=False).split(np.arange(n)))
        nv=len(original[0][1])
        for fold,(trainval,test) in enumerate(original,1):
            split=partitions(n,fold)
            np.testing.assert_array_equal(split['train'],order[trainval][:-nv])
            np.testing.assert_array_equal(split['val'],order[trainval][-nv:])
            np.testing.assert_array_equal(split['test'],order[test])

    def test_streamed_kurtosis_matches_direct_and_excludes_constants(self):
        import torch
        from scripts.probe_kurtosis import embedding_excess
        torch.set_num_threads(1)
        raw=np.random.RandomState(3).normal(size=(3,64,41)).astype(np.float32)
        raw[:,0,:]=7
        centered=raw.astype(np.float64)-raw.mean(-1,dtype=np.float64)[...,None]
        m2=(centered**2).mean(-1);m4=(centered**4).mean(-1);valid=m2>1e-12
        expected=np.mean(m4[valid]/m2[valid]**2-3)
        for chunk in [1,7,100]:
            score,count,total=embedding_excess(torch.nn.Identity(),raw,np.zeros((3,64)),np.ones((3,64)),chunk)
            self.assertAlmostEqual(score,expected,places=12)
            self.assertEqual(count,3*63);self.assertEqual(total,3*64)
        raw=np.tile(np.array([-1,1],np.float32),(1,64,1))
        score,_,_=embedding_excess(torch.nn.Identity(),raw,np.zeros((1,64)),np.ones((1,64)),1)
        self.assertEqual(score,-2.)  # m4 / m2² = 1, excess = 1 − 3

if __name__=='__main__':unittest.main()
