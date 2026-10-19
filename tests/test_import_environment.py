import os
from pathlib import Path
import subprocess
import sys
import unittest


class ImportEnvironmentTests(unittest.TestCase):
    def test_probe_imports_preserve_gpu_visibility(self):
        root=Path(__file__).resolve().parents[1]
        code="""
import importlib, os, sys
sys.path.insert(0, 'scripts')
before=os.environ.get('CUDA_VISIBLE_DEVICES')
for name in ['scripts.frozen_pair_probes', 'scripts.frozen_multiclass_probes',
             'scripts.logit_adjust', 'scripts.xgb_moments',
             'scripts.evaluate_checkpoint', 'scripts.distill_student']:
    importlib.import_module(name)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == before, name
"""
        for value in [None,'','2']:
            with self.subTest(visibility=value):
                env=os.environ.copy()
                env.pop('CUDA_VISIBLE_DEVICES',None)
                if value is not None:env['CUDA_VISIBLE_DEVICES']=value
                subprocess.run([sys.executable,'-c',code],cwd=root,env=env,
                               check=True,capture_output=True,text=True)


if __name__=='__main__':
    unittest.main()
