import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.hf_checkpoints import main, restore, verify
from scripts.package_checkpoints import method


class CheckpointBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.dest = self.root / 'restored'
        self.content = b'checkpoint fixture'
        (self.bundle / 'best.pt').write_bytes(self.content)
        self.manifest = dict(schema_version=2, files=[dict(
            path='best.pt', bytes=len(self.content),
            sha256=hashlib.sha256(self.content).hexdigest(),
            restore_paths=['results/example/Fold_1/best.pt'])])
        self.save_manifest()

    def save_manifest(self):
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_verified_restore_and_repeat(self):
        restore(self.bundle, self.dest)
        restore(self.bundle, self.dest)
        self.assertEqual((self.dest / 'results/example/Fold_1/best.pt').read_bytes(), self.content)

    def test_corrupt_checkpoint_rejected(self):
        (self.bundle / 'best.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            verify(self.bundle)

    def test_conflict_preflight_preserves_local_files(self):
        target = self.dest / 'results/example/Fold_1/best.pt'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'existing experiment')
        self.manifest['files'][0]['restore_paths'].insert(0, 'results/new.pt')
        self.save_manifest()
        with self.assertRaises(ValueError):
            restore(self.bundle, self.dest)
        self.assertEqual(target.read_bytes(), b'existing experiment')
        self.assertFalse((self.dest / 'results/new.pt').exists())

    def test_path_escape_rejected(self):
        self.manifest['files'][0]['restore_paths'] = ['../escape.pt']
        self.save_manifest()
        with self.assertRaises(ValueError):
            restore(self.bundle, self.dest)

    def test_tgt_names(self):
        self.assertEqual(method('results/grouped_scratch/Fold_1/grouped/best.pt'), 'tgt')
        self.assertEqual(method('results/grouped_corrections/Fold_1/temporal/best.pt'), 'tgt-correction')
        self.assertEqual(method('results/cnn_corrections/Fold_1/cnn_temporal/best.pt'), 'cnn-tgt-correction')

    def test_download_accepts_either_visibility_and_pins_revision(self):
        for visibility in [False, True]:
            with self.subTest(private=visibility):
                api=Mock()
                api.repo_info.return_value=SimpleNamespace(private=visibility,sha='fixed-revision')
                download=Mock()
                hub=SimpleNamespace(HfApi=Mock(return_value=api),snapshot_download=download)
                with patch.dict('sys.modules',{'huggingface_hub':hub}), patch(
                        'sys.argv',['hf_checkpoints.py','download','--repo-id','owner/checkpoints',
                                    '--folder',str(self.bundle)]):
                    main()
                download.assert_called_once_with('owner/checkpoints',repo_type='model',
                                                 revision='fixed-revision',local_dir=self.bundle)
                api.create_repo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
