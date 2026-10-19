"""Upload/download checkpoint bundles, or restore verified historical paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe bundle path: {relative}')
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes destination: {relative}')
    return target


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    seen = set()
    for row in manifest['files']:
        file = safe_path(folder, row['path'])
        if row['path'] in seen:
            raise ValueError(f'Duplicate bundle path: {row["path"]}')
        seen.add(row['path'])
        if file.stat().st_size != row['bytes'] or digest(file) != row['sha256']:
            raise ValueError(f'Changed bundle file: {row["path"]}')
    return manifest


def restore(folder, destination):
    manifest = verify(folder)
    planned = {}
    for row in manifest['files']:
        for alias in row.get('restore_paths', [row['path']] if manifest.get('schema_version', 1) == 1 else []):
            target = safe_path(destination, alias)
            if target in planned and planned[target]['sha256'] != row['sha256']:
                raise ValueError(f'Conflicting restore aliases: {alias}')
            if target.exists() and digest(target) != row['sha256']:
                raise ValueError(f'Refusing to overwrite a different local artifact: {alias}')
            planned[target] = row
    # Complete verification and collision checks before changing any local artifacts.
    for target, row in planned.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / row['path'], target)
    print(f'Restored/verified {len(planned)} artifact paths in {destination}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['upload', 'download', 'restore', 'verify'])
    p.add_argument('--repo-id')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--destination', type=Path, default=Path('.'))
    a = p.parse_args()
    if a.action == 'restore':
        restore(a.folder, a.destination)
        return
    if a.action == 'verify':
        print('Verified', len(verify(a.folder)['files']), 'files')
        return
    if not a.repo_id:
        p.error('--repo-id is required for upload/download')
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if a.action == 'upload':
        manifest = verify(a.folder)
        api.create_repo(a.repo_id, repo_type='model', private=True, exist_ok=True)
        if not api.repo_info(a.repo_id, repo_type='model').private:
            raise RuntimeError('Refusing to upload to a public repository')
        api.upload_folder(repo_id=a.repo_id, repo_type='model', folder_path=a.folder,
                          allow_patterns=[r['path'] for r in manifest['files']] + ['manifest.json'],
                          commit_message='Research checkpoints with TGT names and restore manifest')
        if not api.repo_info(a.repo_id, repo_type='model').private:
            raise RuntimeError('Repository privacy changed during upload')
    else:
        info = api.repo_info(a.repo_id, repo_type='model')
        snapshot_download(a.repo_id, repo_type='model', revision=info.sha, local_dir=a.folder)
        verify(a.folder)
    print(a.repo_id)


if __name__ == '__main__':
    main()
