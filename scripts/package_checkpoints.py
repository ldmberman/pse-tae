"""Build a deduplicated, named checkpoint bundle from the audited inventory.

Run IDs identify archived runs, not inferred seeds. Historical paths are retained
as restore aliases; current server files win ambiguous aliases unless an ensemble
manifest pins a particular hash. No checkpoint is deserialized or modified.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_artifacts import canonical, sha


def method(path):
    parts = Path(path).parts
    family = parts[1]
    variant = parts[-2]
    if family.startswith('grouped_corrections'):
        return {'temporal': 'tgt-correction', 'spatial_nopos': 'tgt-spatial-correction',
                'control': 'embedding-correction'}[variant]
    if family.startswith('grouped_scratch') or family in {'complementarity', 'distillation_teachers'}:
        return {'grouped': 'tgt', 'capacity': 'capacity', 'baseline': 'patch-baseline'}[variant]
    if family == 'checkpoint_control':
        return 'patch-baseline'
    if family.startswith('capacity_large'):
        return 'capacity-large'
    if family.startswith('cnn_ta_td_large'):
        return 'cnn-tgt-date-dropout-large'
    if 'cnn_temporal' in path:
        return 'cnn-tgt-correction'
    if family == 'cnn_corrections':
        return 'cnn-spatial-correction' if variant == 'cnn_spatial' else 'embedding-correction'
    if family in {'date_dropout', 'requested_comparison'}:
        return 'all-dates-control' if variant == 'all_dates' else 'date-dropout'
    if family.startswith('distillation'):
        base = 'distilled-baseline' if variant == 'distilled' else 'ce-control'
        return base + ('-large-warmstart' if 'large_warmstart' in family else '')
    if family == 'ensemble_library':
        return Path(path).name.split('_')[0]
    if family == 'pair_probes':
        return 'pair-probe'
    if family == 'multiclass_probes':
        return 'xgboost' if '_xgb_' in path else 'logistic'
    if family.startswith(('frozen_corrections', 'frozen_trajectories', 'pixel_temporal')):
        prefix = ('pixel-correction' if family.startswith('frozen_corrections') else
                  'pixel-probe' if family.startswith('frozen_trajectories') else 'pixel-scratch')
        return prefix + '-' + variant.replace('_', '-')
    return {'inv_sqrt': 'class-weighted', 'tau_decoder': 'decoder-adjusted',
            'raw_kurtosis_geom': 'raw-kurtosis', 'kurtosis_sg': 'kurtosis-stopped-gradient',
            'before_examples_134717': 'baseline-archived'}.get(family, family.replace('_', '-'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, default=Path('records/checkpoints.json'))
    p.add_argument('--source', type=Path, default=Path('.'))
    p.add_argument('--objects', type=Path, required=True, help='Server-only files named by SHA-256')
    p.add_argument('--sidecars', type=Path, required=True, help='Extracted server JSON files')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Use an empty output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(a.inventory.read_text())
    ensemble = json.loads((a.source / 'configs/ensembles.json').read_text())
    pinned = {m['checkpoint']: m['sha256'] for m in ensemble['members'].values()}
    preferred = {m['sha256']: m['checkpoint'] for m in ensemble['members'].values()}
    entries, aliases = [], defaultdict(list)

    def add(source, rel, restore=(), **extra):
        dest = a.output / rel
        if dest.exists():
            raise ValueError(f'Bundle path collision: {rel}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        entry = dict(path=str(rel), bytes=dest.stat().st_size, sha256=sha(dest),
                     restore_paths=list(restore), **extra)
        entries.append(entry)
        return entry

    for row in inventory['files']:
        locations = row['locations']
        paths = {str(canonical(Path(loc['path']))) for loc in locations}
        original = preferred.get(row['sha256'], min(paths, key=lambda s: (
            any(w in s for w in ['training_transition', 'training_pause', 'snapshot', 'ensemble_library']), len(s), s)))
        origins = sorted({loc['source'] for loc in locations if str(canonical(Path(loc['path']))) == original})
        origin = 'server' if 'server' in origins else 'local'
        folds = {int(m.group(1)) for loc in locations for m in re.finditer(r'Fold_(\d+)', loc['path'])}
        if row['sha256'] in preferred:
            folds.add(ensemble['fold'])
        if len(folds) > 1:
            raise ValueError(f'Cross-fold duplicate needs review: {original}')
        fold = next(iter(folds), None)
        parent = str(Path(original).parent)
        run_key = re.sub(r'Fold_\d+', 'Fold_N', parent) + ':' + origin
        run = ('smoke-' if 'smoke' in original else 'run-') + hashlib.sha256(run_key.encode()).hexdigest()[:10]
        name = Path(original).name
        name = re.sub(r'^epoch_(\d+)', r'epoch-\1', name)
        if name == 'model.pth.tar':
            name = 'best.pt'
        elif name.endswith(('_best.pt', '_checkpoint.pt')):
            name = 'best.pt'
        # Preserve stale and current versions of the same historical path.
        rel = Path('checkpoints') / method(original) / run / f'fold-{fold or "unknown"}' / name
        if (a.output / rel).exists():
            rel = rel.with_name(rel.stem + '-' + row['sha256'][:10] + rel.suffix)
        local = [a.source / loc['path'] for loc in locations if loc['source'] == 'local']
        source = next((f for f in local if f.is_file() and sha(f) == row['sha256']), a.objects / row['sha256'])
        if not source.is_file() or sha(source) != row['sha256'] or source.stat().st_size != row['bytes']:
            raise ValueError(f'Missing or changed checkpoint: {original}')
        entry = add(source, rel, checkpoint=True, method=method(original), run=run, fold=fold,
                    seed=None, original_path=original, locations=locations)
        for alias in paths:
            server = any(loc['source'] == 'server' and str(canonical(Path(loc['path']))) == alias for loc in locations)
            aliases[alias].append((server, entry))

    conflicts = []
    for alias, candidates in sorted(aliases.items()):
        required = pinned.get(alias)
        chosen = (next(e for _, e in candidates if e['sha256'] == required) if required else
                  sorted(candidates, key=lambda item: (item[0], item[1]['sha256']))[-1][1])
        chosen['restore_paths'].append(alias)
        if len(candidates) > 1:
            conflicts.append(dict(path=alias, selected_sha256=chosen['sha256'],
                                  alternatives=[e['sha256'] for _, e in candidates]))

    # Prefer server metadata for current paths, preserve alternate JSON separately.
    sidecars = defaultdict(dict)
    for label, root in [('local', a.source), ('server', a.sidecars)]:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not (folder.name == 'results' or folder.name.startswith('results_')):
                continue
            for file in sorted(folder.rglob('*.json')):
                if file.is_file() and not file.is_symlink():
                    alias = str(canonical(file.relative_to(root)))
                    if 'smoke' not in alias.lower():
                        sidecars[alias][label] = file
    for alias, candidates in sorted(sidecars.items()):
        selected = candidates.get('server', candidates.get('local'))
        add(selected, Path('metadata') / alias, [alias], checkpoint=False)
        if len(candidates) == 2 and sha(candidates['local']) != sha(candidates['server']):
            add(candidates['local'], Path('metadata/local-archive') / alias, checkpoint=False)

    add(a.source / 'configs/ensembles.json', Path('ensembles.json'), checkpoint=False)
    index = dict(checkpoints=[e for e in entries if e.get('checkpoint')], alias_conflicts=conflicts)
    index_path = a.output / 'checkpoint_index.json'
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    entries.append(dict(path=index_path.name, bytes=index_path.stat().st_size, sha256=sha(index_path), restore_paths=[], checkpoint=False))
    card = a.output / 'README.md'
    card.write_text('''---
license: mit
tags:
- satellite-imagery
- time-series-classification
---
# PSE+TAE experiments

Checkpoint archive for https://github.com/ldmberman/pse-tae.
TGT means Temporal Group Trajectories; CNN→TGT uses CNN-pooled trajectories.

`checkpoints/<method>/<run-id>/fold-N/<checkpoint>` contains byte-identical original
files, deduplicated by SHA-256. Run IDs identify archived paths, not random seeds.
Unknown seed/fold metadata stays unknown. Partial/interrupted runs are included
for completeness and must not be treated as final measurements.

`checkpoint_index.json` records provenance and maps every checkpoint to original
paths. `ensembles.json` pins the historical ensemble members by hash. `metadata/`
contains experiment JSON sidecars. Datasets and prediction/feature caches are excluded.

Download and restore from the research repository:

```bash
python scripts/hf_checkpoints.py download --repo-id ldmberman/pse-tae-experiments --folder artifacts/hf
python scripts/hf_checkpoints.py restore --folder artifacts/hf --destination .
```

Restoration verifies every file and refuses to overwrite different local artifacts.
Ambiguous historical paths default to the server version, except hash-pinned ensemble
members. All alternate checkpoint versions remain individually accessible in the archive.
See the source README for training, inference, cache generation and plotting commands.
''')
    entries.append(dict(path=card.name, bytes=card.stat().st_size, sha256=sha(card), restore_paths=[], checkpoint=False))
    manifest = dict(schema_version=2, files=entries, checkpoint_count=len(inventory['files']),
                    checkpoint_bytes=sum(r['bytes'] for r in inventory['files']), alias_conflicts=conflicts)
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['checkpoint_count', 'checkpoint_bytes']}, indent=2))
    print('Bundle files:', len(entries), 'alias conflicts preserved:', len(conflicts))


if __name__ == '__main__':
    main()
