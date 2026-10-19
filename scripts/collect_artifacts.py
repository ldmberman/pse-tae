"""Copy research checkpoints/sidecars into a portable bundle, with SHA-256 inventory.

Only result directories are read. Data, environments, credentials, and source
files are excluded. Pass --include-predictions for cached NPZ evaluations/features.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def canonical(rel):
    parts=list(rel.parts)
    if parts[0].startswith('results_'):parts=['results',parts[0][8:],*parts[1:]]
    elif len(parts)>1 and parts[0]=='results' and parts[1].startswith('Fold_'):parts.insert(1,'baseline')
    return Path(*parts)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-predictions',action='store_true');a=p.parse_args()
    source=a.source.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True);rows=[]
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not(folder.name=='results' or folder.name.startswith('results_')):continue
        for file in sorted(folder.rglob('*')):
            if not file.is_file() or file.is_symlink():continue
            checkpoint=file.suffix in {'.pt','.pth','.joblib','.ckpt','.safetensors','.ubj'} or file.name.endswith('.pth.tar')
            if not(checkpoint or file.suffix=='.json' or (a.include_predictions and file.suffix=='.npz')):continue
            rel=canonical(file.relative_to(source));dest=output/rel;h=sha(file)
            if dest.exists() and sha(dest)!=h:raise RuntimeError(f'Artifact collision: {rel}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(file,dest)
            rows.append(dict(path=str(rel),source=str(file.relative_to(source)),bytes=file.stat().st_size,sha256=h,checkpoint=checkpoint))
    (output/'manifest.json').write_text(json.dumps(dict(files=rows),indent=2)+'\n')
    print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),checkpoints=sum(r['checkpoint'] for r in rows)),indent=2))
if __name__=='__main__':main()
