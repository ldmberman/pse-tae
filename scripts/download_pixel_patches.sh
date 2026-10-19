#!/usr/bin/env bash
# S2-Agri Patch, Zenodo record 5815523. No GPU is required.
set -euo pipefail
DEST="${1:-./data}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
ARCHIVE="$DEST/S2-2017-T31TFM-PixelPatch.7z"
URL='https://zenodo.org/records/5815523/files/S2-2017-T31TFM-PixelPatch.7z?download=1'
CHECKSUM='ab6fa97393a664f7aa8a61ba3d362066'
if command -v 7z >/dev/null 2>&1; then
  EXTRACTOR=7z
elif command -v 7zz >/dev/null 2>&1; then
  EXTRACTOR=7zz
else
  echo 'Install 7z/7zz first: Ubuntu apt install p7zip-full; macOS brew install sevenzip.' >&2
  exit 1
fi
verify() {
  python3 - "$ARCHIVE" "$CHECKSUM" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():raise SystemExit(1)
h=hashlib.md5()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
raise SystemExit(0 if h.hexdigest()==sys.argv[2] else 1)
PY
}
if ! verify; then
  curl --fail --location --continue-at - --output "$ARCHIVE" "$URL"
  verify || { echo 'Zenodo checksum mismatch; extraction skipped.' >&2; exit 1; }
fi
"$EXTRACTOR" x -y -o"$DEST" "$ARCHIVE"
python3 - "$DEST" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
expected=root/'s2-2017-IGARSS-NNI-NPY'
if (expected/'DATA').is_dir():
    print('Pixel patches:',expected)
else:
    candidates=[p.parent for p in root.rglob('DATA') if 'NNI' in str(p.parent)]
    if len(candidates)!=1:raise SystemExit('Check extracted archive layout; expected s2-2017-IGARSS-NNI-NPY/DATA')
    if not expected.exists():expected.symlink_to(candidates[0],target_is_directory=True)
    print('Pixel patches:',expected)
PY
