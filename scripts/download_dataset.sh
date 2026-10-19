#!/usr/bin/env bash
# Download and extract the Pixel-Set dataset (Zenodo 10.5281/zenodo.5815488).
# The archive is 10.8 GB; the extracted pixel-set is about 29 GB.
set -euo pipefail

DEST="${1:-./data}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
ARCHIVE="$DEST/S2-2017-T31TFM-PixelSet.7z"
URL="https://zenodo.org/records/5815488/files/S2-2017-T31TFM-PixelSet.7z?download=1"
MD5="9b7b8be46dc541401bb8ad9509546a81"

if command -v 7z >/dev/null 2>&1; then
  EXTRACTOR=7z
elif command -v 7zz >/dev/null 2>&1; then
  EXTRACTOR=7zz
else
  echo "Install 7z/7zz first: Ubuntu apt install p7zip-full; macOS brew install sevenzip." >&2
  exit 1
fi
verify() {
  python3 - "$ARCHIVE" "$MD5" <<'PYCODE'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():raise SystemExit(1)
h=hashlib.md5()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
raise SystemExit(0 if h.hexdigest()==sys.argv[2] else 1)
PYCODE
}
if verify; then
  echo "Archive already present and matches the Zenodo checksum."
else
  echo "Downloading $URL"
  curl -L --fail --continue-at - -o "$ARCHIVE" "$URL"
  verify || { echo 'Zenodo checksum mismatch; extraction skipped.' >&2; exit 1; }
fi
echo "Extracting into $DEST"
"$EXTRACTOR" x -y -o"$DEST" "$ARCHIVE"

PKL="$(find "$DEST" -name 'S2-2017-T31TFM-meanstd.pkl' -print -quit)"
if [[ -z "$PKL" ]]; then
  echo "Extraction finished but S2-2017-T31TFM-meanstd.pkl was not found under $DEST" >&2
  exit 1
fi

echo "Dataset folder: $(dirname "$PKL")"
