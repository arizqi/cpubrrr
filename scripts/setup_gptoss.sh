#!/usr/bin/env bash
# Prepare runtime data for the cputrain engine from a locally-pulled gpt-oss:20b.
#
# Requires: ollama with gpt-oss:20b pulled (`ollama pull gpt-oss:20b`), python3.
# Produces: data/tokens.bin, data/manifest.txt, data/blob_path.txt
#
# The engine reads model weights directly from Ollama's GGUF blob (no copy),
# using the byte offsets recorded in manifest.txt.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HERE/data}"
MANIFEST="${OLLAMA_MODELS:-$HOME/.ollama/models}/manifests/registry.ollama.ai/library/gpt-oss/20b"

if [ ! -f "$MANIFEST" ]; then
  echo "error: gpt-oss:20b not found. Run:  ollama pull gpt-oss:20b" >&2
  exit 1
fi

BLOBS="${OLLAMA_MODELS:-$HOME/.ollama/models}/blobs"
DIGEST=$(python3 -c "
import json,sys
m=json.load(open('$MANIFEST'))
for l in m['layers']:
    if l['mediaType']=='application/vnd.ollama.image.model':
        print(l['digest'].replace('sha256:','sha256-')); break
")
BLOB="$BLOBS/$DIGEST"
[ -f "$BLOB" ] || { echo "error: model blob missing: $BLOB" >&2; exit 1; }

mkdir -p "$OUT"
echo "model blob: $BLOB"
echo "$BLOB" > "$OUT/blob_path.txt"

echo "indexing GGUF tensors..."
python3 "$HERE/scripts/gguf_index.py" "$BLOB" > "$OUT/index.json"

echo "dumping tokenizer..."
python3 "$HERE/scripts/dump_tokenizer.py" "$BLOB" "$OUT"

echo "building tensor manifest..."
python3 -c "
import json
d=json.load(open('$OUT/index.json'))
with open('$OUT/manifest.txt','w') as w:
    w.write(f\"data_start {d['data_start']}\n\")
    for t in d['tensors']:
        n=1
        for x in t['dims']: n*=x
        w.write(f\"{t['name']} {t['type']} {t['offset']} {n}\n\")
"
echo "done. data ready in: $OUT"
echo "run the engine with:  ./target/release/engine $OUT \$(cat $OUT/blob_path.txt) 'Why is the sky blue?'"
