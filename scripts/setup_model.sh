#!/usr/bin/env bash
# Generic runtime-data prep for any Ollama-pulled model.
# Usage: scripts/setup_model.sh <ollama-model>        e.g. qwen3-coder:30b  or  gpt-oss:120b
# Produces: data-<slug>/{tokens.bin, manifest.txt, config.txt, blob_path.txt}
#
# config.txt is arch-agnostic key/value hparams parsed from GGUF metadata, so the
# engine reads model dimensions at runtime instead of hardcoding them.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: setup_model.sh <ollama-model>}"
NAME="${MODEL%%:*}"; TAG="${MODEL##*:}"
SLUG="$(echo "$MODEL" | tr ':/' '__')"
OUT="$HERE/data-$SLUG"
STORE="${OLLAMA_MODELS:-$HOME/.ollama/models}"
MANIFEST="$STORE/manifests/registry.ollama.ai/library/$NAME/$TAG"

[ -f "$MANIFEST" ] || { echo "error: $MODEL not pulled. Run: ollama pull $MODEL" >&2; exit 1; }
DIGEST=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
for l in m['layers']:
    if l['mediaType']=='application/vnd.ollama.image.model':
        print(l['digest'].replace('sha256:','sha256-')); break
")
BLOB="$STORE/blobs/$DIGEST"
[ -f "$BLOB" ] || { echo "error: blob missing: $BLOB" >&2; exit 1; }

mkdir -p "$OUT"
echo "$BLOB" > "$OUT/blob_path.txt"
echo "model: $MODEL"
echo "blob:  $BLOB"

echo "indexing GGUF..."
python3 "$HERE/scripts/gguf_index.py" "$BLOB" > "$OUT/index.json"

echo "dumping tokenizer..."
python3 "$HERE/scripts/dump_tokenizer.py" "$BLOB" "$OUT"

echo "writing manifest + config..."
python3 -c "
import json
d=json.load(open('$OUT/index.json'))
m=d['meta']
# tensor manifest
with open('$OUT/manifest.txt','w') as w:
    w.write(f\"data_start {d['data_start']}\n\")
    for t in d['tensors']:
        n=1
        for x in t['dims']: n*=x
        w.write(f\"{t['name']} {t['type']} {t['offset']} {n}\n\")
# arch-agnostic hparam config: strip the arch prefix so keys are uniform
arch=m.get('general.architecture','?')
def g(*suffixes, default=None):
    for k,v in m.items():
        for s in suffixes:
            if k==f'{arch}.{s}' or k.endswith('.'+s): return v
    return default
# n_vocab: metadata key is often absent; derive from token_embd/output tensor.
n_vocab = g('vocab_size')
if n_vocab is None:
    for t in d['tensors']:
        if t['name'] in ('token_embd.weight','output.weight'):
            n_vocab = max(t['dims']); break
# n_ff_exp: metadata or derive from an expert tensor (ggml dims = [n_embd, n_ff_exp, n_expert])
n_ff_exp = g('expert_feed_forward_length')
if n_ff_exp is None:
    for t in d['tensors']:
        if t['name'].endswith('ffn_gate_exps.weight'):
            n_ff_exp = t['dims'][1]; break
cfg={
  'arch': arch,
  'n_embd': g('embedding_length'),
  'n_layer': g('block_count'),
  'n_head': g('attention.head_count'),
  'n_head_kv': g('attention.head_count_kv'),
  'head_dim': g('attention.key_length'),
  'n_expert': g('expert_count'),
  'n_expert_used': g('expert_used_count'),
  'n_ff_exp': n_ff_exp,
  'sliding_window': g('attention.sliding_window'),
  'rope_freq_base': g('rope.freq_base'),
  'rms_eps': g('attention.layer_norm_rms_epsilon'),
  'n_vocab': n_vocab,
  'context_length': g('context_length'),
}
with open('$OUT/config.txt','w') as w:
    for k,v in cfg.items():
        if v is not None: w.write(f'{k} {v}\n')
print('config:')
for k,v in cfg.items():
    if v is not None: print(f'  {k} = {v}')
"
echo "done -> $OUT"
