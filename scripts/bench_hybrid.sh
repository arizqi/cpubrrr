#!/bin/bash
# E4 (GPU phase): hybrid contention test — cpubrrr on CPU + Ollama on Metal GPU, simultaneously.
# Unified-memory hypothesis (H3): both share DRAM bandwidth, so combined ≈ max(solo), not sum.
# Usage: bench_hybrid.sh
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=qwen3-coder:30b
DATA=data-qwen3-coder_30b
BLOB=$(cat $DATA/blob_path.txt)
PROMPT="Explain in detail how memory hierarchies in modern CPUs affect the performance of large matrix multiplications."

# make sure GPU copy is loaded (fresh, default = full Metal)
ollama stop $MODEL >/dev/null 2>&1 || true
curl -s http://localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"10m\",\"options\":{\"num_predict\":4}}" >/dev/null

echo "=== concurrent: ollama GPU (256 tok) + cpubrrr CPU, started together ==="
( curl -s --max-time 900 http://localhost:11434/api/generate -d "{
    \"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"stream\":false,\"keep_alive\":\"10m\",
    \"options\":{\"num_predict\":256,\"temperature\":0}}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'OLLAMA-GPU concurrent decode {d[\"eval_count\"]/d[\"eval_duration\"]*1e9:6.1f} tok/s')" ) &
OLL=$!

./target/release/engine_qwen2 $DATA "$BLOB" "$PROMPT" 2>&1 | grep -iE "tok/s" | sed 's/^/CPUBRRR concurrent /'
wait $OLL
