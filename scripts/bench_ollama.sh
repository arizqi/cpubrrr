#!/bin/bash
# Benchmark an Ollama model's prefill + decode throughput.
# Usage: bench_ollama.sh <model> <cpu|gpu> [num_predict]
# cpu = num_gpu:0 (llama.cpp CPU kernels), gpu = default Metal offload.
set -euo pipefail

MODEL="$1"
MODE="$2"
NPRED="${3:-96}"
PROMPT="Explain in detail how memory hierarchies in modern CPUs affect the performance of large matrix multiplications, covering caches, prefetching, data layout, and bandwidth limits. Be thorough and technical."

if [ "$MODE" = "cpu" ]; then
  OPTS="{\"num_gpu\":0,\"num_predict\":$NPRED,\"temperature\":0}"
else
  OPTS="{\"num_predict\":$NPRED,\"temperature\":0}"
fi

RESP=$(curl -s --max-time 900 http://localhost:11434/api/generate -d "{
  \"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"stream\":false,\"keep_alive\":0,
  \"options\":$OPTS}")

echo "$RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'error' in d:
    print(f'$MODEL $MODE ERROR: {d[\"error\"]}'); sys.exit(0)
pe, ped = d.get('prompt_eval_count', 0), d.get('prompt_eval_duration', 1)
ec, ed = d.get('eval_count', 0), d.get('eval_duration', 1)
load = d.get('load_duration', 0) / 1e9
print(f'$MODEL [$MODE]  prefill {pe/ped*1e9:7.1f} tok/s ({pe} tok)   decode {ec/ed*1e9:6.1f} tok/s ({ec} tok)   load {load:.1f}s')
"
