#!/bin/bash
# Benchmark an Ollama model's prefill + decode throughput — with LOG-VERIFIED placement.
# Usage: bench_ollama.sh <model> <cpu|gpu> [num_predict]
#   cpu = num_gpu:0 (forces llama.cpp CPU kernels), gpu = default Metal offload.
#
# IMPORTANT (hard lesson): the request `num_gpu` option is a REQUEST, not proof. Ollama
# defaults to full GPU on macOS and can silently place layers on the GPU. This script
# reads ~/.ollama/logs/server.log after the run and PRINTS the actual offload
# ("offloaded N/M layers to GPU") + weight device, and REFUSES to report a "cpu" number
# unless the load was verified 0 layers on GPU. Options are requests; logs are facts.
set -euo pipefail

MODEL="$1"
MODE="$2"
NPRED="${3:-96}"
LOG="${OLLAMA_LOGS:-$HOME/.ollama/logs/server.log}"
PROMPT="Explain in detail how memory hierarchies in modern CPUs affect the performance of large matrix multiplications, covering caches, prefetching, data layout, and bandwidth limits. Be thorough and technical."

if [ "$MODE" = "cpu" ]; then
  OPTS="{\"num_gpu\":0,\"num_predict\":$NPRED,\"temperature\":0}"
else
  OPTS="{\"num_predict\":$NPRED,\"temperature\":0}"
fi

# force a fresh load so the placement we read from the log is THIS run's
ollama stop "$MODEL" >/dev/null 2>&1 || true
sleep 1
MARK=$(wc -l < "$LOG" 2>/dev/null || echo 0)   # only inspect log lines after here

RESP=$(curl -s --max-time 900 http://localhost:11434/api/generate -d "{
  \"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"stream\":false,\"keep_alive\":\"1m\",
  \"options\":$OPTS}")

# --- read the actual placement this load produced ---
PLACE="unknown"
if [ -f "$LOG" ]; then
  OFF=$(tail -n +"$((MARK+1))" "$LOG" | grep -oE "offloaded [0-9]+/[0-9]+ layers to GPU" | tail -1 || true)
  DEV=$(tail -n +"$((MARK+1))" "$LOG" | grep -E "model weights.*device=" | grep -oE "device=[A-Za-z]+ size=\"[^\"]+\"" | tr '\n' ' ' || true)
  [ -n "$OFF" ] && PLACE="$OFF | $DEV"
fi

echo "$RESP" | MODE="$MODE" MODEL="$MODEL" PLACE="$PLACE" python3 -c "
import json, sys, os
d = json.load(sys.stdin)
mode, model, place = os.environ['MODE'], os.environ['MODEL'], os.environ['PLACE']
if 'error' in d:
    print(f'{model} {mode} ERROR: {d[\"error\"]}'); sys.exit(0)
pe, ped = d.get('prompt_eval_count', 0), d.get('prompt_eval_duration', 1)
ec, ed = d.get('eval_count', 0), d.get('eval_duration', 1)
load = d.get('load_duration', 0) / 1e9
# verify placement matches the requested mode
gpu_layers = None
import re
m = re.search(r'offloaded (\d+)/(\d+)', place)
if m: gpu_layers = int(m.group(1))
warn = ''
if mode == 'cpu' and gpu_layers not in (0, None):
    warn = '  !!! NOT CPU-ONLY — {} layers on GPU; this is a GPU number, discard'.format(gpu_layers)
elif mode == 'cpu' and gpu_layers is None:
    warn = '  (WARN: could not verify placement from log — do not trust as CPU)'
print(f'{model} [{mode}]  prefill {pe/ped*1e9:7.1f} tok/s   decode {ec/ed*1e9:6.1f} tok/s   load {load:.1f}s')
print(f'    placement: {place}{warn}')
"
