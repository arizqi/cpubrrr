#!/usr/bin/env bash
# Same-session CPU-vs-CPU: upstream llama.cpp vs cpubrrr, alternating.
#
# Lesson #6 discipline: both sides measured in the SAME session, interleaved, so
# desktop load and thermal state hit them equally. Names the exact binary and flags.
#
# Upstream is forced genuinely CPU-only with `-dev none -ngl 0` -- this homebrew
# build ships a Metal backend (unlike the July -DGGML_METAL=OFF build), and this
# repo has already been burned once by a "CPU" baseline silently using the GPU.
#
# Caveat baked in on purpose: llama-bench's tg-N generates N tokens from an EMPTY
# context, while cpubrrr always prepends its 73-token harmony system prompt, so
# cpubrrr decodes at strictly higher positions (more attention work, more KV traffic)
# for the same N. The handicap runs AGAINST cpubrrr, so the comparison is conservative.
set -u
GGUF="${GGUF:-.upstream_gguf/gpt-oss-20b-MXFP4.gguf}"
DATA="${DATA:-data-gpt-oss_20b}"
ENGINE="${ENGINE:-./target/release/engine_gpt2}"
NGEN="${NGEN:-128}"
ROUNDS="${ROUNDS:-3}"
THREADS="${THREADS:-12}"
BLOB=$(cat "$DATA/blob_path.txt")

echo "== same-session CPU decode: upstream llama.cpp vs cpubrrr =="
echo "   model gpt-oss-20b MXFP4 | ngen=$NGEN | threads=$THREADS | rounds=$ROUNDS"
echo "   upstream: $(llama-bench --version 2>&1 | grep -i version | head -1)"
echo

for r in $(seq 1 "$ROUNDS"); do
  u=$(llama-bench -m "$GGUF" -dev none -ngl 0 -t "$THREADS" -n "$NGEN" -p 0 -r 1 2>/dev/null \
        | grep -E "tg|gen" | tail -1 | sed 's/.*| *\([0-9.]*\) *±.*/\1/')
  c=$("$ENGINE" "$DATA" "$BLOB" "Why is the sky blue?" "$NGEN" 2>/dev/null \
        | grep -o "tok_s=[0-9.]*" | cut -d= -f2)
  echo "  round $r   upstream ${u:-ERR}   cpubrrr ${c:-ERR}"
  echo "${u:-0}" >> /tmp/_vs_up.$$; echo "${c:-0}" >> /tmp/_vs_cp.$$
done

echo
UB=$(sort -rn /tmp/_vs_up.$$ | head -1); CB=$(sort -rn /tmp/_vs_cp.$$ | head -1)
rm -f /tmp/_vs_up.$$ /tmp/_vs_cp.$$
echo "  upstream llama.cpp best : $UB tok/s"
echo "  cpubrrr best            : $CB tok/s"
python3 -c "print(f'  ratio                   : {$CB/$UB:.2f}x') if $UB > 0 else print('  ratio: upstream failed')"
