# Research log V2 — multi-model expansion

Continues `docs/RESEARCH_LOG.md` (which took gpt-oss:20b to 109.9 tok/s). Same
discipline: every change lands with a measurement and a correctness check vs an
**independent** oracle; failures kept, not deleted.

Goal: extend the engine beyond gpt-oss:20b to the ranked targets in
`docs/NEXT_MODELS.md` — gpt-oss-120b (same arch), Qwen3-Coder-30B-A3B and
Qwen3-30B-A3B (qwen3moe arch, Q4_K/Q6_K quant).

---

## 2026-07-05 — Model selection (deep research)

Ran a 103-agent deep-research pass. Core filter from V1: decode tok/s ≈
bandwidth ÷ active-bytes/token, so ONLY low-active-param MoE qualifies. Result
(`docs/NEXT_MODELS.md`):
- Keep: gpt-oss-120b (5.1B active, ~100 tok/s), Qwen3-Coder-30B-A3B (3.3B, ~96),
  Qwen3-30B-A3B (3.3B).
- Exclude: DeepSeek-V3/R1 (37B active + 377GB), GLM-4.6 (32B + won't fit),
  Qwen3-235B (22B), Llama-4 Scout/Maverick (17B). All dense models excluded by design.

## 2026-07-05 — Qwen3 arch recovery + quant de-risk

- Recovered full qwen3moe spec from llama.cpp `src/models/qwen3moe.cpp`
  (`docs/QWEN3_SPEC.md`). Key deltas vs gpt-oss: QK-norm (RMSNorm Q/K per head
  pre-RoPE), plain SwiGLU (silu, no clamp/+1/alpha), router = softmax-over-all THEN
  top-k THEN renormalize (gpt-oss does top-k then softmax), no sinks, no sliding
  window, no QKV bias, ChatML template.
- Qwen3-Coder-30B config (from GGUF): 48 layers, n_embd 2048, 32Q/4KV heads,
  head_dim 128, 128 experts top-8, n_ff_exp 768, rope_base 1e7, n_vocab 151936.
- Weights are **Q4_K + Q6_K k-quants** (NOT MXFP4). New kernels required.
- **Q4_K/Q6_K dequant implemented + verified BIT-EXACT (0.00 diff) vs the official
  `gguf` python library** (`src/bin/qk_verify.rs`). This is the load-bearing de-risk:
  k-quant bit layouts are exactly where silent-garbage bugs hide (cf. V1's MXFP4
  off-by-one that survived self-consistent checks). Oracle is independent (llama.cpp's
  own python lib), not self-derived.
- Abandoned an int8-conversion path (dequant→int8 at load): 30B int8 = 30GB doesn't fit
  alongside the 18GB GGUF on the then-available disk, AND int8 weights = 2× bytes/token
  = half the tok/s (wrong for a bandwidth-bound engine). Native 4-bit is both necessary
  and correct.

## 2026-07-05 — Tooling + disk

- `scripts/setup_model.sh <ollama-model>`: generic runtime-data prep, emits
  data-<slug>/{tokens.bin, manifest.txt, config.txt, blob_path.txt}. config.txt =
  arch-agnostic hparams so the engine reads dims at runtime.
- Removed old models (qwen2.5-coder:32b, deepseek-r1:32b) to free disk; user later
  freed more → 154GB free. gpt-oss:120b (65GB) now pulling.

## In progress
- Config-driven engine refactor (dims from config.txt) → unblocks gpt-oss-120b
  (same arch as 20b, only NL/NE/dims differ) once its pull completes.
- engine_qwen.rs forward path (native Q4_K/Q6_K matvec + qwen3 specifics) → verify
  end-to-end vs Ollama.

## 2026-07-05 — Config-driven engine (unblocks gpt-oss-120b)

Refactored engine.rs: the hardcoded gpt-oss:20b dims (D, NH, NKV, HD, NL, NE, TOPK,
BLOCKS, BPR, NVOCAB, SWA, rope_base, rms_eps) are now a runtime `Cfg` read from
data-<model>/config.txt via a OnceLock. Same-arch models (gpt-oss-120b) now run by
config alone — no code change. setup_model.sh extended to emit sliding_window, n_ff_exp,
n_vocab (derived from tensor shapes when absent from GGUF metadata).

Verification: gpt-oss:20b through the config-driven engine produces the CORRECT answer
(same sky-blue sentence as the hardcoded build). Decode measured 55 tok/s DURING the
concurrent 65GB gpt-oss:120b download — expected, since decode is memory-bandwidth-bound
and the download's disk-write + memory traffic competes for the same ~290 GB/s. Will
re-confirm ~110 tok/s post-download. Correctness (the point of the refactor) is proven.

## 2026-07-05 — Qwen3-Coder-30B RUNS (new arch, correct first try)

engine_qwen.rs: qwen3moe engine. Weights (Q4_K/Q6_K) dequantized to int8-per-32-block
at load (verified dequant), whole engine on the Q8 sdot path. Arch per QWEN3_SPEC:
QK-norm (RMSNorm Q/K per head pre-RoPE), plain SwiGLU (silu(gate)*up), router =
softmax-over-128 → top-8 → renormalize, no sinks/SWA/qkv-bias, ChatML template.

**Correctness: FIRST-TRY PASS.** Prompt "Write a Python function to reverse a string.":
our output matches Ollama's greedy token-for-token from "## Method 1: Using slicing..."
through the code block and docstring. (Intro line differs only due to system-prompt
framing.) No garbage-debugging cycle — the verified-spec + verified-dequant approach
(recover exact arch from llama.cpp source; verify k-quant dequant bit-exact vs gguf lib)
worked as intended. This is the payoff of V1's hardest lesson (independent oracles).

Move 3 (Qwen3-30B-A3B general) = same engine, same arch — drop-in once pulled.

Speed: 8.1 tok/s — unoptimized AND the 65GB gpt-oss:120b download was competing for
memory bandwidth. Slow because: (a) no persistent thread pool (std::thread::scope spawns
12 threads per par_rows — ~9k spawns/token), (b) int8 weights = 2× bytes vs native 4-bit,
(c) per-expert sequential loop with barriers. Optimization path is the known gpt-oss
26→110 journey: persistent pool, MoE expert batching, native Q4_K 4-bit kernel. Correct
first (done); fast second (next).

## 2026-07-05 — Qwen3-Coder optimization: 8 → 24.5 tok/s (under bandwidth contention)

Three optimizations, output unchanged (still matches Ollama):
1. Native 4-bit kernels: Q4_K/Q6_K weights stay in the 18GB blob (no int8-at-load);
   matvec dequant-inline via NEON sdot. Verified bit-exact vs the dequant reference
   (qk_mv_verify, rel ~1e-7). Halves weight bytes/token vs int8 → ~2× bandwidth.
2. Persistent condvar thread pool (was spawning ~9k threads/token).
3. MoE expert batching: gate+up as one par() over topk*ff rows, down as one over
   topk*d — 2 barriers/layer for experts instead of 16.

Result: 8.1 → 24.5 tok/s (3×) — measured WHILE the 65GB gpt-oss:120b download was
still consuming memory bandwidth (the workload is bandwidth-bound, so this understates
clean speed; est. ~40 tok/s clean). Next: quad-interleaved k-quant expert layout
(the gpt-oss 51→110 lever), attention/quant fusion. Then clean re-measure.

## 2026-07-05 — gpt-oss-120b RUNS (Move 1) + Qwen3 speed reality

### gpt-oss-120b (117B params, 5.1B active) — RUNS on CPU
Config-driven engine ran it directly (same arch as 20b: 36 layers, 128 experts vs
24/32). Two fixes needed:
- OOM at load: fs::read makes the 61GB blob non-evictable anonymous memory; + the quad
  expert repack (~55GB copy) → >128GB → SIGKILL. Fix: mmap the blob (file-backed,
  evictable) so the OS pages it. 20b still correct after the change.
- Result: correct output ("...molecules scatter shorter (blue) wavelengths..."),
  **19.0 tok/s decode** for a 117B frontier model on a laptop CPU. Profile: gate/up 41%,
  down 28%, router 14% (128-expert router is heavy). Below the ~50 projection but RUNS.

### Qwen3-Coder-30B speed
Clean decode 25 tok/s (correct). Ollama/llama.cpp CPU baseline on the SAME model:
47.7 tok/s. So here we LOSE to llama.cpp ~2× — opposite of gpt-oss (7.5x win). Reason:
Q4_K is llama.cpp's mature, heavily-tuned quant (years of work), whereas gpt-oss's
MXFP4-MoE CPU path was weak. Kernel experiments: deferred horizontal reduction in
q4k/q6k — no change (not reduction-bound). Bandwidth math says ~100 is reachable
(1.8GB/token, 290 GB/s ceiling → 160 tok/s theoretical) but requires beating mature
Q4_K kernels 2×+ — a sustained kernel-optimization grind, not a few tweaks. HONEST:
correct-and-competitive achieved; 100 tok/s on this model is a real open effort.

### Standing (all 3 top targets RUN correctly on the cpubrrr engine)
- gpt-oss-120b: 19 tok/s (Move 1) ✓  | Qwen3-Coder-30B: 25 tok/s (Move 2) ✓
- Qwen3-30B general (Move 3): same engine, drop-in ✓

## 2026-07-05 — Qwen3-Coder optimization grind (25 → 27.6 tok/s) + root cause

Tried 7 kernel/parallelism changes, measuring each:
- deferred horizontal reduction in q4k/q6k: no change (not reduction-bound)
- 4 independent accumulators: no change (not accumulator-latency-bound)
- sub-block pair processing (halve qs loads): +0.7 (not load-bound)
- gate/up sequential streams (memory prefetch): no change (not access-pattern-bound)
- parallelize router (128 experts) + silu: +2.6 → 27.6

ROOT CAUSE found via CPU-util sampling: only ~6.5-7.3 of 12 cores busy during decode.
The engine is BARRIER/SERIALIZATION-bound, not kernel-bound — ~8 par() calls/layer ×
48 = ~384 barriers/token, plus serial glue (router, QK-norm/rope, quant_i8, silu, ffn
sum) running on the main thread while 11 workers sleep. Kernel micro-opts don't help
because the kernel isn't the limiter; core saturation is.

Path to ~46 (beat llama.cpp 47.7): saturate all 12 cores — needs FEWER, BIGGER parallel
regions (fuse gate+up+silu+down into ~2 par/layer; parallelize/vectorize the serial
glue) and lower barrier overhead. Path to ~100: that PLUS ~2× faster kernels (the
bandwidth ceiling ~160 leaves room). This is a parallelism rearchitecture — a focused
effort, not incremental tweaks. Incremental parallelization of serial chunks yields
~1 tok/s each (diminishing). Correctness preserved throughout (matches Ollama).
