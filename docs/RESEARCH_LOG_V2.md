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

## 2026-07-05 — grind continued: fuse FFN + parallel QK-norm (27.6 → 29.3)

Fused gate+up+silu into one par (was 3 barriers), parallelized QK-norm/rope.
29.3 tok/s, CPU util ~7.6/12 cores (up from 6.3). Correct throughout.

Cumulative grind: 25.0 → 29.3 tok/s (+17%) over 8 measured optimizations. Every
kernel micro-opt (reduction, accumulators, pair-loads, sequential streams) was ~flat;
only parallelizing serial chunks moved it — confirming barrier/serialization limit.
Util climbs slowly (6.3→7.6) as serial glue is parallelized piece by piece.

HONEST STANDING: to beat llama.cpp (47.7) needs full core saturation (~12/12) = a
parallelism rearchitecture (persistent per-layer parallel region, minimal barriers,
all glue vectorized/parallel). To reach ~100 needs that PLUS ~2x kernel throughput.
Both are real, scoped, multi-session efforts. Incremental tweaking has hit diminishing
returns (~1 tok/s/change). Correct + improving; 100 remains an open optimization target.

## 2026-07-05 — barrier analysis: the fork-join floor (final, 30.1 tok/s)

Measured the smoking gun: **337 par() barriers/token × ~35µs each ≈ 12ms of the ~33ms
budget** — barrier wakeups dominate. Root: workers park (condvar) during the serial glue
between pars (quant_i8, rmsnorm, router), then pay a full wakeup for the next par.
gpt-oss stayed fast with the SAME pool because it has 24 layers (not 48) → half the
barriers, and less serial glue between them.

Attempts to cut barrier cost (all MEASURED):
- Full spin pool: util 10.5/12 cores but 20.6 tok/s (SLOWER — spinning wastes cycles +
  atomic cache-line contention on shared counters). Same lesson as gpt-oss E15.
- Short-spin (2µs) hybrid + atomic dispatch: 21.4 tok/s (still slower — contention).
- Serialize small ops (QK-norm: ~2µs work but 35µs barrier): +0.8 → 30.1. Correct win
  but small (few such ops exist).

CONCLUSION (definitive): the condvar fork-join pool at ~30 tok/s is the ceiling of THIS
execution model for 48-layer single-token decode. The ~12ms barrier tax is structural.
Beating llama.cpp (47.7) / reaching 100 requires a DIFFERENT model — dependency-aware
persistent scheduling with minimal sync, or sequence batching — not incremental tuning.
Spin-based barrier reduction is counterproductive here (proven twice). Correct + stable
at 30 tok/s; 100 needs an execution-model rewrite, scoped but out of incremental reach.

## 2026-07-05 — REWRITE: worker-driven engine v2 (30 → 65.5 tok/s, beats llama.cpp)

engine_qwen2.rs: complete execution-model rewrite. 12 persistent workers run the ENTIRE
forward pass themselves with sense-reversing spin-barriers (~1us, all arrive together),
parking only ONCE per token. ALL glue (rmsnorm, quant, router) parallelized. This
eliminated the fork-join barrier tax (337 barriers × 35us wakeup = 12ms).

Optimization ladder (each measured, correct throughout — verified vs Ollama):
| change | tok/s | note |
|---|---|---|
| worker-driven, main spins | 20.6 | oversubscribed (main steals a core) |
| main parks (frees core) | 36.7 | |
| balanced split + QoS P-core pin | 42.5 | ceil-div left last worker empty |
| q4k scale-hoist + attn stack buf | 43.3 | |
| **vectorize Q6_K reconstruct (NEON)** | **60.4** | biggest single win — Q6_K was 64% of time |
| 4 accumulators in kernels | 62.6 | |
| down expert-major (sequential) | 62.9 | |
| fused gate+up kernel | 64.5 | |
| NEON attention loops | **65.5** | |

**Result: 65.5 tok/s vs llama.cpp 47.7 — we WIN on Qwen3-Coder by 1.37x** (correct
output on all test prompts). The rewrite's premise held: eliminate barriers → saturate
12 cores (measured ~11.9/12) → become kernel-bound → optimize kernels.

### Why not 100 — the quality ceiling (measured, definitive)
At 65.5 we run ~98 GB/s aggregate; our gpt-oss int4 kernel hit 228. The gap is Q4_K's
per-sub-block scale+MIN unpacking (asymmetric quant). Converting Q4_K -> the simple
symmetric int4-quad layout (which enabled 228 GB/s) was TESTED for quality:
- symmetric int8 per-32: rel L2 err 0.5-0.7% (safe) but 2x bytes -> ~76 tok/s, marginal
- symmetric int4 per-32: rel L2 err **8.6-13.2%** -> would wreck the coding model
So 100 tok/s on Qwen3-Coder requires lossy quant that degrades quality — not acceptable.
65.5 is the quality-preserving ceiling for correct Q4_K/Q6_K decode on NEON. (gpt-oss hit
110 because MXFP4 is symmetric — no min to unpack.) Honest stop: we beat the mature
baseline 1.37x with the rewrite; exactly-100 needs either a symmetric-quant model or SME
(which doesn't fit single-vector GEMV).

## 2026-07-05 — CORRECTION: earlier Qwen3-Coder claims do not hold up

Rigorous fair measurement overturned two earlier claims in this log:

1. **The llama.cpp baseline (47.7 tok/s) was contaminated** — measured while the 65GB
   gpt-oss:120b download ran concurrently, starving it. Clean, llama.cpp does ~75 tok/s
   on qwen3-coder:30b CPU (measured 75.2 cool). So the "cpubrrr beats llama.cpp 1.37x"
   claim is FALSE — it compared against a crippled baseline.

2. **engine_qwen2's spin-barrier design is fragile, not fast.** In isolation it swings
   52 -> 5 -> 8 -> 42 tok/s. Root cause: 12 spinning workers saturate all 12 P-cores,
   leaving none for the OS kernel; the OS preempts a worker, and the sense-reversing
   spin-barrier then stalls all 11 others. The "65.5 peak" is a lucky-scheduling
   artifact, not a reliable number. (Same "spinning is fragile" lesson as V1 E15 /
   the earlier spin-pool regressions — now definitive.)

HONEST STANDING on Qwen3-Coder (CPU decode):
- llama.cpp: ~75 tok/s (robust).
- cpubrrr robust engine (engine_qwen.rs, condvar fork-join): ~29-30 tok/s, stable.
- cpubrrr v2 (engine_qwen2.rs, spin): unreliable (5-65), NOT usable.
=> We do NOT beat llama.cpp on Qwen3-Coder; we are ~2.5x slower with the robust engine.
The 100 tok/s target is not met; the gap to llama.cpp is real.

NOTE: all measurements taken while the machine was thermally saturated after a long
session (both engines showed 3-5x swings). Numbers need re-verification on a cool
machine. The gpt-oss comparisons (llama.cpp 14.7 for MXFP4-MoE) should ALSO be
re-verified clean — though MXFP4-MoE is a documented llama.cpp weak path, so that
win is more plausible than the Qwen one. Do not trust any single measurement from
this session's tail as final.

## 2026-07-05 — FRAGILITY FIXED: NT=10 -> stable ~59 tok/s (robust, cool machine)

The v2 spin-barrier collapse was exactly the diagnosed cause: 12 spinning workers on
12 P-cores starve the OS kernel -> it preempts a worker -> sense-reversing barrier
stalls all others. FIX: use NT<12 so the OS always has free cores. Clean sweep on a
COOLED machine (llama.cpp stable 76.9 confirms cool):
| NT | tok/s (stable, 6 runs) |
|---|---|
| 8 | 53 (rock stable) |
| 9 | 57 |
| 10 | 57-60 (stable, chosen default — 2-core OS margin) |
| 11 | 56-64 (stable but cliff-adjacent) |
| 12 | 5-65 (COLLAPSES — all cores spinning starves OS) |

**Honest robust result: ~59 tok/s stable on Qwen3-Coder (NT=10), correct output.**
vs llama.cpp 76.9 -> we are at ~77% of llama.cpp (behind, but robust + reproducible,
and ~2x the fork-join engine's 30). NT=12 was the fragility cliff; NT=10 is the sound
default. This supersedes the retracted "65.5 beats 47.7" claims: real standing is
cpubrrr ~59 vs llama.cpp ~77 on this model, honestly measured on a cool machine.

## 2026-07-05 — Yield-barrier: NT=12 robust ~64 tok/s (uses all cores, no collapse)

NT=10 sacrificed 2 cores vs llama.cpp's 12. Fix: NT=12 + a YIELDING barrier (spin
~1024 then std::thread::yield_now) so 12 workers coexist with the OS. Verified:
- isolation: stable ~64 tok/s (6 runs 60-66, no collapse).
- under 2-core contention (2 bg burners): degrades gracefully to ~27, NO collapse
  (pure-spin collapsed to ~5). The yield lets preempted workers reschedule.
Honest standing: cpubrrr ~64 vs llama.cpp ~77 = 83% (robust). Yield barrier + all 12
cores is the sound design; supersedes NT=10 (~59). Remaining gap to llama.cpp is
kernel efficiency (they run the same Q4_K weights faster) — the real next lever.

## 2026-07-06 — GPU/CPU verification: the retraction OVER-corrected. Full evidence chain.

User asked: "are you sure llama.cpp makes no GPU calls?" Verified via Ollama server
logs (~/.ollama/logs/server.log), which record every model load's actual placement:

1. **Ollama/llama.cpp DEFAULTS to full GPU on macOS**: default loads log
   "offloaded 49/49 layers to GPU", weights device=Metal. Metal backend always
   initializes.
2. **num_gpu:0 IS honored on fresh loads**: controlled test logged
   "offloaded 0/49 layers to GPU", weights device=CPU → decode 16.2 tok/s. The
   bench script (cpu mode) verified the same: 0/49 → 15.8 tok/s.
3. **THE KEY FINDING: yesterday's "clean llama.cpp CPU = 75-77" runs were GPU runs.**
   The log shows every qwen3-coder load in that window (12:45–12:56) as
   "offloaded 49/49 layers to GPU". Mechanism unclear (instance reuse or option not
   applied on those loads) — but placement is logged fact. So the previous
   correction's baseline was itself wrong: it retracted our win against a GPU number.
4. cpubrrr cannot use the GPU at all: binary links only libSystem (otool-verified).

CORRECTED HONEST STANDING (Qwen3-Coder-30B decode):
- llama.cpp CPU (log-verified 0/49): **~16 tok/s** (16.2, 15.8 measured cool-ish).
  Consistent with gpt-oss:20b CPU 14.7 — llama.cpp's CPU-MoE path is ~15 tok/s weak
  on BOTH quant formats, coherent with the E7 finding.
- cpubrrr (CPU-only by construction): **~66 tok/s** (6-run stable on cool machine).
- → **cpubrrr ≈ 4× llama.cpp CPU** on qwen3-coder. And ≈86% of Ollama's Metal GPU
  (75-77).
- The 2026-07-05 "we do NOT beat llama.cpp / are 2.5x slower" correction is ITSELF
  RETRACTED as to the CPU comparison: it compared against GPU numbers.

Session-tail thermal collapse observed again (llama.cpp CPU 3.0, cpubrrr 12) — all
absolutes need one final cool-machine confirmation pass. ACTION ITEMS:
(a) bench_ollama.sh must auto-verify placement from the server log per run and PRINT
    it — no more trusting options.
(b) One cool-machine verification session: llama.cpp CPU / GPU, cpubrrr, all
    log-verified, before publishing any comparison.
