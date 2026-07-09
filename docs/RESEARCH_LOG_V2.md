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

## 2026-07-06 — DEFINITIVE measurement (cool, Chrome closed, log-verified placement)

Resolved the number-confusion saga. Root causes of all prior swings: (1) thermal
throttling after sustained load, (2) unverified GPU/CPU placement, (3) BACKGROUND CPU
CONTENTION — Chrome (~7 tabs, ~60% CPU) + a runaway pairedsyncd (44%) were stealing
~2 cores from our all-12-core engine every measurement, silently varying. Closed Chrome,
confirmed no throttle. cpubrrr numbers stabilized tightly.

FINAL reproducible table (5-run/3-run stable, llama.cpp placement verified from logs):
| model (CPU decode) | cpubrrr | llama.cpp CPU (verified) | ratio |
|---|---|---|---|
| gpt-oss:20b (MXFP4)   | ~55 tok/s (55.0-56.3) | 13.7 (0/25 GPU) | **cpubrrr 4.0x faster** |
| Qwen3-Coder-30B (Q4_K)| ~71 tok/s (69.2-71.8) | 85.7 (0/49 GPU) | llama.cpp 1.2x faster |

HONEST FINAL STORY (this supersedes ALL prior speed claims/corrections in this log):
- On MXFP4 MoE (gpt-oss), cpubrrr is ~4x faster than llama.cpp's CPU path (its weak spot).
- On mature Q4_K (qwen), llama.cpp CPU is ~1.2x faster than cpubrrr (~83% of it).
- Note: the earlier "gpt-oss ~110" is NOT reproducible on this machine now; both the
  original and current binaries measure ~55 stably. Not a code regression (verified by
  building the initial-release commit — also 55). Likely the M4 Max lacks a deep cooldown
  after a week of sustained load; a reboot may restore higher. Using the CONSERVATIVE
  reproducible 55 for all public docs. Even at 55, the 4x MXFP4 win holds.
- cpubrrr is contention-sensitive (wants all 12 cores); llama.cpp tolerates background
  load better. Peak cpubrrr numbers require a quiet machine.

## 2026-07-06 — WE BEAT LLAMA.CPP ON Q4_K (the win). Int-accumulation Q8_K kernel.

Back to the drawing board on the Q4_K gap (was ~71 vs ~86). Read llama.cpp's actual ARM
kernel (ggml_vec_dot_q4_K_q8_K) + surveyed 2025-26 research (T-MAC/Vec-LUT LUT kernels,
Arm i8mm/smmla work, arXiv 2501.00032 group-quant GEMV kernels). Found llama.cpp's real
edge: NOT the instructions (they use vdotq/sdot too on the NEON path) but the ALGORITHM —
they quantize activations to **Q8_K (one scale per 256-superblock + per-32 bsums)** and
**accumulate the sub-block integer dot products weighted by the 6-bit scales in INT32**,
doing only ONE float convert + 2 float ops per superblock. Our old kernel did a float
vcvt+vfma PER sub-block (8 per superblock) — ~6 extra float ops/superblock.

Adopted it: rewrote q4k_dot/q6k_dot to int-accumulation, changed activation quant to
Q8_K (per-256 scale, per-32 sums). Verified new kernels bit-exact vs dequant-f64 (rel
~1e-7, src/bin/q8k_verify.rs). Also tried a 2-row smmla (i8mm) variant — verification
FAILED (lane-mapping bug), dropped it; the int-accum win alone was enough.

Result: **68 -> 94 tok/s** (+38%). Output quality unchanged (coherent fibonacci code,
Four/Tokyo/cba all correct — Q8_K per-256 activation is fine, same as llama.cpp uses).

DEFINITIVE head-to-head (alternating, cool machine, llama.cpp placement log-verified 0/49):
| round | cpubrrr | llama.cpp CPU |
|---|---|---|
| 1 | 93.7 | 87.6 |
| 2 | 94.4 | 74.7 |
| 3 | 90.8 | 77.8 |
| 4 | 89.6 | 86.4 |
=> cpubrrr ~92 vs llama.cpp ~82. **WE WIN EVERY ROUND on Q4_K — llama.cpp's mature home turf.**

FINAL STANDING — cpubrrr beats llama.cpp CPU on BOTH quant families:
- gpt-oss:20b (MXFP4): ~55 vs ~14 -> ~4x
- Qwen3-Coder-30B (Q4_K): ~92 vs ~82 -> ~1.1-1.2x
Both CPU-only, log-verified, correct output. The Q4_K win = their own algorithm +
our worker-driven execution model (better core utilization). This supersedes the earlier
"llama.cpp wins on Q4_K" finding — that was true until we adopted the int-accum kernel.

---

## 2026-07-08 — GPU PHASE OPENS: what does the GPU actually buy on unified memory?

New question (user): does putting the GPU in the middle double/triple throughput?
Hypothesis-first, then measured. Machine note: Chrome open during these runs (draft in
progress) — GPU-side numbers tolerate that; CPU numbers here match the 92 win anyway.

### Hypotheses (stated BEFORE measuring)
- **H1 decode:** M4 Max = unified memory; CPU + GPU share ~546 GB/s DRAM. Decode is
  bandwidth-bound, so GPU ceiling = its achievable-bandwidth fraction vs CPU's
  (~293 GB/s measured). Predict ~1.4-1.6x, NOT 2-3x.
- **H2 prefill:** compute-bound -> GPU should win big (5-20x).
- **H3 hybrid:** CPU + GPU concurrently contend for the same DRAM -> combined ≈
  max(solo), not sum.
- **H4 headroom:** if Metal-Ollama sits far below the GPU bandwidth ceiling, a
  hand-written Metal engine could beat it (same playbook as the CPU win).

### E1 — GPU streaming bandwidth (metal/gpu_bw.swift, new)
| probe | GB/s |
|---|---|
| GPU compute read (decode-like) | **486.9** |
| GPU blit copy | 437.0 |
| GPU compute copy | 422.6 |
| CPU cluster (prior measured) | ~293 |
GPU streams **1.66x** the CPU cluster. That ratio IS the decode story.

### E2 — Ollama Metal GPU baselines (fresh load, log-verified FULL offload)
| model | decode | prefill | placement |
|---|---|---|---|
| gpt-oss:20b | 94.7 / 94.1 | 740 / 748 | 25/25 layers GPU |
| qwen3-coder:30b | 101.2 / 102.1 | 256-353 | 49/49 layers GPU |

### E3 — ceilings + verdicts
- qwen effective traffic ≈ 3.1 GB/token (from 92 tok/s @ ~290 GB/s CPU) ->
  GPU ceiling ≈ 487/3.1 ≈ **155 tok/s**. Ollama Metal = 102 = **66% of ceiling**.
- gpt-oss ≈ 2.6 GB/token -> ceiling ≈ **185 tok/s**. Ollama Metal = 94 = **~50%**.
- **H1 CONFIRMED**: GPU-vs-our-CPU decode = 1.1x (qwen, 102 vs 91) and 1.7x
  (gpt-oss, 94 vs 55). No doubling. Physics, not magic.
- **H2 CONFIRMED**: prefill 740 GPU vs ~91 CPU (qwen measured 31tok/0.34s) — ~8x,
  and our CPU prefill is unoptimized.
- **H4 SUPPORTED**: Metal-Ollama leaves 34-50% of GPU bandwidth on the table.
  A hand-written Metal engine has real room (predict ~150 qwen / ~180 gpt-oss).

### E4 — hybrid contention test (scripts/bench_hybrid.sh, new) — **H3 REFUTED**
Solo: cpubrrr CPU 91.3; Ollama GPU 101.7. Concurrent (started together):
| engine | solo | concurrent | delta |
|---|---|---|---|
| Ollama GPU | 101.7 | **99.9** | -2% |
| cpubrrr CPU | 91.3 | **61.6** | -33% |
| **combined** | (best single 102) | **161.5** | **1.59x best single** |
Aggregate DRAM pull ≈ 100x3.1 + 62x3.1 ≈ **~500 GB/s ≈ the SoC's ~546 GB/s ceiling.**
The fabric genuinely serves both masters: combined throughput is ~84% of the naive
sum, NOT max(solo). H3 was wrong in the good direction. Caveat: CPU run is short
(64 tok, ~1s) inside the GPU's 256-tok window; overlap partial — directional, rerun
longer before quoting externally.

### What this opens (next frontier, in order of ROI)
1. **Hybrid single-stream decode**: split MoE experts CPU/GPU per token; ceiling =
   546/3.1 ≈ **~176 tok/s qwen** (1.7x GPU-only). Hard part: per-token sync latency
   between clusters.
2. **Own Metal decode kernels**: approach 487 GB/s -> ~150+ solo GPU (beat
   Metal-Ollama the way we beat CPU-llama.cpp).
3. **Aggregate serving**: two independent streams (GPU + CPU engine) = ~160 tok/s
   total on one laptop, today, zero new code.

---

## 2026-07-08 (cont) — G-phase: per-layer hybrid REFUTED; built our own full-GPU engine instead

### G1/G2 — the sync-latency wall (hybrid feasibility gate)
- Classic dispatch round-trip (commit+wait): **105us median**. Dead for 48 syncs/token.
- **Persistent-kernel mailbox: IMPOSSIBLE on AGX.** Kernel spinning on a shared-memory
  atomic never sees CPU stores mid-kernel (and its own stores stay invisible to the CPU
  — heartbeat counter read 0 the whole run). Writes flush only at command boundaries.
  Two implementations (Swift, then ObjC with proper C volatile atomics + RMW) both hang.
  Also learned: Swift -O hoists spin-loop loads (no volatile) — use C for lock-free probes.
- **MTLSharedEvent ping-pong: 63us round-trip** (p99 98us), data visibility correct
  (metal/sync_lat3.m). That's the floor.
- Per-layer FFN split math: FFN ≈ 108us/layer on CPU; best-case parallel saving ≈
  40us/layer < 63us sync cost. **Single-stream per-layer CPU+GPU hybrid: REFUTED.**
  Hybrid needs <=10us sync; Apple hardware won't give it.

### Pivot: full-GPU decode engine (metal/engine_metal.m) — sync amortized to 1/token
Whole qwen3moe forward pass on GPU: ~580 dispatches in ONE command buffer per token,
weights zero-copy via mmap + newBufferWithBytesNoCopy (verified full-speed: file-backed
mmap streams at 489 GB/s, metal/mmap_bw.m). Same int-accumulation Q4_K/Q6_K algorithm
as the CPU win, ported to MSL (carried in float4 fma — products <=15*127 are exact in
f32). Kernels verified vs CPU scalar reference (rel 5.7e-5 / 3.6e-6) BEFORE the forward
pass; stage sums then matched engine_qwen2 bit-for-bit through L5 on first run.

**The one bug (and it echoes the CPU build):** first run produced ".SIG" garbage. Stage
bisection: perfect through L5, x=NaN at L6, all L6 inputs sane, CPU recompute from the
GPU's own inputs ALSO NaN -> weights, not kernels. `blk.6.ffn_down_exps` is **Q4_K** —
the down tensors are MIXED Q4_K/Q6_K across layers (24/24; attn_v likewise). We had
hardcoded Q6_K. LESSON (again): NEVER assume a tensor's quant type; dispatch by the
manifest's type field per tensor per layer.

### Optimization ladder (KTIME = per-kernel-class GPU ms for 48 layers)
| step | decode tok/s | what |
|---|---|---|
| v1 correct | 40 | scalar-ish kernels, 1 simdgroup/row |
| vectorized loads | 40 (no change) | uchar4/char4 + float4 fma — NOT ALU-bound |
| attention two-pass | 55.5 | 10.54 -> 1.30 ms: scores to threadgroup mem, 128 thr/head (old: 1024 threads TOTAL + serial simd_sum chain) |
| parallel top-k | 74.7 | ktopk was ONE GPU THREAD scanning 128x8; 32-lane simd argmax -> "small" 6.7 -> 2.3 ms |
| fused rms+quant | (incl above) | krmsq: one dispatch instead of two |
| kglu4 4-row | 79.1 | activation loads amortized over 8 dot streams |
| kdown flattened (expert,unit) | 81.6 | independent streams per lane -> ILP |
| head 4-row + GPU-chained decode | **85.9** | argmax writes outtok[slot], next embed READS it on-GPU; all cmdbufs committed ahead; CPU sync off critical path |
Also measured: dispatch chains are ~1us each (metal/disp_cost.m) — dispatch count was
never the bottleneck; occupancy shape was.

### Standing (decode, same prompt, warm)
| engine | tok/s |
|---|---|
| cpubrrr CPU (engine_qwen2) | ~92 |
| **cpubrrr GPU (engine_metal, ours)** | **~86** |
| Ollama/llama.cpp Metal (full offload) | ~102 |
Our first-day Metal engine reaches 84% of llama.cpp's mature Metal path; kernels sit at
173-260 GB/s vs the 487 ceiling -> clear headroom (qkv+o worst at 173). Both engines
running together (ours+ours, one laptop): 75.6 + 43.7 ≈ **119 tok/s aggregate**
(partial-overlap measurement, rerun matched-duration before quoting).

### Open next steps
- qkv single-dispatch fusion; 2-row middle ground for qkv/o; concurrent-dispatch
  encoder with stage barriers; target >102 (beat Metal-Ollama), ceiling ~155.
- GPU prefill for the CPU engine (740 tok/s class) — one handoff, no sync wall.

---

## 2026-07-09 — THE "110 -> 55" MYSTERY: solved enough, and a 52 -> 77 recovery

User rebooted to chase the gpt-oss regression. Systematic elimination on a FRESH,
COOL, 91%-idle machine:
- reboot/thermal: REFUTED (still 54 after fresh boot, cold chip)
- power: found a 30W charger + 28% battery (M4 Max needs ~90W+ under load) — plausible!
  but REFUTED: at 94W wall power, still 52
- code: REFUTED again, properly this time (initial-release commit in a worktree on the
  clean machine: 57.3)
- OS: no update since January
- machine: EXONERATED — bench_moe still streams 294 GB/s (projects ~185 tok/s)

So the gap was engine-side all along. Root causes found and fixed, in order:

| fix | tok/s |
|---|---|
| (start, clean machine) | 52 |
| **condvar pool -> spin/yield pool** (measured: condvar dispatch = 39us x ~192/token = **7.5 ms/token of pure wakeup overhead**; pool_lat.rs) | 66.5 (NT sweep: 13 hot threads = preemption storm; main thread now works a chunk instead of spinning) |
| **router: serial scalar 92k MACs -> par_rows + slice-zip** (bounds checks had killed autovectorization; was 2.2 ms/token) | 74.5 |
| **act+quant: serial 11.5k exp()/layer on main thread -> parallel over (expert, 32-block)** | 77 |
| work-stealing par_rows chunks (P-cores absorb E-core lag) + hoisted per-layer scratch allocs | 77 (neutral-to-small, kept for robustness) |

Output verified identical (same Rayleigh sentence, greedy) after every change.

**On the original "110":** decode-only stage profile today: experts (gate/up+down)
6.0 ms/token at 210 GB/s — MATCHING E17's expert budget. But E17's claimed 9.1 ms
total requires attn+qkv+o+head+glue in ~3 ms, while those stages measure ~6 ms of
bandwidth-bound Q8 traffic today. E17's total doesn't close against its own
components as measured now. Verdict: 110 is not reproducible in this configuration
and its composition can't be reconstructed; **77 tok/s is the honest, reproducible,
clean-machine number** (5.3x llama.cpp CPU's ~14).

LESSON (benchmark-integrity #4 and #5):
4. Check the WALL SOCKET: a 30W charger on a 90W laptop silently halves peak CPU
   clocks. It wasn't the cause here but it WILL be someone's 2x mystery.
5. A historical peak you can't reproduce is not a baseline. Log enough environment
   (charger wattage, battery %, load, thermals, exact binary hash + data hash) that
   future-you can either reproduce a number or retire it. We now retire "110".

Sub-lesson: condvar wakeup latency is not a constant of nature — 39us today vs
implied ~10us on 07-04, same OS build. Spin/yield pools remove that variable.
