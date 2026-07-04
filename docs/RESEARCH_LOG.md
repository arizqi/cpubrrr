# Research log

## 2026-07-02 — Phase 0: fp32 matmul ceiling ladder (M4 Max, 12P+4E, 128 GB)

Bench: `cargo run --release` (src/main.rs). Square fp32 GEMM, best-of-10, correctness
checked vs naive reference.

| kernel | 512 | 1024 | 2048 | 4096 | % ceiling @4096 |
|---|---|---|---|---|---|
| naive ijk | 3.4 | 2.6 | — | — | ~0.1% |
| reordered ikj (autovec) | 34.6 | 33.0 | 32.6 | 32.5 | 1.0% |
| cache-blocked ikj | 33.2 | 33.2 | 28.9 | 28.1 | 0.9% |
| NEON microkernel, 1 thread | 112.2 | 110.9 | 108.3 | 108.2 | 3.4% |
| NEON microkernel, 12 threads | 319.9 | 790.5 | 1002.7 | 1097.4 | 34.4% |
| Accelerate sgemm (AMX/SME) | 2180.9 | 3302.6 | 3292.3 | 3192.5 | 100% |

(GFLOPS fp32)

### Findings

1. **Measured silicon ceiling: ~3.3 TFLOPS fp32** via Accelerate — above the 2–3 TFLOPS
   estimate in PLAN.md. The matrix units, not NEON, are the machine's real compute.
2. **NEON cannot reach the ceiling, ever.** 12 P-cores of NEON peak at ~1.7 TFLOPS
   theoretical; we measured 1.1 TFLOPS (65% of NEON peak, 34% of machine ceiling).
   The AMX/SME units are ~3× the entire NEON P-core complex. → Phase 1 "own the
   microkernel to 80% of ceiling" is **impossible NEON-only**; SME/AMX kernels (or
   linking Accelerate) are mandatory, not optional. Phase 2 promoted in priority.
3. **The naive→tuned gap is ~1000×** (2.6 → 3300 GFLOPS). This is the entire leverage
   pool of "byte-level" software optimization on this chip, now quantified.
4. Single-thread NEON microkernel hits 108 GFLOPS = 77% of per-core NEON peak —
   packing + `vfmaq_laneq` pattern works; remaining 23% is loop overhead/pipelining.
5. Thread scaling only 10×/12 threads at 4096 and much worse at 512 — per-thread B-panel
   packing is duplicated work and small sizes are sync-bound. Shared packed-B + better
   scheduling is a known fix; matters for training (many small-ish GEMMs).
6. Autovectorized ikj stalls at ~33 GFLOPS regardless of blocking — compiler won't
   register-block; hand microkernel is worth 3.3× over autovec at equal effort class.

### A100 context (measured, honest)

3.3 TFLOPS fp32 = ~1% of A100 bf16 (312 TFLOPS). If SME fp16 doubles throughput:
~2%. Low-bit int paths might reach 3–5%. The 80% target of the original brief remains
physically out of reach; the interesting research is maximizing tok/s within these
ceilings.

### Next (Phase 1/2 reordered)

- [ ] SME2 investigation: can we emit streaming-mode outer-product asm from Rust
      (`.inst` or global asm)? Compare vs Accelerate on same sizes.
- [ ] fp16 path: Accelerate has no public hgemm; test `appleblas_sgemm`-alternatives /
      BNNS / our own SME fp16 widening kernels.
- [ ] Fix thread scaling: shared packed B, persistent thread pool, MC/KC retune for
      16 MB shared L2.
- [ ] Non-square shapes typical of transformer training (e.g. 4096×(4·4096) MLP,
      batch·seq × d_model activations GEMMs).

## 2026-07-02 — Phase 1: direct SME2 programming (src/bin/bench_sme.rs)

Hypotheses stated before running:
- H1: 1 fmopa/cycle/unit → ~2.2 TFLOPS fp32 per SME unit.
- H2: 2 SME units (one per 6-P-core cluster, from `cpusperl2: 6`) → raw doubles at 2 threads.
- H3: direct SME asm from Rust reaches ≥80% of unit peak, no Accelerate dependence.

### E1 smoke
SVL = 64 B = 16 fp32 lanes; `fmopa` executes in userspace from Rust `asm!` with
`.arch armv9.2-a+sme2`. No entitlement, no SIGILL. Direct programmability: **proven**.

### E2 raw fmopa (register-only)
| config | GFLOPS |
|---|---|
| 1 tile, 1 thread (latency chain) | 501 |
| 4 tiles, 1 thread (ILP) | 2003 |
| 4 tiles, 2 threads | 3851 |
| 4 tiles, 3 threads | 3302 |
| 4 tiles, 4–12 threads | ~4100–4225 |

- **H1 confirmed:** ~2.0 TFLOPS/unit; 4-tile/1-tile ratio = 4.0 → fmopa throughput
  1/cycle, latency 4 cycles (must keep ≥4 independent ZA tiles in flight).
- **H2 confirmed:** 2 threads → 1.92×; plateau ~4.2 TFLOPS → exactly 2 SME units.
  3-thread dip = two threads contending for one cluster's unit.
- **Machine raw SME peak = ~4.2 TFLOPS fp32 > Accelerate's 3.3 sustained.**
  Accelerate is NOT at silicon peak — ~25% headroom exists to beat it.

### E3 first SME GEMM (32×32 ZA kernel, packed panels, full-k accumulate)
Correctness: exact match vs Accelerate (max abs diff 0).
| size | 1T | 2T | 4T | Accelerate |
|---|---|---|---|---|
| 512 | 940 | 969 | 950 | 2766 |
| 1024 | 1031 | 1039 | 1260 | 3282 |
| 2048 | 1068 | 1685 | 1811 | 3282 |
| 4096 | 701 | 1216 | 1522 | 3186 |

(GFLOPS fp32)

- H3 partial: 1T GEMM = 51% of unit peak (1031/2003), not 80% yet. k-loop is
  issue-limited: 11 instructions per 4 fmopa (4 loads, 2 address adds, sub+branch).
- 4096 1T regression (1068→701): single-thread strided packing + no cache blocking;
  B pack (64 MB) exceeds L2.
- Best own-kernel: 1.81 TFLOPS = 55% of Accelerate, day one.

### Known fixes queued (in expected-value order)
1. k-loop unroll ×4 + SME2 multi-vector loads (`ld1w {z0.s-z1.s}`) + indexed
   addressing → cut non-fmopa instructions/iter.
2. Parallel + cache-blocked packing (KC/NC blocking, reuse phase-0 structure).
3. Thread scheduling: 2 GEMM threads (one per cluster) with M-split, packing threads
   separate — more GEMM threads than units is pure contention.
4. Then fp16/bf16 fmopa (SME_B16F32) → 2× peak candidate; int8 (SME_I8I32) → 4×.

## 2026-07-02 — Kernel v2: issue-budget optimization + instruction-level trace

Hypothesis: v1's 51%-of-peak was instruction-issue-limited (11 instructions per
4 fmopa). Unroll k×4 + SME2 multi-vector loads (`ld1w {z0.s-z3.s}`: 4 registers,
1 instruction) → 23 instructions per 16 fmopa (70% payload) → predicted ≥1.6 TFLOPS 1T.

| size | v1 1T | v2 1T | v2 2T | v2 4T | Accelerate |
|---|---|---|---|---|---|
| 512 | 936 | 761 | 811 | 803 | 2715 |
| 1024 | 998 | 1135 | 1169 | 1193 | 3254 |
| 2048 | 939 | 1250 | 2397 | 1624 | 3231 |
| 4096 | 688 | 1055 | 1939 | 2295 | 3187 |

(GFLOPS fp32, correctness exact vs Accelerate)

### Findings
- Hypothesis PARTIALLY confirmed: v2 1T +17% at 2048 (1068→1250) — issue budget was
  *a* limiter, not *the* limiter. 62% of unit peak, not the predicted 80%.
- New prime suspect: L2 streaming bandwidth per core. Kernel needs 8 FLOP/byte;
  2 TFLOPS ⇒ 250 GB/s sustained from L2 — likely above single-core L2 bandwidth.
  Next fix: KC blocking so the A panel lives in L1 (32×512×4 = 64 KB vs 128 KB L1d),
  and prefetch.
- **Best result: 2.40 TFLOPS (v2, 2T @2048) = 73% of Accelerate** — up from 55%.
- 512 regression (936→761): packing thread-spawn overhead dominates small GEMMs
  (~50 µs spawn vs ~300 µs total). Fix: persistent thread pool, serial packing under
  a size threshold.
- 2T@2048 (2397) > 4T@2048 (1624): confirms 2-threads-one-per-cluster rule; 4T adds
  contention. But 4T@4096 wins (2295 vs 1939) — at DRAM-resident sizes extra threads
  help hide memory latency. Scheduling policy must be size-aware.

### Instruction-level artifacts (src/bin/sme_trace.rs)
Traced C(16×16) = A(16×2)·B(2×16) on hardware, dumping ZA after each fmopa:
outer product #1 (A col [0..15] ⊗ B row [1,1,..]) → C[i][j]=i; fmopa #2 accumulates
(A col [2,2,..] ⊗ B row [0..15]) → C[i][j]=i+2j. Verified EXACT vs scalar.
Disassembly (llvm-objdump --mattr=+sme2): fmopa za0.s = machine word 0x80810000.

## 2026-07-02 — E4 precision sweep + v3 blocking experiment + clobber bug

### Bug found the hard way: streaming-mode register clobbering
Benchmarks printed 0.0 after a refactor. Cause: `smstart`/`smstop` architecturally
zero ALL 32 vector registers; our asm! blocks declared only the registers we used.
rustc kept its own f64 temporaries in callee-saved v8–v15 across the asm call → zeroed
→ silent data corruption (manifested as 0.0 GFLOPS; could have corrupted anything).
Fix: every streaming-mode asm block now clobbers v0–v31. Law: asm clobber lists
describe the ISA's behavior, not your instruction usage.

### E4: precision multipliers — HYPOTHESIS REFUTED
Predicted bf16/fp16 = 2×, int8 = 4×. Measured (raw mopa, register-only, 1 thread):

| precision | G(FL)OPS | vs fp32 |
|---|---|---|
| fp32 fmopa | 2003 | 1.0× |
| fp16 fmopa (widening) | 1998 | **1.0×** |
| bf16 bfmopa (widening) | 2002 | **1.0×** |
| int8 smopa | 4011 | **2.0×** |

2 threads: ~3.9 TFLOPS float (any), 7.8 TOPS int8.

M4's SME unit runs widening (2-deep/4-deep dot) mopa at reduced instruction rate:
fp16/bf16 do 2× work per instruction at 1/2 rate (net 1.0×), int8 4× at 1/2 (net 2×).
Machine ceilings: **~4.2 TFLOPS in ANY float precision; ~8 TOPS int8.**

Consequences for the project thesis:
- Low-bit compute multiplier is 2× (int8), not the hoped 4–8×. The "3B in months"
  bet weakens on compute; revise scale expectations accordingly.
- BUT: bf16/fp16 inputs halve load bytes at identical FLOPS. Our GEMM is load-bound
  (raw 2003 vs GEMM ~1300) → bf16 kernel should close the gap to peak without any
  numerical downside vs fp32 inputs. Next kernel experiment.
- int8 2× compute + 4× smaller data = still the right endgame for forward passes.

### E3: v3 (KC=512 L1 blocking + single thread wave) — mostly negative result
| size | v2 1T | v3 1T | v3 2T | v3 4T | Accelerate |
|---|---|---|---|---|---|
| 512 | 755 | 962 | 1157 | 1209 | 2753 |
| 1024 | 1089 | 1002 | 1296 | 1286 | 3263 |
| 2048 | 1377 | 1090 | 1465 | 2249 | 3150 |
| 4096 | 1069 | 788 | 1576 | 1936 | 3097 |

- L1 k-blocking HURT 1T at k≥1024 (1377→1090 @2048): splitting k means k/512× more
  kernel invocations, each paying smstart/smstop (full vector-state flush, now with
  v0–v31 save/restore) + 300-instruction ZA writeback. Overhead > L1 locality win.
- Single-wave threading (pack+barrier+compute in one spawn) fixed the 512 regression
  (755→962 @1T-equivalent path, 1209 @4T).
- Conclusion: keep full-k ZA accumulation; attack smstart/smstop frequency instead —
  hoist streaming mode to once per thread (requires the panel loop inside asm, or
  verified FP-free Rust between kernels). Combine with bf16 loads.

## 2026-07-02 — Inference track: E5 bandwidth ceiling + E6 int4 GEMV (bench_infer.rs)

Goal: assess ~100 tok/s decode for frontier-class models on M4 Max CPU.
Decode is GEMV = zero weight reuse → bandwidth-bound: tok/s ≈ BW / active-bytes-per-token.

### E5: CPU-side read bandwidth (streaming NEON sum, 512 MB/thread)
| threads | GB/s |
|---|---|
| 1 | 86 |
| 2 | 166 |
| 4 | 272 |
| 8 | 282 |
| 16 | 293 |

- H-A confirmed: **CPU peak ≈ 293 GB/s** = 54% of the 546 GB/s chip spec (rest is
  GPU/ANE fabric allocation). Saturates at ~4 threads; single core pulls 86 GB/s.
- Decode ceiling on this machine: 293 GB/s ÷ active-GB/token, minus KV/attention.

### E6: int4 GEMV decode kernel (NEON sdot, nibble unpack; correctness EXACT)
| threads | effective GB/s |
|---|---|
| 1 | 26 |
| 4 | 86 |
| 8 | **131** |
| 12 | 127 |

- H-B FAILED: 131 GB/s = 45% of raw read BW (hypothesis ≥70%). Per-core the kernel is
  instruction-bound: unpack (and/shr/sub ×4) + 4 sdot per 32 B crowds out loads
  (1T: 26 vs 86 GB/s raw). 12T < 8T: E-core scheduling contention.
- Known fixes: fold the -8 bias out of the inner loop (y = sdot(w_u4, x) - 8·Σx per
  row — removes 4 vsub per 32 B), deeper unroll, prefetch, possibly SME streaming
  loads. Target ≥230 GB/s (80% of raw).

### Projections (4-bit weights ≈ 0.56 B/param incl. scales; ~10% KV/attn overhead not included)
| model (fits 128 GB?) | active params | GB/token | tok/s @131 (today) | tok/s @260 (kernel goal) |
|---|---|---|---|---|
| GPT-OSS-120B (66 GB ✓) | 5.1 B | ~2.9 | ~45 | **~90 ✓ in window** |
| GLM-4.5-Air (60 GB ✓) | 12 B | ~6.8 | ~19 | ~38 |
| Llama-4-Scout-class (61 GB ✓) | 17 B | ~9.6 | ~14 | ~27 |
| dense 70B (39 GB ✓) | 70 B | ~39 | ~3 | ~7 |
| DeepSeek/Kimi/Fable5-class (✗ >300 GB) | — | — | does not fit | does not fit |

### Verdict on "100 tok/s ± 15% frontier inference"
- **Plausible** for ~5 B-active frontier MoE (GPT-OSS-120B class): needs GEMV kernel
  at ~85–90% of raw CPU bandwidth. Gap today: 2.2×, all in kernel efficiency — exactly
  this project's kind of problem.
- **Not reachable** for 12 B+ active models (bandwidth physics: ceiling ~40 tok/s) or
  any dense 70 B (~7 tok/s). Proprietary-frontier scale doesn't fit in RAM at all.
- Levers beyond 4-bit: 3-bit/2.5-bit quant of a 12 B-active model trades quality for
  ~1.5–1.8× tok/s; speculative decoding turns decode into small-GEMM batches (SME
  helps verify step) for ~1.5–2.5×.

## 2026-07-02 — E7: Ollama/llama.cpp baseline benchmarks (scripts/bench_ollama.sh)

Harness: `scripts/bench_ollama.sh <model> <cpu|gpu> [num_predict]` — hits the Ollama
API, cpu mode forces num_gpu:0 (pure llama.cpp CPU kernels), reports prefill + decode
tok/s. Reusable regression baseline as our kernels land.

Measured (M4 Max, 96-token decode, temp 0, keep_alive 0):

| model | mode | prefill tok/s | decode tok/s | implied eff. BW |
|---|---|---|---|---|
| gpt-oss:20b (MoE 3.6B act, MXFP4) | cpu | 144 | 14.7 | ~32 GB/s (11% of CPU raw) |
| gpt-oss:20b | gpu | 732 | 94.4 | ~208 GB/s |
| qwen3.5:35b-a3b (MoE ~3B act) | cpu | 74 | 27.5 | ~60 GB/s (20%) |
| qwen3.5:35b-a3b | gpu | 216 | 45.2 | ~100 GB/s |
| deepseek-r1:32b (dense, 19 GB) | cpu | 7.5 | 12.7 | **~236 GB/s (80% of CPU raw)** |
| deepseek-r1:32b | gpu | 100 | 22.9 | **~426 GB/s (78% of fabric)** |

### Predictions vs measured
- Dense: predicted 7–10 CPU / 18–22 GPU; measured 12.7 / 22.9. Bandwidth model
  VALIDATED — llama.cpp dense Q4 kernels run at ~80% of raw BW on both engines.
- MoE: predicted 50–70 CPU; measured 14.7–27.5. Model missed because llama.cpp's MoE
  decode path is NOT bandwidth-bound — expert routing/scatter overhead caps it at
  11–20% of raw BW on CPU (and only ~25–48% on GPU). gpt-oss worst: MXFP4 CPU dequant
  path is poor.

### Consequences
1. 236 GB/s effective dense CPU decode EXISTS in production code → our E6 GEMV target
   (≥230 GB/s) is proven reachable; llama.cpp currently beats our 131. Study their
   dense path (thread count ~8, unpack trick), then beat it.
2. **MoE CPU decode is wide open**: llama.cpp leaves 4–7× on the table for exactly the
   model class (5B-active MoE) that our 100-tok/s target needs. Batched expert GEMV +
   MXFP4-native unpack = the differentiating kernel work.
3. Revised 100 tok/s outlook: gpt-oss:20b CPU ceiling = 293/2.2 ≈ 133 tok/s (llama.cpp
   gets 14.7). GPT-OSS-120B CPU ceiling ≈ 101 tok/s — needs near-perfect BW efficiency
   AND fixed MoE path. Metal already does 94 tok/s on the 20b today.
4. Benchmark discipline going forward: run bench_ollama.sh (cpu) on gpt-oss:20b +
   deepseek-r1:32b after each inference-kernel milestone; our kernels' end-to-end
   numbers go in the same table.

## 2026-07-03 — E6 GEMV optimization ladder v2→v4 (dense int4 decode kernel)

Ladder (8192×8192 int4 layer, correctness EXACT at every step):

| version | change | 1T GB/s | peak GB/s (threads) |
|---|---|---|---|
| v1 | baseline nibble+sdot | 27 | 117 (8T) |
| v2 | bias-fold (dot(w,x)−8Σx) + 4 rows/x-load | 33 | 151 (10T) |
| v3 | 2 accumulators/row (break sdot chains) + prfm | 45 | 180 (8T) |
| v4 | quad-interleaved weight layout → 1 sequential stream/thread | **65** | **197 (8T)** |

Findings:
- Data layout was the big rock: v4's single-stream layout lifted single-core from
  45→65 GB/s = 75% of raw per-core read bandwidth (86). Prefetchers reward
  sequential; 4×4KB parallel streams were restarting them. Confirms "bytes-layout
  below instructions" thesis with hard numbers.
- Dependency chains second: 2 accs/row bought +35% (sdot latency ~3-4 cyc).
- Bias-fold algebra third: +23% by deleting the subtract stage.
- Multithread saturation ~200 GB/s vs 293 raw — scaling loss above 4T; suspects:
  no QoS pinning (threads land on E-cores), scheduler migration. Next: QoS
  USER_INTERACTIVE per worker + persistent pool; target 240+.
- vs llama.cpp dense implied ~236 GB/s: at 197 = 84% of their effective. One more
  round should pass them.

Projections at current 197 GB/s: gpt-oss:20b CPU ≈ 90 tok/s (llama.cpp: 14.7 —
6× win if MoE path built); GPT-OSS-120B ≈ 68 tok/s (window needs ≥85 → requires
~250 GB/s + MoE-efficient routing).

Next queue: thread QoS pinning; MoE expert-batched GEMV (the 9× llama.cpp gap);
per-block scales (real Q4 format, adds ~6% traffic); end-to-end single layer test.

## 2026-07-03 — v5 QoS pinning: hypothesis REFUTED

One-line change: pthread_set_qos_class_self_np(USER_INTERACTIVE) per worker.
Result: v5 ≈ v4 (225 vs 228 GB/s @8T); only 12T improved (203→214). Scheduler was
already placing ≤8 busy workers on P-cores; QoS not the limiter. Also observed
run-to-run variance ~±10-15% (v4 measured 197 yesterday, 228 today — thermal/machine
state); the "scaling stall" was partly noise. Benchmark discipline update: report
best-of-5 AND note variance; long warmup before ladder comparisons.
Current best: ~228 GB/s = 97% of llama.cpp dense implied (236), 78% of raw (293).
Projections @228: gpt-oss:20b ≈ 104 tok/s potential; GPT-OSS-120B ≈ 79 tok/s.

## 2026-07-03 — E8: MoE expert-batched decode — hypothesis CONFIRMED at ceiling

Sim: gpt-oss:20b-shaped layer (32 experts, top-4, d=2880, 3 int4 mats/expert,
398 MB/layer, 49.8 MB active/token). Design: active experts' GEMVs = one flat
quad work list per token, single thread wave, v4 quad-interleaved layout per mat.
No per-op dispatch/sync (llama.cpp's disease).

| threads | GB/s on expert weights | layer ms | full-model proj (24L, FFN+attn mats) |
|---|---|---|---|
| 4 | 243 | 0.205 | ~152 tok/s |
| 8 | 293 | 0.170 | ~184 tok/s |
| 10 | 297 | 0.168 | ~186 tok/s |

Control: 64-expert pool (796 MB, same active bytes) → 290 GB/s. Cache inflation ≈2%.
**MoE routing overhead ≈ 0 in this design; expert streaming runs AT the measured
293 GB/s machine ceiling** (vs llama.cpp's effective ~32 GB/s on the same model class).

Caveats (sim vs real model): no activation math between mats (silu/requant), no
attention compute/KV reads, no routing logits, no per-block dequant scales (+~6%
traffic), x reused across mats. Expect real end-to-end 15–30% below projection →
~120–160 tok/s for 20b class vs llama.cpp CPU 14.7 (~10×), and GPT-OSS-120B class
(≈2.9 GB active/token incl. attention) ≈ 90–100 tok/s — the 100±15% target is live.

Next: real end-to-end path = load actual gpt-oss weights (GGUF/MXFP4 → our quad-int4
layout), implement attention + router + activations, measure true tok/s, put it in
the E7 table against Ollama.

## 2026-07-03 — E9: REAL gpt-oss:20b expert weights through our kernel

Pipeline: parsed Ollama's GGUF (scripts/gguf_index.py), extracted blk.0 gate/up/down
MoE tensors (3×141 MB MXFP4, 32 experts) (scripts/extract_experts.py), repacked to
quad-interleaved nibbles + per-block e8m0 scales, ran MoE decode bench (bench_real.rs).

Kernel novelty: exact MXFP4 on integer units — e2m1 nibble values ×2 are all int8
({0,±1,±2,±3,±4,±6,±8,±12}) → vqtbl1q lookup → sdot → per-block scale 2^(e-128) via
fmla (the ×2 folded into the exponent). No float dequant of weights, ever.

- Correctness: **max rel err 0.00e0 vs f64 dequant reference** (first 8 rows, real data).
- Throughput: 243 GB/s on real MXFP4 (83% of 293 ceiling; tbl+scale costs ~17% vs
  the int4 sim's 293).
- **Full-model projection ~144 tok/s** (24 layers, +4 attn-mat allowance) vs
  llama.cpp CPU 14.7 on the same model = **~10×**. Still missing from projection:
  attention compute/KV, activations, router — expect real end-to-end 100–130.

Remaining to a true generating engine: bf16 attention path (type 30 tensors), KV
cache, router (ffn_gate_inp), RMSNorm/SwiGLU/rope, tokenizer+sampling, per-token
activation quant. Then bench_ollama.sh side-by-side becomes apples-to-apples.

## 2026-07-03 — E10: exact gpt-oss MoE-FFN forward reproduced on real weights

Recovered the exact gpt-oss compute spec from llama.cpp (src/models/openai-moe.cpp,
ggml-cpu/ops.cpp) — the correctness-critical constants that guessing would get wrong:
- Router: logits = x·W_gate_inp + b; take top-4 of 32; softmax OVER THE 4 SELECTED
  (gating type SOFTMAX_WEIGHT), not over all 32. No extra expert_weights_scale for 20b.
- SwiGLU-OAI: alpha=1.702, limit=7.0;  xg=min(gate,limit); yu=clamp(up,-limit,limit);
  h = (xg / (1+exp(-alpha*xg))) * (yu + 1).   (note the (+1) and the asym clamp)
- Attention (for next phase): head_count 64, kv 8, head_dim 64, scale 1/sqrt(64),
  RoPE NeoX + YaRN, learned per-head attention SINKS appended to softmax denom,
  sliding-window 128 on alternating layers (swa_period 2), pre+post attn RMSNorm.

Implementation (src/bin/ffn_forward.rs) + numpy f64 reference (scripts/ffn_reference.py),
fixed pseudo-random activation, layer-0 real MXFP4 weights:
- top experts [22,25,12,0], softmax wts [.315,.246,.238,.201] — Rust == numpy.
- **out vs numpy f64: max abs 2e-4, rel L2 1.0e-6 → PASS.**
The full differentiated block (routing → per-expert gate/up → SwiGLU-OAI → down →
weighted combine) is now verified correct on real weights.

Status: two halves proven separately — E9 (speed: 243 GB/s MXFP4 kernel, ~144 tok/s
proj) and E10 (correctness: exact FFN math). Uniting them = quantize activations to
int8 per token and feed the E9 kernel; requires activation-quant error budget (next).
Then attention path (spec above) → full generating engine → head-to-head vs
bench_ollama.sh on identical prompts.

## 2026-07-03 — E11: fused speed+correctness — int8 activations keep gpt-oss faithful

United E9 (243 GB/s MXFP4 int8 kernel) with E10 (exact FFN) by quantizing activations
to per-token int8 (scale = max|x|/127) and running the full FFN through the fast integer
path. Router kept in f32 (expert-selection errors flip experts = catastrophic; router is
tiny, ~0.1% of bytes). src/bin/ffn_quant.rs.

Result vs f64 reference (real layer-0 weights):
- cosine similarity 0.999921
- rel L2 error 0.36%
- max abs elem 2.4 on ref norm 654.6

Cosine >0.999 → next-token logit distribution effectively unchanged. **The fast path
is accurate enough to run the model faithfully.** Speed (E9) and correctness (E10) now
demonstrated to coexist in one kernel.

Caveats: single fixed activation (Gaussian); real hidden states have channel outliers
that per-tensor int8 handles worse — production needs per-channel or outlier-aware quant
(known QAT technique), and end-to-end perplexity vs llama.cpp is the real acceptance
test. Also int8 x doubles activation bytes vs int4 but activations are <2% of traffic.

### Engine remaining (unchanged, spec in E10)
Attention (GQA 64/8 heads, sinks, sliding-window-128 alt layers, RoPE+YaRN) → RMSNorm
plumbing → token embed/unembed → sampler → generation loop. Then bench_ollama.sh
head-to-head on identical prompts for the true tok/s + a perplexity check.

## 2026-07-03 — E12: exact gpt-oss attention reproduced on real weights

Implemented layer-0 attention (src/bin/attn_forward.rs) vs numpy reference
(scripts/attn_reference.py), single decode step over synthetic 8-position KV cache,
real bf16 weights. Verified mechanics:
- GQA: 64 query heads share 8 KV heads (head hd -> kv hd/8).
- Learned attention SINKS: per-head extra logit sinks[h] added to softmax denominator
  only (no value contribution) — the gpt-oss-specific detail.
- Causal softmax, scale 1/sqrt(64), QKV biases, NeoX RoPE @ freq_base 150000.
- bf16 weight load (top-16-bits-of-f32).

Result: cosine 1.000000, max abs 4e-4 on ref norm 520 → PASS.

Now BOTH halves of a transformer block are exact on real weights: attention (E12) +
MoE-FFN (E10), plus the fast-path accuracy proof (E11). A full layer = attn_norm →
attn (E12) → residual → post_norm → FFN (E10/E11) → residual.

Remaining exactness gap (flagged, honest): RoPE uses plain NeoX, not YaRN
(scaling.factor 32, orig_ctx 4096). At small positions the ramp effect is bounded but
mscale (attn_factor) is not applied — must add for long-context faithfulness. True
acceptance gate stays: end-to-end logits/perplexity vs llama.cpp on real tokens.

### Engine assembly remaining
KV cache management (sliding-window 128 alt layers), tokenizer, embed/unembed (bf16),
sampler, 24-layer generation loop, YaRN-exact RoPE. Then bench_ollama.sh head-to-head:
measured tok/s + perplexity delta.

## 2026-07-03 — E13: FULL ENGINE GENERATES CORRECT TEXT — 1.8× llama.cpp CPU

src/bin/engine.rs: complete gpt-oss:20b decoder — GGUF loaded directly (no conversion),
bf16 attention w/ YaRN RoPE + sinks + GQA + sliding window, f32 router, int8-block
activations into native-layout MXFP4 expert kernel, o200k tokenizer (validated vs
tiktoken), harmony template, greedy sampling.

### Debug trail (kept honest, each step logged)
1. First output garbage at llama.cpp-equal speed. Bisect ladder:
2. dot_bf16 verified vs numpy on real rows (exact). Token table verified vs tiktoken
   (exact). Per-layer norms at pos 0 matched f64 reference after switching activation
   quant from per-tensor to per-32-block Q8_0-style (per-tensor int8 diverged 26% by
   layer 9 — real hidden states have massive outliers; E11's caveat confirmed).
3. Multi-position still diverged (0.07% → 4.7% → 17.7% by pos 2). Eliminated rope
   (NO_ROPE both sides), tokenizer, template. Oracle test: llama.cpp CPU with
   identical raw prompt bytes → perfect output ⇒ engine bug.
4. **Root cause: MXFP4 dequant exponent off by one — 2^(e-127) with the ×2 integer
   table; correct is 2^(e-128) (ggml e8m0_to_fp32_half). All expert weights 2× too
   large; nonlinear cascade → garbage. My E10/E11 python references had the SAME
   error, so engine-vs-reference checks passed while both were wrong vs reality.
   Lesson: verification references must be independently grounded (ggml source /
   tiktoken / llama.cpp oracle), not self-derived.**

### Result (M4 Max CPU, greedy, identical raw harmony prompt)
- Output: token-for-token IDENTICAL to llama.cpp/Ollama CPU ground truth, including
  analysis channel: "...The sky appears blue because molecules in Earth's atmosphere
  scatter shorter-wavelength (blue) sunlight more efficiently..."
- **Decode: 26.3 tok/s vs Ollama CPU 14.7 = 1.8× faster, day one.**
- Prefill: 16 tok/s (sequential decode-mode; batching TBD — Ollama does 144).
- Effective traffic ~3.7 GB/token at 26.3 tok/s ≈ 97 GB/s — big headroom vs our
  228 GB/s kernel ceiling. Known next: quantize attention+head paths (bf16 currently
  2.4 GB/token of the 3.7), persistent thread pool, batch prefill, sampling temp.

## 2026-07-04 — E14: engine optimization ladder — 26.3 → 48.6 tok/s (3.3× llama.cpp)

Same prompt, same correct output at every step (one synonym-level word difference vs
bf16 attn path — quant near-tie, both fluent/correct).

| step | change | tok/s |
|---|---|---|
| baseline (E13) | bf16 attn+head | 26.3 |
| +Q8 weights | attn+head bf16→Q8_0 at load (traffic 3.7→2.5 GB/tok, int8 sdot path) | 31.9 |
| +thread pool | persistent workers, condvar park/wake (was ~1350 spawns/token) | 35.5 |
| +parallel attention | heads across pool (was serial 64-head scalar loop) | 40.9 |
| +NT 8→12 | more workers now that jobs are cheap | **48.6** |

vs Ollama/llama.cpp CPU on identical model+prompt: 14.7 tok/s → **3.3×**.
Effective bandwidth: 2.5 GB/token @ 48.6 = ~120 GB/s (kernel ceiling 228 → headroom ~2×).

Remaining known drains (next round): serial per-layer quant_i8 + rmsnorm; 7 pool
jobs/layer (fuse gate/up/down passes); attention V-accumulate scalar (NEON it);
prefill still sequential decode-mode (3.6s for 78 tok — batch it); KV f32→f16 traffic.
Target: 70–90 tok/s. Beyond that: speculative decoding (SME GEMM verify step).

## 2026-07-04 — E15: 48.6 → 51.2 tok/s; spin-pool hypothesis REFUTED

| change | tok/s |
|---|---|
| spin-wait pool (workers busy-poll atomics) | 34.6 ❌ regression |
| shorter spin before yield | 39.1 still worse |
| revert to condvar pool + keep NEON attn + fused down-proj | **51.2** |

- Spin pool refuted: 12 busy-spinning workers steal cores from the main thread's
  serial sections (rmsnorm, router, activation, quant) between jobs. Parked-thread
  wake cost (~10µs) < core theft. Latency-vs-occupancy tradeoff went the other way.
- NEON score-dot + value-accumulate (16-reg fma) and fusing 4 down-proj pool jobs
  into 1: +2.6 tok/s over E14 on top of the revert.
- Standing: **51.2 tok/s = 3.5× llama.cpp CPU (14.7), identical-quality output.**
  2.46 GB/token → ~126 GB/s effective vs 228 kernel ceiling.
- Next candidates: quad-interleave head matrix (largest single matvec), batch prefill,
  fp16 KV at long context, speculative decoding (SME verify).

## 2026-07-04 — E17: TARGET HIT — 109.9 tok/s decode (7.5× llama.cpp)

Profiled first (new discipline after E16): experts = 78% of decode time, running at
~89 GB/s in-engine vs 197+ standalone → cause: native GGUF row layout, not the E6
quad-interleaved layout. Fix: repack all expert tensors at load (~10 GB, threaded,
one-time) + dot4 quad kernel (4 rows/pass, laneq scale application).

| metric | before | after |
|---|---|---|
| decode | 52.2 tok/s | **109.9 tok/s** |
| prefill | 21 tok/s | 63.9 tok/s |
| vs llama.cpp CPU (14.7 / 144 prefill) | 3.5× | **7.5× decode** |

Output: identical correct sentence. Bandwidth math closes: 1.2 GB expert bytes/token
at quad-kernel rates + attn/head ≈ 9.1 ms/token ✓ consistent with E6/E8 predictions.

**The 100 tok/s ±15% goal is achieved on gpt-oss:20b (frontier-adjacent class), CPU
only.** GPT-OSS-120B (fits in 128 GB, ~2.4× active bytes) projects ~45-55 tok/s on
this engine; reaching 100 there needs ~2× more (speculative decoding — SME verify
GEMM — plus head/attn quad treatment).

Remaining stage profile: gate/up 34%, down 18%, qkv 14%, o-proj 13%, router 10%
(router suspiciously slow for 92k MACs — f32 serial, easy NEON win), attention 7%.
