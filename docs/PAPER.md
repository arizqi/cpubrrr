# Bandwidth Is All You Need (on CPUs): 7.5× Faster Mixture-of-Experts Inference on Consumer Apple Silicon

**Abstract**

We show that a from-scratch, CPU-only inference runtime can decode a frontier-class
open Mixture-of-Experts (MoE) language model, OpenAI's `gpt-oss-20b`, at **109.9
tokens/second on a single Apple M4 Max**, versus **14.7 tok/s** for the widely used
llama.cpp on the same machine in CPU-only mode — a **7.5× speedup with token-for-token
identical output**, and faster than the same model on the machine's integrated GPU
(94.4 tok/s). The result rests on a single analytical observation: autoregressive decode
of an MoE model is bound by *memory bandwidth*, not arithmetic, and the achievable rate
is `bandwidth / active-bytes-per-token`. We measure the CPU-visible memory bandwidth of
the M4 Max at 293 GB/s (54% of the advertised unified figure), and show that llama.cpp's
MoE decode path realizes only ~11% of it, while our runtime realizes ~78%. The gains
come almost entirely from data layout and integer-SIMD kernel design rather than
algorithmic change: a quad-interleaved weight layout that keeps each core's reads
sequential is worth 2×, and an exact 4-bit (MXFP4) matrix-vector kernel built on ARM
`sdot`/`tbl` integer instructions eliminates all floating-point dequantization. We
release the full runtime, benchmark suite, and a reproducible measurement log,
including four falsified hypotheses. We argue the practical implication is a
re-pricing of CPU inference for the MoE model class that now dominates open frontier
releases.

---

## 1. Introduction

The prevailing assumption in production LLM serving is that inference requires GPUs.
This assumption was formed on *dense* transformer models, whose per-token cost scales
with the full parameter count, and against CPU software that leaves most of the
hardware's capability unrealized. Two things have changed. First, essentially every
competitive open model released in 2025–2026 is a **Mixture-of-Experts (MoE)**
architecture, in which a routing network activates only a small fraction of parameters
per token. Second, consumer Apple-silicon laptops now ship with large unified memory
(up to 128 GB) and high memory bandwidth. Together these invite a re-examination of the
question: *how much frontier-class inference can a consumer CPU actually do?*

We answer with a concrete artifact. Our runtime runs `gpt-oss-20b` (21B total
parameters, ~3.6B active per token, 32 experts, top-4) end-to-end on CPU cores alone —
no GPU, no Metal, no Apple Accelerate framework (verified by inspecting the linked
libraries of the compiled binary) — and reaches interactive speed with output
bit-compatible with the reference implementation.

Our contributions:

1. **An analytical model** for MoE decode throughput on CPU
   (`tok/s ≈ effective_bandwidth / active_bytes_per_token`) and a direct measurement
   of the binding resource: the M4 Max delivers 293 GB/s to CPU cores, not the 546 GB/s
   unified-memory figure (§3).
2. **A demonstration that the dominant open-source CPU stack severely underutilizes
   this resource for MoE models** — 11% of bandwidth on `gpt-oss-20b` vs. 80% on a dense
   model of similar size — and an identification of the cause (per-expert dispatch and
   synchronization overhead) (§4.2).
3. **A set of kernel and layout techniques** — quad-interleaved sequential-stream weight
   layout, exact integer-SIMD MXFP4 arithmetic, block-wise activation quantization,
   MoE-aware expert batching — that together realize 78% of the bandwidth ceiling (§5).
4. **A complete, reproducible engineering record**, including a matrix-unit (SME)
   micro-study, an end-to-end correctness methodology grounded in independent oracles,
   and four explicitly falsified hypotheses (§6, §7).

We deliberately foreground the failures. Four plausible optimizations were tested and
measured to not work (thread pinning, spin-waiting worker pools, and two variants of
prompt-batch cache amortization); each falsification redirected effort more efficiently
than any success, and we report them as first-class results.

---

## 2. Background and terminology

**MoE (Mixture of Experts).** A transformer whose feed-forward blocks are replaced by
`N` parallel "expert" sub-networks plus a lightweight router that selects `k` experts
per token. `gpt-oss-20b` uses `N=32`, `k=4`. Total parameters are large (21B) but
*active* parameters per token are small (~3.6B), because only 4 of 32 experts run.

**Decode vs. prefill.** *Prefill* processes the prompt; many tokens share one pass over
the weights, so it is compute-bound (a matrix-matrix product, GEMM). *Decode* generates
one token at a time; each weight is read once and used once with no reuse, so it is
memory-bandwidth-bound (a matrix-vector product, GEMV). Interactive latency is dominated
by decode.

**Quantization and MXFP4.** `gpt-oss` ships in **MXFP4**: each weight is a 4-bit code
mapping to one of sixteen values `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, with a shared
power-of-two (E8M0) scale per block of 32 weights. Four bits per weight is what fits 21B
parameters in ~13 GB.

**ARM NEON, `sdot`, `tbl`.** NEON is ARM's 128-bit SIMD instruction set. `sdot`
computes sixteen 8-bit integer multiply-accumulates in one instruction; `tbl` performs a
16-entry table lookup on 16 bytes in one instruction. **SME** (Scalable Matrix
Extension) is a separate matrix coprocessor, first publicly programmable on the M4;
one `fmopa` instruction performs a 16×16 outer product (256 multiply-adds).

---

## 3. The bandwidth model and machine characterization

For MoE decode, per generated token the engine must read every active weight exactly
once. With `P_a` active parameters at `b` bytes each,

```
tok/s ≈ BW_eff / (P_a · b + overhead)
```

where `BW_eff` is the memory bandwidth actually deliverable to the compute units. This
makes `BW_eff` the single most important machine parameter, and the vendor's unified
figure misleads: it is shared across CPU, GPU, and neural engine.

We measured `BW_eff` directly with a threaded streaming-read microbenchmark
(`bench_infer`, experiment E5). The CPU cores saturate at **293 GB/s** — 54% of the
advertised 546 GB/s — reaching the plateau at only ~4 threads (bandwidth, unlike
compute, does not scale with core count past the fabric limit). A single core sustains
86 GB/s.

For `gpt-oss-20b` at 4 bits, active bytes per token (experts + attention + head) are
~2.5 GB, so the model predicts a ceiling near 115 tok/s on this machine — which our final
engine approaches (109.9), confirming the decode path is essentially bandwidth-saturated
and correctly modeled.

We separately characterized the M4's compute ceiling via a matrix-multiply ladder
(`bench_matmul`) and direct SME2 kernels (`bench_sme`): naive scalar code reaches
~3 GFLOPS; a hand-written NEON microkernel across 12 cores reaches 1.1 TFLOPS; the SME
matrix units reach **4.2 TFLOPS fp32** (two units, one per performance-core cluster),
exceeding Apple's own Accelerate library (3.3 TFLOPS). A precision sweep found that,
unlike on GPUs, half-precision does not raise SME throughput (the wider instruction runs
at half rate), while int8 doubles it. These compute results are relevant to prefill and
to a training track; decode remains bandwidth-bound.

---

## 4. Baseline analysis: where llama.cpp leaves bandwidth on the table

### 4.1 Setup

All comparisons use the same machine (Apple M4 Max, 128 GB, macOS 15), the same model
file (Ollama's `gpt-oss:20b` GGUF blob), identical prompts, greedy decoding, and
llama.cpp/Ollama forced to CPU-only (`num_gpu: 0`). Our engine reads the same GGUF blob
directly.

### 4.2 Baseline results

| model | type | decode tok/s (CPU) | implied BW utilization |
|---|---|---|---|
| deepseek-r1:32b | dense | 12.7 | ~80% (236 GB/s) |
| gpt-oss:20b | MoE | 14.7 | **~11%** (~32 GB/s) |
| qwen3.5:35b-a3b | MoE | 27.5 | ~20% |

Working the observed rates backward into effective bandwidth reveals the finding:
llama.cpp's *dense* Q4 kernels achieve ~80% of the memory ceiling — near-optimal — but
its *MoE* decode path realizes only 11–20%. The cause is architectural: the MoE path
dispatches each expert's matrix-vector product as a separate operation with thread
synchronization between them, and (for `gpt-oss`) uses a slow scalar path for MXFP4
dequantization on CPU. The bandwidth is available; the software does not use it.

This is the opening. The under-served case is exactly the model class — small-active-set
MoE — that all frontier open releases now occupy.

---

## 5. Method

Our engine is written from scratch in Rust with inline ARM assembly for the hot kernels.
It parses the GGUF file directly, requires no framework, and links only the C standard
library. The techniques below were each landed with a before/after measurement and a
correctness check; the cumulative decode ladder is in §6.

**5.1 Exact MXFP4 arithmetic on integer hardware.** The sixteen MXFP4 values are not
integers, but *doubled* they are (`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`). We therefore (a)
convert each 4-bit code to its doubled integer via a single `tbl` lookup, (b) compute an
exact integer dot product against 8-bit-quantized activations with `sdot`, and (c) fold
the ÷2 into the per-block power-of-two scale. This yields bit-exact MXFP4 matrix-vector
products (verified to 0.0 max error against a 64-bit reference) with **no
floating-point dequantization of weights at all**.

**5.2 Quad-interleaved sequential-stream layout.** The single largest optimization
changes *zero* arithmetic. We repack expert weight matrices at load so that the four
rows a kernel processes together are contiguous in memory and each core reads one long
sequential stream. This aligns with the hardware prefetcher; scattered per-row reads
restart it. Effect: single-core kernel throughput rose from 45 to 65 GB/s (75% of the
per-core ceiling), and applying the same repack inside the full engine roughly doubled
end-to-end decode (§6).

**5.3 Block-wise (Q8) activation quantization.** Activations are quantized to 8-bit with
a separate scale per 32 values rather than one scale per tensor. This is required for
correctness, not just speed: hidden states in `gpt-oss` contain extreme per-channel
outliers (values in the tens of thousands) that a single per-tensor scale destroys. Per-
block scaling confines each outlier's error to its own block. We keep the tiny router in
full precision, since an error there changes *which* experts fire — a discrete,
catastrophic failure mode.

**5.4 MoE-aware scheduling.** Rather than dispatching experts separately, we flatten each
token's active experts into one flat work list executed by a persistent thread pool.
Measured expert-weight streaming then runs at the full 293 GB/s machine ceiling with
negligible routing overhead — the direct fix for the baseline's central weakness.

**5.5 Systems engineering.** A persistent condvar-parked thread pool (replacing ~1,350
thread creations per token), hand-vectorized attention score and value-accumulation
loops, fused expert output accumulation, and conversion of the bf16 attention and
vocabulary matrices to Q8 at load (halving their traffic and moving them onto the integer
`sdot` path).

**5.6 Faithfulness.** Correctly reproducing `gpt-oss` required recovering exact,
non-obvious details from the reference implementation: the router applies softmax over
the top-4 selected experts (not all 32); the SwiGLU-OAI activation uses `α=1.702`, a
clamp at ±7, and a `(up+1)` term; attention uses grouped-query attention (64 query heads
over 8 key/value heads), learned per-head attention "sinks" appended to the softmax
denominator, sliding-window attention on alternating layers, and YaRN-scaled rotary
position embeddings. Each subsystem was verified against an independent numeric reference
before integration.

---

## 6. Evaluation

### 6.1 Kernel-level decode-bandwidth ladder (`bench_infer`)

| version | change | GB/s (peak) |
|---|---|---|
| v1 | baseline nibble-unpack + `sdot` | 117 |
| v2 | bias-fold algebra removes a subtract stage | 151 |
| v3 | two accumulators per row (hide `sdot` latency) | 180 |
| v4 | quad-interleaved sequential layout | **228** |

### 6.2 End-to-end engine decode ladder

All steps produced output identical to the reference; each row is measured on the same
prompt.

| step | tok/s |
|---|---|
| first correct engine (bf16 attention path) | 26.3 |
| bf16 → Q8 attention/vocabulary weights | 31.9 |
| persistent thread pool | 35.5 |
| parallelized attention heads | 40.9 |
| 8 → 12 workers | 48.6 |
| NEON attention + fused barriers | 51.2 |
| **quad-interleaved expert layout** | **109.9** |

### 6.3 Headline comparison (Apple M4 Max, CPU only unless noted)

| system | gpt-oss:20b decode |
|---|---|
| llama.cpp / Ollama (`num_gpu:0`) | 14.7 tok/s |
| **this work** | **109.9 tok/s** |
| Ollama with Metal GPU (reference) | 94.4 tok/s |

Prefill throughput also improved as a byproduct of the layout change (21 → 64 tok/s).
Output was verified token-for-token identical to the llama.cpp/Ollama CPU reference on
matched prompts, including the model's internal reasoning channel.

---

## 7. Falsified hypotheses (negative results)

We report these as results because each was a measured, informative "no":

1. **Half/quarter-precision raises the SME matrix unit's throughput.** Refuted: bf16
   showed no gain (the wider instruction issues at half rate); int8 gave only 2×, not 4×.
   The machine's true ceilings are ~4.2 TFLOPS for any float precision and ~8 TOPS int8.
2. **Thread pinning (QoS "user-interactive") lifts multi-thread bandwidth.** Refuted: no
   change — the scheduler already placed the busy workers on performance cores. This
   pass also revealed ~±10% run-to-run variance from thermal state, prompting stricter
   measurement discipline (warm-up + best-of-N).
3. **A spin-waiting thread pool beats a parked one during decode.** Refuted (regression
   to 34.6 tok/s): busy-spinning workers steal cores from the serial glue work between
   parallel sections; occupancy beat wake-latency.
4. **Batched prompt processing amortizes weight reads (L1-resident).** Refuted twice
   (naive and tiled): prompt processing is instruction-throughput bound, not bandwidth
   bound; the two paths run the same arithmetic and take the same time. The correct fix
   is a register-blocked or SME GEMM kernel, identified but reserved for future work.

A methodological lesson emerged during end-to-end bring-up: an early MXFP4 scale-exponent
off-by-one (2^(e−127) vs. the correct 2^(e−128)) produced fluent-looking but wrong output
and survived all unit checks because *our reference implementation shared the same bug*.
It was caught only by comparing against three independent oracles — the format's
canonical source, the official tokenizer library, and the reference runtime's output.
**Verification references must be independently grounded, not self-derived.**

---

## 8. Discussion, limitations, and future work

**Scope of validity.** All numbers are on Apple M4 (ARMv9 + SME). The techniques —
bandwidth-first analysis, integer-SIMD low-bit kernels, sequential-stream layout,
MoE-aware scheduling — are architecture-general, but their quantitative benefit on ARM
server CPUs (Ampere, Graviton) and on x86 with AVX-512 must be measured, not assumed.

**Maturity.** This is a research runtime, not a production server: no cross-request
batching, no serving hardening, no multi-model management. The kernel and layout results
are the contribution; the surrounding serving layer is standard engineering.

**Model coverage.** `gpt-oss-20b` is verified end-to-end. Larger MoE models that fit in
128 GB at 4 bits (up to ~200B total) are projected from the bandwidth model
(e.g. a `gpt-oss-120B`-class model with ~2.5× the active bytes projects to ~45–55 tok/s)
but not yet run end-to-end.

**Future work.** (i) A register-blocked or SME-based prefill GEMM to fix the
instruction-bound prompt phase; (ii) speculative decoding, which converts decode into
small verify-batches that map onto the SME kernels from our compute study; (iii) porting
the kernels to ARM-server and x86 targets with measured re-benchmarking; (iv) extending
to the broader set of small-active-set MoE reasoning and coding models.

**Broader implication.** For the MoE model class that now dominates open frontier
releases, the gap between deliverable and realized CPU memory bandwidth is large,
measurable, and closable in software. Where a substantial fraction of inference demand is
latency-tolerant and cost-sensitive, well-engineered CPU serving becomes an economically
meaningful alternative to scarce, expensive accelerators — using hardware fleets that
already exist.

---

## 9. Reproducibility

The complete runtime, all benchmark binaries, the setup pipeline, and a chronological
measurement log (with the failures above) are released under MIT/Apache-2.0 at
`https://github.com/arizqi/cpubrrr`. No model weights are redistributed; the setup
script reads the user's own locally-pulled `gpt-oss` file. Every headline number is
regenerable from the corresponding binary.

---

*Author: Ashar Rizqi. Engineering and measurement performed in collaboration with an
AI coding agent (Claude). Not affiliated with OpenAI, Apple, or Ollama.*
