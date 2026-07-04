# cputrain — CPU-first LLM training runtime

Research goal: how large a transformer can you *train* (not just run) on consumer
Apple-silicon hardware in a reasonable wall-clock time, by optimizing from vector-math
primitives upward — register blocking, SIMD/SME instruction selection, cache-aware data
layout, and low-bit arithmetic. CPU-only first; Metal backend is a later phase.

## 1. First-principles ceiling (M4 Max, 128 GB)

Measured/spec'd hardware facts (this machine):

| Resource | Value |
|---|---|
| P-cores | 12 (M4, ~4.5 GHz) + 4 E-cores |
| NEON per P-core | 4×128-bit FMA pipes → 32 fp32 FLOP/cycle → ~144 GFLOPS/core |
| NEON all P-cores (fp32) | ~1.7 TFLOPS theoretical |
| AMX/SME matrix units | reached via Accelerate; measured by phase-0 bench (expect 2–3 TFLOPS fp32) |
| Unified memory bandwidth | 546 GB/s |
| L2 per P-cluster | 16 MB |
| RAM | 128 GB |

A100 40 GB reference: 312 TFLOPS bf16, 1.55 TB/s. Compute gap ≈ 80–150×; bandwidth gap
≈ 3×. Training is compute-bound (matmul arithmetic intensity ≫ 3.7 FLOP/byte roofline
knee), so **the compute gap is the binding constraint**. No software layer closes 100×;
kernel work buys us the distance between naive code and silicon peak (often 50–500×),
and low-bit arithmetic buys another integer factor on top.

### What fits in 128 GB (training state, not inference)

Per-parameter cost of standard training: bf16 weights (2 B) + bf16 grads (2 B) +
8-bit Adam moments (2 B) ≈ 6 B/param, plus activations (bounded via checkpointing).

→ **~15–20 B params is the hard in-RAM training ceiling** with known-convergent recipes.
4-bit *frozen* weights only help fine-tuning (QLoRA); pure 4-bit pretraining state is an
open research problem (all published low-bit training keeps higher-precision master
weights and gradient accumulators).

### What finishes in reasonable time (Chinchilla ≈ 20 tokens/param)

tokens/sec ≈ sustained FLOPS ÷ (6·N). At an optimistic 3 TFLOPS sustained fp32/fp16:

| Model | tok/s | Chinchilla tokens | Wall clock |
|---|---|---|---|
| 125 M | ~4,000 | 2.5 B | ~7 days |
| 350 M | ~1,400 | 7 B | ~2 months |
| 1 B | ~500 | 20 B | ~15 months |
| 3 B | ~170 | 60 B | ~11 years |
| 7 B | ~70 | 140 B | ~60 years |

Honest conclusion: CPU-only pretraining sweet spot today is **~100 M–1 B params**.
Low-bit compute (int8 SME ≈ 2–4×, ternary/LUT forward ≈ 4–8× on the forward pass) could
drag ~3 B into "months" territory — that is the research bet of this project. 7 B+
pretraining needs the Metal phase or distributed CPUs. Fine-tuning (frozen quantized
base + LoRA) reaches 30 B+ in RAM and is a supported later milestone.

## 2. The "below the kernel" thesis — where byte-level leverage actually lives

There is no bytecode/JIT layer on this path (we compile AOT Rust + asm); the exploitable
layers, top to bottom:

1. **Loop order & cache blocking** — tile for 128 KB L1d / 16 MB shared L2; pack panels
   so the microkernel streams contiguous, aligned data.
2. **Register blocking / instruction selection** — hand-written NEON microkernel:
   `vfmaq_laneq_f32` broadcast-FMA pattern, 24+ live vector registers, software
   pipelining of loads under FMAs.
3. **SME/SME2 direct** (M4 has SME2) — streaming-mode outer-product tiles, hand asm;
   compare against Accelerate's use of the same silicon.
4. **Precision engineering** — fp16 NEON FMA (2× fp32 rate), int8 dot products
   (`sdot`/`smmla`, 4–8×), with fp32 accumulate + master weights.
5. **Bit-level formats** — lookup-table matmul (T-MAC style) and popcount/bit-serial
   tricks for 1–4-bit weight forward passes; interleaved quant layouts designed for the
   load unit, not for readability.
6. **Memory hierarchy beyond RAM** — 8-bit optimizer states, activation checkpointing,
   fused backward, SSD streaming for >RAM experiments (8 GB/s NVMe vs 546 GB/s RAM —
   only viable for cold optimizer shards).

## 3. Experiment ladder

- **Phase 0 (this commit): ceiling bench.** fp32 matmul ladder — naive → loop-reorder →
  cache-blocked → NEON microkernel → 12-thread NEON → Accelerate `cblas_sgemm`.
  Establishes the machine's real peak and how much of it hand-written code recovers.
  Success: numbers on the table below, correctness-checked.
- **Phase 1: own the microkernel.** Reach ≥80% of Accelerate fp32 without Accelerate;
  add fp16 microkernel (target ~2× fp32 rate). Threaded packing, NUMA-free scheduling
  across P-cores only.
- **Phase 2: SME2 direct.** Hand-asm streaming-mode kernels; measure vs Accelerate.
  Decide build-vs-use (Accelerate is closed but ships on every target Mac).
- **Phase 3: quantized training kernels.** int8 (`smmla`) forward/backward with fp32
  master weights; LUT matmul for ≤4-bit frozen-weight paths; numerical-fidelity harness
  (train tiny model, compare loss curves vs fp32 reference).
- **Phase 4: training loop.** llm.c-style hand-written transformer (no framework):
  fused QKV/attention/MLP kernels, flash-attention-style tiling for CPU caches,
  fused cross-entropy. Train GPT-2-124M on C4/FineWeb subset end-to-end; report tok/s
  vs PyTorch-CPU, vs llm.c, vs A100.
- **Phase 5: scale.** 8-bit Adam, checkpointing, biggest-trainable-model search on
  128 GB; 350 M → 1 B → 3 B (low-bit compute). Publish measured wall-clock table.
- **Phase 6: Metal backend** (deferred by design decision 2026-07-02).

## 4. Benchmark discipline

Every optimization lands with: before/after GFLOPS (or tok/s), % of Accelerate ceiling,
and a correctness check vs a reference kernel. Failed experiments get logged in
`docs/RESEARCH_LOG.md`, not deleted.

## 5. Open questions carried forward

- Can SME2 hand-asm beat Accelerate, or is Accelerate already at silicon peak?
- Does int8-forward/int8-backward with fp32 master weights match bf16 loss curves at
  125 M scale? (Gate for the 3 B bet.)
- LUT matmul is bandwidth-bound — profitable for training forward passes, or
  inference-only in practice?
- E-cores: worth scheduling packing/dataloader work on, or pure interference?
