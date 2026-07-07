# cpubrrr, Part II: A Config-Driven CPU-Only Inference Engine Across Three MoE Architectures — and a Cautionary Study in Benchmark Integrity

**Abstract.** We extend a from-scratch CPU-only LLM inference engine (Part I: gpt-oss:20b at interactive speed on an Apple M4 Max) to three additional Mixture-of-Experts models spanning a 6× parameter range and two architecture families: gpt-oss-120b (117B params, MXFP4), and Qwen3-Coder-30B / Qwen3-30B (30B params, Q4_K/Q6_K, `qwen3moe`). We contribute: (1) a config-driven engine that runs a 117B frontier model on a 128 GB laptop CPU via memory-mapped weights; (2) native Q4_K/Q6_K matvec kernels verified bit-exact against the reference `gguf` library; (3) a worker-driven execution model with a *yielding* spin-barrier that saturates all cores robustly, replacing a barrier-bound fork-join design; and (4) a format-dependent performance characterization showing our engine is several-fold faster than llama.cpp's CPU path on symmetric MXFP4 MoE but *slower* than its mature asymmetric-Q4_K path. Most importantly, we document — with full evidence — a sequence of **benchmark-integrity failures** (a contaminated baseline, unverified GPU/CPU placement, and thermal instability) that led us to publicly overclaim and then repeatedly correct. We argue that verified, adversarial benchmarking is a first-class engineering discipline, and release a placement-verified benchmark harness. All code, measurements, and corrections are public.

---

## 1. Introduction

Part I demonstrated that aggressive low-level optimization (hand-written NEON/SME kernels, MoE-aware scheduling, memory-layout tuning) makes frontier-*class* Mixture-of-Experts (MoE) inference viable on a consumer CPU. A single-model result, however, is not an engine. Part II asks whether the approach **generalizes** across model sizes and architectures, and — equally important — whether our performance claims **survive rigorous measurement**.

Our findings are mixed and, we believe, more valuable for it:

- **Generalization: success.** One config-driven engine runs gpt-oss-120b (6× larger, same family) and the `qwen3moe` family (a distinct architecture and quantization format), producing verified-correct output in both cases.
- **Performance: format-dependent.** We are several-fold faster than llama.cpp's CPU path on MXFP4 MoE (a format for which its CPU kernels are weak) and slower on Q4_K (its mature, heavily-tuned format).
- **Process: a cautionary tale.** We overclaimed a general "we beat llama.cpp" result three times, each traceable to a distinct benchmarking error. We document the mechanism of each error and the fix.

We consider (4) the most transferable contribution. The open-model community rightly scrutinizes benchmark methodology; our experience is a concrete case study in how confident, well-intentioned engineers fool themselves.

---

## 2. Generalizing the engine

### 2.1 Config-driven dimensions

The Part-I engine hardcoded gpt-oss:20b's dimensions. We refactored all model dimensions (layer count, expert count/top-k, head geometry, vocab, RoPE base, RMS epsilon, sliding-window) into a runtime `Cfg` read from a per-model config emitted at setup. Same-family models then require **no code changes**.

**gpt-oss-120b (117B/5.1B-active, 36 layers, 128 experts):** ran directly on the config-driven engine. Naively reading the 61 GB weight file into an anonymous buffer, combined with a working-copy repack, exceeded 128 GB and was OOM-killed. Replacing `read` with a memory-mapped (file-backed, evictable) buffer fixed this: the OS pages the blob under memory pressure. The 117B model then generated correct output on CPU.

### 2.2 A distinct architecture, verified before implementation

`qwen3moe` differs from gpt-oss in attention (QK-norm — per-head RMSNorm on Q/K pre-RoPE), router gating (softmax-over-all → top-k → renormalize, vs. gpt-oss's top-k → softmax), activation (plain SiLU-SwiGLU vs. clamped SwiGLU-OAI), presence of bias/sinks/sliding-window (none), and quantization (Q4_K/Q6_K k-quants vs. MXFP4).

Following Part I's central lesson (verify against independent oracles), we (a) recovered the exact computation graph from llama.cpp's `qwen3moe` source, and (b) implemented Q4_K/Q6_K dequantization in Rust and verified it **bit-exact** (max abs diff 0) against the official `gguf` Python library on real tensors, *before* writing the forward pass. The engine produced correct code output on its first execution, with no output-divergence debugging cycle.

---

## 3. Kernels

### 3.1 Native 4-bit k-quant matvec

Weights remain in their native GGUF k-quant form in the mapped blob (no dequant-to-int8 at load, which would double memory traffic). Per-row matvec dequantizes inline via NEON: `Q4_K` (256-element super-blocks: 6-bit packed scales+mins, 4-bit quants) and `Q6_K` (interleaved 6-bit signed quants, per-16 int8 scales). Both dot against per-32-block int8-quantized activations using `sdot`. Correctness was established transitively: our dequant matches `gguf` bit-exact, and our fused matvec matches dequant-then-dot to ~1e-7 relative error.

Two kernel results are notable. First, **vectorizing the Q6_K reconstruct** (replacing a 256-element scalar unpacking loop with 16-lane NEON) was the single largest kernel win, roughly halving the down-projection and head cost. Second, adopting llama.cpp's **u32 bit-trick for unpacking all 8 Q4_K scales/mins at once** (vs. 8 branchy scalar extractions) gave a further gain.

### 3.2 The quality ceiling for Q4_K on NEON

Our gpt-oss MXFP4 kernel reaches ~228 GB/s (near memory bandwidth) because MXFP4 is *symmetric* — one scale per block, trivial unpack. Q4_K is *asymmetric* (scale + min per sub-block); the unpack is inherently more instructions, capping throughput well below bandwidth. Converting Q4_K to the fast symmetric layout is possible but lossy: we measured symmetric-int4 requantization at **8–13% relative error** (vs. 0.5% for int8), which degrades a coding model. Thus Q4_K decode on NEON is instruction-bound at a *quality-preserving* ceiling; matching bandwidth would require either a symmetric-quant model or a matrix-unit path that does not map to single-vector GEMV.

---

## 4. Execution model

The Qwen engine was initially fork-join (main thread dispatches per-stage work to a condvar pool). Profiling showed CPU utilization of only ~6–7 of 12 cores: with ~8 parallel regions/layer × 48 layers ≈ 340 barriers/token and ~35 µs per condvar wakeup, barrier overhead — not the kernels — dominated.

We rewrote to a **worker-driven** model: 12 persistent workers execute the entire forward pass, synchronizing at sense-reversing barriers and parking only once per token. All previously-serial glue (RMSNorm reductions, activation quantization, routing) is parallelized.

A pure spin-barrier saturated cores in isolation (~65 tok/s) but **collapsed to 5 tok/s under any scheduling jitter**: 12 spinning workers on 12 performance cores leave none for the OS, which must preempt a worker, stalling all others at the barrier. The fix is a **yielding barrier** (bounded spin, then `yield_now`): the engine now (a) uses all cores, (b) is stable in isolation, and (c) degrades *gracefully* under contention (~27 tok/s with two cores externally loaded) rather than collapsing. This is a concrete instance of a general principle: **busy-wait synchronization is fragile on a non-realtime OS; cooperative yielding is required for robustness.**

---

## 5. Performance: honest and format-dependent

Figures below are from a cool, quiet (no background browser), contention-controlled pass, 3–5 runs each (§6). All llama.cpp figures are **placement-verified** from server logs (`offloaded N/M layers to GPU`); the "CPU" figures are confirmed 0 layers on GPU.

| Model (CPU decode) | llama.cpp CPU (verified) | cpubrrr (CPU-only by construction) | Direction |
|---|---|---|---|
| gpt-oss:20b (MXFP4) | 13.7 tok/s (stable) | ~55 tok/s (stable) | **cpubrrr ~4.0× faster** |
| Qwen3-Coder-30B (Q4_K) | 85.7 tok/s (mature) | ~71 tok/s | **llama.cpp faster (~1.2×)** |

The MXFP4 gap is large and robust: our stable ~55 exceeds llama.cpp's ~14 by ~4×, consistent with llama.cpp's documented weak CPU-MoE path for MXFP4. The Q4_K gap favors llama.cpp: Q4_K is its mature format, and our quality-preserving ceiling (§3.2) sits below it. cpubrrr uses only the C standard library (`otool`-verified), so it *cannot* use the GPU; llama.cpp on macOS defaults to full Metal offload and must be explicitly and verifiably constrained to CPU.

**We explicitly do not claim a general "beats llama.cpp" result.** We win where its CPU path is weak and lose where it is strong.

---

## 6. Benchmark integrity: a cautionary study

We publicly claimed, retracted, re-asserted, and re-corrected the Qwen CPU comparison. Each error is documented in the lab log with evidence; we summarize the mechanisms because they are common and instructive.

1. **Contaminated baseline.** An early llama.cpp "47 tok/s" baseline was measured while a 65 GB download consumed memory bandwidth. A bandwidth-bound workload measured under bandwidth contention understates the competitor. *Fix: identical, quiescent conditions for all systems under comparison.*
2. **Unverified placement.** `num_gpu:0` is a *request*. On macOS, Ollama defaults to full GPU and, across instance reuse, our "CPU" runs were in some cases logged as `offloaded 49/49 layers to GPU` — i.e., GPU numbers. *Fix: read the runtime's own placement log after every run; refuse to label a run "CPU" unless the log confirms zero GPU layers. We release this hardened harness.*
3. **Thermal instability.** After sustained load, the same binary produced 66→42→5 tok/s across runs (thermal throttling). Single-sample benchmarks on a hot machine are noise. *Fix: cool, rested machine; multiple samples; report spread.*

The meta-lesson: **results that flatter the author deserve more skepticism, not less.** An unverified benchmark provides false confidence and is worse than none. We caught these errors ourselves — but only after overclaiming publicly first. The discipline we recommend: verify placement and conditions *before* claiming, treat every favorable number as a hypothesis to be falsified, and publish the methodology and raw spread, not a headline.

---

## 7. Limitations and future work

- **Figures in §5 are the reproducible cool-machine, log-verified, contention-controlled numbers** (5/3-run stable). One caveat: an earlier ~110 tok/s for gpt-oss on a fresh machine was not reproducible after a week of sustained load (both original and current binaries measure ~55; not a code regression); the conservative ~55 is used, and the 4× MXFP4 win holds regardless.
- The gpt-oss path shows no code regression (the initial-release binary measures identically to current, ~55); the earlier ~110 peak appears tied to a deeper machine-cool state we could not recreate. cpubrrr is contention-sensitive (targets all 12 cores); peak numbers require a quiet machine.
- Q4_K decode is at a quality-preserving instruction ceiling on NEON; closing to llama.cpp likely needs a different arithmetic path.
- Not a production server: no cross-request batching or serving hardening.
- Results are Apple M4 (ARMv9+SME); portability to other ARM and to x86 (AVX-512) must be measured, not assumed.

## 8. Conclusion

One config-driven CPU-only engine runs three MoE models across a 6× size range and two architectures, correctly and GPU-free. It is several-fold faster than llama.cpp's CPU path on symmetric MXFP4 MoE and slower on mature asymmetric Q4_K — an honest, format-dependent result. The execution-model rewrite (yielding barrier) is a robust, transferable systems result. And our most durable contribution may be negative: a documented, evidence-backed account of how easily confident engineers overclaim, and a verified benchmarking discipline to prevent it.

---

*Artifacts: engine and kernels (`src/bin/`), verified dequant (`src/bin/qk_*`), placement-verified benchmark (`scripts/bench_ollama.sh`), and complete lab notes with all corrections (`docs/RESEARCH_LOG.md`, `docs/RESEARCH_LOG_V2.md`). Not affiliated with OpenAI, Alibaba, Apple, or the llama.cpp/Ollama projects; model weights are neither included nor redistributed.*
