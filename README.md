# cpubrrr

**Frontier-class LLM inference on a laptop CPU — 7.5× faster than llama.cpp, no GPU.**

`cpubrrr` is a from-scratch research runtime that runs OpenAI's open
[`gpt-oss:20b`](https://huggingface.co/openai/gpt-oss-20b) model at **~110 tokens/second
on an Apple M4 Max CPU**, versus **~15 tok/s** for llama.cpp on the same machine in
CPU-only mode — producing **token-for-token identical output**. Everything below is
measured on real hardware and reproducible from this repo.

| gpt-oss:20b decode, M4 Max, CPU only | tok/s |
|---|---|
| llama.cpp / Ollama (`num_gpu: 0`) | 14.7 |
| **cpubrrr engine** | **109.9** |
| *(reference) Ollama with Metal GPU* | 94.4 |

The engine links only the C standard library — no Metal, no CoreML, no Accelerate, no
GPU of any kind (`otool -L target/release/engine` to verify).

## Why it's fast

Modern frontier models are **mixture-of-experts (MoE)**: a 21B-parameter model only
activates ~3.6B parameters per token. Token generation is therefore **memory-bandwidth
bound**, not compute bound. llama.cpp's MoE path uses only ~11% of the CPU's memory
bandwidth (measured); cpubrrr streams expert weights at the full ~293 GB/s the CPU
cores can sustain. The core techniques:

- Hand-written **NEON** (ARM SIMD) integer kernels using `sdot`/`tbl` for exact MXFP4
  4-bit arithmetic on integer hardware — zero float dequant.
- **Quad-interleaved weight layout** so each core reads one sequential stream (the
  single biggest win — a byte-reordering, not a code change).
- MoE-aware scheduling: each token's active experts flattened into one work list, no
  per-expert dispatch overhead.
- Block-wise **Q8** activation quantization (outlier-safe), persistent thread pool,
  vectorized attention.

There's also a **training/kernel-research track** that programs the M4's undocumented
matrix coprocessor (**SME/AMX**) directly from Rust assembly — ~4.2 TFLOPS fp32,
measured to exceed Apple's own Accelerate library.

Full derivation, every experiment (including four refuted hypotheses), and a
student-level teaching walkthrough:

- **[docs/TEACHING.md](docs/TEACHING.md)** — explains everything from first principles,
  every acronym defined. Start here.
- **[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)** — chronological lab log, all measured
  numbers, failures kept.
- **[docs/PLAN.md](docs/PLAN.md)** — first-principles ceilings and the experiment ladder.

## Requirements

- Apple silicon Mac (M-series; developed on **M4 Max**, ARMv9 + SME). Other ARM/x86
  targets need a port — see caveats in the teaching doc.
- [Rust](https://rustup.rs/) (stable), Python 3, and [Ollama](https://ollama.com/) with
  the model pulled: `ollama pull gpt-oss:20b`.

## Quick start

```bash
# 1. build (uses target-cpu=native)
cargo build --release

# 2. prepare runtime data from your local Ollama copy (no weights are copied)
./scripts/setup_gptoss.sh          # writes data/tokens.bin, data/manifest.txt

# 3. generate
./target/release/engine data "$(cat data/blob_path.txt)" "Why is the sky blue?"
```

### Side-by-side demo

A local web page streams cpubrrr vs. llama.cpp/Ollama (CPU) with live tok/s counters:

```bash
python3 scripts/demo_server.py     # then open http://localhost:8642
```

## Benchmarks & kernels

Each is a standalone binary (`cargo build --release`, then `./target/release/<name>`):

| binary | what it measures |
|---|---|
| `bench_matmul` | fp32 GEMM ladder: naive → NEON → Accelerate ceiling |
| `bench_sme` | direct SME2 matrix-unit kernels + precision sweep |
| `bench_infer` | int4 GEMV (decode inner loop) optimization ladder |
| `bench_moe` | MoE expert-batched decode bandwidth |
| `bench_real` | real gpt-oss MXFP4 weights through the decode kernel |
| `engine` | the full generation engine |

`scripts/bench_ollama.sh <model> <cpu|gpu>` produces the llama.cpp baseline.

## Status & honest limits

This is a **research engine**, not a production server (no cross-user batching, no
serving hardening). Results are on Apple M4 silicon; the techniques port to other ARM
and (with AVX-512) x86, but those numbers must be measured, not assumed. gpt-oss:20b is
verified end-to-end; larger MoE models are projected from measured bandwidth. Consumer
CPUs can realistically *train* models up to ~1B parameters — a trillion-parameter model
is ~47,000 years of compute (physics, not pessimism). See the teaching doc for the full
boundary discussion.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

Not affiliated with OpenAI, Apple, or Ollama. `gpt-oss` is released by OpenAI under its
own license; this project neither includes nor redistributes model weights.
