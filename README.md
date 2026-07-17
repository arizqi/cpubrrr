# cpubrrr


https://github.com/user-attachments/assets/a390cb6e-86b8-41e5-9ef6-957c94dabe19


**From-scratch CPU-only LLM inference that beats llama.cpp's CPU path — on both quant formats it runs, no GPU.**

`cpubrrr` is a research runtime that runs frontier-class **mixture-of-experts** models
on an Apple M4 Max **CPU only** — the binary links nothing but the C standard library,
so it *physically cannot* touch the GPU (`otool -L target/release/engine` to verify). It
started as a one-model experiment (gpt-oss:20b) and now runs four MoE models across two
architectures and two quantization formats from one config-driven engine — and it is
**faster than llama.cpp's CPU path on both formats**, including Q4_K, llama.cpp's own
hand-tuned home turf.

## Numbers (Apple M4 Max, CPU only, log-verified 0 GPU layers, cool + quiet machine)

| model | quant | cpubrrr | llama.cpp CPU | |
|---|---|---|---|---|
| **gpt-oss:20b** | MXFP4 | **~77 tok/s** | ~14 tok/s | **~5×** |
| **Qwen3-Coder-30B** | Q4_K/Q6_K | **~92 tok/s** | ~82 tok/s | **~1.1–1.2×** |

Decode throughput, several runs each. Output verified correct in both cases. llama.cpp
placement confirmed CPU-only from Ollama's own server logs (it defaults to Metal GPU on
macOS — `num_gpu:0` is a *request*, not a fact; see the benchmark-integrity note below).

These numbers replace this repo's earlier, over-optimistic claims (a "7.5× / 110 tok/s"
headline that did not survive rigorous re-measurement — the reproducible figure is
~77, recovered from a 52 tok/s regression by fixing dispatch overhead the hard way).
The story of *how* the early numbers were wrong — a contaminated baseline, unverified
GPU/CPU placement, thermal throttling, and a thread-pool whose condvar wakeups silently
cost 7.5 ms/token — is documented in full, in order, with evidence. We consider that
the most transferable part of the project.

## Why it's fast

Token generation for an MoE model is **memory-bandwidth bound**, not compute bound: a
21B model activates only ~3.6B params per token, so decode speed ≈ memory bandwidth ÷
active-bytes-per-token. llama.cpp's MXFP4 MoE CPU path uses a small fraction of the
available bandwidth; cpubrrr streams expert weights at close to what the cores can
sustain. Core techniques:

- Hand-written **NEON** (ARM SIMD) integer kernels using `sdot`/`tbl` for exact 4-bit
  arithmetic on integer hardware — no float dequant in the inner loop.
- **Quad-interleaved weight layout** so each core reads one sequential stream (the
  single biggest MXFP4 win — a byte-reordering, not a code change).
- **Integer-accumulation Q8_K kernel** for Q4_K/Q6_K — llama.cpp's own algorithm,
  found by reading its ARM source: quantize activations to Q8_K, accumulate sub-block
  integer dot products weighted by the 6-bit scales in int32, and convert to float
  *once per 256-value superblock*. This is what took Q4_K from losing (~71 vs ~86) to
  winning (~92 vs ~82). Kernels verified bit-exact against a dequant reference.
- **Worker-driven execution**: 12 persistent workers run the whole forward pass with a
  *yielding* spin-barrier (spin briefly, then let the OS in) — saturates all cores
  without the collapse-under-jitter that a pure spin-barrier suffers.
- MoE-aware scheduling, block-wise Q8 activation quantization, `mmap`'d weights (so a
  117B model pages in under memory pressure instead of OOM-killing).

The lesson behind the Q4_K win: to beat a mature kernel, don't out-clever it with
micro-tweaks — read its source and the research, find the *algorithmic* edge, adopt it,
then out-schedule it.

## Models

One config-driven engine (dimensions read from the model at setup — same-family models
run with zero code changes):

- **gpt-oss:20b** (MXFP4) — the original target, verified end-to-end.
- **gpt-oss-120b** (117B / ~5.1B active, MXFP4) — 6× bigger, same family; runs on the
  laptop CPU via `mmap`'d weights.
- **Qwen3-Coder-30B** (`qwen3moe`, Q4_K/Q6_K) — a different architecture *and* a
  different quant format; writes correct code.
- **Qwen3-30B** general — same arch, drop-in.

Architecture math was recovered from llama.cpp source and the 4-bit unpacking verified
bit-for-bit against the official `gguf` library *before* writing the forward pass — so
the new architecture produced correct output on the first run.

## Docs

- **[docs/RESEARCH_LOG_V2.md](docs/RESEARCH_LOG_V2.md)** — chronological lab log: every
  measured number, every wrong number, every correction, in order.
- **[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)** — the original single-model log.
- **[docs/PLAN.md](docs/PLAN.md)** — first-principles ceilings and the experiment ladder.

## Requirements

- Apple silicon Mac (M-series; developed on **M4 Max**, ARMv9 + SME). Other ARM/x86
  targets need a port — see the honest-limits section below.
- [Rust](https://rustup.rs/) (stable), Python 3, and [Ollama](https://ollama.com/) with
  a model pulled (e.g. `ollama pull gpt-oss:20b` or `ollama pull qwen3-coder:30b`).

## Quick start

```bash
# 1. build (uses target-cpu=native)
cargo build --release

# 2. prepare runtime data from your local Ollama copy (no weights are copied)
./scripts/setup_model.sh gpt-oss:20b     # or qwen3-coder:30b, etc.
                                         # writes data-<slug>/{tokens.bin,config.txt,...}

# 3. generate (gpt-oss uses the `engine` binary; qwen uses `engine_qwen2`)
./target/release/engine data-gpt-oss_20b "$(cat data-gpt-oss_20b/blob_path.txt)" "Why is the sky blue?"
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
| `engine` | full gpt-oss (MXFP4) generation engine |
| `engine_qwen2` | full Qwen3 (Q4_K/Q6_K) generation engine — the worker-driven, integer-accum path |
| `q8k_verify` | Q4_K/Q6_K integer-accum kernels vs a dequant-f64 reference |
| `qk_verify` | Q4_K/Q6_K dequant vs the official `gguf` library (bit-exact) |
| `bench_real` | real gpt-oss MXFP4 weights through the decode kernel |
| `bench_moe` | MoE expert-batched decode bandwidth |
| `bench_sme` | direct SME2 matrix-unit kernels + precision sweep |

`scripts/bench_ollama.sh <model> <cpu|gpu>` produces the llama.cpp baseline **and reads
Ollama's server log to confirm actual GPU/CPU placement** — it refuses to report a "CPU"
number if any layers ran on the GPU.

## Benchmark integrity (read this before quoting numbers)

Bandwidth-bound CPU benchmarks are easy to get wrong, and this project got them wrong
three times before getting them right:

1. **Contaminated baseline** — the first llama.cpp number was measured while a 65 GB
   download ate memory bandwidth in the background.
2. **Unverified placement** — on macOS, Ollama defaults to full Metal GPU; some "CPU"
   runs were silently 49/49 layers on the GPU. Only the server log (`offloaded N/M
   layers to GPU`) reveals the truth.
3. **Thermal throttling** — after days of load the *same binary* gave 66 → 42 → 5 tok/s
   on back-to-back runs.

cpubrrr also saturates all 12 cores, so its peak needs a quiet machine; llama.cpp
tolerates background load and heat better. Meta-lesson: **be more skeptical of
benchmarks that flatter you, not less.** Re-verify on your own hardware.

## Status & honest limits

This is a **research engine**, not a production server (no cross-user batching, no
serving hardening). Numbers are Apple M4; the techniques port to other ARM and (with
AVX-512) x86, but those numbers must be measured, not assumed. A `training/kernel`
research track programs the M4's SME/AMX matrix unit directly from Rust assembly
(~4.2 TFLOPS fp32, measured to exceed Accelerate) — see the research log. On training
scale: consumer CPUs can realistically train models up to ~1B params; a
trillion-parameter model is ~47,000 years of compute (physics, not pessimism).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

Not affiliated with OpenAI, Apple, or Ollama. `gpt-oss` and `Qwen` are released by their
authors under their own licenses; this project neither includes nor redistributes model
weights.
