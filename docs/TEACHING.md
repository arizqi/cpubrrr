# How We Made a Laptop Run a Frontier AI Model 7.5× Faster Than the Standard Software

### A teaching walkthrough — written for a high-school or early-undergraduate reader

*The test for this document: after reading it, you should be able to explain to someone else — accurately, in your own words — what was built, why it worked, and what every technical term means. No prior knowledge is assumed beyond basic algebra and the idea that computers run programs.*

*(A second section at the end, "For the CTO/CPO," translates the results into business terms.)*

---

## Part 1 — The question we asked

Modern AI models like ChatGPT run in giant datacenters on specialized chips called **GPUs** that cost tens of thousands of dollars each. Our question was simple:

> **How much of that can an ordinary consumer laptop do — using only its CPU?**

The laptop in question is an Apple MacBook with an **M4 Max** chip and 128 GB of memory. The AI model is **gpt-oss:20b** — a real, freely downloadable model released by OpenAI, from the same family as their commercial products.

The headline result: we wrote our own software from scratch and ran that model at **109.9 tokens per second**, versus **14.7 tokens per second** for the standard open-source software (llama.cpp) on the exact same machine — **7.5× faster**, producing *word-for-word identical output*. No GPU involved at any point.

The rest of this document explains, step by step, how — including the failures, because four of our hypotheses turned out to be wrong, and the wrong ones taught us the most.

---

## Part 2 — The vocabulary you need (every acronym, explained)

Read this section once; everything later builds on it.

**CPU (Central Processing Unit).** The general-purpose "brain" of every computer. It executes instructions one after another, very fast. The M4 Max CPU has 16 **cores** — 12 fast "performance" cores (P-cores) and 4 slower, power-efficient ones (E-cores). A core is essentially an independent mini-CPU; more cores means more things happening at once.

**GPU (Graphics Processing Unit).** A chip originally designed to draw video-game graphics, which turned out to be excellent at the kind of math AI needs. Datacenter GPUs (like Nvidia's A100) are the standard AI workhorse. Our project's whole point was to *not* use one.

**FLOP / FLOPS (Floating-Point Operation / ...per Second).** A "floating-point number" is how computers store decimals (like 3.14159). One FLOP is a single addition or multiplication of such numbers. FLOPS measures speed: a **GFLOPS** is a billion per second, a **TFLOPS** is a trillion per second. AI is, at bottom, an astronomical number of multiplications — so FLOPS is the currency of AI computing.

**Parameter.** A single learned number inside an AI model. gpt-oss:20b has about 21 billion parameters. Think of them as the model's memory of everything it learned during training.

**Token.** Models don't read letters or whole words; they read *tokens* — chunks of text, usually a word or part of one ("understand" might be one token; "understandable" might be two). A model generates its answer one token at a time. **Tokens per second (tok/s)** is the standard speed measure for AI text generation. Around 20 tok/s feels like fluent reading speed; 100+ feels instant.

**Inference vs. Training.** *Training* is teaching the model (adjusting billions of parameters — needs enormous compute). *Inference* is using the trained model to answer questions. This document is mostly about making inference fast; the project also has a training track.

**Matrix multiplication.** The single mathematical operation that dominates AI. A matrix is a grid of numbers. Multiplying two matrices means computing lots of "dot products" — multiply pairs of numbers and add them up. Roughly 95%+ of the work in running an AI model is matrix multiplication, which is why the whole project is about making it fast.

- **GEMM** (GEneral Matrix Multiply): matrix × matrix. Happens when processing many tokens at once.
- **GEMV** (GEneral Matrix-Vector multiply): matrix × single column of numbers. Happens when generating one token at a time.

This distinction turns out to be the key to everything (Part 4).

**Memory bandwidth (GB/s — gigabytes per second).** How fast data can flow from the computer's memory (RAM) into the CPU. Our machine's memory system can theoretically move 546 GB/s; we measured that the CPU cores alone can actually pull **293 GB/s**. This number ends up mattering more than FLOPS for text generation — a central lesson.

**Cache (L1, L2).** Small, extremely fast memory *inside* the CPU chip. **L1** is tiny (128 KB per core) and fastest; **L2** is bigger (16 MB shared by a cluster of cores) and slower; main RAM is huge and slowest. The CPU automatically keeps recently-used data in cache. A huge fraction of performance engineering is arranging your data so the math finds its numbers already sitting in cache instead of waiting on RAM.

**SIMD (Single Instruction, Multiple Data).** Normally one CPU instruction does one operation ("multiply these two numbers"). SIMD instructions do the same operation on a whole batch at once ("multiply these *four pairs* of numbers"). It's how CPUs got fast at math.

**NEON.** ARM's brand name for the SIMD instructions in chips like Apple's. NEON works on 128-bit registers — a register is a tiny storage slot inside the CPU — so one NEON instruction can process four 32-bit decimal numbers, or sixteen 8-bit integers, at once.

**`sdot` and `tbl`.** Two specific NEON instructions we leaned on. `sdot` ("signed dot product") multiplies sixteen pairs of small integers and adds them up — in one instruction. `tbl` ("table lookup") replaces each of 16 bytes with a value looked up from a 16-entry table — also one instruction. Together they let us do exotic AI math using plain integer hardware.

**SME (Scalable Matrix Extension) and AMX.** Beyond SIMD, Apple's chips contain a *matrix coprocessor* — dedicated silicon whose only job is multiplying matrices. For years it was Apple-proprietary and undocumented (called **AMX**); the M4 is the first Apple chip that exposes it through an official, public instruction set called **SME**. One SME instruction (`fmopa`) performs **256 multiply-adds at once** — versus 4 for a NEON instruction. We measured this unit at ~2 trillion operations/second, and found the chip has two of them (~4.2 TFLOPS total). Our training-track kernels drive it directly; it's the biggest untapped headroom for the inference engine too.

**Kernel.** In this context, not the operating system — a *compute kernel* is a small, hand-optimized inner-loop function that does one mathematical job (like "multiply this matrix by this vector") as fast as the silicon allows. The project is essentially a collection of kernels plus the logic that connects them.

**Quantization.** Model parameters are naturally stored as 16- or 32-bit decimals. Quantization stores them with fewer bits — 8, or even 4 — accepting a tiny accuracy loss for a huge size reduction. Formats we used:
- **FP32 / BF16**: 32-bit and 16-bit decimal formats. BF16 ("brain float 16") is FP32 with the least-significant half chopped off — same range, less precision, half the size.
- **INT8 / Q8**: 8-bit integers. Our "Q8" scheme stores 32 weights as integers plus one shared scaling factor — so real value = integer × scale.
- **MXFP4**: the 4-bit format gpt-oss actually ships in. Each parameter is one of just 16 possible values (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6), with a shared power-of-two scale for every block of 32. Four bits per parameter is what lets a 21-billion-parameter model fit in 13 GB.

**MoE (Mixture of Experts).** A model architecture trick. Instead of one giant network, the model contains many small "expert" networks (gpt-oss:20b has 32 per layer) plus a tiny "router" that picks which 4 experts to use *for each token*. So a 21B-parameter model only *activates* ~3.6B parameters per token — it has a big library but only opens four books at a time. Nearly all frontier models now use MoE, and it's the loophole that makes fast CPU inference possible.

**Transformer, and its parts.** The neural-network architecture behind all modern language models. Per layer (gpt-oss has 24 layers), a token's data flows through:
- **Attention** — the mechanism that lets the current token "look back" at previous tokens and decide which ones matter. Involves Query/Key/Value (Q/K/V) vectors and a **softmax** (a formula that turns raw scores into percentages summing to 100%).
- **KV cache** — stored Key/Value vectors from all previous tokens, so they aren't recomputed for every new token.
- **GQA (Grouped-Query Attention)** — a memory-saving variant: 64 "question-asking" heads share 8 sets of stored keys/values.
- **Attention sinks** — a gpt-oss quirk: each head has a learned "null option" it can attend to when nothing in the text is relevant.
- **RoPE (Rotary Position Embedding)** — how the model knows *where* each token sits in the sentence: token vectors are literally rotated by an angle proportional to their position. **YaRN** is a modification that stretches RoPE so the model can handle much longer texts than it was trained on.
- **FFN (Feed-Forward Network)** — the other half of each layer; in gpt-oss it's the MoE expert block, using an activation function called **SwiGLU** (a smooth gate that decides how much signal passes through).

**GGUF.** The standard file format for downloadable AI models (used by llama.cpp and Ollama). We read gpt-oss's 13 GB GGUF file directly, byte by byte, with our own parser.

**llama.cpp / Ollama.** llama.cpp is the most popular open-source program for running AI models on ordinary computers; Ollama is a friendly wrapper around it. This is the "standard software" we benchmarked against — always in CPU-only mode, so the comparison is fair.

---

## Part 3 — The journey, experiment by experiment

Everything below was run on the actual machine, measured, and logged — including the failures. Each experiment started with a written hypothesis *before* the code ran.

### Chapter 1: Measure the machine (don't trust the spec sheet)

We first built a "ladder" of matrix-multiplication implementations, from naive to expert, to find the machine's true limits:

| implementation | speed |
|---|---|
| Naive three-nested-loops code | 3 GFLOPS |
| Loop-reordered (compiler auto-vectorizes) | 33 GFLOPS |
| Hand-written NEON kernel, 1 core | 108 GFLOPS |
| Hand-written NEON, 12 cores | 1,100 GFLOPS |
| Apple's own library (secretly using the matrix unit) | **3,300 GFLOPS** |

Two lessons. First, the gap between naive and expert code is about **1000×** — that entire gap is what "low-level optimization" means. Second, Apple's library was 3× faster than *all twelve CPU cores combined*, which proved a hidden matrix unit existed and set our next goal: program it ourselves.

We also measured memory: the CPU cores can pull at most **293 GB/s** from RAM (the chip's advertised 546 GB/s includes the GPU's share). Remember this number.

### Chapter 2: Driving the hidden matrix unit (SME)

The M4 is the first Apple chip where the matrix unit speaks a public language (SME). We wrote raw assembly instructions — the lowest level a programmer can go — and discovered, by direct measurement:

- One `fmopa` instruction = 256 multiply-adds. The unit executes **one per clock cycle** but each takes 4 cycles to finish — so you must keep 4 independent computations "in flight" or waste 75% of the hardware. (We proved this: using 1 accumulator tile gave 501 GFLOPS; using 4 gave 2,003.)
- The chip has exactly **two** SME units (one per cluster of 6 P-cores): two threads gave 1.92× the speed; more threads added nothing.
- Total: **4.2 TFLOPS** — more than Apple's own library achieves (3.3), meaning even Apple leaves ~25% unused.

We also verified it end to end: a tiny matrix multiplication traced instruction by instruction, printing the actual contents of the hardware tile after each step, matching pencil-and-paper math exactly.

**A surprise (hypothesis refuted):** we predicted half-precision (bf16) inputs would double throughput and 8-bit integers would quadruple it. Measurement said: bf16 = **no gain at all** (the unit runs the fancier instruction at half rate, cancelling out), int8 = only **2×**. The machine's real ceilings: ~4.2 TFLOPS for any decimal format, ~8 trillion ops/s for 8-bit integers.

### Chapter 3: The physics of generating text

Here's the insight that shaped everything after. When a model **generates one token**, it must read *every active parameter once*, use each exactly once, and throw it away. There's no reuse — so speed isn't limited by math (FLOPS) but by how fast bytes flow from memory:

> **tokens/second ≈ memory bandwidth ÷ bytes of parameters read per token**

For gpt-oss:20b, the 4 active experts (of 32) mean about 1.2 GB must be read per token. Against our measured 293 GB/s ceiling, the physics allows up to ~130 tokens/second. Everything else is engineering toward that ceiling.

(Processing the *prompt* is different — many tokens can share one pass over the weights, so it's math-limited, not memory-limited. Different bottleneck, different tools.)

### Chapter 4: Building the memory-speed kernel (a five-step ladder)

We built the core "read 4-bit weights and multiply by a vector" kernel and improved it in measured steps:

| version | the one change | result |
|---|---|---|
| v1 | baseline (`sdot` + unpacking) | 117 GB/s |
| v2 | **algebra beats instructions**: dot(w−8, x) = dot(w, x) − 8·Σx, so the "subtract 8 from every weight" step vanishes from the inner loop | 151 GB/s |
| v3 | **two accumulators per row**, so consecutive `sdot`s never wait for each other (each takes ~4 cycles to finish) | 180 GB/s |
| v4 | **reorder the bytes on disk** so each core reads one long, continuous stream — the CPU's prefetcher (which guesses what you'll read next) rewards this massively | 228 GB/s |
| v5 | pin threads to fast cores (hypothesis) | **refuted — no change**; the OS already did this |

Note v4: the largest single win came from *changing zero instructions* — only the order of bytes in memory. That's the project's deepest recurring lesson: **below a certain level, performance is decided by data layout, not code.**

### Chapter 5: Finding the opening — MoE is where the standard software is weak

We benchmarked the user's own Ollama installation (CPU mode) and worked the numbers backward into effective memory bandwidth:

| model | type | decode speed | % of memory ceiling used |
|---|---|---|---|
| deepseek-r1:32b | dense (all params active) | 12.7 tok/s | **80%** — near perfect |
| gpt-oss:20b | MoE | 14.7 tok/s | **11%** — terrible |

llama.cpp is excellent at dense models and *leaves ~9× on the table* for MoE models — the exact class all frontier models now belong to. The cause: it dispatches each expert's math separately, re-synchronizing threads constantly. We designed the alternative — flatten each token's 4 active experts into one continuous work list — and measured expert weights streaming at the **full 293 GB/s machine ceiling**. Routing cost: zero.

### Chapter 6: Real weights, exact math

Simulations prove speed; correctness needs the real thing. We:

1. Parsed Ollama's 13 GB GGUF file directly and extracted the real expert matrices.
2. Invented the kernel trick the MXFP4 format needed: its 4-bit values (0, ±0.5, ±1 …±6) aren't integers, but **doubled they all are** — so a one-instruction table lookup (`tbl`) converts each 4-bit code to a small integer, `sdot` does exact integer math, and the ÷2 hides inside the block's power-of-two scale. Result: **bit-exact** MXFP4 arithmetic on integer hardware, zero floating-point conversion, verified at 0.00 error against a slow-but-perfect reference.
3. Recovered gpt-oss's exact formulas from llama.cpp's source code (the router picks top-4 *then* softmaxes over just those 4; the SwiGLU activation uses α=1.702, a clamp at ±7, and a "+1" — details that produce garbage if guessed wrong) and reproduced the full expert block to 1-part-in-a-million accuracy.
4. Reproduced the attention block (GQA + sinks + RoPE) to matching precision.

### Chapter 7: The engine — and the bug that taught the best lesson

We then wired everything into a complete program: tokenizer (validated against OpenAI's official `tiktoken` library), all 24 layers, KV cache, sliding windows, YaRN, sampling.

First output: **fluent-looking garbage.** The debugging hunt that followed is the most instructive part of the whole project:

- Every kernel passed its tests. The tokenizer was verified perfect. Layer-by-layer numbers matched our reference.
- Yet llama.cpp, fed the identical bytes, produced a perfect answer. So the bug was ours.
- Root cause: **one character.** Our lookup table stores weight values doubled, so the scale exponent needed to be 2^(e−128), not 2^(e−127). Every expert weight was exactly 2× too large — and because the network is nonlinear, that didn't just scale the answer, it destroyed it.
- Why did our checks miss it? **Our reference implementation contained the same mistake.** The engine agreed with the reference while both disagreed with reality.

> **The lesson, in bold in our lab log: verification references must be independently grounded** — checked against something you didn't build (the format's official source code, OpenAI's tokenizer, llama.cpp's output), never only against your own derivations.

Two other real fixes from this hunt: hidden values inside the network contain extreme "outlier" spikes (values in the tens of thousands), so quantizing activations needs a separate scale per 32 numbers, not one per vector; and the model demands its exact chat template ("harmony" format) or it rambles.

After the fix: the engine's output matched llama.cpp's **token for token** — same analysis, same final sentence — at 26.3 tok/s vs. their 14.7. Correct first, already 1.8× faster.

### Chapter 8: The climb from 26 to 110

With correctness locked (and re-verified after every change), we optimized in measured steps:

| step | idea | tok/s |
|---|---|---|
| baseline | — | 26.3 |
| Q8 weights | convert 16-bit attention/vocabulary matrices to 8-bit at load: half the bytes, and integer `sdot` math | 31.9 |
| thread pool | we were creating and destroying ~1,350 operating-system threads *per token*; persistent workers wake in microseconds instead | 35.5 |
| parallel attention | the 64 attention heads were computed on one core | 40.9 |
| more workers | 8 → 12 | 48.6 |
| spin-waiting workers | **refuted** — workers that busy-wait steal cores from the serial glue work between jobs; parked threads are better | (34.6 ❌, reverted) |
| NEON attention + fused barriers | hand-vectorize the last scalar loops | 51.2 |
| batched prompt processing | **two refuted hypotheses** — the prompt phase turned out to be limited by instruction count, not memory | (no change) |
| **profile first, then act** | measurement showed experts = 78% of time, running at half our proven kernel speed — because the engine still read them in the file's native byte order | — |
| **quad-interleaved expert layout** | apply the Chapter-4 v4 lesson: repack 10 GB of expert weights once at load into the sequential-stream layout | **109.9** |

That last row doubled the speed in one change — the same byte-layout insight from the micro-benchmark, applied at full scale. Prompt processing also tripled (21 → 64 tok/s) as a side effect.

### Chapter 9: Final results

All on one consumer laptop CPU — no GPU, verified by inspecting the binary (it links only the C standard library; no graphics frameworks anywhere):

| metric | llama.cpp/Ollama (CPU) | our engine | ratio |
|---|---|---|---|
| gpt-oss:20b decode | 14.7 tok/s | **109.9 tok/s** | **7.5×** |
| output quality | reference | token-for-token identical | = |
| For scale: Ollama *with* the GPU | 94.4 tok/s | (we beat it on CPU alone) | 1.2× |

And the honest boundaries, because measured limits are part of the result:
- The training track showed a laptop CPU can realistically *train* models up to ~1 billion parameters (a trillion-parameter model would take ~47,000 years — physics, not pessimism).
- The biggest model that fits in 128 GB at 4 bits is ~200B parameters; gpt-oss-120B would run at roughly 45–55 tok/s on this engine today.
- Scoreboard of scientific honesty: 4 hypotheses refuted (thread pinning, spin-waiting, two prompt-batching theories), each logged; every success number is reproducible from code in this repository.

### Chapter 10: The five transferable lessons

1. **Measure the machine, not the datasheet.** (546 GB/s advertised; 293 real for CPU cores.)
2. **Layout beats code.** The two largest wins were byte-reordering, with zero new instructions.
3. **Keep independent work in flight.** Whether SME tiles or `sdot` accumulators — hardware finishes slowly but starts fast; never make it wait on itself.
4. **Bottlenecks are layered and must be profiled, not guessed.** Four refutations came from acting on plausible theory instead of measurement.
5. **Ground your verification outside yourself.** A reference you derived can share your bug. tiktoken, ggml's source, and llama.cpp-as-oracle caught what self-consistency never would.

---

## Glossary (quick reference, A→Z)

| term | meaning |
|---|---|
| AMX | Apple's undocumented matrix coprocessor (pre-M4 name for the silicon SME now exposes) |
| bandwidth | data-flow rate from memory to CPU, in GB/s |
| BF16 | 16-bit "brain float" decimal format (FP32 with half the bits) |
| cache (L1/L2) | small fast memory inside the CPU; L1 smallest/fastest |
| core (P/E) | one independent processor within a CPU; Performance vs Efficiency variants |
| CPU / GPU | general-purpose processor / parallel math processor born from graphics |
| FLOP(S) | floating-point operation(s per second); GFLOPS=10⁹/s, TFLOPS=10¹²/s |
| FFN | feed-forward network — half of each transformer layer (the MoE part in gpt-oss) |
| GEMM / GEMV | matrix×matrix / matrix×vector multiply |
| GGUF | file format for downloadable models |
| GQA | grouped-query attention (64 query heads share 8 key/value sets) |
| inference / training | using a model / teaching a model |
| int8 / Q8 | 8-bit integer storage; Q8 = int8 blocks with shared scale factors |
| kernel | a hand-optimized inner-loop compute function |
| KV cache | stored keys/values of past tokens so attention doesn't recompute them |
| llama.cpp / Ollama | the standard open-source model-running software / its friendly wrapper |
| MoE | mixture of experts — many small networks, few active per token |
| MXFP4 | 4-bit float format (16 possible values + per-32 block scale) used by gpt-oss |
| NEON | ARM's SIMD instruction set (128-bit vectors) |
| parameter | one learned number in a model |
| prefill / decode | processing the prompt / generating tokens one by one |
| quantization | storing parameters in fewer bits |
| RoPE / YaRN | rotation-based position encoding / its long-context extension |
| `sdot` / `tbl` / `fmopa` | key instructions: 16-way integer dot product / 16-byte table lookup / 256-multiply-add matrix op |
| SIMD | single instruction, multiple data — one instruction, many numbers |
| SME | Scalable Matrix Extension — the public interface to Apple's matrix unit (M4+) |
| softmax | formula converting scores to probabilities that sum to 1 |
| SWA | sliding-window attention (some layers only look back 128 tokens) |
| token / tok/s | text chunk models read and write / generation speed |
| transformer | the neural architecture of modern language models |

---
---

## For the CTO / CPO — what this means for an infrastructure business

*(Written for DigitalOcean leadership. Five minutes, minimal jargon, business first.)*

### What happened, in one paragraph

On a single consumer CPU — no GPU — we ran OpenAI's open 20-billion-parameter model at **110 tokens/second with output identical to the standard stack**, where that standard stack (llama.cpp, the engine underneath most open-source AI serving today) achieves **14.7**. The 7.5× is pure software: the standard stack wastes ~90% of the CPU's memory bandwidth on exactly the model architecture (mixture-of-experts) that every frontier lab has now converged on. We measured the gap, built kernels that close it, and verified correctness token-for-token. Every number is reproducible from the repository.

### Why this matters to an infrastructure company

**1. It reprices CPU inference.** The industry assumption is "AI inference = GPU." That assumption was built on dense models and unoptimized CPU software. For the MoE model class — which now includes essentially all competitive open models — a well-engineered CPU runtime delivers interactive speeds (100+ tok/s on a 20B-class model, ~50 on a 120B-class) on hardware you already own and already know how to operate, bill, and secure. GPUs remain unbeatable for training and high-throughput batch serving; but a large share of real inference demand is latency-tolerant, small-batch, and cost-sensitive — precisely where idle CPU fleets become sellable AI capacity.

**2. It's a differentiation story, not a parity story.** Everyone resells the same Nvidia hardware at the same constrained supply. Software that extracts 7× more from commodity CPUs is proprietary leverage: an "AI inference tier" on standard droplets/instances, priced under GPU offerings, with capacity that scales with your existing fleet rather than with Nvidia's allocation schedule.

**3. The unit economics are the pitch.** Tokens-per-dollar, not tokens-per-second, is what customers buy. A CPU instance costs a small fraction of a GPU instance; at 7.5× the standard CPU throughput, the cost-per-million-tokens for open MoE models on optimized CPU serving becomes competitive with — and for smaller models cheaper than — GPU serving, with zero supply risk. (Exact ratios depend on your fleet mix; the engineering result is the throughput multiplier.)

**4. It fits three growing demand curves.** (a) *Data-sovereign / on-prem AI* — customers who cannot ship data to a GPU cloud can run frontier-class open models on CPU hardware they control. (b) *Edge and regional inference* — CPU capacity exists in every region; H100s don't. (c) *The open-model wave* — gpt-oss, DeepSeek, Qwen, GLM: the models people increasingly self-host are exactly the MoE class where this advantage is largest.

### The honest caveats (we log our failures — here are the real ones)

- **Results are from Apple M4 silicon (ARM).** The techniques — memory-layout optimization, integer-SIMD kernels, MoE-aware scheduling — translate to ARM server chips (Ampere, Graviton-class) directly and to x86 (AMD/Intel with AVX-512) with a porting effort, but those numbers must be measured, not assumed. This is the first diligence step.
- **One model class proven end-to-end.** gpt-oss:20b is fully verified; 120B-class is projected (~50 tok/s) from measured bandwidth, not yet run end-to-end.
- **This is a research engine, not a product.** No batching across users, no server hardening, no multi-model management yet. The kernel advantage is real; the serving layer around it is standard engineering.
- **The moat is a head start, not a patent.** llama.cpp could adopt these techniques; the durable advantage is the demonstrated capability to find and close such gaps repeatedly — this project went from empty directory to 7.5× in days, with a documented method.

### What we'd propose exploring

1. **Two-week port-and-measure** on your actual fleet hardware (one ARM target, one x86 target): reproduce the benchmark ladder, get real tokens-per-dollar numbers.
2. **A pilot "CPU inference endpoint"** serving one open MoE model from existing capacity, priced against your GPU offering.
3. **Publish the benchmark methodology** — the measured-and-honest framing (including refuted hypotheses) is credible developer marketing in itself.

The one-sentence takeaway: **the software gap on CPU inference for modern AI models is large, measurable, and closable — and whoever closes it converts ordinary compute fleets into AI capacity.**

---

*All experiments, failures included, are logged in `docs/RESEARCH_LOG.md` in this repository. Every kernel, benchmark, and the full engine are in `src/`. A live side-by-side demo is one command away: `python3 scripts/demo_server.py`.*
