# Teaching Walkthrough, Part 2 — Running *Many* Models, and the Hardest Lesson of All: Measuring Honestly

*A continuation of [TEACHING.md](TEACHING.md). Same audience: a high-school or early-undergraduate reader. Part 1 explained how we made one model (gpt-oss:20b) run fast on a laptop CPU. Part 2 is about what happened when we tried to run **three** models — and the humbling, essential lesson we learned about not fooling ourselves with benchmarks.*

*The test for this document is unchanged: after reading it, you should be able to explain — accurately — what was built, why it worked, where it **didn't**, and what every term means.*

---

## Part 1 — The goal this time

Part 1 ended with one model running well. The natural next question: **does the engine generalize?** A one-model trick isn't an engine. So we picked three new targets and tried to run them all on the same CPU-only runtime:

- **gpt-oss-120b** — a 117-billion-parameter model (6× bigger than the 20b).
- **Qwen3-Coder-30B** — a *completely different architecture* from a different company, specialized for writing code.
- **Qwen3-30B** — the general-purpose sibling of the coder.

Two things had to be true for this to count: the models had to **produce correct output** (not fluent nonsense), and they had to run at a **usable speed**.

---

## Part 2 — Vocabulary you'll need (building on Part 1)

Part 1 defined CPU, GPU, SIMD, NEON, matmul, tokens, quantization, MoE, and the transformer. Here are the new terms for Part 2.

**MoE, revisited (Mixture of Experts).** A model made of many small sub-networks ("experts") plus a tiny "router" that picks a few experts per token. Only the picked experts do work, so a huge model stays cheap to run. All three new models are MoE — because MoE is the *only* kind of large model that runs fast on a CPU (you read few parameters per token).

**Active parameters.** The number of parameters actually used for one token. gpt-oss-120b has 117 **billion** total but only ~5 billion **active** per token. This is the number that decides speed, because generating a token means reading every *active* parameter from memory once.

**Quantization format.** *How* the model's numbers are packed into few bits. This turned out to matter enormously:
- **MXFP4** — a 4-bit *floating-point* format (gpt-oss uses it). It's **symmetric**: each little block of numbers has one scale factor, and that's it. Simple to unpack → fast kernels.
- **Q4_K** — a 4-bit format from the `llama.cpp` project (Qwen uses it). It's **asymmetric**: each block has *both* a scale *and* a minimum offset. More accurate, but there's more math to unpack each number → slower kernels.
- **Q6_K** — a 6-bit cousin of Q4_K, used for the parts of the model that need more precision.

The symmetric-vs-asymmetric difference is the whole plot of Part 2's performance story. Remember it.

**GGUF.** The standard file format models are shipped in. We read these files directly, byte by byte.

**llama.cpp.** The dominant open-source engine for running models on normal computers. It's our benchmark to beat (or lose to). Crucially: **on a Mac, llama.cpp defaults to using the GPU.** Making it use *only* the CPU requires an explicit setting — and *verifying* that setting actually took effect turned out to be the hardest lesson of the whole project (Part 6).

**Barrier (in parallel computing).** When you split work across 12 CPU cores, they periodically have to *wait for each other* to finish a step before starting the next — like rowers who must all finish a stroke together. That waiting point is a "barrier." Barriers are necessary but expensive, and how you implement them can make or break performance (Part 5).

**Thermal throttling.** When a chip gets hot, it deliberately slows itself down to avoid damage. On a laptop under sustained heavy load, the *same code* can run at half speed simply because the chip is hot. This wrecked our ability to measure anything reliably (Part 6).

---

## Part 3 — Making one engine run three different models

### Config-driven, not hardcoded

The Part-1 engine had gpt-oss:20b's exact dimensions baked into the code — 24 layers, 32 experts, and so on. To run a *different* model, those numbers can't be constants. So we made the engine read every dimension from the model file at startup: layer count, expert count, head sizes, everything. Now the same code runs any model of that architecture family by reading its config.

**Payoff:** gpt-oss-120b has the *same architecture* as the 20b, just bigger (36 layers, 128 experts). Once the engine was config-driven, the 120b ran with **zero code changes** — it just read bigger numbers. A 117-billion-parameter frontier model, generating correct text on a laptop CPU. (One bug had to be fixed: loading a 61 GB file the naive way used too much memory and the operating system killed the program. The fix was **memory-mapping** — telling the OS "this data lives in the file; page it in and out as needed" instead of copying all 61 GB into RAM at once.)

### A genuinely new architecture

Qwen3-Coder is *not* the same architecture. It has features gpt-oss doesn't:
- **QK-norm** — an extra normalization step inside attention.
- A **different router math** (softmax-then-pick vs. pick-then-softmax).
- A **plainer activation function** (no special clamping).
- And it's stored in **Q4_K/Q6_K**, not MXFP4.

Here's the discipline that mattered: before writing a single line, we **recovered the exact recipe from llama.cpp's source code** and **verified our 4-bit unpacking was bit-for-bit identical** to the official reference library. This is Part 1's hardest lesson applied again — *check against something you didn't build.* The payoff: when we ran the new architecture for the first time, it produced **correct code on the very first try**, with no frustrating "why is it outputting garbage" debugging. Getting the spec exactly right up front is faster than guessing and debugging.

**Result of Part 3:** all three models run and produce correct output on CPU. The engine generalized across a 6× size jump *and* a new architecture. That part is a solid, verified success.

---

## Part 4 — The performance surprise: one format is friendly, one is not

Now the speed story, and it has a genuine twist.

On **gpt-oss (MXFP4)**, our engine is **about 4× faster than llama.cpp running on the CPU.** Why? Because MXFP4 is *symmetric* (just a scale, no offset), our unpacking kernel is simple and fast — and, as it happens, llama.cpp's CPU path for this newer MoE format is not well-optimized (it manages only ~15 tokens/second). We comfortably beat it.

On **Qwen3-Coder (Q4_K)**, the story flips: **llama.cpp's CPU is faster than ours.** Q4_K is *asymmetric* (scale *and* offset per block), so every number costs more instructions to unpack — and Q4_K is llama.cpp's *bread and butter*, a format they've hand-tuned for years. We got close (~80% of their speed after a lot of work) but did not beat their mature kernel.

**The honest takeaway:** we win big where the competition's CPU code is weak (MXFP4 MoE), and we lose where it's strong (mature Q4_K). That's a real, defensible, *nuanced* result — not a clean "we beat everyone." Pretending otherwise would be dishonest, and it would fall apart the moment someone else ran the benchmark.

---

## Part 5 — The rewrite: how to keep 12 cores busy without them tripping over each other

Our Qwen engine started slow (~30 tokens/sec) even though the *kernels* were fine. We profiled and found the culprit: **barriers**. The engine used a "fork-join" pattern — the main thread hands out work to 12 helper threads, waits for all of them at a barrier, then hands out the next piece. With hundreds of these hand-offs per token, and each hand-off costing ~35 microseconds to wake sleeping threads, the *waiting* dominated. The cores were idle more than half the time.

So we **rewrote the execution model.** Instead of the main thread bossing the workers around step-by-step, the 12 workers run the *entire* forward pass themselves, syncing at cheap "spin-barriers" — where a waiting worker just loops in place for a moment instead of going to sleep. Workers only truly sleep *once per token* instead of hundreds of times.

**It worked... and then it didn't.** In isolation it hit ~65 tokens/sec. But it would randomly **collapse to 5 tokens/sec.** The cause is a beautiful, humbling lesson: with 12 spinning workers pinned to 12 CPU cores, there's **no core left for the operating system.** The OS *must* interrupt one of our workers to run its own housekeeping — and the moment it does, that one paused worker holds up all 11 others at the barrier. The whole engine stalls.

**The fix:** leave the OS some breathing room. Use a **yielding barrier** — a waiting worker spins briefly, then politely says "OS, you can run something else on my core for a moment" (`yield`). That single change made the engine both fast *and* robust: it now degrades *gracefully* under load (dropping to ~27 tok/s when two cores are stolen) instead of collapsing to 5. Lesson: **spinning is fragile on a general-purpose computer; always leave room for the operating system.**

---

## Part 6 — The most important lesson: how we fooled ourselves with benchmarks (three times)

This is the part every young engineer should read twice.

Over this project, we made a confident public claim — *"our engine beats llama.cpp"* — and had to **retract it, then retract the retraction, then correct again.** Not because the engine changed, but because we kept measuring wrong. Here is exactly how, because the mistakes are common and instructive:

**Mistake 1 — a contaminated baseline.** We measured llama.cpp at "47 tokens/sec" and declared victory. But that measurement was taken *while a 65 GB file was downloading in the background*, starving llama.cpp of the memory bandwidth it needed. It's like timing a competitor's sprint while someone stands on their foot. The real number was much higher. **Lesson: measure competing systems under identical, clean conditions.**

**Mistake 2 — options are requests, not facts.** We told Ollama "use CPU only" (`num_gpu: 0`) and trusted it. But on a Mac, the tool *defaults to the GPU*, and in some of our runs the setting silently didn't take — so our "llama.cpp CPU" numbers were actually the **GPU** running. We only caught this by reading the tool's own log files, which record where each layer actually ran ("offloaded 49/49 layers to GPU"). **Lesson: verify what actually happened from ground-truth logs; never trust that your request was honored.** We then hardened our benchmark script to read the log after every run and refuse to report a "CPU" number unless the log confirms zero layers went to the GPU.

**Mistake 3 — a thermally exhausted machine.** After days of continuous benchmarking, the laptop was so hot that the *same binary* produced 66, then 42, then 5 tokens/sec on consecutive runs. We were reading tea leaves. **Lesson: a hot chip is a lying chip. Benchmark on a rested, cool machine, run multiple times, and report the spread — not a lucky single number.**

The meta-lesson, now written in three places in our lab notes: **an unverified benchmark is worse than no benchmark**, because it gives false confidence. Real engineering means being *more* skeptical of results that flatter you than of results that don't. We caught our own errors by staying skeptical — but only after publicly overclaiming first. Do better than we did: verify *before* you claim.

---

## Part 7 — Honest scorecard

What is **solidly true and verified**:
- One CPU-only engine runs three real models (gpt-oss-120b, Qwen3-Coder-30B, Qwen3-30B) and produces correct output — verified against reference libraries and matching outputs.
- It uses **no GPU at all** — the program links only the basic system library; using a GPU is physically impossible for it.
- On **MXFP4 MoE models** (gpt-oss), it is **~4× faster than llama.cpp's CPU path** (~55 vs ~14 tok/s — this format is llama.cpp's CPU weak spot).
- The execution-model rewrite and the yielding-barrier fix are real, reproducible engineering wins.

What is **honest but less flattering**:
- On **mature Q4_K models** (Qwen3-Coder), llama.cpp's CPU is **faster than ours** (~86 vs ~71 tok/s — we reached ~83% of its speed, not more).
- Our engine is **thermally fragile** (it runs the cores hot and throttles), where llama.cpp is thermally steady.
- **Exact tokens/second figures need a final clean-machine, log-verified measurement** before anyone should quote them. The *directions* above are solid; the precise numbers are pending.

That last bullet is not a cop-out — it's the whole point of Part 6. We would rather hand you an accurate "here's what we know and here's what we still need to verify" than a shiny number that evaporates under scrutiny.

---

## Glossary (Part 2 additions)

| term | meaning |
|---|---|
| active parameters | parameters actually used per token (decides speed) |
| barrier | a point where parallel threads must wait for each other |
| config-driven | reading model dimensions from a file instead of hardcoding them |
| fork-join | a parallel pattern: hand out work, wait for all, repeat |
| GGUF | the standard model file format |
| llama.cpp | the dominant open-source model-running engine (our benchmark) |
| memory-mapping (mmap) | letting the OS page a file in/out instead of copying it to RAM |
| MXFP4 | 4-bit *symmetric* float format (gpt-oss) — simple, fast to unpack |
| Q4_K / Q6_K | 4/6-bit *asymmetric* formats (Qwen) — accurate, costlier to unpack |
| QK-norm | an extra normalization inside Qwen's attention |
| spin-barrier | a barrier where waiting threads loop in place instead of sleeping |
| thermal throttling | a chip slowing itself when hot |
| yielding barrier | a spin-barrier that periodically lets the OS use the core |

---

*Full lab notes, including every failed measurement and correction, are in [RESEARCH_LOG_V2.md](RESEARCH_LOG_V2.md). The engine and kernels are in `src/bin/`. The benchmark methodology (now log-verified) is in `scripts/bench_ollama.sh`.*
