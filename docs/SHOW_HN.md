# Show HN draft — cpubrrr (post-flag rewrite, 2026-08-02)

Voice: matches the Reddit post — lowercase, first person, plain. No section headers in
the posted body, no hype, no comparative claim in the title (the flagged one had
"beats llama.cpp" in the title, which is editorializing per HN's title guideline).

Framing change vs the flagged version: the speed ratio is no longer the point. The
measurement discipline is. That is both more honest and more interesting, and it is
the part that can't be dismissed as a rigged benchmark.

---

**Title:**

Show HN: Cpubrrr – a CPU-only LLM inference engine in Rust

---

**Body:**

the thing i keep thinking about: tokens/sec of reasoning is going to matter
enormously, and it won't be evenly distributed. a kid with 15 tok/s and a kid with
several hundred have very different tools to think with. open weights were step one.
throughput feels like step two — and decode throughput is a kernel and scheduling
problem, on hardware people already own. that's why i wanted it on the CPU: it's the
compute everyone already has.

so i built cpubrrr, an inference engine in rust where the binary links only
libSystem. it physically cannot touch the GPU or Accelerate. one config-driven
engine runs 4 mixture-of-experts models.

on an M4 Max it decodes gpt-oss-20b (MXFP4) at 90 tok/s. upstream llama.cpp on the
same machine, same session, same weights, CPU-only, tuned to its own best thread
count, does 77. so about 1.17x, at GSM8K 98/100 — the same score those weights get
through ollama.

that 1.17x is most of what i want to talk about, because earlier today the same two
binaries measured 1.62x, and the difference was not the code.

i had been benchmarking llama.cpp at -t 12, because this machine has 12 performance
cores and that seemed obviously right. it is close to the worst setting:

    -t 6      66 tok/s
    -t 8      78          <- its actual best
    -t 10     70
    -t 12     48          <- what i had been using
    -t 16     11

upstream at -t 12 runs at 61% of its own best. llama.cpp's barrier degrades sharply
once threads get preempted or land on efficiency cores. cpubrrr pins to P-cores via
QoS and work-steals, so it doesn't care about oversubscription — but that is a
robustness advantage, not a throughput advantage, and reporting it as throughput
would have handed me a 1.6x headline built entirely on someone else's
misconfiguration.

so: a baseline is not a binary, it's a *configuration*. sweep the baseline's knobs
before you quote a ratio. this also corrects my own earlier numbers — every previous
comparison i published used -t 12, which means my old "parity with upstream" claim
was measured against a mistuned opponent, and the honest reading is that my earlier
engine was behind, not level.

what's actually in it, for anyone who cares about the guts:

- decode is bandwidth-bound, so the wins are bytes and stragglers, not flops. i wrote
  a probe shaped like the real access pattern instead of trusting a synthetic loop:
  12 workers each walking their own sequential run sustain 271 GB/s — and 336
  barriers per token costs 27% of that. that reframed everything.
- the output head was 25% of all bytes moved per token, more than all 24 attention
  blocks combined. it's now two-stage: a Q4 pass screens the vocabulary, and the
  surviving candidates get recomputed from the Q8 weights, so the emitted token
  provably cannot depend on Q4 precision. there's a flag that checks this against a
  full-Q8 argmax every step; it reports 129/129.
- 12 persistent workers run the whole forward pass and sync at ~1us barriers instead
  of a fork-join pool. i got 336 barriers/token down to 220.
- i predicted the quantization sensitivity exactly backwards. i assumed the output
  projection would be fragile (it writes the residual stream directly, errors compound
  over 24 layers) and the query projection robust (its error passes through a
  softmax). it's the reverse: 4-bit on wq costs 3 GSM8K questions, on wo costs 1. i
  only found that by testing them separately instead of reasoning about it.

i also could not hit the target i set. i wanted 2x. the roofline says no: at 2311
MB/token and 271 GB/s the ceiling is ~117 tok/s, and 2x of a properly tuned upstream
needs more than that. every configuration that does clear the bar costs measurable
accuracy. so the honest answer is that 2x of upstream on this model, on this machine,
is something you buy with quality rather than engineer for free.

the repo has a research log with every wrong number and its correction, in order —
including a contaminated baseline, ollama silently running on the GPU during what i
thought was a CPU test, a 110 tok/s peak that never reproduced, and a "5x llama.cpp"
claim that was really 5x ollama's bundled runner. be more skeptical of benchmarks
that flatter you; mine flattered me four separate ways.

on how it was built: i'm an engineer, not a kernel expert. i directed claude through
the kernel work across many sessions. my job was the goal, the verification gates
(bit-matched layer sums, greedy-output identity, GSM8K parity, the head-exactness
check), and retiring every number that didn't reproduce. the log marks which is which.

limits: apple silicon only (x86 is the top ask), and prompt processing still loses to
llama.cpp's Accelerate GEMM — the win is token generation, which is bandwidth-bound
and never touches BLAS.

to try it: clone, cargo build --release, and the setup script reuses weights you
already have from ollama. MIT/Apache-2.0. i'd like this to be a track rather than a
one-off — poke holes in the methodology, or port the kernels off apple silicon.

github.com/arizqi/cpubrrr

---

## Notes before posting

1. **Do not resubmit the URL cold.** The previous Show HN was flagged and the URL is
   likely penalized. Reply to the existing hn@ycombinator.com thread (sent Jul 23,
   never answered) before submitting — mention the benchmark was found to be wrong
   and has been corrected publicly, and ask for a second-chance placement.
2. **Scrub the README first.** Any surviving "5x" or "beats llama.cpp" language will
   be the first thing a commenter checks, and it now contradicts the log.
3. Numbers quoted are same-session, both sides CPU-only (`-dev none -ngl 0` for
   upstream, which matters because the homebrew build ships a Metal backend), both at
   their own best thread count, on a desktop that was NOT idle. Say that if asked.
4. cpubrrr decodes with its 73-token harmony prompt in context (pos 73→201) while
   llama-bench tg128 runs pos 0→128, so cpubrrr is measured at higher positions. The
   handicap runs against us; don't quietly drop it.
