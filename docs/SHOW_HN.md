# Show HN draft (2026-08-02, short version)

Framing (2026-08-02, after the batched-prefill session): the MISSION is the post.
Tokens/sec of reasoning on hardware people already own. Absolute numbers are the
claim (90 decode / ~140 prefill / 98 GSM8K), llama.cpp is a reference point not the
headline, and the measurement-honesty record is the evidence. Under ~230 words, no
em dashes, sound like the Reddit post.

---

**Title:**

Show HN: Cpubrrr – a CPU-only LLM inference engine in Rust

---

**Body:**

i keep thinking that tokens/sec of reasoning won't be evenly distributed. a kid with 15 tok/s and a kid with several hundred have very different tools to think with. open weights were step one. step two is throughput on hardware people already own. that's what this project tries to prove: how much reasoning speed is actually in a consumer chip if you treat decode as a memory problem and refuse to pay for speed with quality.

current state on an M4 Max, CPU only (the binary links nothing but libSystem, it physically can't touch the GPU): gpt-oss-20b decodes at 90 tok/s and prefills at ~140. GSM8K 98/100, the same score those weights get served through ollama, and the output head is provably exact.

for reference, upstream llama.cpp on the same machine, same session, tuned to its own best thread count, decodes at 77. i'm deliberately not leading with that ratio. i've published bad ratios before: my first "5x llama.cpp" was really 5x ollama's slow bundled runner, and this week i caught myself at "1.62x upstream" because i'd left llama.cpp at nearly its worst thread setting. the repo's research log records every wrong number and its correction, in order. the absolute numbers are the claim; the log is the evidence.

i'm not a kernel engineer. i directed claude through the kernel work. my job was verification gates and retiring numbers that didn't reproduce.

apple silicon only for now. MIT/Apache-2.0.

github.com/arizqi/cpubrrr

---

## Before posting

1. Reply to the unanswered hn@ycombinator.com thread (Jul 23) before resubmitting.
   The flagged URL is likely penalized without mod intervention.
2. README is already corrected (no stale 5x / parity claims).
3. If asked in comments: upstream = homebrew b9860, -dev none -ngl 0, thread-swept.
   cpubrrr decodes at higher positions than llama-bench tg128 (73-token prompt in
   context), handicap runs against us.
