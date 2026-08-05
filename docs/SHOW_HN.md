# Show HN draft (2026-08-02, short version)

Checked the last 10 Show HN posts: most have no body text at all, and the ones that
do run 90-375 words. Keep it under ~220. No em dashes, no bullet symmetry, no
"here's the thing" cadence. Sound like the Reddit post.

---

**Title:**

Show HN: Cpubrrr – a CPU-only LLM inference engine in Rust

---

**Body:**

i keep thinking that tokens/sec of reasoning won't be evenly distributed. a kid with
15 tok/s and a kid with several hundred have very different tools to think with.
open weights were step one. throughput on hardware people already own feels like
step two. so i built a CPU-only inference engine in rust. the binary links only
libSystem, it physically can't touch the GPU.

on an M4 Max it decodes gpt-oss-20b at 90 tok/s vs 77 for upstream llama.cpp, same
session, both CPU-only, both tuned. GSM8K 98/100, same score as ollama serving the
same weights.

the embarrassing part: earlier that day i measured 1.62x. i had llama.cpp at -t 12
because the machine has 12 P-cores. that's nearly its worst setting (48 tok/s).
its best was -t 8 (78.5). half my "win" was somebody else's misconfiguration, and
every number i'd published before was against the mistuned baseline too. the repo's
research log records each wrong number and its correction, in order. there are seven
lessons in there now and most of them are about measurement, not kernels.

i'm not a kernel engineer. i directed claude through the kernel work and my job was
verification gates and retiring numbers that didn't reproduce.

apple silicon only for now. prefill still loses to llama.cpp. MIT/Apache-2.0.

github.com/arizqi/cpubrrr

---

## Before posting

1. Reply to the unanswered hn@ycombinator.com thread (Jul 23) before resubmitting.
   The flagged URL is likely penalized without mod intervention.
2. README is already corrected (no stale 5x / parity claims).
3. If asked in comments: upstream = homebrew b9860, -dev none -ngl 0, thread-swept.
   cpubrrr decodes at higher positions than llama-bench tg128 (73-token prompt in
   context), handicap runs against us.
