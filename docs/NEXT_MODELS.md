# Next models to target — decision doc

**Core rule (from cpubrrr):** decode tok/s ≈ effective_bandwidth (~290 GB/s CPU) ÷
(active_params × 0.55 bytes at 4-bit). So the ONLY thing that matters for speed is
**active parameters per token**, not total size. Pick low-active MoE. Exclude everything
with ≥17B active — bandwidth physics caps it below usable speed regardless of quality.

## Ranked shortlist

| # | model | total / **active** | proj. tok/s | 4-bit size | fits 128GB | focus | effort |
|---|---|---|---|---|---|---|---|
| **1** | **Qwen3-Coder-30B-A3B-Instruct** | 30.5B / **3.3B** | **~96** | ~18 GB | ✅ easily | coding (agentic, tools, 256K–1M ctx) | **low** — standard SwiGLU/RoPE MoE, no MXFP4, no sinks |
| **2** | **Qwen3-30B-A3B** (general) | 30.5B / **3.3B** | ~96 | ~18 GB | ✅ | reasoning/general | low — same arch as #1, drop-in |
| **3** | **gpt-oss-120b** | 117B / **5.1B** | ~100 | ~63 GB | ✅ | reasoning | **near-zero** — reuses our MXFP4 + sinks + SWA path |
| 4 | GLM-4.5-Air *(if specs verify)* | ~106B / ~12B | ~44 | ~67 GB | ✅ | general | medium |
| 5 | gpt-oss-20b | 21B / 3.6B | ~110 | ~13 GB | ✅ | baseline (done) | — |

## Excluded (high active params → bandwidth-bound death)

| model | active | proj. tok/s | why out |
|---|---|---|---|
| DeepSeek-V3 / R1 | 37B | ~14 | too slow **and** Q4 = 377 GB (won't fit) |
| GLM-4.6 / REAP-252B | 32B | ~16 | too slow; Q4 191–216 GB (won't fit) |
| Qwen3-235B-A22B | 22B | ~24 | too slow; Q4 ~130–140 GB (borderline) |
| Llama-4 Scout / Maverick | 17B | ~31 | slow; Maverick ~400B won't fit |
| Mixtral 8x22B | ~39B | ~15 | old + slow |
| any dense model | =total | ~18 (30B) | activates all params — inherently wrong fit |

## Recommended sequence

**Move 1 — gpt-oss-120b (fastest to land, proves scale).** Reuses everything already
built: MXFP4 kernel, attention sinks, sliding-window, YaRN. Only changes = layer count,
expert count, dims (all config, no new kernels). Ships a 6× bigger reasoning model at
~100 tok/s for near-zero marginal work. Validates the engine at scale.

**Move 2 — Qwen3-Coder-30B-A3B (proves generalization + adds coding).** Standard MoE
(no MXFP4/sinks) — actually *simpler* than gpt-oss. Landing it proves the engine isn't
gpt-oss-specific, adds a high-value coding capability, and covers the "standard SwiGLU
MoE" arch that most open models (Qwen3, GLM, etc.) share — so it's reusable scaffolding.
Needs: a Q4_K/Q8 GGUF loader path (we have Q8; add Q4_K block format) + generic-MoE
router/activation (plain softmax-top-k + standard SwiGLU, both simpler than gpt-oss's).

Together: one move = scale on known arch; one move = new arch + new capability. After
both, the engine covers the two dominant open-MoE families (gpt-oss + Qwen3-style),
which most other models (GLM-Air, etc.) resemble.

*Full citations and per-model detail in the research task output; every spec is
primary-source verified (official HF model cards) with adversarial 3-vote checks.*
