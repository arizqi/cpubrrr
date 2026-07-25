# Contributing to cpubrrr

Thanks for being here. The mission: free and cheap models should have kernel-level-up
tooling so fast reasoning runs on hardware people already own. Everything below exists
to keep contributions fast to review and honest to measure.

## Ground rules (the short version)

1. **Verify before you wire.** Any new or changed kernel gets checked against an
   independent oracle *before* it goes into a forward pass. Follow the existing
   patterns: `src/bin/qk_verify.rs` (dequant vs the official `gguf` library,
   bit-exact) and `src/bin/q8k_verify.rs` (fused matvec vs a dequant-f64 reference,
   ~1e-7 rel). A kernel that "seems right" in generated text is not verified.
2. **Every perf claim ships with numbers.** Before/after decode tok/s on a cool,
   quiet machine, several runs, and — if a llama.cpp/Ollama baseline is involved —
   placement log-verified via `scripts/bench_ollama.sh` (it refuses to report a "CPU"
   number if any layers ran on GPU). Add the numbers to `docs/RESEARCH_LOG_V2.md`.
   Failed experiments are logged too; refuted hypotheses are contributions.
3. **Output must not regress.** Same greedy output on the standard prompts before and
   after (or an explained, quantified quality delta if you change numerics).
4. **Be skeptical of benchmarks that flatter you.** The research log documents the four
   ways this project fooled itself. Don't add a fifth silently.

## Getting started

```bash
cargo build --release
./scripts/setup_model.sh gpt-oss:20b        # needs Ollama with the model pulled
./target/release/engine data-gpt-oss_20b "$(cat data-gpt-oss_20b/blob_path.txt)" "Why is the sky blue?"
```

Read `docs/RESEARCH_LOG_V2.md` before optimizing anything — it records what has already
been tried, what worked, and what was refuted. The fastest way to waste a weekend is to
re-run a refuted experiment.

## Where help is most valuable

- **x86 port (AVX-512/VNNI)** — [#1](https://github.com/arizqi/cpubrrr/issues/1). Keep
  the integer-accumulation structure; gate with `#[cfg(target_arch = "x86_64")]`.
- **BLAS-build baselines** — [#2](https://github.com/arizqi/cpubrrr/issues/2).
  Self-contained measurement work; a good first issue.
- **Prefill throughput** — [#3](https://github.com/arizqi/cpubrrr/issues/3).
- New MoE architectures on the config-driven engine, kernel ideas with numbers,
  and reproductions of our results on other hardware (including failed ones).

## PR checklist

- [ ] Kernel changes verified against an oracle (say which one in the PR)
- [ ] Before/after numbers in the PR description and appended to `docs/RESEARCH_LOG_V2.md`
- [ ] Output unchanged on the standard prompts (or delta explained)
- [ ] No model weights, no `data-*` dirs, no local drafts committed

## Conduct

Attack measurements, not people. Skepticism about numbers is the house style — aim it
at everyone's numbers, including ours and your own.
