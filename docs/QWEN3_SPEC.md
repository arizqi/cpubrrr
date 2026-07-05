# Qwen3-MoE build spec (verified from llama.cpp `src/models/qwen3moe.cpp`)

Target: Qwen3-Coder-30B-A3B (coding) and Qwen3-30B-A3B (general) — same `qwen3moe` arch.
Recovered exactly to avoid the guess-and-garbage trap (cf. the gpt-oss MXFP4 off-by-one).

## Architecture (per layer, N=48 layers for 30B-A3B)

**Attention** (`build_attn_inp_kv` — plain causal, NO sliding window, NO sinks):
1. `attn_norm` (RMSNorm, eps from metadata ~1e-6).
2. QKV projection — **NO bias** (qwen3 dropped qkv bias; wo_b/wo_s are null).
   Heads: n_head query, n_head_kv (GQA); head_dim = n_embd_head_k (=128).
3. **QK-norm (the key new op):** RMSNorm applied to **Q and K per head** (over head_dim)
   using `attn_q_norm.weight` / `attn_k_norm.weight` (dim = head_dim), BEFORE RoPE.
4. RoPE (NeoX type), `freq_base` from metadata (Qwen3 ~1e6), no YaRN for base ctx.
5. Attention: scale `1/sqrt(head_dim)`, standard softmax (no sink term).
6. Output proj `wo` (no bias).

**MoE FFN** (`build_moe_ffn`, `LLM_FFN_SILU`, gating `SOFTMAX`, norm=true):
1. `ffn_norm` (RMSNorm).
2. Router: `logits = x · ffn_gate_inp`; **softmax over ALL N experts**; take top-k;
   **renormalize** the k selected weights to sum to 1 (norm=true). Optional
   `expert_weights_scale`.  ⚠ differs from gpt-oss (which does top-k THEN softmax).
3. Each expert: `h = silu(gate·x) * (up·x)`; standard SwiGLU — **no clamp, no +1, no
   alpha** (unlike gpt-oss SwiGLU-OAI).
4. `down·h`, weighted-sum over the k experts.
5. **No shared experts** in plain qwen3moe (the hybrid `qwen35moe` has them; not this).

**Head:** final RMSNorm → `output.weight` (may be **tied to `token_embd`** if absent).

## Config to read from GGUF metadata (`qwen3moe.*`)
`block_count` (48), `embedding_length` (n_embd), `attention.head_count`,
`attention.head_count_kv`, `attention.key_length` (head_dim), `expert_count`,
`expert_used_count`, `expert_feed_forward_length` (n_ff_exp), `rope.freq_base`,
`attention.layer_norm_rms_epsilon`, `vocab_size`.

## Quantization — the main new kernel work
Ollama's qwen3-coder:30b default is a **k-quant** (Q4_K_M ≈ 4.7 bpw), NOT MXFP4. Needed:
- **Q4_K** dequant: 256-elem superblock = 8 sub-blocks of 32; 6-bit per-subblock scales
  and mins (packed), 4-bit quants; value = q·scale + min. Verify exact layout vs ggml
  `dequantize_row_q4_K`.
- **Q6_K** for some tensors; **Q8_0** for others (histogram showed mixed types).
- Kernel: dequant-to-int8-blocks then reuse `sdot`, OR direct k-quant·int8. Confirm
  which tensors are which quant from the actual GGUF (next step).

## Build plan
1. Config-driven engine: dims from metadata (shared with gpt-oss-120b path).
2. Q4_K/Q6_K/Q8_0 dequant + matvec kernels (reuse quad-interleave lesson where it helps).
3. Qwen3 forward: QK-norm, SILU SwiGLU, softmax-topk-renorm router, no sinks/SWA/bias,
   handle tied lm_head.
4. Verify per-layer vs an independent numpy reference AND vs Ollama end-to-end on
   matched prompts (independent oracles — non-negotiable).

## Chat template (Qwen3 ChatML, not gpt-oss harmony)
`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
(confirm exact special-token ids from the tokenizer dump.)
