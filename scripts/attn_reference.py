#!/usr/bin/env python3
"""Exact gpt-oss layer-0 attention reference (numpy f64), single decode step over a
synthetic KV cache. Verifies GQA mapping, learned sinks, causal softmax, scale, and
NeoX RoPE at freq_base=150000. YaRN mscale/ramp NOT applied (positions < orig ctx;
flagged as the remaining exactness gap vs llama.cpp)."""
import json, struct, sys, numpy as np

BLOB, IDX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
D, NH, NKV, HD = 2880, 64, 8, 64
FREQ_BASE = 150000.0
idx = json.load(open(IDX)); tn = {t["name"]: t for t in idx["tensors"]}; ds = idx["data_start"]
f = open(BLOB, "rb")

def get(name):
    t = tn[name]; f.seek(ds + t["offset"]); dims = t["dims"]; n = int(np.prod(dims))
    if t["type"] == 0:      # f32
        a = np.frombuffer(f.read(n*4), np.float32)
    elif t["type"] == 30:   # bf16
        raw = np.frombuffer(f.read(n*2), np.uint16).astype(np.uint32) << 16
        a = raw.view(np.float32)
    else:
        raise ValueError(t["type"])
    # ggml dims are reversed vs numpy: dims=[cols, rows] -> (rows, cols)
    return a.astype(np.float64).reshape(list(reversed(dims)))

anorm = get("blk.0.attn_norm.weight")          # (2880,)
wq = get("blk.0.attn_q.weight");  bq = get("blk.0.attn_q.bias")   # (4096,2880),(4096,)
wk = get("blk.0.attn_k.weight");  bk = get("blk.0.attn_k.bias")   # (512,2880),(512,)
wv = get("blk.0.attn_v.weight");  bv = get("blk.0.attn_v.bias")
wo = get("blk.0.attn_out.weight"); bo = get("blk.0.attn_out.bias") # (2880,4096),(2880,)
sinks = get("blk.0.attn_sinks")                 # (64,)

rng = np.random.default_rng(1)
T = 8  # existing cache length; new token at position T
h = rng.standard_normal(D) * 0.5
kcache = rng.standard_normal((T, NKV, HD)) * 0.3
vcache = rng.standard_normal((T, NKV, HD)) * 0.3

def rmsnorm(x, w):
    return x / np.sqrt((x*x).mean() + 1e-5) * w

def rope(vec, pos):  # NeoX: split halves, rotate
    out = vec.copy()
    half = HD // 2
    inv = FREQ_BASE ** (-(np.arange(half)*2)/HD)
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    a, b = vec[..., :half], vec[..., half:]
    out[..., :half] = a*c - b*s
    out[..., half:] = b*c + a*s
    return out

x = rmsnorm(h, anorm)
q = (wq @ x + bq).reshape(NH, HD)
k = (wk @ x + bk).reshape(NKV, HD)
v = (wv @ x + bv).reshape(NKV, HD)
q = rope(q, T)
k = rope(k, T)
# append new k/v
kfull = np.concatenate([kcache, k[None]], 0)   # (T+1, NKV, HD)
vfull = np.concatenate([vcache, v[None]], 0)
scale = 1.0/np.sqrt(HD)
attnout = np.zeros((NH, HD))
for hd in range(NH):
    kv = hd // (NH // NKV)          # GQA: 8 q-heads share each kv-head
    scores = (kfull[:, kv, :] @ q[hd]) * scale   # (T+1,)
    # learned sink: extra logit sinks[hd], no value
    m = max(scores.max(), sinks[hd])
    ex = np.exp(scores - m); es = np.exp(sinks[hd] - m)
    denom = ex.sum() + es
    w = ex / denom
    attnout[hd] = w @ vfull[:, kv, :]
o = wo @ attnout.reshape(-1) + bo   # (2880,)

for name, arr in [("h",h),("kcache",kcache),("vcache",vcache),("attn_yref",o),
                  ("anorm",anorm),("wq",wq),("bq",bq),("wk",wk),("bk",bk),
                  ("wv",wv),("bv",bv),("wo",wo),("bo",bo),("sinks",sinks)]:
    arr.astype(np.float32).tofile(f"{OUT}/aw_{name}.f32")
print("T", T, "attn out[:6]", [round(float(v),4) for v in o[:6]])
print("attn out norm", round(float(np.linalg.norm(o)),4))
