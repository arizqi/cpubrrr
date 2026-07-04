#!/usr/bin/env python3
"""Multi-token forward reference with KV cache, plain RoPE. Prints per-position
final-hidden norm + argmax. Compare against engine PLAIN_ROPE=1 DUMP_POS=1."""
import json, sys, numpy as np

BLOB, IDX = sys.argv[1], sys.argv[2]
TOKS = [200006, 1428, 200008]
D, NH, NKV, HD, NL, NE, TOPK = 2880, 64, 8, 64, 24, 32, 4
BPR = D // 32 * 17
KVT = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], np.float64)
idx = json.load(open(IDX)); tn = {t["name"]: t for t in idx["tensors"]}; ds = idx["data_start"]
f = open(BLOB, "rb")
CACHE = {}
def get(name):
    if name in CACHE: return CACHE[name]
    t = tn[name]; f.seek(ds + t["offset"]); dims = t["dims"]; n = int(np.prod(dims))
    if t["type"] == 0:
        a = np.frombuffer(f.read(n*4), np.float32).astype(np.float64)
    else:
        a = (np.frombuffer(f.read(n*2), np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    a = a.reshape(list(reversed(dims))); CACHE[name] = a; return a
def expert(name, e):
    key = (name, e)
    if key in CACHE: return CACHE[key]
    t = tn[name]; f.seek(ds + t["offset"] + e * D * BPR)
    raw = np.frombuffer(f.read(D * BPR), np.uint8).reshape(D, BPR)
    out = np.empty((D, D))
    for b in range(D // 32):
        blk = raw[:, b*17:(b+1)*17]
        s = np.exp2(blk[:, 0].astype(np.float64) - 127)
        n = blk[:, 1:17]
        out[:, b*32:b*32+16] = KVT[n & 0xF] * s[:, None]
        out[:, b*32+16:b*32+32] = KVT[n >> 4] * s[:, None]
    CACHE[key] = out; return out
def rms(x, w): return x / np.sqrt((x*x).mean() + 1e-5) * w
def rope(v, pos):
    half = HD // 2
    inv = 150000.0 ** (-(np.arange(half)*2)/HD)
    ang = pos * inv; c, s = np.cos(ang), np.sin(ang)
    a, b = v[..., :half].copy(), v[..., half:].copy()
    v[..., :half] = a*c - b*s; v[..., half:] = b*c + a*s
    return v

kc = [np.zeros((0, NKV, HD)) for _ in range(NL)]
vc = [np.zeros((0, NKV, HD)) for _ in range(NL)]
ALPHA, LIM = 1.702, 7.0
for pos, tok in enumerate(TOKS):
    x = get("token_embd.weight")[tok].copy()
    for il in range(NL):
        p = f"blk.{il}."
        xn = rms(x, get(p+"attn_norm.weight"))
        q = rope((get(p+"attn_q.weight") @ xn + get(p+"attn_q.bias")).reshape(NH, HD), pos)
        k = rope((get(p+"attn_k.weight") @ xn + get(p+"attn_k.bias")).reshape(NKV, HD), pos)
        v = (get(p+"attn_v.weight") @ xn + get(p+"attn_v.bias")).reshape(NKV, HD)
        kc[il] = np.concatenate([kc[il], k[None]]); vc[il] = np.concatenate([vc[il], v[None]])
        sinks = get(p+"attn_sinks")
        ao = np.zeros((NH, HD))
        for h in range(NH):
            kv = h // (NH // NKV)
            sc = (kc[il][:, kv, :] @ q[h]) / np.sqrt(HD)
            m = max(sc.max(), sinks[h])
            ex = np.exp(sc - m); denom = ex.sum() + np.exp(sinks[h] - m)
            ao[h] = (ex / denom) @ vc[il][:, kv, :]
        x = x + get(p+"attn_out.weight") @ ao.reshape(-1) + get(p+"attn_out.bias")
        xn2 = rms(x, get(p+"ffn_norm.weight"))
        lg = get(p+"ffn_gate_inp.weight") @ xn2 + get(p+"ffn_gate_inp.bias")
        top = np.argsort(-lg)[:TOPK]
        w = lg[top]; w = np.exp(w - w.max()); w /= w.sum()
        ffn = np.zeros(D)
        for e, we in zip(top, w):
            g = expert(p+"ffn_gate_exps.weight", e) @ xn2 + get(p+"ffn_gate_exps.bias")[e]
            u = expert(p+"ffn_up_exps.weight", e) @ xn2 + get(p+"ffn_up_exps.bias")[e]
            xg = np.minimum(g, LIM); yu = np.clip(u, -LIM, LIM)
            hh = (xg / (1 + np.exp(-ALPHA * xg))) * (yu + 1)
            ffn += we * (expert(p+"ffn_down_exps.weight", e) @ hh + get(p+"ffn_down_exps.bias")[e])
        x = x + ffn
        print(f"p{pos} layer {il:2d} |x| {np.linalg.norm(x):10.4f}")
    xo = rms(x, get("output_norm.weight"))
    lgt = get("output.weight") @ xo
    am = int(np.argmax(lgt))
    print(f"pos {pos} tok {tok}: |x| {np.linalg.norm(x):.2f} argmax {am} logit {lgt[am]:.3f}")
