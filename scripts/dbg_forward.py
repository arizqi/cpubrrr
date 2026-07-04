#!/usr/bin/env python3
"""Single-token full forward reference (f64, plain RoPE at pos 0 = identity).
Prints per-layer hidden norms + final argmax for bisecting the engine."""
import json, sys, numpy as np

BLOB, IDX = sys.argv[1], sys.argv[2]
TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 200006
D, NH, NKV, HD, NL, NE, TOPK = 2880, 64, 8, 64, 24, 32, 4
BPR = D // 32 * 17
KV = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], np.float64)

idx = json.load(open(IDX)); tn = {t["name"]: t for t in idx["tensors"]}; ds = idx["data_start"]
f = open(BLOB, "rb")

def get(name):
    t = tn[name]; f.seek(ds + t["offset"]); dims = t["dims"]; n = int(np.prod(dims))
    if t["type"] == 0:
        a = np.frombuffer(f.read(n*4), np.float32).astype(np.float64)
    elif t["type"] == 30:
        a = (np.frombuffer(f.read(n*2), np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    else:
        raise ValueError
    return a.reshape(list(reversed(dims)))

def get_expert_rows(name, e, rows_needed=None):
    t = tn[name]; base = ds + t["offset"] + e * D * BPR
    f.seek(base); raw = np.frombuffer(f.read(D * BPR), np.uint8).reshape(D, BPR)
    out = np.empty((D, D))
    for b in range(D // 32):
        blk = raw[:, b*17:(b+1)*17]
        s = np.exp2(blk[:, 0].astype(np.float64) - 127)
        n = blk[:, 1:17]
        out[:, b*32:b*32+16] = KV[n & 0xF] * s[:, None]
        out[:, b*32+16:b*32+32] = KV[n >> 4] * s[:, None]
    return out

def rms(x, w): return x / np.sqrt((x*x).mean() + 1e-5) * w

x = get("token_embd.weight")[TOK].copy()
print(f"embed norm {np.linalg.norm(x):.4f}")
ALPHA, LIM = 1.702, 7.0
for il in range(NL):
    p = f"blk.{il}."
    xn = rms(x, get(p+"attn_norm.weight"))
    q = (get(p+"attn_q.weight") @ xn + get(p+"attn_q.bias")).reshape(NH, HD)
    v = (get(p+"attn_v.weight") @ xn + get(p+"attn_v.bias")).reshape(NKV, HD)
    # pos 0, plain rope = identity; single token: softmax over [sink, score_self]
    k = (get(p+"attn_k.weight") @ xn + get(p+"attn_k.bias")).reshape(NKV, HD)
    sinks = get(p+"attn_sinks")
    ao = np.zeros((NH, HD))
    for h in range(NH):
        kv = h // (NH // NKV)
        sc = (k[kv] @ q[h]) / np.sqrt(HD)
        m = max(sc, sinks[h])
        e1, e0 = np.exp(sc - m), np.exp(sinks[h] - m)
        ao[h] = (e1 / (e1 + e0)) * v[kv]
    x = x + get(p+"attn_out.weight") @ ao.reshape(-1) + get(p+"attn_out.bias")
    xn2 = rms(x, get(p+"ffn_norm.weight"))
    logits = get(p+"ffn_gate_inp.weight") @ xn2 + get(p+"ffn_gate_inp.bias")
    top = np.argsort(-logits)[:TOPK]
    w = logits[top]; w = np.exp(w - w.max()); w /= w.sum()
    gb, ub, db = get(p+"ffn_gate_exps.bias"), get(p+"ffn_up_exps.bias"), get(p+"ffn_down_exps.bias")
    ffn = np.zeros(D)
    for e, we in zip(top, w):
        g = get_expert_rows(p+"ffn_gate_exps.weight", e) @ xn2 + gb[e]
        u = get_expert_rows(p+"ffn_up_exps.weight", e) @ xn2 + ub[e]
        xg = np.minimum(g, LIM); yu = np.clip(u, -LIM, LIM)
        h = (xg / (1 + np.exp(-ALPHA * xg))) * (yu + 1)
        ffn += we * (get_expert_rows(p+"ffn_down_exps.weight", e) @ h + db[e])
    x = x + ffn
    print(f"layer {il:2d} |x| {np.linalg.norm(x):10.4f} top{list(top)}")
xn = rms(x, get("output_norm.weight"))
head = get("output.weight")
lg = head @ xn
am = int(np.argmax(lg))
print(f"final |x| {np.linalg.norm(x):.4f} argmax {am} logit {lg[am]:.4f}")
