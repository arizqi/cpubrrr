#!/usr/bin/env python3
"""Exact gpt-oss layer-0 MoE-FFN reference (numpy f64) from extracted GGUF tensors.
Recovered spec (llama.cpp openai-moe.cpp / ops.cpp):
  router: logits = x @ gate_inp + gate_inp_b ; top-4 ; softmax over the 4 selected
  expert: g = gate_e @ x + gate_b ; u = up_e @ x + up_b
  act (SwiGLU-OAI, alpha=1.702, limit=7): xg=min(g,limit); yu=clamp(u,-limit,limit)
          h = (xg / (1+exp(-alpha*xg))) * (yu + 1)
  out += weight_e * (down_e @ h + down_b)
"""
import json, struct, sys, numpy as np

BLOB, IDX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
D, NE, TOPK = 2880, 32, 4
BPR = D // 32 * 17
KV = np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], np.float64)

idx = json.load(open(IDX)); tn = {t["name"]: t for t in idx["tensors"]}; ds = idx["data_start"]
f = open(BLOB, "rb")

def raw(name, nbytes):
    t = tn[name]; f.seek(ds + t["offset"]); return f.read(nbytes), t

def deq_mxfp4(buf, rows):  # -> (rows, D) f64
    a = np.frombuffer(buf, np.uint8).reshape(rows, BPR)
    out = np.empty((rows, D), np.float64)
    for b in range(D // 32):
        blk = a[:, b*17:(b+1)*17]
        scale = np.exp2(blk[:, 0].astype(np.float64) - 127)
        n = blk[:, 1:17]
        lo = KV[n & 0xF]; hi = KV[n >> 4]
        out[:, b*32:b*32+16] = lo * scale[:, None]
        out[:, b*32+16:b*32+32] = hi * scale[:, None]
    return out

def f32(name, shape):
    t = tn[name]; f.seek(ds + t["offset"])
    return np.frombuffer(f.read(int(np.prod(shape))*4), np.float32).astype(np.float64).reshape(shape)

gate = deq_mxfp4(raw("blk.0.ffn_gate_exps.weight", NE*D*BPR)[0], NE*D).reshape(NE, D, D)
up   = deq_mxfp4(raw("blk.0.ffn_up_exps.weight",   NE*D*BPR)[0], NE*D).reshape(NE, D, D)
down = deq_mxfp4(raw("blk.0.ffn_down_exps.weight", NE*D*BPR)[0], NE*D).reshape(NE, D, D)
gate_b = f32("blk.0.ffn_gate_exps.bias", (NE, D))
up_b   = f32("blk.0.ffn_up_exps.bias",   (NE, D))
down_b = f32("blk.0.ffn_down_exps.bias", (NE, D))
ginp   = f32("blk.0.ffn_gate_inp.weight", (NE, D))   # dims [2880,32] stored row-major -> (32,2880)
ginp_b = f32("blk.0.ffn_gate_inp.bias", (NE,))

# fixed pseudo-random activation
rng = np.random.default_rng(0)
x = (rng.standard_normal(D) * 0.5)

logits = ginp @ x + ginp_b
top = np.argsort(-logits)[:TOPK]
w = logits[top]; w = np.exp(w - w.max()); w /= w.sum()
ALPHA, LIM = 1.702, 7.0
out = np.zeros(D)
for e, we in zip(top, w):
    g = gate[e] @ x + gate_b[e]
    u = up[e] @ x + up_b[e]
    xg = np.minimum(g, LIM); yu = np.clip(u, -LIM, LIM)
    h = (xg / (1 + np.exp(-ALPHA * xg))) * (yu + 1)
    out += we * (down[e] @ h + down_b[e])

gate_b.astype(np.float32).tofile(f"{OUT}/gate_b.f32")
up_b.astype(np.float32).tofile(f"{OUT}/up_b.f32")
down_b.astype(np.float32).tofile(f"{OUT}/down_b.f32")
ginp.astype(np.float32).tofile(f"{OUT}/ginp.f32")
ginp_b.astype(np.float32).tofile(f"{OUT}/ginp_b.f32")
np.save(f"{OUT}/ffn_x.npy", x)
x.astype(np.float32).tofile(f"{OUT}/ffn_x.f32")
out.astype(np.float32).tofile(f"{OUT}/ffn_yref.f32")
np.array(top, np.int32).tofile(f"{OUT}/ffn_top.i32")
np.array(w, np.float32).tofile(f"{OUT}/ffn_w.f32")
print("top experts:", top.tolist())
print("softmax weights:", [round(float(v),4) for v in w])
print("out[:6]:", [round(float(v),4) for v in out[:6]])
print("out norm:", round(float(np.linalg.norm(out)),4))
