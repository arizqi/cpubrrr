#!/usr/bin/env python3
"""Extract blk.0 MoE expert tensors (MXFP4) from gpt-oss GGUF + reference dot products."""
import json, struct, sys

BLOB = sys.argv[1]
IDX = sys.argv[2]
OUT = sys.argv[3]
D = 2880
BPR = D // 32 * 17  # bytes per row: 90 blocks x 17

idx = json.load(open(IDX))
tensors = {t["name"]: t for t in idx["tensors"]}
data_start = idx["data_start"]

KV = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

def deq_row(raw, r):
    out = []
    row = raw[r * BPR:(r + 1) * BPR]
    for b in range(D // 32):
        blk = row[b * 17:(b + 1) * 17]
        e = blk[0]
        scale = 2.0 ** (e - 128)
        for j in range(16):
            out.append(KV[blk[1 + j] & 0x0F] * scale)
        for j in range(16):
            out.append(KV[blk[1 + j] >> 4] * scale)
    return out

f = open(BLOB, "rb")
for mat in ["gate", "up", "down"]:
    t = tensors[f"blk.0.ffn_{mat}_exps.weight"]
    assert t["type"] == 39, t
    f.seek(data_start + t["offset"])
    raw = f.read(32 * D * BPR)
    open(f"{OUT}/{mat}.mxfp4", "wb").write(raw)
    print(mat, "extracted", len(raw), "bytes")

# reference: x = deterministic int8, y_ref over first 8 rows of gate expert 0
x = [((i * 7 + 3) % 11) - 5 for i in range(D)]
open(f"{OUT}/x.bin", "wb").write(struct.pack(f"{D}b", *x))
gate = open(f"{OUT}/gate.mxfp4", "rb").read(D * BPR)  # expert 0
yref = []
for r in range(8):
    w = deq_row(gate, r)
    yref.append(sum(wi * xi for wi, xi in zip(w, x)))
open(f"{OUT}/yref.bin", "wb").write(struct.pack("8f", *yref))
print("yref:", [round(v, 3) for v in yref])
