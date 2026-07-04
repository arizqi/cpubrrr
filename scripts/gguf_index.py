#!/usr/bin/env python3
"""Minimal GGUF v3 parser: prints tensor index (name, dims, ggml type, offset)."""
import struct, sys, json

def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")

def read_value(f, t):
    S = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    if t in S:
        return struct.unpack(S[t], f.read(struct.calcsize(S[t])))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"bad kv type {t}")

def main(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    meta = {}
    for _ in range(n_kv):
        k = read_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        v = read_value(f, t)
        if not isinstance(v, list) or len(v) <= 8:
            meta[k] = v
    infos = []
    for _ in range(n_tensors):
        name = read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ttype, off = struct.unpack("<IQ", f.read(12))
        infos.append({"name": name, "dims": list(dims), "type": ttype, "offset": off})
    align = meta.get("general.alignment", 32)
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    print(json.dumps({"version": ver, "data_start": data_start,
                      "meta": {k: v for k, v in meta.items() if isinstance(v, (int, float, str))},
                      "tensors": infos}, indent=1))

main(sys.argv[1])
