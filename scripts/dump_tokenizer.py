#!/usr/bin/env python3
"""Dump GGUF tokenizer to flat binary (u32 count, then per-token u16 len + raw bytes)
and a tensor manifest (name type offset nelems)."""
import struct, sys

BLOB, OUT = sys.argv[1], sys.argv[2]

def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8)); return f.read(n)

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip([chr(c) for c in cs], bs))  # unicode char -> byte

U2B = bytes_to_unicode()

def tok_bytes(s):
    try:
        t = s.decode("utf-8")
    except UnicodeDecodeError:
        return s
    out = bytearray()
    for ch in t:
        if ch in U2B:
            out.append(U2B[ch])
        else:
            return s  # special token or non-bpe: keep literal
    return bytes(out)

f = open(BLOB, "rb")
assert f.read(4) == b"GGUF"
f.read(4)
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
tokens = None
for _ in range(n_kv):
    k = read_str(f).decode()
    (t,) = struct.unpack("<I", f.read(4))
    if t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            vals = [read_str(f) for _ in range(n)]
            if k == "tokenizer.ggml.tokens":
                tokens = vals
        else:
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[et]
            f.seek(n * sz, 1)
    else:
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
        f.seek(sz, 1)

assert tokens, "no tokens found"
w = open(f"{OUT}/tokens.bin", "wb")
w.write(struct.pack("<I", len(tokens)))
for s in tokens:
    b = tok_bytes(s)
    w.write(struct.pack("<H", len(b)))
    w.write(b)
print("tokens:", len(tokens))
for i, s in enumerate(tokens):
    d = s.decode("utf-8", "replace")
    if d in ("<|start|>", "<|end|>", "<|message|>", "<|return|>", "<|channel|>", "<|endoftext|>"):
        print(d, "=", i)
