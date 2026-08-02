#!/usr/bin/env python3
"""Fast quality proxy: greedy token-stream agreement between two engines.

GSM8K is the real correctness gate but costs ~10 minutes a run, which is too slow to
iterate a quantizer against. Both engines decode greedily, so their token streams are
deterministic and directly comparable: this reports, per prompt, how many tokens match
prefix-wise before the first divergence, plus overall exact-match rate.

Divergence is NOT automatically a defect -- a different-but-equally-good token can
appear once error is nonzero -- so treat this as a screen that ranks variants, and
confirm anything that looks good with parity_eval.

Usage: python3 scripts/agree.py --ref target/release/A --new target/release/B --ngen 96
"""
import argparse, os, re, subprocess, sys

PROMPTS = [
    "Why is the sky blue?",
    "Write a haiku about CPUs.",
    "Explain how a hash map works.",
    "What is 17 * 23? Show your reasoning.",
    "Name three causes of the French Revolution.",
    "Write a Python function that reverses a linked list.",
    "If a train leaves at 3pm going 60mph, how far in 2.5 hours?",
    "Summarize the plot of Hamlet in two sentences.",
]

def start(binpath, datadir, ngen):
    blob = open(os.path.join(datadir, "blob_path.txt")).read().strip()
    p = subprocess.Popen([binpath, datadir, blob, "--serve", str(ngen)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    while True:
        line = p.stderr.readline().decode("utf-8", "replace")
        if "[READY]" in line:
            return p
        if not line:
            raise RuntimeError(f"{binpath} died during load")

def ask(p, prompt, ngen):
    p.stdin.write(f"0\t1\t{ngen}\t{prompt}\n".encode())
    p.stdin.flush()
    out = b""
    while b"[DONE]" not in out:
        c = os.read(p.stdout.fileno(), 65536)
        if not c:
            raise RuntimeError("engine died mid-request")
        out += c
    return out.decode("utf-8", "replace").split("[STATS]")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--data", default="data-gpt-oss_20b")
    ap.add_argument("--ngen", type=int, default=96)
    a = ap.parse_args()

    ra, rb = start(a.ref, a.data, a.ngen), start(a.new, a.data, a.ngen)
    exact = 0
    fracs = []
    for pr in PROMPTS:
        ta, tb = ask(ra, pr, a.ngen), ask(rb, pr, a.ngen)
        if ta == tb:
            exact += 1
            fracs.append(1.0)
            print(f"  identical              | {pr[:44]}")
            continue
        # characters matching before first divergence, as a fraction of the ref length
        i = 0
        while i < min(len(ta), len(tb)) and ta[i] == tb[i]:
            i += 1
        f = i / max(len(ta), 1)
        fracs.append(f)
        print(f"  diverges at {i:5}/{len(ta):5} ({f:5.1%}) | {pr[:44]}")
    print(f"\nexact matches {exact}/{len(PROMPTS)}   mean prefix agreement {sum(fracs)/len(fracs):.1%}")
    ra.kill(); rb.kill()

if __name__ == "__main__":
    main()
