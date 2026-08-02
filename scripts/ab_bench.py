#!/usr/bin/env python3
"""Same-session A/B decode bench: alternate warm --serve engines on one prompt set.

Each engine loads ONCE, then the harness interleaves requests round-robin so both
engines see the same machine interference (desktop load, thermal state). Reports
per-engine best and median; best is the least-contaminated estimate of what the
machine can do, median shows what you get under the ambient load.

Usage: python3 scripts/ab_bench.py --engines A=target/release/x B=target/release/y
                                   --data data-gpt-oss_20b --rounds 5 --ngen 256
"""
import argparse, os, re, statistics, subprocess, sys, time

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
    fd = p.stdout.fileno()
    while b"[DONE]" not in out:
        c = os.read(fd, 65536)
        if not c:
            raise RuntimeError("engine died mid-request")
        out += c
    t = out.decode("utf-8", "replace")
    m = re.search(r"tok_s=([\d.]+)", t)
    return float(m.group(1)), t.split("[STATS]")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--data", default="data-gpt-oss_20b")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ngen", type=int, default=256)
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--diff", action="store_true", help="compare generated text across engines")
    args = ap.parse_args()

    eng = {}
    for spec in args.engines:
        name, path = spec.split("=", 1)
        print(f"loading {name} ({path})...", file=sys.stderr)
        eng[name] = start(path, args.data, args.ngen)

    res = {n: [] for n in eng}
    texts = {}
    for r in range(args.rounds):
        for n, p in eng.items():                      # round-robin: same interference
            tok_s, text = ask(p, args.prompt, args.ngen)
            res[n].append(tok_s)
            texts.setdefault(n, text)
            print(f"  round {r+1} {n:12} {tok_s:6.1f} tok/s", file=sys.stderr)

    print()
    base = None
    for n, v in res.items():
        b, med = max(v), statistics.median(v)
        if base is None:
            base = b
        print(f"{n:12} best {b:6.1f}  median {med:6.1f}  all {[round(x,1) for x in v]}")
    names = list(res)
    if len(names) == 2:
        a, b = names
        print(f"\n{b} vs {a}: best {max(res[b])/max(res[a]):.3f}x  median "
              f"{statistics.median(res[b])/statistics.median(res[a]):.3f}x")
    if args.diff and len(names) == 2:
        same = texts[names[0]] == texts[names[1]]
        print(f"generated text identical: {same}")
    for p in eng.values():
        p.kill()

if __name__ == "__main__":
    main()
