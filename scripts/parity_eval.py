#!/usr/bin/env python3
"""Correctness-parity eval: run the same benchmark questions through cpubrrr and
through Ollama (same model, same quantized weights) and compare scores.

An inference engine cannot make a model smarter -- but a broken engine can make it
dumber. Matching scores on a real benchmark is evidence the speed doesn't come from
corrupted math. This is a parity check, not a leaderboard entry: both engines get the
same prompts, greedy decoding, and the same answer extraction.

Usage:
  python3 scripts/parity_eval.py --engine qwen --n 100
  python3 scripts/parity_eval.py --engine gptoss --n 100

Requires: ollama running with the model pulled; engines built (cargo build --release).
Dataset: GSM8K test split, downloaded once to .eval_cache/.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".eval_cache")
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

ENGINES = {
    "qwen": {
        "bin": os.path.join(ROOT, "target/release/engine_qwen2"),
        "data": os.path.join(ROOT, "data-qwen3-coder_30b"),
        "ollama_model": "qwen3-coder:30b",
    },
    "gptoss": {
        "bin": os.path.join(ROOT, "target/release/engine"),
        "data": os.path.join(ROOT, "data-gpt-oss_20b"),
        "ollama_model": "gpt-oss:20b",
    },
    "gptoss2": {
        "bin": os.path.join(ROOT, "target/release/engine_gpt2"),
        "data": os.path.join(ROOT, "data-gpt-oss_20b"),
        "ollama_model": "gpt-oss:20b",
    },
}

PROMPT_SUFFIX = "\n\nSolve step by step, then give the final numeric answer on the last line as: #### <number>"

def load_gsm8k(n):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "gsm8k_test.jsonl")
    if not os.path.exists(path):
        print("downloading GSM8K test split...", file=sys.stderr)
        urllib.request.urlretrieve(GSM8K_URL, path)
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"q": d["question"], "gold": gold})
    return items[:n]

def extract_answer(text):
    # prefer '#### <number>'; fall back to last number in the text
    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")

def norm(x):
    if x is None:
        return None
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x

class CpubrrrServe:
    """Warm --serve engine speaking the [READY]/[STATS]/[DONE] protocol."""
    def __init__(self, binpath, datadir, npred=320):
        blob = open(os.path.join(datadir, "blob_path.txt")).read().strip()
        self.npred = npred
        self.p = subprocess.Popen([binpath, datadir, blob, "--serve", str(npred)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        while True:
            line = self.p.stderr.readline().decode("utf-8", "replace")
            if "[READY]" in line:
                break
            if not line:
                raise RuntimeError("engine died during warmup")

    def ask(self, prompt):
        # TSV serve protocol: temp \t seed \t ngen \t prompt (both engines accept it;
        # qwen is greedy-only and ignores temp/seed)
        line = f"0\t1\t{self.npred}\t" + prompt.replace("\t", " ").replace("\n", " ") + "\n"
        self.p.stdin.write(line.encode())
        self.p.stdin.flush()
        out = b""
        fd = self.p.stdout.fileno()
        while b"[DONE]" not in out:
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("engine died mid-request")
            out += chunk
        text = out.decode("utf-8", "replace")
        return text.split("[STATS]")[0]

    def close(self):
        self.p.kill()

def ollama_ask(model, prompt, npred=320):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "keep_alive": "30m",
                       "options": {"temperature": 0, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(ENGINES), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--npred", type=int, default=320,
                    help="max new tokens BOTH sides; raise for reasoning models so "
                         "the thinking channel can't eat the budget before the answer")
    ap.add_argument("--skip-ollama", action="store_true")
    args = ap.parse_args()
    cfg = ENGINES[args.engine]
    items = load_gsm8k(args.n)
    print(f"GSM8K parity, {len(items)} questions, model {cfg['ollama_model']}")

    results = {"cpubrrr": [], "ollama": []}
    eng = CpubrrrServe(cfg["bin"], cfg["data"], args.npred)
    t0 = time.time()
    for i, it in enumerate(items):
        text = eng.ask(it["q"] + PROMPT_SUFFIX)
        ok = norm(extract_answer(text)) == norm(it["gold"])
        results["cpubrrr"].append(ok)
        print(f"  cpubrrr {i+1}/{len(items)} {'ok' if ok else 'WRONG'}", file=sys.stderr)
    eng.close()
    t_cpu = time.time() - t0

    if not args.skip_ollama:
        t0 = time.time()
        for i, it in enumerate(items):
            text = ollama_ask(cfg["ollama_model"], it["q"] + PROMPT_SUFFIX, args.npred)
            ok = norm(extract_answer(text)) == norm(it["gold"])
            results["ollama"].append(ok)
            print(f"  ollama  {i+1}/{len(items)} {'ok' if ok else 'WRONG'}", file=sys.stderr)
        t_oll = time.time() - t0

    n = len(items)
    c = sum(results["cpubrrr"])
    print(f"\ncpubrrr : {c}/{n} = {100*c/n:.1f}%  ({t_cpu:.0f}s)")
    if results["ollama"]:
        o = sum(results["ollama"])
        print(f"ollama  : {o}/{n} = {100*o/n:.1f}%  ({t_oll:.0f}s)")
        print(f"delta   : {100*(c-o)/n:+.1f} points")

if __name__ == "__main__":
    main()
