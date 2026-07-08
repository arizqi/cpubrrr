#!/usr/bin/env python3
"""cpubrrr demo server — tabbed side-by-side races, token streaming.

Tabs served by demo/demo.html:
  1. CPU gpt-oss:20b   : cpubrrr engine (CPU)      vs llama.cpp/Ollama (CPU)
  2. CPU Qwen3 (Q4_K)  : cpubrrr engine_qwen2 (CPU) vs llama.cpp/Ollama (CPU)
  3. GPU Qwen3         : cpubrrr engine_metal (GPU) vs Ollama Metal (GPU)
  4. Finale            : BOTH our engines at once (CPU + GPU), aggregate tok/s

Usage: python3 scripts/demo_server.py  ->  open http://localhost:8642
Requires: ollama running; engines built:
  cargo build --release
  clang -O2 -fobjc-arc metal/engine_metal.m -framework Metal -framework Foundation -o metal/engine_metal
"""
import json, os, subprocess, threading, urllib.request, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def blob_of(data_dir):
    f = os.path.join(data_dir, "blob_path.txt")
    return open(f).read().strip() if os.path.exists(f) else ""

# --- gpt-oss (tab 1) ---
GPTOSS_ENGINE = os.environ.get("CPUBRRR_ENGINE", os.path.join(ROOT, "target/release/engine"))
_gptoss_default = next((d for d in ("data", "data-gpt-oss_20b")
                        if os.path.exists(os.path.join(ROOT, d, "blob_path.txt"))), "data")
GPTOSS_DIR = os.environ.get("CPUBRRR_DATA", os.path.join(ROOT, _gptoss_default))
GPTOSS_BLOB = os.environ.get("CPUBRRR_BLOB") or blob_of(GPTOSS_DIR)
# --- qwen (tabs 2/3/4) ---
QWEN_ENGINE = os.path.join(ROOT, "target/release/engine_qwen2")
METAL_ENGINE = os.path.join(ROOT, "metal/engine_metal")
QWEN_DIR = os.environ.get("CPUBRRR_QWEN_DATA", os.path.join(ROOT, "data-qwen3-coder_30b"))
QWEN_BLOB = os.environ.get("CPUBRRR_QWEN_BLOB") or blob_of(QWEN_DIR)
QWEN_MODEL = "qwen3-coder:30b"
NPRED = 256

HARMONY = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
           "Knowledge cutoff: 2024-06\nCurrent date: 2026-07-03\n\nReasoning: low\n\n"
           "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
           "<|start|>user<|message|>{p}<|end|><|start|>assistant")

class ServeEngine:
    """A warm --serve engine process ([READY] / streamed tokens / [STATS] / [DONE])."""
    def __init__(self, argv, name):
        self.argv, self.name = argv, name
        self.lock = threading.Lock()
        self.proc = None

    def ensure(self):
        if self.proc is None or self.proc.poll() is not None:
            print(f"warming {self.name}...")
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            while True:
                line = self.proc.stderr.readline().decode("utf-8", "replace")
                if "[READY]" in line or not line:
                    break
            print(f"{self.name} ready")
        return self.proc

gptoss = ServeEngine([GPTOSS_ENGINE, GPTOSS_DIR, GPTOSS_BLOB, "--serve"], "gpt-oss CPU engine") \
    if GPTOSS_BLOB and os.path.exists(GPTOSS_BLOB) else None
qwen = ServeEngine([QWEN_ENGINE, QWEN_DIR, QWEN_BLOB, "--serve"], "qwen CPU engine") \
    if QWEN_BLOB and os.path.exists(QWEN_BLOB) else None

def sse(w, obj):
    w.write(f"data: {json.dumps(obj)}\n\n".encode())
    w.flush()

def stream_serve_engine(w, eng, prompt):
    """One request against a warm --serve engine; forwards tokens + final stats.
    CRITICAL: always drain the engine to [DONE] even if the browser disconnects,
    otherwise the NEXT request reads this run's leftover output (stale responses)."""
    with eng.lock:
        p = eng.ensure()
        alive = [True]
        def send(obj):
            if not alive[0]:
                return
            try:
                sse(w, obj)
            except (BrokenPipeError, ConnectionResetError, OSError):
                alive[0] = False
        send({"t": "status", "s": "generating (model resident)..."})
        p.stdin.write((prompt.replace("\n", " ") + "\n").encode())
        p.stdin.flush()
        fd = p.stdout.fileno()
        buf, sent = b"", 0
        MARK = "\n[STATS]"
        while True:
            chunk = os.read(fd, 256)
            if not chunk:
                break
            buf += chunk
            text = buf.decode("utf-8", "replace")
            mi = text.find(MARK)
            if mi >= 0:
                if mi > sent:
                    send({"t": "tok", "s": text[sent:mi]})
                    sent = mi
                if "[DONE]" in text:
                    break
            else:
                safe = max(sent, len(text) - len(MARK))
                if safe > sent:
                    send({"t": "tok", "s": text[sent:safe]})
                    sent = safe
        stats = {}
        for line in buf.decode("utf-8", "replace").splitlines():
            if line.startswith("[STATS]"):
                for kv in line.split()[1:]:
                    k, v = kv.split("=")
                    stats[k] = float(v)
        send({"t": "done", "stats": stats})

metal_lock = threading.Lock()

def stream_metal(w, prompt):
    """One-shot engine_metal run: mmap load ~0.1s, stream stdout, parse final line.
    One-shot process -> safe to kill on client disconnect."""
    with metal_lock:
        sse(w, {"t": "status", "s": "loading (zero-copy mmap)..."})
        p = subprocess.Popen([METAL_ENGINE, QWEN_DIR, QWEN_BLOB, prompt.replace("\n", " "), str(NPRED)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        def drain_status():
            for line in iter(p.stderr.readline, b""):
                s = line.decode("utf-8", "replace").strip()
                if s.startswith("prompt:"):
                    try: sse(w, {"t": "status", "s": "prefill..."})
                    except Exception: pass
                elif s.startswith("prefill"):
                    try: sse(w, {"t": "status", "s": "decoding (GPU-chained)..."})
                    except Exception: pass
        threading.Thread(target=drain_status, daemon=True).start()
        fd = p.stdout.fileno()
        buf, sent = b"", 0
        MARK = "\n---"
        try:
            while True:
                chunk = os.read(fd, 256)
                if not chunk:
                    break
                buf += chunk
                text = buf.decode("utf-8", "replace")
                mi = text.find(MARK)
                if mi >= 0:
                    if mi > sent:
                        sse(w, {"t": "tok", "s": text[sent:mi]})
                        sent = mi
                    if "tok/s" in text[mi:]:
                        break
                else:
                    safe = max(sent, len(text) - len(MARK))
                    if safe > sent:
                        sse(w, {"t": "tok", "s": text[sent:safe]})
                        sent = safe
        except (BrokenPipeError, ConnectionResetError, OSError):
            p.kill()
            return
        p.wait(timeout=60)
        stats = {}
        for line in buf.decode("utf-8", "replace").splitlines():
            # "decode: 256 tokens in 2.98s = 85.9 tok/s"
            if line.startswith("decode:"):
                f = line.split()
                stats = {"decode_tok": float(f[1]), "decode_s": float(f[4].rstrip("s")), "tok_s": float(f[6])}
        sse(w, {"t": "done", "stats": stats})

def stream_ollama(w, prompt, model, gpu):
    if model == "gpt-oss:20b":
        body = {"model": model, "raw": True, "stream": True, "keep_alive": "15m",
                "prompt": HARMONY.format(p=prompt),
                "options": {"num_predict": NPRED, "temperature": 0}}
    else:
        body = {"model": model, "stream": True, "keep_alive": "15m", "prompt": prompt,
                "options": {"num_predict": NPRED, "temperature": 0}}
    if not gpu:
        body["options"]["num_gpu"] = 0
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sse(w, {"t": "status", "s": f"loading model ({'GPU' if gpu else 'CPU'})..."})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            d = json.loads(line)
            if d.get("response"):
                sse(w, {"t": "tok", "s": d["response"]})
            if d.get("done"):
                ec, ed = d.get("eval_count", 0), d.get("eval_duration", 1)
                pc, pd = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1)
                sse(w, {"t": "done", "stats": {
                    "prefill_tok": pc, "prefill_s": pd / 1e9,
                    "decode_tok": ec, "decode_s": ed / 1e9, "tok_s": ec / (ed / 1e9)}})

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        prompt = (q.get("prompt") or ["Why is the sky blue?"])[0]
        try:
            if u.path == "/":
                html = open(os.path.join(ROOT, "demo/demo.html"), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html)
            elif u.path == "/run_gptoss_ours":
                self._sse_headers()
                stream_serve_engine(self.wfile, gptoss, prompt)
            elif u.path == "/run_qwen_ours":
                self._sse_headers()
                stream_serve_engine(self.wfile, qwen, prompt)
            elif u.path == "/run_metal":
                self._sse_headers()
                stream_metal(self.wfile, prompt)
            elif u.path == "/run_ollama":
                model = (q.get("model") or ["gpt-oss:20b"])[0]
                gpu = (q.get("gpu") or ["0"])[0] == "1"
                self._sse_headers()
                stream_ollama(self.wfile, prompt, model, gpu)
            else:
                self.send_response(404)
                self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

if __name__ == "__main__":
    for eng, what in ((qwen, "tabs 2+4"), (gptoss, "tab 1")):
        if eng is None:
            print(f"note: engine for {what} not configured (missing data dir/blob) — that tab will error")
        else:
            eng.ensure()
    if not os.path.exists(METAL_ENGINE):
        print("note: metal/engine_metal not built — GPU tabs will error "
              "(clang -O2 -fobjc-arc metal/engine_metal.m -framework Metal -framework Foundation -o metal/engine_metal)")
    print("demo ready: http://localhost:8642")
    ThreadingHTTPServer(("127.0.0.1", 8642), H).serve_forever()
