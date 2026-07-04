#!/usr/bin/env python3
"""Side-by-side demo server: our engine vs Ollama/llama.cpp CPU, token streaming.
Usage: python3 scripts/demo_server.py  ->  open http://localhost:8642
Requires: ollama running; engine built (cargo build --release)."""
import json, os, subprocess, threading, urllib.request, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.environ.get("CPUTRAIN_ENGINE", os.path.join(ROOT, "target/release/engine"))
GPTOSS_DIR = os.environ.get("CPUTRAIN_DATA", os.path.join(ROOT, "data"))
_blobfile = os.path.join(GPTOSS_DIR, "blob_path.txt")
BLOB = os.environ.get("CPUTRAIN_BLOB") or (
    open(_blobfile).read().strip() if os.path.exists(_blobfile) else "")
if not BLOB or not os.path.exists(BLOB):
    raise SystemExit(
        "model blob not found. Run scripts/setup_gptoss.sh first, "
        "or set CPUTRAIN_BLOB to the gpt-oss GGUF path.")
NPRED = 256

HARMONY = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
           "Knowledge cutoff: 2024-06\nCurrent date: 2026-07-03\n\nReasoning: low\n\n"
           "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
           "<|start|>user<|message|>{p}<|end|><|start|>assistant")

engine_lock = threading.Lock()
engine_proc = None

def get_engine():
    global engine_proc
    if engine_proc is None or engine_proc.poll() is not None:
        engine_proc = subprocess.Popen(
            [ENGINE, GPTOSS_DIR, BLOB, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        # wait for [READY] on stderr
        while True:
            line = engine_proc.stderr.readline().decode("utf-8", "replace")
            if "[READY]" in line or not line:
                break
    return engine_proc

def sse(w, obj):
    w.write(f"data: {json.dumps(obj)}\n\n".encode())
    w.flush()

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
        if u.path == "/":
            html = open(os.path.join(ROOT, "demo/demo.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)
        elif u.path == "/run_ours":
            self._sse_headers()
            self.run_ours(prompt)
        elif u.path == "/run_ollama":
            self._sse_headers()
            self.run_ollama(prompt)
        else:
            self.send_response(404)
            self.end_headers()

    def run_ours(self, prompt):
        with engine_lock:
            p = get_engine()
            sse(self.wfile, {"t": "status", "s": "generating (model resident)..."})
            p.stdin.write((prompt.replace("\n", " ") + "\n").encode())
            p.stdin.flush()
            fd = p.stdout.fileno()
            buf = b""
            sent = 0
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
                        sse(self.wfile, {"t": "tok", "s": text[sent:mi]})
                        sent = mi
                    if "[DONE]" in text:
                        break
                else:
                    safe = max(sent, len(text) - len(MARK))
                    if safe > sent:
                        sse(self.wfile, {"t": "tok", "s": text[sent:safe]})
                        sent = safe
            text = buf.decode("utf-8", "replace")
            stats = {}
            for line in text.splitlines():
                if line.startswith("[STATS]"):
                    for kv in line.split()[1:]:
                        k, v = kv.split("=")
                        stats[k] = float(v)
            sse(self.wfile, {"t": "done", "stats": stats})

    def run_ollama(self, prompt):
        body = json.dumps({
            "model": "gpt-oss:20b", "raw": True, "stream": True, "keep_alive": "15m",
            "prompt": HARMONY.format(p=prompt),
            "options": {"num_gpu": 0, "num_predict": NPRED, "temperature": 0},
        }).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        sse(self.wfile, {"t": "status", "s": "loading model (CPU)..."})
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                d = json.loads(line)
                if d.get("response"):
                    sse(self.wfile, {"t": "tok", "s": d["response"]})
                if d.get("done"):
                    ec, ed = d.get("eval_count", 0), d.get("eval_duration", 1)
                    pc, pd = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1)
                    sse(self.wfile, {"t": "done", "stats": {
                        "prefill_tok": pc, "prefill_s": pd / 1e9,
                        "decode_tok": ec, "decode_s": ed / 1e9, "tok_s": ec / (ed / 1e9)}})

if __name__ == "__main__":
    print("warming engine (load + repack, ~15s)...")
    get_engine()
    print("demo ready: http://localhost:8642")
    ThreadingHTTPServer(("127.0.0.1", 8642), H).serve_forever()
