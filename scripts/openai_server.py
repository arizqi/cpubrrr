#!/usr/bin/env python3
"""OpenAI-compatible API server for cpubrrr — drop the engine into any tool that
speaks the OpenAI chat API (chat UIs, IDE plugins, agent frameworks).

  python3 scripts/openai_server.py            # serves http://localhost:8643/v1
  curl http://localhost:8643/v1/models
  curl http://localhost:8643/v1/chat/completions -d '{
    "model": "cpubrrr/qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Write a haiku about CPUs"}],
    "stream": true}'

Implemented: /v1/models, /v1/chat/completions (streaming + non-streaming),
max_tokens, usage accounting from engine [STATS]. Honest limits: the engines apply
their own chat template, so multi-turn history is flattened into the prompt;
temperature is honored by the gpt-oss engine and ignored (greedy) by the qwen
engine; one request at a time per model (the engine owns all cores anyway).

Stdlib only. Requires engines built (cargo build --release) and model data prepared
(scripts/setup_model.sh <model>).
"""
import json, os, re, subprocess, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("CPBRR_PORT", "8643"))

CANDIDATES = [
    {"id": "cpubrrr/qwen3-coder-30b", "bin": "target/release/engine_qwen2",
     "data": "data-qwen3-coder_30b", "greedy_only": True},
    {"id": "cpubrrr/gpt-oss-20b", "bin": "target/release/engine",
     "data": "data-gpt-oss_20b", "greedy_only": False},
]

class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.proc = None

    def ensure(self):
        if self.proc is None or self.proc.poll() is not None:
            data = os.path.join(ROOT, self.cfg["data"])
            blob = open(os.path.join(data, "blob_path.txt")).read().strip()
            print(f"warming {self.cfg['id']}...")
            self.proc = subprocess.Popen(
                [os.path.join(ROOT, self.cfg["bin"]), data, blob, "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0)
            # [READY] goes to stderr which we discard; give the load a beat, then
            # probe with a 1-token request so we know it's alive
            self.request("hi", 1, 0.0, lambda _t: None)
            print(f"{self.cfg['id']} ready")
        return self.proc

    def request(self, prompt, max_tokens, temperature, on_text):
        """Stream generated text chunks to on_text; return final stats dict."""
        p = self.proc
        line = f"{temperature}\t1\t{max_tokens}\t" + prompt.replace("\\", " ").replace("\t", " ").replace("\n", "\\n") + "\n"
        p.stdin.write(line.encode())
        p.stdin.flush()
        fd = p.stdout.fileno()
        buf, sent = b"", 0
        MARK = "\n[STATS]"
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("engine died")
            buf += chunk
            text = buf.decode("utf-8", "replace")
            mi = text.find(MARK)
            if mi >= 0:
                if mi > sent:
                    on_text(text[sent:mi])
                    sent = mi
                if "[DONE]" in text:
                    break
            else:
                safe = max(sent, len(text) - len(MARK))
                if safe > sent:
                    on_text(text[sent:safe])
                    sent = safe
        stats = {}
        for l in buf.decode("utf-8", "replace").splitlines():
            if l.startswith("[STATS]"):
                for kv in l.split()[1:]:
                    k, v = kv.split("=")
                    stats[k] = float(v)
        return stats

ENGINES = {}
for c in CANDIDATES:
    if os.path.exists(os.path.join(ROOT, c["data"], "blob_path.txt")) and \
       os.path.exists(os.path.join(ROOT, c["bin"])):
        ENGINES[c["id"]] = Engine(c)

def flatten_messages(messages):
    """Engines apply their own chat template around a single prompt; flatten
    history in a transparent, documented way."""
    parts = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if isinstance(content, list):  # OpenAI content-parts form
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            parts.append(f"[system] {content}")
        elif role == "assistant":
            parts.append(f"[assistant said] {content}")
        else:
            parts.append(content)
    return "\n".join(parts)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": mid, "object": "model", "owned_by": "cpubrrr"} for mid in ENGINES]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        model = req.get("model") or next(iter(ENGINES), None)
        if model not in ENGINES:
            return self._json(404, {"error": {"message":
                f"model '{model}' not loaded; available: {list(ENGINES)}"}})
        eng = ENGINES[model]
        prompt = flatten_messages(req.get("messages", []))
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
        temperature = float(req.get("temperature", 0.0))
        stream = bool(req.get("stream", False))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        with eng.lock:
            eng.ensure()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": model, "choices": [{"index": 0,
                     "delta": {"role": "assistant"}, "finish_reason": None}]})
                def on_text(t):
                    sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                         "model": model, "choices": [{"index": 0,
                         "delta": {"content": t}, "finish_reason": None}]})
                stats = eng.request(prompt, max_tokens, temperature, on_text)
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": model, "choices": [{"index": 0, "delta": {},
                     "finish_reason": "stop"}],
                     "usage": {"completion_tokens": int(stats.get("decode_tok", 0)),
                               "cpubrrr_tok_s": stats.get("tok_s")}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                chunks = []
                stats = eng.request(prompt, max_tokens, temperature, chunks.append)
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "".join(chunks).strip()}}],
                    "usage": {"completion_tokens": int(stats.get("decode_tok", 0)),
                              "prompt_tokens": int(stats.get("prefill_tok", 0)),
                              "cpubrrr_tok_s": stats.get("tok_s")}})

if __name__ == "__main__":
    if not ENGINES:
        raise SystemExit("no engines available — build with cargo and run scripts/setup_model.sh first")
    print(f"models: {list(ENGINES)}")
    first = next(iter(ENGINES.values()))
    with first.lock:
        first.ensure()   # warm the first model so the first request is instant
    print(f"OpenAI-compatible server: http://localhost:{PORT}/v1")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
