#!/usr/bin/env python3
# Logging proxy between OpenCode and koboldcpp: forwards every request, streams the response back as it arrives,
# and appends {path, request, status, response} per request to a JSONL capture file.
#
# A chat request whose prompt plus max_tokens exceeds the context gets OpenAI's context_length_exceeded error
# instead: koboldcpp would silently cut the prompt's start to fit, and OpenCode, which sizes sessions from usage
# that koboldcpp's streamed tool replies don't carry, would never compact.
#
# usage: proxy.py LISTEN_PORT UPSTREAM_URL CAPTURE.jsonl [CONTEXT]

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kcpp  # noqa: E402

PORT, UPSTREAM, CAPTURE = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
CONTEXT = int(sys.argv[4]) if len(sys.argv) > 4 else 65536
lock = threading.Lock()
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def overflow(self, body):
        """prompt tokens when the request can't fit the context, else None"""
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return None
        try:
            req = json.loads(body)
            n = len(kcpp.tokenize(kcpp.render_request(req)))
        except Exception:
            return None
        max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or 0
        return n if n + max_tokens > CONTEXT else None

    def record(self, rec, body):
        try:
            rec["request"] = json.loads(body) if body else None
        except json.JSONDecodeError:
            rec["request"] = body.decode("utf-8", "replace")
        with lock, open(CAPTURE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def forward(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        t0 = time.time()
        over = self.overflow(body) if method == "POST" else None
        if over is not None:
            err = json.dumps({"error": {"message": f"This model's maximum context length is {CONTEXT} tokens. However, your messages "
                                                  f"resulted in {over} tokens. Please reduce the length of the messages.",
                                        "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            self.record({"t0": t0, "t1": time.time(), "method": method, "path": self.path, "status": 400, "overflow": over,
                         "response": err.decode()}, body)
            return
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            r = requests.request(method, UPSTREAM + self.path, data=body, headers=headers, stream=True, timeout=7200)
        except requests.RequestException as e:
            self.send_error(502, str(e))
            return
        self.send_response(r.status_code)
        for k, v in r.headers.items():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.end_headers()
        chunks = []
        try:
            for c in r.iter_content(chunk_size=None):
                chunks.append(c)
                self.wfile.write(c)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.record({"t0": t0, "t1": time.time(), "method": method, "path": self.path, "status": r.status_code,
                     "response": b"".join(chunks).decode("utf-8", "replace")}, body)

    def do_GET(self):
        self.forward("GET")

    def do_POST(self):
        self.forward("POST")


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
