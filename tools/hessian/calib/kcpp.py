import json
import os
import subprocess
import threading
import time
from pathlib import Path

import requests
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from common import CALIB, GEN_MODEL, ROOT, WORK, log

URL = "http://127.0.0.1:5002"
MODE = "gen"
_session = requests.Session()
_restart_lock = threading.Lock()


class ServerDown(Exception):
    pass


def server_alive():
    try:
        os.kill(int((WORK / "server.pid").read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def ensure_server():
    """restarts koboldcpp if it died (its batched generation has crashed on this setup)"""
    with _restart_lock:
        if server_alive():
            return
        log("koboldcpp is down, restarting it")
        subprocess.run([str(CALIB / "serve.sh"), MODE], check=True)


def _post(path, body, timeout=3600):
    for attempt in range(120):
        try:
            r = _session.post(URL + path, json=body, timeout=timeout)
            if r.status_code == 503:  # koboldcpp's request queue is full
                time.sleep(15)
                continue
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as e:
            if not server_alive():
                ensure_server()
            raise ServerDown(str(e))
    raise ServerDown("koboldcpp stayed busy")


class _Tokenizer:
    """hessian-tokenize: llama_tokenize with the model's vocab, as hessian-collect tokenizes"""

    def __init__(self, detok=False):
        self.args = [str(ROOT / "hessian-tokenize"), str(GEN_MODEL)] + (["--detokenize"] if detok else [])
        self.p = None
        self.lock = threading.Lock()

    def __call__(self, x):
        with self.lock:
            for attempt in range(2):
                if self.p is None or self.p.poll() is not None:
                    self.p = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
                self.p.stdin.write(json.dumps(x) + "\n")
                self.p.stdin.flush()
                line = self.p.stdout.readline()
                if line:
                    out = json.loads(line)
                    if isinstance(out, dict):
                        raise ValueError(f"hessian-tokenize: {out.get('error')}")
                    return out
            raise RuntimeError("hessian-tokenize died")


_tok = None
_detok = None


def clean(text):
    """lone UTF-16 surrogates (which some source texts contain) can't be encoded as UTF-8, by koboldcpp either"""
    return text.encode("utf-8", "replace").decode("utf-8")


def tokenize(text):
    global _tok
    if _tok is None:
        _tok = _Tokenizer()
    return _tok(clean(text))


def detokenize(ids):
    global _detok
    if _detok is None:
        _detok = _Tokenizer(detok=True)
    return _detok(ids)


def complete(prompt, max_tokens, temperature=1.0, top_p=0.95, top_k=20):
    j = _post("/v1/completions", {
        "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "min_p": 0.0, "rep_pen": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    })
    c = j["choices"][0]
    return c["text"], c.get("finish_reason"), j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]


class _IgnoreGenerationTags(Extension):
    tags = {"generation"}

    def parse(self, parser):
        parser.stream.skip(1)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


_template = None


def chat_template():
    global _template
    if _template is None:
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_IgnoreGenerationTags, loopcontrols])
        env.globals["raise_exception"] = lambda msg: ""
        env.globals["strftime_now"] = lambda fmt="%Y-%m-%d %H:%M:%S": time.strftime(fmt)
        env.filters["tojson"] = _tojson
        _template = env.from_string((WORK / "chat_template.jinja").read_text())
    return _template


def render(messages, tools=None, **kwargs):
    """The prompt koboldcpp's format_jinja builds for these messages (no assistant prefill)."""
    msgs = json.loads(json.dumps(messages))
    for m in msgs:
        if m.get("content") is None:
            m["content"] = ""
    if tools:
        return clean(chat_template().render(messages=msgs, tools=tools, add_generation_prompt=True, bos_token="", eos_token="", **kwargs))
    return clean(chat_template().render(messages=msgs, add_generation_prompt=True, bos_token="", eos_token="", **kwargs))


def think_kwargs(think):
    if think == "off":
        return {"enable_thinking": False}
    return {"reasoning_effort": think}


def _kcpp_messages(messages):
    """koboldcpp's format_jinja message clean-up: None content, list content flattened to text with media
    placeholders, tool-call arguments parsed from JSON strings"""
    msgs = json.loads(json.dumps(messages))
    for m in msgs:
        if m.get("content") is None:
            m["content"] = ""
    media = 1
    for m in msgs:
        if isinstance(m.get("content"), list):
            normalized, text = [], ""
            for item in m["content"]:
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") in ("image_url", "image"):
                    text += f"\n(Attached Image {media})\n"
                    media += 1
                elif item.get("type") == "input_audio":
                    text += f"\n(Attached Audio {media})\n"
                    media += 1
                else:
                    normalized.append(item)
            if text:
                normalized.append({"type": "text", "text": text})
            m["content"] = normalized
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except json.JSONDecodeError:
                    pass
    return msgs


def request_kwargs(req):
    kw = dict(req.get("chat_template_kwargs") or {})
    if req.get("reasoning_effort") is not None:
        kw["reasoning_effort"] = req["reasoning_effort"]
    if "enable_thinking" in kw and not kw["enable_thinking"]:
        kw["reasoning_strength"] = "none"
    elif kw.get("reasoning_effort"):
        kw["reasoning_strength"] = kw["reasoning_effort"]
    return kw


def render_request(req):
    """the prompt koboldcpp renders for an OpenAI chat-completions request (jinja mode)"""
    msgs = _kcpp_messages(req.get("messages", []))
    tools = req.get("tools") or []
    kw = request_kwargs(req)
    prefill = ""
    if msgs and msgs[-1]["role"].lower() == "assistant" and isinstance(msgs[-1]["content"], str) and msgs[-1]["content"].strip():
        prefill = msgs[-1]["content"]
        msgs = msgs[:-1] if len(msgs) > 1 else msgs
    t = chat_template()
    args = dict(messages=msgs, add_generation_prompt=True, bos_token="", eos_token="", **kw)
    if tools:
        args["tools"] = tools
    return t.render(**args) + prefill


def render_turn(req, assistant):
    """the request's prompt followed by an assistant turn, as the template renders it in the next request's history"""
    msgs = _kcpp_messages(req.get("messages", []) + [assistant])
    tools = req.get("tools") or []
    args = dict(messages=msgs, add_generation_prompt=False, bos_token="", eos_token="", **request_kwargs(req))
    if tools:
        args["tools"] = tools
    return chat_template().render(**args)
