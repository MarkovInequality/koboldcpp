#!/usr/bin/env python3
# End-to-end koboldcpp check: fixed greedy prompts against a real model, recording text hashes, draft counts and speeds.
#
#   kcpp-e2e.py record GOLDEN.json [options]   run and write the golden file
#   kcpp-e2e.py check  GOLDEN.json [options]   run and compare: hashes must match, speeds are reported against it
#   kcpp-e2e.py run [options]                  run and print only
#
# --build DIR    tree with koboldcpp.py and koboldcpp_cublas.so (default: this repo)
# --lib FILE     A/B a scratch library: runs a symlinked copy of --build with FILE as koboldcpp_cublas.so
# --configs      comma list of mtp, nomtp, guidance, grammar, media, deep, agentic, checkpoints (default: the first four)
# --server-args  extra koboldcpp arguments for every server, e.g. "--mtpvocab 65536"
#
# deep: MTP generation after ~32k- and ~96k-token prompts (pinned sources), at a 131072-token context
#
# agentic: a conversation that grows like an agent's: a ~27k-token first prompt, then turns that each append the
# reply and ~3k tokens of new input, 150 tokens generated per turn, SmartCache on as in a real setup; reports each
# turn's wall time (prompt processing, generation and SmartCache snapshots)
#
# media: a server with MTP, the vision projector, whisper and TTS on the GPU; an image description, a TTS clip, a
# transcription, and a TTS clip made while an MTP generation runs (both must equal their solo results)
#
# checkpoints: SmartCache restore points on a hybrid model (MTP, jinja chat completions with tools, 2 slots, the admin
# endpoints): OpenCode-style tool steps, regenerate, a changed generation prompt, a restore to latest - 32 compared
# bitwise, edits near the end and far back, two interleaved conversations, a new conversation on the same system block,
# admin save and load, and tool steps at ~90k tokens; a second server without --smartcache checks the default slot
# count and an image at the end of a prompt (no checkpoint inside it, restores across it equal a cold run). Reports
# processed tokens, slot saves and loads, and wall time per request; the expectations that need check_state's
# checkpoint lists run only on a build that reports them
#
# A config's texts may differ from another config's (MTP verifies batches through other kernels); the cross-config
# checks (guidance and grammar give the same text with and without MTP) and --ref accept a divergence only where the
# two tokens' logprobs are within NEAR_TIE, probed at the first differing token.

import argparse, base64, hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DEFAULT_MODEL = os.path.expanduser("~/AI/qwen3/Qwen3.8-27B-HQ4_K_M.gguf")
MEDIA = {
    "mmproj":       os.path.expanduser("~/AI/qwen3/mmproj-F16.gguf"),
    "whisper":      os.path.expanduser("~/AI/whisper-base.en-q5_1.bin"),
    "tts":          os.path.expanduser("~/AI/qwen3/qwen3-tts-0.6b-q8_0.gguf"),
    "wavtokenizer": os.path.expanduser("~/AI/qwen3/qwen3-tts-tokenizer-q8_0.gguf"),
    "speech":       os.path.expanduser("~/AI/qwen3/Voices/Vivian.wav"),
    "image":        os.path.join(REPO, "media", "preview.png"),
}
TTS_TEXT = "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun."
LONG_SRC = ("53ed051ce", "src/llama-graph.cpp", 30000)  # pinned, so edits to the tree don't change the prompt
DEEP_SRC = ("53ed051ce", ["src/llama-context.cpp", "src/llama-graph.cpp", "ggml/src/ggml.c", "src/llama-vocab.cpp"])
DEEP_CHARS = {"deep32k": 100000, "deep96k": 300000}
AGENTIC_SRC = ("53ed051ce", ["src/llama-model.cpp", "src/llama-vocab.cpp"])
AGENTIC_TURNS, AGENTIC_TURN_CHARS = 5, 10000
# a divergence counts as rounding when, at the first differing token, the two runs' tokens are this close in logprob:
# on the 27B HQ4_K_M a batch-size change alone (1024 -> 512) diverged at gaps of 0.06-0.29 (2026-10-06)
NEAR_TIE = 0.3

SHORT = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into a timedelta, with tests.",
    "Explain how a B-tree insertion works, step by step, including node splitting.",
    "Write a bash script that finds the 10 largest files under a directory and prints them in human-readable sizes.",
]
GUIDANCE = ("Describe a quiet morning in a small coastal town.", "sad, gloomy, rain")
GRAMMAR = ("List three primary colors as a JSON array of lowercase strings.",
           'root ::= "[" ws item ("," ws item)* ws "]"\nitem ::= "\\"" [a-z]+ "\\""\nws ::= " "?')
# long enough for MTP to draft under the grammar
GRAMMAR_LONG = ("Describe three fictional books as a JSON array of objects with a title, an author, a year and a one-sentence summary.",
                'root ::= "[" ws book ("," ws book)* ws "]"\n'
                'book ::= "{" ws "\\"title\\": " str "," ws "\\"author\\": " str "," ws "\\"year\\": " [0-9]+ "," ws "\\"summary\\": " str ws "}"\n'
                'str ::= "\\"" [^"\\\\\\n]* "\\""\n'
                'ws ::= [ \\n]*')

def chat(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0, "rep_pen": 1.0, "sampler_seed": 42}
SAMPLED = {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "rep_pen": 1.0, "sampler_seed": 42,
           "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "logit_bias": {"13": -2.0, "271": 1.5}}
# the other sampler paths, 120 tokens each
SAMPLERS = {
    "mirostat": {"temperature": 0.8, "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "sampler_seed": 42},
    "xtc":      {"temperature": 1.0, "top_k": 40, "top_p": 1.0, "xtc_threshold": 0.1, "xtc_probability": 0.5, "sampler_seed": 42},
    "dynatemp": {"temperature": 0.9, "top_k": 40, "dynatemp_range": 0.4, "dynatemp_exponent": 1.0, "nsigma": 1.5, "sampler_seed": 42},
    "adaptive": {"temperature": 0.8, "top_k": 40, "adaptive_target": 0.4, "adaptive_decay": 0.9, "sampler_seed": 42},
    "bans":     {"temperature": 0.7, "top_k": 40, "rep_pen": 1.1, "rep_pen_range": 256, "ban_eos_token": True,
                 "custom_token_bans": "13,271", "sampler_seed": 42},
}

SERVERS = {
    "mtp":            ["--usemtp", "--draftamount", "4"],
    "nomtp":          [],
    "guidance-mtp":   ["--usemtp", "--draftamount", "4", "--enableguidance"],
    "guidance-nomtp": ["--enableguidance"],
    "deep":           ["--usemtp", "--draftamount", "4", "--contextsize", "131072"],
    "agentic":        ["--usemtp", "--draftamount", "4", "--contextsize", "131072", "--smartcache", "2"],
    "media":          ["--usemtp", "--draftamount", "4", "--mmproj", MEDIA["mmproj"], "--whispermodel", MEDIA["whisper"],
                       "--ttsmodel", MEDIA["tts"], "--ttswavtokenizer", MEDIA["wavtokenizer"], "--ttsgpu"],
    "checkpoints":    ["--usemtp", "--draftamount", "4", "--contextsize", "131072", "--smartcache", "2", "--jinja", "--jinja_tools"],
    # f16 KV: with q5_1 a follow-up after an image differs from the same prompt processed from the start, before
    # checkpoints already (what this server checks is that a checkpoint near an image keeps the text right)
    "checkpoints-default": ["--usemtp", "--draftamount", "4", "--jinja", "--jinja_tools", "--mmproj", MEDIA["mmproj"], "--quantkv", "f16"],
}

CKPT_GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0, "rep_pen": 1.0, "seed": 42}
CKPT_SRC = ("53ed051ce", ["docs/build.md", "src/llama-batch.cpp", "src/llama-context.cpp", "tools/server/README.md"])
CKPT_TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}}
              for n, d, p in [("read_file", "Read a file of the repository.", {"path": {"type": "string", "description": "path from the repository root"}}),
                              ("list_dir", "List a directory of the repository.", {"path": {"type": "string"}}),
                              ("grep", "Search the repository for a regular expression.", {"pattern": {"type": "string"}, "path": {"type": "string"}})]]
CKPT_CAP, CKPT_TAIL = 8, 32
CKPT_CHAINS = {"regenerate": "steps", "changed-tail": "steps", "interleaved": "edit-far", "admin": "same-system"} # prompts made of another's replies

def h(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]

class Server:
    def __init__(self, tree, model, flags, port, log):
        self.port = port
        self.log = log
        cmd = [sys.executable, os.path.join(tree, "koboldcpp.py"), "--model", model, "--usecuda", "normal", "0",
               "--gpulayers", "99", "--contextsize", "32768", "--quantkv", "q5_1", "--batchsize", "1024",
               "--port", str(port), "--quiet", "--skiplauncher", "--nopipelineparallel"] + flags
        self.logf = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, cwd=tree, start_new_session=True)
        for _ in range(600):
            time.sleep(1)
            if self.proc.poll() is not None:
                raise RuntimeError(f"koboldcpp exited during load, see {log}")
            if "Please connect to custom endpoint" in open(log, errors="replace").read():
                return
        raise RuntimeError(f"koboldcpp did not come up, see {log}")

    def post(self, path, body=None, raw=False):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        data = urllib.request.urlopen(req, timeout=1800).read()
        return data if raw else json.loads(data)

    def gen(self, prompt, max_length, params, **extra):
        body = dict(params, prompt=prompt, max_length=max_length, logprobs=True, **extra)
        t = time.monotonic()
        r = self.post("/api/v1/generate", body)
        wall = time.monotonic() - t
        perf = self.post("/api/extra/perf")
        text = r["results"][0]["text"]
        content = (r["results"][0].get("logprobs") or {}).get("content", [])
        tokens = [c["token"] for c in content]
        tops = [{t["token"]: t["logprob"] for t in c.get("top_logprobs", [])} for c in content]
        return {"hash": h(text), "text": text, "tokens": tokens, "tops": tops, "prompt_text": prompt, "probe_extra": extra,
                "greedy": params.get("top_k") == 1, "wall": wall, "n_in": perf["last_input_count"],
                "n_out": perf["last_token_count"], "pp_s": perf["last_process_time"], "eval_s": perf["last_eval_time"],
                "draft_ok": perf.get("last_draft_success"), "draft_fail": perf.get("last_draft_failed")}

    # (tokens, seconds) of the last request's prompt processing from koboldcpp's summary line; the perf endpoint
    # counts reused prompt tokens as processed
    def last_processed(self):
        lines = [l for l in open(self.log, errors="replace").read().splitlines() if "CtxLimit:" in l]
        m = re.search(r"Processed:(\d+) in ([\d.]+)s", lines[-1]) if lines else None
        return (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)

    def mark(self):
        return os.path.getsize(self.log)

    # the slots saved and loaded, and the prompt tokens processed, since mark
    def events(self, mark):
        with open(self.log, "rb") as f:
            f.seek(mark)
            t = f.read().decode(errors="replace")
        saves = [int(m) for m in re.findall(r"KV Save State (\d+): Created SaveState", t)]
        loads = [int(m) for m in re.findall(r"KV Load SaveState (\d+): Restored", t)]
        pp = re.findall(r"Processed:(\d+) in ([\d.]+)s", t)
        return saves, loads, (int(pp[-1][0]), float(pp[-1][1])) if pp else (0, 0.0)

    def _record(self, mark, t0, content):
        wall = time.monotonic() - t0
        perf = self.post("/api/extra/perf")
        saves, loads, (n_pp, pp_s) = self.events(mark)
        ids = [c["token_id"] for c in content]
        return {"hash": h(json.dumps(ids)), "tokens": ids, "tops": [{t["token_id"]: t["logprob"] for t in c["top_logprobs"]} for c in content],
                "greedy": True, "wall": wall, "n_in": perf["last_input_count"], "n_out": len(ids), "n_pp": n_pp, "pp_s": pp_s,
                "eval_s": perf["last_eval_time"], "draft_ok": perf.get("last_draft_success"),
                "draft_fail": perf.get("last_draft_failed"), "saves": saves, "loads": loads}

    def chat(self, messages, max_tokens, **extra):
        body = dict(CKPT_GREEDY, messages=messages, tools=CKPT_TOOLS, max_tokens=max_tokens, logprobs=True, top_logprobs=5, **extra)
        rendered = self.post("/api/extra/tokenize", dict(extra, messages=messages, tools=CKPT_TOOLS))
        mark, t0 = self.mark(), time.monotonic()
        ch = self.post("/v1/chat/completions", body)["choices"][0]
        r = self._record(mark, t0, ch["logprobs"]["content"])
        images = [i["image_url"]["url"].split(",", 1)[1] for m in messages if isinstance(m.get("content"), list)
                  for i in m["content"] if i.get("type") == "image_url"]
        r.update(msg=ch["message"], ids_prompt=rendered["ids"], prompt_text=rendered["prompt"], probe_extra={"images": images} if images else {})
        return r

    def gen_ids(self, prompt, max_length, **extra):
        mark, t0 = self.mark(), time.monotonic()
        res = self.post("/api/v1/generate", dict(GREEDY, prompt=prompt, max_length=max_length, logprobs=True, **extra))["results"][0]
        r = self._record(mark, t0, res["logprobs"]["content"])
        r.update(prompt_text=prompt, probe_extra=extra)
        return r

    def tokenize(self, prompt=None, messages=None, **extra):
        body = dict(extra, prompt=prompt) if messages is None else dict(extra, messages=messages, tools=CKPT_TOOLS)
        return self.post("/api/extra/tokenize", body)["ids"]

    def state(self):
        try:
            return self.post("/api/admin/check_state", {})
        except Exception:
            return None

    # the largest peak resident set of the server's processes (admin mode runs the model in a child)
    def peak_rss_gb(self):
        peak = 0
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if os.getpgid(int(pid)) != self.proc.pid:
                    continue
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmHWM:"):
                        peak = max(peak, int(line.split()[1]))
            except (OSError, ProcessLookupError):
                pass
        return peak / (1 << 20)

    def stop(self):
        print(f"  peak RSS {self.peak_rss_gb():.1f} GB", flush=True)
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.logf.close()

def farm(build, lib, workdir):
    tree = os.path.join(workdir, "tree")
    os.makedirs(tree)
    for name in os.listdir(build):
        if name in ("koboldcpp.py", "koboldcpp_cublas.so"):
            continue
        os.symlink(os.path.join(build, name), os.path.join(tree, name))
    shutil.copy(os.path.join(build, "koboldcpp.py"), tree)  # it finds its library through realpath(__file__)
    os.symlink(os.path.realpath(lib), os.path.join(tree, "koboldcpp_cublas.so"))
    return tree

def long_prompt(run):
    sha, path, n = LONG_SRC
    code = subprocess.run(["git", "-C", REPO, "show", f"{sha}:{path}"], capture_output=True, text=True, check=True).stdout[:n]
    return chat(f"Run {run}. Summarize this code in three sentences:\n{code}")

def deep_prompt(n_chars):
    sha, paths = DEEP_SRC
    text = "".join(subprocess.run(["git", "-C", REPO, "show", f"{sha}:{p}"], capture_output=True, text=True, check=True).stdout
                   for p in paths)[:n_chars]
    return chat(f"Summarize what this code does in three sentences:\n{text}")

def agentic_turns():
    sha, paths = AGENTIC_SRC
    text = "".join(subprocess.run(["git", "-C", REPO, "show", f"{sha}:{p}"], capture_output=True, text=True, check=True).stdout
                   for p in paths)
    return [text[i*AGENTIC_TURN_CHARS:(i+1)*AGENTIC_TURN_CHARS] for i in range(AGENTIC_TURNS)]

# a w x h RGB gradient PNG
def gradient_png(w, h):
    import struct, zlib
    rows = b"".join(b"\0" + bytes(v for x in range(w) for v in (x*255//w, y*255//h, 128)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")

def common_prefix(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)

# the first token where r's text differs from ref_tokens, and the logprob gap there between the two runs' tokens,
# probed with r's prompt and its text up to that token, without samplers or grammar (the logprobs a request reports
# are its samplers' output: greedy keeps one token; a grammar would restart at the probe)
def tie_probe(s, r, ref_tokens):
    ta = r["tokens"]
    k = common_prefix(ta, ref_tokens)
    if k >= min(len(ta), len(ref_tokens)):
        return k, 0.0
    ids = isinstance(ta[0], int)
    prefix = s.post("/api/extra/detokenize", {"ids": ta[:k]})["result"] if ids else "".join(ta[:k])
    extra = {f: v for f, v in r.get("probe_extra", {}).items() if f != "grammar"}
    res = s.post("/api/v1/generate", dict({"prompt": r["prompt_text"] + prefix, "max_length": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                                           "min_p": 0.0, "rep_pen": 1.0, "sampler_seed": 1, "logprobs": True}, **extra))
    tops = res["results"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {(t["token_id"] if ids else t["token"]): t["logprob"] for t in tops}
    floor = min(lp.values())
    return k, abs(lp.get(ta[k], floor) - lp.get(ref_tokens[k], floor))

# each request whose text differs from the reference run's: the first differing token and the top-2 gap there; a
# greedy request must diverge at a near-tie, a sampled one is reported only. A conversation is probed at its first
# divergence (later prompts contain the diverged replies).
def probe_against(s, name, out, ref, res):
    diverged = set()
    for k, r in out.items():
        g = ref.get(k)
        if not isinstance(r, dict) or not isinstance(g, dict) or "hash" not in r or r["hash"] == g["hash"] or not g.get("tokens"):
            continue
        chain = name if name == "agentic" else CKPT_CHAINS.get(k.split("/")[0], k.split("/")[0]) if name.startswith("checkpoints") else None
        if chain in diverged:
            continue
        if chain:
            diverged.add(chain)
        i, gap = tie_probe(s, r, g["tokens"])
        r["tie"] = [i, gap]
        good = gap < NEAR_TIE or not r.get("greedy")
        print(f"  {k}: differs from the reference at token {i}, logprob gap {gap:.2e}"
              f"{'' if r.get('greedy') else ' (sampled)'}{'' if good else '  FAIL'}", flush=True)
        if not good:
            res.setdefault("_cross", {})[f"{name}/{k}: not a near-tie"] = False

def reply_msg(m):
    a = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content") is not None:
        a["reasoning_content"] = m["reasoning_content"]
    if m.get("tool_calls"):
        a["tool_calls"] = m["tool_calls"]
    return a

# the checkpoint list policy (otherarch/kcpp_smartcache.h): a cut drops checkpoints past the restore point r; a
# request adds the system position S if r < S < L, L - 32 if more than 32 tokens are new and L - 32 > S, and L; past
# the capacity it evicts, sparing the 3 newest and the system checkpoint (or the oldest), first those before 40 % of
# L (earliest first), else the one whose neighbors are closest (older first on ties)
def ckpt_predict(lst, serial, r, L, S):
    lst[:] = [c for c in lst if c[0] <= r]
    if L <= CKPT_TAIL:
        return serial
    def add(pos, kind):
        nonlocal serial
        serial += 1
        lst[:] = [c for c in lst if c[0] != pos] + [(pos, serial, kind)]
        if len(lst) <= CKPT_CAP:
            return
        by_serial = sorted(lst, key=lambda c: c[1])
        keep = {c[1] for c in by_serial[-3:]}
        sysc = [c for c in lst if c[2] == "system"]
        keep.add(sysc[0][1] if sysc else by_serial[0][1])
        cand = [c for c in lst if c[1] not in keep]
        below = [c for c in cand if c[0]*10 < L*4]
        if below:
            lst.remove(min(below))
            return
        byp = sorted(lst)
        gap = lambda c: (byp[byp.index(c)+1][0] if byp.index(c)+1 < len(byp) else L) - (byp[byp.index(c)-1][0] if byp.index(c) else 0)
        lst.remove(min(cand, key=lambda c: (gap(c), c[1])))
    if S and r < S < L:
        add(S, "system")
    if L - r > CKPT_TAIL and L - CKPT_TAIL > (S or 0):
        add(L - CKPT_TAIL, "tail")
    if L > r:
        add(L, "latest")
    else:
        lst[:] = [(p, serial + 1 if p == L else q, k) for p, q, k in lst]
        serial += 1
    lst.sort()
    return serial

def run_checkpoints(s, name, out, res):
    sha, paths = CKPT_SRC
    docs, batch_cpp, ctx_cpp, server_md = (subprocess.run(["git", "-C", REPO, "show", f"{sha}:{p}"], capture_output=True, text=True,
                                                          check=True).stdout for p in paths)
    # what the build reports: the live context's checkpoints, then the slots' (an expectation runs only on a build
    # that has what it checks)
    st = s.state()
    have = {None} | ({"ckpt"} if st and "checkpoints" in st else set()) | ({"slots"} if st and "slots" in st else set())
    new = "ckpt" in have
    print(f"  check_state reports: {', '.join(sorted(h for h in have if h)) or 'no checkpoint lists'}", flush=True)

    def expect(label, cond, detail="", needs="ckpt"):
        if needs not in have:
            return
        print(f"    {label}: {detail}  {'ok' if cond else 'FAIL'}", flush=True)
        if not cond:
            res.setdefault("_cross", {})[f"{name}: {label}"] = False

    def live():
        st = s.state() if new else None
        return [tuple(c) for c in st["checkpoints"]] if st else None

    def ask(key, msgs, max_tokens, **kw):
        r = s.chat(msgs, max_tokens, **kw)
        r["ckpts"], r["r"] = live(), r["n_in"] - r["n_pp"]
        out[key] = r
        print(f"  {key}: [{r['hash']}] {r['n_in']} tok, processed {r['n_pp']} in {r['pp_s']:.2f}s, {r['n_out']} out in {r['eval_s']:.2f}s, "
              f"wall {r['wall']:.2f}s (other {r['wall'] - r['pp_s'] - r['eval_s']:.2f}s), saves {r['saves']} loads {r['loads']}, "
              f"drafts {r['draft_ok']}/{r['draft_fail']}" + (f", checkpoints {[c[0] for c in r['ckpts']]}" if r["ckpts"] is not None else ""),
              flush=True)
        return r

    # a request that continues prev's conversation: it continues the live context if the reply re-renders into the
    # generated tokens, else it restores prev's latest checkpoint
    def continues(label, prev, r, slot_switch=False):
        full = prev["ids_prompt"] + prev["tokens"]
        d = common_prefix(full, r["ids_prompt"])
        if d >= len(full):
            want, how = r["n_in"] - (prev["n_in"] + prev["n_out"]) + 1, "reply re-rendered as generated: continues"
        else:
            want, how = r["n_in"] - prev["n_in"], f"re-rendered reply differs at its token {d - len(prev['ids_prompt'])}: restores latest"
        expect(f"{label}", r["n_pp"] == want, f"{how}, processed {r['n_pp']} (want {want})",
               needs="slots" if slot_switch else "ckpt" if d < len(full) else None)
        if not slot_switch:
            expect(f"{label}: no slot saved or loaded", not r["saves"] and not r["loads"], f"saves {r['saves']} loads {r['loads']}")

    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    if name == "checkpoints-default":
        st = s.state()
        print(f"  slots without --smartcache: {len(st['old_states']) if st else '?'}", flush=True)
        expect("default slot count", st is not None and len(st["old_states"]) == 2, f"{len(st['old_states']) if st else '?'} slots", needs="slots")
        # 32 x 24 vision positions at the end of the prompt, then 9 tokens: latest - 32 falls inside the image
        img = base64.b64encode(gradient_png(1024, 768)).decode()
        E = [{"role": "system", "content": "You describe images for a project's documentation."},
             {"role": "user", "content": [{"type": "text", "text": docs[:6000] + "\n\nThe image below is from this project."},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}]
        e1 = ask("media/e1", E, 64, **nothink)
        expect("media: no checkpoint inside the image", e1["ckpts"] is not None and all(k != "tail" for _, k in e1["ckpts"]),
               f"checkpoints {e1['ckpts']} for {e1['n_in']} tokens")
        r = ask("media/regenerate", E, 64, **nothink)
        expect("media: regenerate restores latest after the image", r["n_pp"] == 0 and r["hash"] == e1["hash"], f"processed {r['n_pp']}")
        E2 = E + [reply_msg(e1["msg"]), {"role": "user", "content": "Which colors dominate the image?"}]
        e2 = ask("media/e2", E2, 64, **nothink)
        s.post("/api/v1/generate", dict(GREEDY, prompt="Unrelated text about cats.", max_length=4))
        s.post("/api/admin/clear_state", {})
        cold = ask("media/e2-cold", E2, 64, **nothink)
        # the tools block is a shared prefix: this one restores the other conversation's checkpoint, then the image
        ask("media/other", [{"role": "user", "content": "Say hello."}], 8, **nothink)
        s.post("/api/admin/clear_state", {})
        r = ask("media/e2-restored", E2, 64, **nothink)
        k, gap = tie_probe(s, r, cold["tokens"])
        expect("media: restored from a checkpoint before the image, equals a cold run", 0 < r["r"] < r["n_in"] and gap < NEAR_TIE,
               f"restore point {r['r']}, " + ("same text" if r["hash"] == cold["hash"] else f"differs at token {k}, top-2 gap {gap:.2e}"))
        k, gap = tie_probe(s, e2, cold["tokens"])
        print(f"    (report only) media: the follow-up continuing the live context after an MTP generation "
              f"{'equals the cold run' if e2['hash'] == cold['hash'] else f'differs from the cold run at token {k}, top-2 gap {gap:.2e}'}", flush=True)
        return

    # OpenCode-style tool steps: the reply goes back with its reasoning and tool calls, then the tool results
    sys_a = ("You are a coding agent in the llama.cpp repository. Inspect files with the tools before you answer, and keep "
             "answers short.\n\n# Build notes\n" + docs[:9000])
    A = [{"role": "system", "content": sys_a},
         {"role": "user", "content": "How does llama_batch_allocr split a batch into ubatches? Read src/llama-batch.cpp first."}]
    chunk = lambda i: batch_cpp[i*6000:(i+1)*6000]
    prev = None
    for k in range(4):
        r = ask(f"steps/{k}", A, 300, reasoning_effort="medium")
        if prev:
            continues(f"steps/{k}", prev, r)
        last, prev = list(A), r
        A = A + [reply_msg(r["msg"])]
        A += ([{"role": "tool", "tool_call_id": tc.get("id", ""), "content": chunk(k)} for tc in r["msg"]["tool_calls"]]
              if r["msg"].get("tool_calls") else [{"role": "user", "content": "Here is more of it:\n" + chunk(k)}])

    r = ask("regenerate", last, 300, reasoning_effort="medium")
    expect("regenerate restores latest, nothing to decode", r["n_pp"] == 0, f"processed {r['n_pp']}")
    expect("regenerate: no slot saved or loaded", not r["saves"] and not r["loads"], f"saves {r['saves']} loads {r['loads']}", needs="slots")
    expect("regenerate: same text and drafts", r["hash"] == prev["hash"] and (r["draft_ok"], r["draft_fail"]) == (prev["draft_ok"], prev["draft_fail"]),
           f"drafts {r['draft_ok']}/{r['draft_fail']} vs {prev['draft_ok']}/{prev['draft_fail']}, text {'same' if r['hash'] == prev['hash'] else 'differs'}")

    L = prev["n_in"]
    r = ask("changed-tail", last, 120, reasoning_effort="medium", **nothink)
    d = common_prefix(prev["ids_prompt"], r["ids_prompt"]) - (len(prev["ids_prompt"]) - L)
    expect("changed tail restores latest - 32", d >= L - CKPT_TAIL and r["r"] == L - CKPT_TAIL, f"diverges at {d} of {L}, restore point {r['r']}")
    expect("changed tail: no slot saved or loaded", not r["saves"] and not r["loads"], f"saves {r['saves']} loads {r['loads']}", needs="slots")

    # a restore to latest - 32 that decodes the same last 32 tokens as the request that made it
    P = chat("Answer in one sentence: what does this code do?\n" + ctx_cpp[:8000])
    ip, iy = s.tokenize(prompt=P), s.tokenize(prompt=P[:-1])
    expect("bitwise: the second prompt differs in its last token only", len(ip) == len(iy) and ip[:-1] == iy[:-1] and ip[-1] != iy[-1],
           f"{len(ip)} and {len(iy)} tokens", needs=None)
    for key, prompt in (("bitwise/x", P), ("bitwise/y", P[:-1]), ("bitwise/z", P)):
        r = s.gen_ids(prompt, 48)
        r["ckpts"], r["r"] = live(), r["n_in"] - r["n_pp"]
        out[key] = r
        print(f"  {key}: [{r['hash']}] {r['n_in']} tok, processed {r['n_pp']}, wall {r['wall']:.2f}s, saves {r['saves']} loads {r['loads']}, "
              f"drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
    x, y, z = out["bitwise/x"], out["bitwise/y"], out["bitwise/z"]
    expect("bitwise: both later requests restore latest - 32", y["n_pp"] == CKPT_TAIL and z["n_pp"] == CKPT_TAIL,
           f"processed {y['n_pp']} and {z['n_pp']}")
    expect("bitwise: logits at L equal the first request's", z["tops"][0] == x["tops"][0], f"top-5 {'equal' if z['tops'][0] == x['tops'][0] else 'differ'}")
    expect("bitwise: same text and drafts", z["hash"] == x["hash"] and (z["draft_ok"], z["draft_fail"]) == (x["draft_ok"], x["draft_fail"]),
           f"drafts {z['draft_ok']}/{z['draft_fail']} vs {x['draft_ok']}/{x['draft_fail']}")

    # an edit near the end keeps >= 70 % of the live context: no save, the latest checkpoint before the edit
    B = [{"role": "system", "content": "You are a helpful assistant reviewing documentation. Answer in two sentences."}]
    for k in range(4):
        B.append({"role": "user", "content": f"Part {k+1} of the server's README:\n{server_md[k*4000:(k+1)*4000]}\nWhat does this part describe?"})
        r = ask(f"edit-last/{k}", B, 48, **nothink)
        B.append(reply_msg(r["msg"]))
    B = B[:-2] + [dict(B[-2], content=B[-2]["content"].replace("What does this part describe?", "Which options does this part list?"))]
    before = r["ckpts"]
    r = ask("edit-last/edit", B, 48, **nothink)
    d = common_prefix(out["edit-last/3"]["ids_prompt"], r["ids_prompt"]) - (len(r["ids_prompt"]) - r["n_in"])
    want = max((p for p, _ in before or [] if p <= d), default=0)
    expect("edit-last: the latest checkpoint before the edit", r["r"] == want, f"edit at {d}, restore point {r['r']} (want {want})")
    expect("edit-last: keeps >= 70 %, no slot saved or loaded", not r["saves"] and not r["loads"], f"saves {r['saves']} loads {r['loads']}", needs="slots")

    # an edit far back keeps < 70 %: the old version is saved, then the latest checkpoint before the edit
    C = [{"role": "system", "content": "You are a careful technical writer. Answer in two sentences.\n\n# Reference\n" + docs[9000:15000]}]
    for k in range(8):
        C.append({"role": "user", "content": f"Section {k+1}:\n{ctx_cpp[k*5000:(k+1)*5000]}\nSummarize this section."})
        r = ask(f"edit-far/{k}", C, 48, **nothink)
        C.append(reply_msg(r["msg"]))
    if new:
        S = next((p for p, kind in r["ckpts"] if kind == "system"), None)
        pred, serial = [], 0
        for k in range(8):
            q = out[f"edit-far/{k}"]
            serial = ckpt_predict(pred, serial, q["r"], q["n_in"], S)
        expect("edit-far: checkpoint positions follow the eviction rule", [c[0] for c in pred] == [c[0] for c in r["ckpts"]],
               f"{[c[0] for c in r['ckpts']]} (predicted {[c[0] for c in pred]})")
    before = r["ckpts"]
    C[9] = dict(C[9], content=C[9]["content"].replace("Summarize this section.", "List the functions this section defines."))
    C.pop()
    e = ask("edit-far/edit", C, 48, **nothink)
    d = common_prefix(out["edit-far/7"]["ids_prompt"], e["ids_prompt"]) - (len(e["ids_prompt"]) - e["n_in"])
    want = max((p for p, _ in before or [] if p <= d), default=0)
    expect("edit-far: the latest checkpoint before the edit", e["r"] == want, f"edit at {d} of {r['n_in'] + r['n_out']}, restore point {e['r']} (want {want})")
    expect("edit-far: keeps < 70 %, the old version saved", len(e["saves"]) == 1 and not e["loads"], f"saves {e['saves']} loads {e['loads']}", needs="slots")

    # two interleaved conversations: each switch saves over the leaving conversation's older snapshot
    s.post("/api/admin/clear_state", {})
    M = C + [reply_msg(e["msg"])]
    O = [{"role": "system", "content": "You translate English technical text into French."}, {"role": "user", "content": docs[15000:17000]}]
    want = [("o1", [0], []), ("m2", [1], [0]), ("o2", [0], [1]), ("m3", [1], [0])]
    last_of = {"m": e}
    for key, saves, loads in want:
        conv = M if key[0] == "m" else O
        if key != "o1":
            conv.append({"role": "user", "content": "Continue with the next part, in two sentences." if key[0] == "m" else docs[17000:18000]})
        r = ask(f"interleaved/{key}", conv, 48, **nothink)
        expect(f"interleaved/{key}: saves {saves}, loads {loads}", (r["saves"], r["loads"]) == (saves, loads), f"saves {r['saves']} loads {r['loads']}",
               needs="slots")
        if key[0] in last_of:
            continues(f"interleaved/{key}", last_of[key[0]], r, slot_switch=True)
        last_of[key[0]] = r
        conv.append(reply_msg(r["msg"]))

    # a new conversation on the same system block restores the system checkpoint, which sits right after the
    # rendered system block
    S = next((p for p, kind in (r["ckpts"] or []) if kind == "system"), None)
    if S is not None:
        im_end, off = s.tokenize(prompt="<|im_end|>")[-1], len(s.tokenize(prompt=""))
        want = r["ids_prompt"].index(im_end) + 2 - off
        expect("system checkpoint at the end of the system block", S == want, f"at {S}, the system block ends at {want}", needs="slots")
    N = [C[0], {"role": "user", "content": "Explain the build options in two sentences."}]
    n1 = ask("same-system", N, 48, **nothink)
    if S is None:
        print("    same-system: no system checkpoint reported, skipped", flush=True)
    else:
        expect("same-system: the previous conversation saved, the system checkpoint restored", n1["r"] == S and len(n1["saves"]) == 1,
               f"restore point {n1['r']} (system at {S}), saves {n1['saves']}", needs="slots")

    # admin save and load by slot number carry the checkpoints
    s.post("/api/admin/save_state", {"slot": 1})
    ask("admin/other", [{"role": "user", "content": "Say hello in French."}], 16, **nothink)
    s.post("/api/admin/load_state", {"slot": 1})
    st = s.state()
    if "slots" in have:
        expect("admin: a loaded slot brings its checkpoints", st["checkpoints"] == st["old_states"][1]["checkpoints"] and
               [tuple(c) for c in st["checkpoints"]] == n1["ckpts"], f"live {st['checkpoints']}, slot 1 {st['old_states'][1]['checkpoints']}",
               needs="slots")
    r = ask("admin/regenerate", N, 48, **nothink)
    expect("admin: regenerate after the load restores latest", r["n_pp"] == 0 and r["hash"] == n1["hash"], f"processed {r['n_pp']}", needs="slots")

    # /api/v1/generate: the memory is the system part
    mem = "Notes on building llama.cpp:\n" + docs[40000:46000]
    r = s.gen_ids(chat("Summarize the notes above in one sentence."), 32, memory=mem)
    r["ckpts"] = live()
    out["memory"] = r
    want = len(s.tokenize(prompt=mem)) - len(s.tokenize(prompt=""))
    print(f"  memory: [{r['hash']}] {r['n_in']} tok, processed {r['n_pp']}, checkpoints {r['ckpts']}", flush=True)
    if any(kind == "system" for _, kind in r["ckpts"] or []):
        expect("memory: a system checkpoint at the memory's end", (want, "system") in r["ckpts"], f"want {want}", needs="slots")
    else:
        print("    memory: no system checkpoint reported, skipped", flush=True)

    # OpenCode-style steps at ~90k tokens
    D = [{"role": "system", "content": sys_a},
         {"role": "user", "content": "Here are three files of the repository.\n" + server_md + ctx_cpp + docs[:40000] +
                                     "\n\nWhere is the KV cache cleared? Use grep to confirm."}]
    prev = None
    for k in range(3):
        r = ask(f"deep-steps/{k}", D, 200, reasoning_effort="medium")
        if prev:
            continues(f"deep-steps/{k}", prev, r)
        prev = r
        D = D + [reply_msg(r["msg"])]
        D += ([{"role": "tool", "tool_call_id": tc.get("id", ""), "content": chunk(k)} for tc in r["msg"]["tool_calls"]]
              if r["msg"].get("tool_calls") else [{"role": "user", "content": "Now grep for seq_rm:\n" + chunk(k)}])

def run_server(name, args, tree, workdir, res):
    log = os.path.join(workdir, f"{name}.log")
    print(f"== {name}: loading", flush=True)
    flags = SERVERS[name] + args.server_args.split()
    if name.startswith("checkpoints"):
        os.makedirs(os.path.join(workdir, "admin"), exist_ok=True)
        flags += ["--admin", "--admindir", os.path.join(workdir, "admin")]
    s = Server(tree, args.model, flags, args.port, log)
    try:
        out = res.setdefault(name, {})
        if name.startswith("checkpoints"):
            run_checkpoints(s, name, out, res)
        elif name == "deep":
            for k, n in DEEP_CHARS.items():
                r = s.gen(deep_prompt(n), 200, GREEDY)
                out[k] = r
                print(f"  {k}: [{r['hash']}] {r['n_in']} tok, process {r['pp_s']:.1f}s -> {r['n_in']/r['pp_s']:.0f} t/s; "
                      f"{r['n_out']} tok -> {r['n_out']/r['eval_s']:.1f} t/s, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
        elif name == "agentic":
            prompt = deep_prompt(DEEP_CHARS["deep32k"])
            for i, chunk in enumerate([None] + agentic_turns()):
                if chunk is not None:
                    prompt += f"<|im_end|>\n<|im_start|>user\nNow this part, in three sentences:\n{chunk}<|im_end|>\n" \
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                r = s.gen(prompt, 150, GREEDY)
                r["n_pp"], r["pp_s"] = s.last_processed()
                prompt += r["text"]
                out[f"turn{i}"] = r
                print(f"  turn{i}: [{r['hash']}] {r['n_in']} tok, wall {r['wall']:.2f}s: processed {r['n_pp']} in {r['pp_s']:.2f}s, "
                      f"{r['n_out']} tok in {r['eval_s']:.2f}s, other {r['wall'] - r['pp_s'] - r['eval_s']:.2f}s, "
                      f"drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
        elif name == "media":
            img = base64.b64encode(open(MEDIA["image"], "rb").read()).decode()
            r = s.gen(chat("Describe this image in two sentences."), 80, GREEDY, images=[img])
            out["vision"] = r
            print(f"  vision: [{r['hash']}] {r['text'][:70]!r}", flush=True)
            tts = lambda: s.post("/api/extra/tts", {"input": TTS_TEXT, "voice": "vivian", "seed": 42}, raw=True)
            wav = tts()
            out["tts"] = {"hash": hashlib.md5(wav).hexdigest()[:12], "bytes": len(wav)}
            print(f"  tts: [{out['tts']['hash']}] {len(wav)} bytes", flush=True)
            speech = base64.b64encode(open(MEDIA["speech"], "rb").read()).decode()
            t = s.post("/api/extra/transcribe", {"audio_data": speech})
            out["transcribe"] = {"hash": h(t.get("text", "")), "text": t.get("text", "")}
            print(f"  transcribe: [{out['transcribe']['hash']}] {t.get('text', '')[:70]!r}", flush=True)
            solo = s.gen(chat(SHORT[1]), 200, GREEDY)
            both = {}
            th = threading.Thread(target=lambda: both.update(gen=s.gen(chat(SHORT[1]), 200, GREEDY)))
            th.start()
            time.sleep(1.0)
            wav2 = tts()
            th.join()
            same = both["gen"]["hash"] == solo["hash"] and hashlib.md5(wav2).hexdigest()[:12] == out["tts"]["hash"]
            out["tts+gen"] = {"hash": h(both["gen"]["hash"] + hashlib.md5(wav2).hexdigest()[:12]), "same_as_solo": same}
            print(f"  tts during MTP generation: {'same as solo' if same else 'DIFFERS from solo'} "
                  f"(gen [{both['gen']['hash']}] vs [{solo['hash']}])", flush=True)
            if not same:
                res.setdefault("_cross", {})["tts+gen"] = False
        elif name in ("mtp", "nomtp"):
            for run in range(2):
                r = s.gen(long_prompt(run), 64, GREEDY)
                out[f"long{run}"] = r
                print(f"  long{run}: [{r['hash']}] {r['n_in']} tok, process {r['pp_s']:.2f}s -> {r['n_in']/r['pp_s']:.0f} t/s, wall {r['wall']:.2f}s", flush=True)
            for i, p in enumerate(SHORT):
                r = s.gen(chat(p), 400, GREEDY)
                out[f"short{i}"] = r
                print(f"  short{i}: [{r['hash']}] {r['n_out']} tok -> {r['n_out']/r['eval_s']:.1f} t/s, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
            r = s.gen(chat(SHORT[0]), 200, SAMPLED)
            out["sampled"] = r
            print(f"  sampled: [{r['hash']}] {r['n_out']} tok -> {r['n_out']/r['eval_s']:.1f} t/s, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
            for sname, sp in SAMPLERS.items():
                r = s.gen(chat(SHORT[2]), 120, sp)
                out[sname] = r
                print(f"  {sname}: [{r['hash']}] {r['n_out']} tok, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
            if "grammar" in args.configs:
                r = s.gen(chat(GRAMMAR[0]), 64, GREEDY, grammar=GRAMMAR[1])
                out["grammar"] = r
                print(f"  grammar: [{r['hash']}] {r['text']!r}, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
                r = s.gen(chat(GRAMMAR_LONG[0]), 300, GREEDY, grammar=GRAMMAR_LONG[1])
                out["grammar_long"] = r
                print(f"  grammar_long: [{r['hash']}] {r['n_out']} tok -> {r['n_out']/r['eval_s']:.1f} t/s, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
        else:
            r = s.gen(chat(GUIDANCE[0]), 200, GREEDY, negative_prompt=chat(GUIDANCE[1]), guidance_scale=1.5)
            out["guidance"] = r
            print(f"  guidance: [{r['hash']}] {r['n_out']} tok, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
        if args.ref_data and name in args.ref_data:
            probe_against(s, name, out, args.ref_data[name], res)
    except Exception:
        s.stop()
        raise
    return s

def same_or_near_tie(server, prompt, a, b, label, **extra):
    if a["hash"] == b["hash"]:
        print(f"  {label}: same text")
        return True
    ta, tb = a["tokens"], b["tokens"]
    k, gap = tie_probe(server, dict(b, prompt_text=prompt, probe_extra=extra), ta)
    ok = gap < NEAR_TIE
    print(f"  {label}: texts differ at token {k} ({ta[k:k+1]} vs {tb[k:k+1]}), logprob gap there {gap:.2e} -> "
          f"{'near-tie, accepted' if ok else 'FAIL'}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check", "run"])
    ap.add_argument("golden", nargs="?")
    ap.add_argument("--build", default=REPO)
    ap.add_argument("--lib")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--configs", default="mtp,nomtp,guidance,grammar")
    ap.add_argument("--port", type=int, default=5098)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--min-ratio", type=float, default=0.0, help="check: fail when a speed falls below this ratio of the golden")
    ap.add_argument("--out", help="also write this run's results here")
    ap.add_argument("--keep-tokens", action="store_true", help="--out keeps every request's tokens")
    ap.add_argument("--ref", help="a --keep-tokens run of another build: where a text differs from it, probe for a near-tie")
    args = ap.parse_args()
    args.configs = args.configs.split(",")
    args.ref_data = json.load(open(args.ref)) if args.ref else None
    if args.mode != "run" and not args.golden:
        ap.error("record and check need a golden file")

    workdir = tempfile.mkdtemp(prefix="kcpp-e2e-")
    tree = farm(os.path.realpath(args.build), args.lib, workdir) if args.lib else os.path.realpath(args.build)
    names = [n for n in ("mtp", "nomtp") if n in args.configs]
    if "guidance" in args.configs:
        names += ["guidance-mtp", "guidance-nomtp"]
    if "media" in args.configs:
        names += ["media"]
    names += [n for n in ("deep", "agentic") if n in args.configs]
    if "checkpoints" in args.configs:
        names += ["checkpoints", "checkpoints-default"]

    res, ok = {}, True
    for name in names:
        s = run_server(name, args, tree, workdir, res)
        cross = None
        if name == "guidance-nomtp" and "guidance-mtp" in res:
            cross = ("guidance", res["guidance-mtp"]["guidance"], res["guidance-nomtp"]["guidance"], chat(GUIDANCE[0]),
                     {"negative_prompt": chat(GUIDANCE[1]), "guidance_scale": 1.5})
        crosses = [cross] if cross else []
        if name == "nomtp" and "mtp" in res and "grammar" in res["mtp"]:
            crosses = [("grammar", res["mtp"]["grammar"], res["nomtp"]["grammar"], chat(GRAMMAR[0]), {"grammar": GRAMMAR[1]})]
            if "grammar_long" in res["mtp"]:
                crosses.append(("grammar_long", res["mtp"]["grammar_long"], res["nomtp"]["grammar_long"], chat(GRAMMAR_LONG[0]),
                                {"grammar": GRAMMAR_LONG[1]}))
        try:
            for cross in crosses:
                good = same_or_near_tie(s, cross[3], cross[1], cross[2], f"{cross[0]}: MTP vs no MTP", **cross[4])
                if cross[0] == "grammar_long" and cross[1]["draft_ok"] == 0:
                    print("  grammar_long: MTP did not draft under the grammar  FAIL")
                    good = False
                res.setdefault("_cross", {})[cross[0]] = good
                ok &= good
        finally:
            s.stop()
    ok &= all(res.get("_cross", {}).values())
    print(f"logs in {workdir}")

    # the checkpoints configs keep their token ids: a later build's texts may differ at a near-tie (the cut before the
    # last 32 tokens changes batch shapes), which the check finds from the ids
    def strip(r):
        drop = lambda c, f: f in ("text", "msg", "ids_prompt", "tops", "prompt_text", "probe_extra") or (f == "tokens" and not keep and not c.startswith("checkpoints"))
        return {c: {k: {f: v for f, v in e.items() if not drop(c, f)} if isinstance(e, dict) else e for k, e in d.items()}
                if isinstance(d, dict) else d for c, d in r.items()}

    if args.out:
        keep = args.keep_tokens
        json.dump(strip(res), open(args.out, "w"), indent=1)
    keep = False
    if args.mode == "record":
        json.dump(strip(res), open(args.golden, "w"), indent=1)
        print(f"recorded {args.golden}")
    elif args.mode == "check":
        gold = json.load(open(args.golden))
        for c, d in res.items():
            if c.startswith("_") or c not in gold:
                continue
            if c.startswith("checkpoints"):
                diverged = set()
                for k, r in d.items():
                    g = gold[c].get(k)
                    if not g or "n_pp" not in r:
                        continue
                    conv = CKPT_CHAINS.get(k.split("/")[0], k.split("/")[0])
                    if r["hash"] == g["hash"]:
                        what = "same"
                    elif conv in diverged:
                        what = "after a divergence"
                    else:
                        diverged.add(conv)
                        what = f"DIFF at token {r['tie'][0]}, top-2 gap {r['tie'][1]:.1e}" if r.get("tie") else "DIFF (no --ref probe)  FAIL"
                        ok &= bool(r.get("tie")) and r["tie"][1] < NEAR_TIE
                    print(f"  {c:19s} {k:22s} {what}; processed {r['n_pp']:6d} vs {g['n_pp']:6d}, wall {r['wall']:6.2f}s vs {g['wall']:6.2f}s, "
                          f"saves {r['saves']} vs {g['saves']}, loads {r['loads']} vs {g['loads']}, drafts {r['draft_ok']}/{r['draft_fail']} vs "
                          f"{g['draft_ok']}/{g['draft_fail']}")
                continue
            for k, r in d.items():
                g = gold[c].get(k)
                if not g:
                    continue
                same = g["hash"] == r["hash"]
                drafts = (g.get("draft_ok"), g.get("draft_fail")) == (r.get("draft_ok"), r.get("draft_fail"))
                if "n_out" not in r:
                    ok &= same
                    print(f"  {c:15s} {k:10s} {'same' if same else 'DIFF'} hash{'' if same else '  FAIL'}")
                    continue
                if k.startswith("long"):
                    speed, gs = r["n_in"]/r["pp_s"], g["n_in"]/g["pp_s"]
                    what = "prompt t/s"
                elif k.startswith("turn"):
                    speed, gs = 1/r["wall"], 1/g["wall"]
                    what = "turns/s"
                else:
                    speed, gs = r["n_out"]/r["eval_s"], g["n_out"]/g["eval_s"]
                    what = "gen t/s"
                ratio = speed/gs
                bad = not same or (args.min_ratio and ratio < args.min_ratio)
                ok &= not bad
                print(f"  {c:15s} {k:8s} {'same' if same else 'DIFF'} hash, drafts {'same' if drafts else 'differ'}, "
                      f"{what} {speed:8.1f} vs {gs:8.1f} ({ratio:5.3f}x){'  FAIL' if bad else ''}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
