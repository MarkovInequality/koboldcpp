#!/usr/bin/env python3
# End-to-end koboldcpp check: fixed greedy prompts against a real model, recording text hashes, draft counts and speeds.
#
#   kcpp-e2e.py record GOLDEN.json [options]   run and write the golden file
#   kcpp-e2e.py check  GOLDEN.json [options]   run and compare: hashes must match, speeds are reported against it
#   kcpp-e2e.py run [options]                  run and print only
#
# --build DIR    tree with koboldcpp.py and koboldcpp_cublas.so (default: this repo)
# --lib FILE     A/B a scratch library: runs a symlinked copy of --build with FILE as koboldcpp_cublas.so
# --configs      comma list of mtp, nomtp, guidance, grammar (default: all)
#
# A config's texts may differ from another config's (MTP verifies batches through other kernels); the cross-config
# checks (guidance and grammar give the same text with and without MTP) accept a divergence only at a near-tie,
# probed with top-2 logprobs at the first differing token.

import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DEFAULT_MODEL = os.path.expanduser("~/AI/qwen3/Qwen3.8-27B-HQ4_K_M.gguf")
LONG_SRC = ("53ed051ce", "src/llama-graph.cpp", 30000)  # pinned, so edits to the tree don't change the prompt
NEAR_TIE = 1e-3

SHORT = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into a timedelta, with tests.",
    "Explain how a B-tree insertion works, step by step, including node splitting.",
    "Write a bash script that finds the 10 largest files under a directory and prints them in human-readable sizes.",
]
GUIDANCE = ("Describe a quiet morning in a small coastal town.", "sad, gloomy, rain")
GRAMMAR = ("List three primary colors as a JSON array of lowercase strings.",
           'root ::= "[" ws item ("," ws item)* ws "]"\nitem ::= "\\"" [a-z]+ "\\""\nws ::= " "?')

def chat(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0, "rep_pen": 1.0, "sampler_seed": 42}
SAMPLED = {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "rep_pen": 1.0, "sampler_seed": 42,
           "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "logit_bias": {"13": -2.0, "271": 1.5}}

SERVERS = {
    "mtp":            ["--usemtp", "--draftamount", "4"],
    "nomtp":          [],
    "guidance-mtp":   ["--usemtp", "--draftamount", "4", "--enableguidance"],
    "guidance-nomtp": ["--enableguidance"],
}

def h(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]

class Server:
    def __init__(self, tree, model, flags, port, log):
        self.port = port
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

    def post(self, path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=1800).read())

    def gen(self, prompt, max_length, params, **extra):
        body = dict(params, prompt=prompt, max_length=max_length, logprobs=True, **extra)
        t = time.time()
        r = self.post("/api/v1/generate", body)
        wall = time.time() - t
        perf = self.post("/api/extra/perf")
        text = r["results"][0]["text"]
        tokens = [c["token"] for c in (r["results"][0].get("logprobs") or {}).get("content", [])]
        return {"hash": h(text), "text": text, "tokens": tokens, "wall": wall, "n_in": perf["last_input_count"],
                "n_out": perf["last_token_count"], "pp_s": perf["last_process_time"], "eval_s": perf["last_eval_time"],
                "draft_ok": perf.get("last_draft_success"), "draft_fail": perf.get("last_draft_failed")}

    def top2_gap(self, prompt, **extra):
        r = self.post("/api/v1/generate", dict({"prompt": prompt, "max_length": 1, "temperature": 1.0, "top_k": 2, "top_p": 1.0,
                                                "rep_pen": 1.0, "sampler_seed": 1, "logprobs": True}, **extra))
        tops = r["results"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = sorted((t["logprob"] for t in tops), reverse=True)
        return lp[0] - lp[1] if len(lp) > 1 else float("inf")

    def stop(self):
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

def run_server(name, args, tree, workdir, res):
    log = os.path.join(workdir, f"{name}.log")
    print(f"== {name}: loading", flush=True)
    s = Server(tree, args.model, SERVERS[name], args.port, log)
    try:
        out = res.setdefault(name, {})
        if name in ("mtp", "nomtp"):
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
            if "grammar" in args.configs:
                r = s.gen(chat(GRAMMAR[0]), 64, GREEDY, grammar=GRAMMAR[1])
                out["grammar"] = r
                print(f"  grammar: [{r['hash']}] {r['text']!r}, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
        else:
            r = s.gen(chat(GUIDANCE[0]), 200, GREEDY, negative_prompt=chat(GUIDANCE[1]), guidance_scale=1.5)
            out["guidance"] = r
            print(f"  guidance: [{r['hash']}] {r['n_out']} tok, drafts {r['draft_ok']}/{r['draft_fail']}", flush=True)
    except Exception:
        s.stop()
        raise
    return s

def same_or_near_tie(server, prompt, a, b, label, **extra):
    if a["hash"] == b["hash"]:
        print(f"  {label}: same text")
        return True
    ta, tb = a["tokens"], b["tokens"]
    k = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
    gap = server.top2_gap(prompt + "".join(ta[:k]), **extra)
    ok = gap < NEAR_TIE
    print(f"  {label}: texts differ at token {k} ({ta[k:k+1]} vs {tb[k:k+1]}), top-2 logprob gap there {gap:.2e} -> "
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
    ap.add_argument("--min-ratio", type=float, default=0.0, help="check: fail when a speed falls below this ratio of the golden")
    ap.add_argument("--out", help="also write this run's results here")
    args = ap.parse_args()
    args.configs = args.configs.split(",")
    if args.mode != "run" and not args.golden:
        ap.error("record and check need a golden file")

    workdir = tempfile.mkdtemp(prefix="kcpp-e2e-")
    tree = farm(os.path.realpath(args.build), args.lib, workdir) if args.lib else os.path.realpath(args.build)
    names = [n for n in ("mtp", "nomtp") if n in args.configs]
    if "guidance" in args.configs:
        names += ["guidance-mtp", "guidance-nomtp"]

    res, ok = {}, True
    for name in names:
        s = run_server(name, args, tree, workdir, res)
        cross = None
        if name == "guidance-nomtp" and "guidance-mtp" in res:
            cross = ("guidance", res["guidance-mtp"]["guidance"], res["guidance-nomtp"]["guidance"], chat(GUIDANCE[0]),
                     {"negative_prompt": chat(GUIDANCE[1]), "guidance_scale": 1.5})
        if name == "nomtp" and "mtp" in res and "grammar" in res["mtp"]:
            cross = ("grammar", res["mtp"]["grammar"], res["nomtp"]["grammar"], chat(GRAMMAR[0]), {"grammar": GRAMMAR[1]})
        try:
            if cross:
                good = same_or_near_tie(s, cross[3], cross[1], cross[2], f"{cross[0]}: MTP vs no MTP", **cross[4])
                res.setdefault("_cross", {})[cross[0]] = good
                ok &= good
        finally:
            s.stop()
    print(f"logs in {workdir}")

    def strip(r):
        return {c: {k: {f: v for f, v in e.items() if f not in ("text", "tokens")} if isinstance(e, dict) else e for k, e in d.items()}
                if isinstance(d, dict) else d for c, d in r.items()}

    if args.out:
        json.dump(strip(res), open(args.out, "w"), indent=1)
    if args.mode == "record":
        json.dump(strip(res), open(args.golden, "w"), indent=1)
        print(f"recorded {args.golden}")
    elif args.mode == "check":
        gold = json.load(open(args.golden))
        for c, d in res.items():
            if c.startswith("_") or c not in gold:
                continue
            for k, r in d.items():
                g = gold[c].get(k)
                if not g:
                    continue
                same = g["hash"] == r["hash"]
                drafts = (g.get("draft_ok"), g.get("draft_fail")) == (r.get("draft_ok"), r.get("draft_fail"))
                if k.startswith("long"):
                    speed, gs = r["n_in"]/r["pp_s"], g["n_in"]/g["pp_s"]
                    what = "prompt t/s"
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
