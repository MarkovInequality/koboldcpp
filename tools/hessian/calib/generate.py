#!/usr/bin/env python3
# Generates the assistant turns for work/gen/prompts.jsonl with koboldcpp (serve.sh gen): each prompt is rendered
# as koboldcpp's format_jinja would, sent to the raw completion endpoint, and the document is prompt + completion
# + <|im_end|>, the text the model processed. Multi-turn conversations are generated turn by turn. Buckets
# (category, source, language) fill to their share of the category's target, calibration and held-out sides
# separately. Resumable: finished documents and drops are appended to work/gen/.

import argparse
import collections
import concurrent.futures as cf
import fcntl
import signal
import threading
import time
import zlib

import kcpp
from common import *

DOCS = GEN / "docs.jsonl"
DROPS = GEN / "drops.jsonl"
COMPLETIONS = GEN / "completions.jsonl"
STATE = GEN / "state.json"

MAX_NEW = 6144
LOOP_SPAN = 64
N_CALIBRATE = 200


def bucket_of(p):
    return (p["category"], p["source"], p.get("lang"))


def key_of(p):
    """calibration buckets per (category, source, language); the held-out side is pooled per category"""
    return ((p["category"], "heldout", None), True) if p["heldout"] else (bucket_of(p), False)


def target_of(p):
    return TARGETS[p["category"]] * p.get("share", 1.0) * (HELDOUT if p["heldout"] else 1.0)


def ratio(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b, 9)))


def has_loop(ids):
    """a LOOP_SPAN-token span that occurs 3 or more times (twice is normal: code drafted while thinking, then given)"""
    seen = {}
    for i in range(0, len(ids) - LOOP_SPAN + 1):
        key = hash(tuple(ids[i:i + LOOP_SPAN]))
        last = seen.get(key)
        if last is None:
            seen[key] = (i, 1)
        elif i - last[0] >= LOOP_SPAN:
            if last[1] + 1 >= 3:
                return True
            seen[key] = (i, last[1] + 1)
    return False


def split_turn(completion, think):
    if think == "off":
        return "", completion.strip()
    if "</think>" not in completion:
        return None
    reasoning, content = completion.split("</think>", 1)
    return reasoning.strip(), content.strip()


class Generator:
    def __init__(self, prompts):
        self.lock = threading.Lock()
        self.docs = read_jsonl(DOCS)
        self.drops = read_jsonl(DROPS)
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {"ratio_threshold": None, "gen_tokens": 0, "gen_seconds": 0.0}
        decontam = set(json.loads((GEN / "decontam.json").read_text())) if (GEN / "decontam.json").exists() else set()
        done = {d["id"] for d in self.docs} | {d["id"] for d in self.drops}
        self.queue = collections.defaultdict(list)
        self.target = collections.Counter()
        seen_buckets = set()
        for p in prompts:
            key = key_of(p)
            if (bucket_of(p), p["heldout"]) not in seen_buckets:
                seen_buckets.add((bucket_of(p), p["heldout"]))
                self.target[key] += target_of(p)
            if p["id"] not in done:
                self.queue[key].append(p)
        self.filled = collections.Counter()
        self.n_docs = collections.Counter()
        for d in self.docs:
            if d["id"] in decontam:
                continue
            key = key_of(d)
            self.filled[key] += d["n_tokens"]
            self.n_docs[key] += 1
        self.inflight = collections.Counter()
        self.inflight_est = {}
        self.draining = False
        self.t0 = time.time()
        self.tokens0 = self.state["gen_tokens"]

    def estimate(self, key):
        return self.filled[key] / self.n_docs[key] if self.n_docs[key] else 2000

    def next_prompt(self):
        with self.lock:
            if self.draining:
                return None
            best, best_fill = None, None
            for key, q in self.queue.items():
                if not q:
                    continue
                t = self.target[key]
                fill = (self.filled[key] + self.inflight[key]) / t
                # start another document only if the bucket then likely ends closer to its target
                if self.filled[key] + self.inflight[key] + self.estimate(key)/2 >= t:
                    continue
                if best is None or fill < best_fill:
                    best, best_fill = key, fill
            if best is None:
                return None
            p = self.queue[best].pop(0)
            self.inflight_est[p["id"]] = self.estimate(best)
            self.inflight[best] += self.inflight_est[p["id"]]
            return p

    def requeue(self, p, at_end=False):
        key = key_of(p)
        with self.lock:
            self.inflight[key] -= self.inflight_est.pop(p["id"])
            if at_end:
                self.queue[key].append(p)
            else:
                self.queue[key].insert(0, p)

    def finish(self, p, doc=None, reason=None, gen_tokens=0):
        key = key_of(p)
        with self.lock:
            self.inflight[key] -= self.inflight_est.pop(p["id"])
            self.state["gen_tokens"] += gen_tokens
            if doc is not None:
                self.docs.append(doc)
                self.filled[key] += doc["n_tokens"]
                self.n_docs[key] += 1
                append_jsonl(DOCS, doc)
                self.calibrate_ratio()
            else:
                d = {"id": p["id"], "category": p["category"], "bucket": list(bucket_of(p)), "reason": reason}
                self.drops.append(d)
                append_jsonl(DROPS, d)
            self.state["gen_seconds"] = self.state.get("gen_seconds_base", 0.0) + time.time() - self.t0
            STATE.write_text(json.dumps(self.state))

    def calibrate_ratio(self):
        """once N_CALIBRATE documents exist, fixes the compression-ratio threshold and applies it to all of them"""
        if self.state["ratio_threshold"] is None and len(self.docs) >= N_CALIBRATE:
            rs = sorted(d["ratio"] for d in self.docs[:N_CALIBRATE])
            self.state["ratio_threshold"] = round(1.5 * rs[int(0.95*(len(rs) - 1))], 3)
            log(f"compression-ratio threshold {self.state['ratio_threshold']}")
            self.recheck()

    def recheck(self):
        th = self.state["ratio_threshold"]
        keep = []
        for d in self.docs:
            if d["ratio"] > th:
                key = key_of(d)
                self.filled[key] -= d["n_tokens"]
                self.n_docs[key] -= 1
                dr = {"id": d["id"], "category": d["category"], "bucket": d["bucket"], "reason": "loop_ratio"}
                self.drops.append(dr)
                append_jsonl(DROPS, dr)
            else:
                keep.append(d)
        if len(keep) != len(self.docs):
            self.docs = keep
            write_jsonl(DOCS, keep)

    def run_prompt(self, p):
        msgs = json.loads(json.dumps(p["messages"]))
        kwargs = kcpp.think_kwargs(p["think"])
        limit = p.get("max_prompt", MAX_LEN - 512)
        turns = [None] + p.get("followups", [])
        doc_text, meta, gen_tokens = None, {"prompt_tokens": [], "completion_tokens": []}, 0
        for i, follow in enumerate(turns):
            if follow is not None:
                msgs.append({"role": "user", "content": follow})
            prompt = kcpp.render(msgs, p.get("tools"), **kwargs)
            n_prompt = len(kcpp.tokenize(prompt))
            if n_prompt > limit:
                if doc_text is None:
                    return None, "prompt_too_long", gen_tokens
                break
            completion, finish, pt, ct = kcpp.complete(prompt, min(MAX_NEW, MAX_LEN - n_prompt))
            gen_tokens += ct
            append_jsonl(COMPLETIONS, {"id": p["id"], "turn": i, "finish": finish, "prompt_tokens": pt, "n_prompt": n_prompt,
                                       "completion_tokens": ct, "completion": completion})
            if pt != n_prompt:
                return None, "token_mismatch", gen_tokens
            if finish not in ("stop", "length"):
                return None, f"finish_{finish}", gen_tokens
            if has_loop(kcpp.tokenize(completion)):
                return None, "loop_span", gen_tokens
            meta["prompt_tokens"].append(n_prompt)
            meta["completion_tokens"].append(ct)
            if finish == "length":
                # capped: kept as a document that ends at the cap, and the conversation ends here
                doc_text = prompt + completion
                meta["truncated"] = True
                break
            parts = split_turn(completion, p["think"])
            if parts is None:
                return None, "no_think_end", gen_tokens
            doc_text = prompt + completion + "<|im_end|>"
            msgs.append({"role": "assistant", "content": parts[1], "reasoning_content": parts[0]})
        return self.make_doc(p, doc_text, prompt, meta), None, gen_tokens

    def make_doc(self, p, doc_text, prompt, meta):
        n_tokens = len(kcpp.tokenize(doc_text))
        if n_tokens > MAX_LEN:
            return None
        doc = {"id": p["id"], "category": p["category"], "unit": p["unit"], "heldout": p["heldout"], "templated": True,
               "bucket": list(bucket_of(p)), "source": p["source"], "think": p["think"], "turns": len(meta["prompt_tokens"]),
               "n_tokens": n_tokens, "ratio": round(ratio(doc_text[len(prompt):]), 4), **meta, "text": doc_text}
        for k in ("lang", "code_lang", "task", "scenario", "numina_source"):
            if p.get(k):
                doc[k] = p[k]
        return doc

    def worker(self):
        while True:
            p = self.next_prompt()
            if p is None:
                return
            try:
                doc, reason, n = self.run_prompt(p)
                if doc is None and reason is None:
                    reason = "too_long"
            except kcpp.ServerDown:
                self.requeue(p)
                kcpp.ensure_server()
                continue
            except Exception as e:
                log(f"{p['id']}: {e!r}")
                p["_errors"] = p.get("_errors", 0) + 1
                if p["_errors"] < 3:
                    self.requeue(p, at_end=True)
                    continue
                doc, reason, n = None, "error", 0
            if doc is not None and self.state["ratio_threshold"] is not None and doc["ratio"] > self.state["ratio_threshold"]:
                doc, reason = None, "loop_ratio"
            self.finish(p, doc, reason, n)

    def report(self):
        by_cat = collections.defaultdict(lambda: [0, 0, 0, 0])
        for (b, h), t in self.target.items():
            by_cat[b[0]][2 + h] += t
        for (b, h), f in self.filled.items():
            by_cat[b[0]][h] += f
        el = time.time() - self.t0
        rate = (self.state["gen_tokens"] - self.tokens0) / max(el, 1)
        s = " | ".join(f"{c} {v[0]/1000:.0f}/{v[2]/1000:.0f}k h {v[1]/1000:.0f}/{v[3]/1000:.0f}k" for c, v in sorted(by_cat.items()))
        log(f"[{el/60:.0f} min, {rate:.1f} tok/s, {len(self.docs)} docs, {len(self.drops)} drops] {s}")


def recover_capped(g, prompts):
    """documents from the capped replies dropped before they were kept, replayed from completions.jsonl"""
    by_id = {p["id"]: p for p in prompts}
    attempts = collections.defaultdict(list)
    for c in read_jsonl(COMPLETIONS):
        if c["turn"] == 0:
            attempts[c["id"]] = []
        attempts[c["id"]].append(c)
    keep_drops, n_rec = [], 0
    for d in g.drops:
        p = by_id.get(d["id"])
        turns = attempts.get(d["id"], [])
        if d["reason"] != "cap" or p is None or not turns or turns[-1]["finish"] != "length":
            keep_drops.append(d)
            continue
        msgs = json.loads(json.dumps(p["messages"]))
        kwargs = kcpp.think_kwargs(p["think"])
        follow = [None] + p.get("followups", [])
        meta = {"prompt_tokens": [], "completion_tokens": [], "truncated": True}
        ok = True
        for i, c in enumerate(turns):
            if follow[i] is not None:
                msgs.append({"role": "user", "content": follow[i]})
            prompt = kcpp.render(msgs, p.get("tools"), **kwargs)
            if len(kcpp.tokenize(prompt)) != c["n_prompt"] or has_loop(kcpp.tokenize(c["completion"])):
                ok = False
                break
            meta["prompt_tokens"].append(c["n_prompt"])
            meta["completion_tokens"].append(c["completion_tokens"])
            if i < len(turns) - 1:
                parts = split_turn(c["completion"], p["think"])
                if parts is None:
                    ok = False
                    break
                msgs.append({"role": "assistant", "content": parts[1], "reasoning_content": parts[0]})
        doc = g.make_doc(p, prompt + turns[-1]["completion"], prompt, meta) if ok else None
        th = g.state["ratio_threshold"]
        if doc is None or (th is not None and doc["ratio"] > th):
            keep_drops.append(d)
            continue
        g.docs.append(doc)
        append_jsonl(DOCS, doc)
        n_rec += 1
    write_jsonl(DROPS, keep_drops)
    log(f"recovered {n_rec} capped replies as documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--categories", default="")
    ap.add_argument("--recover-capped", action="store_true")
    args = ap.parse_args()
    lock = open(GEN / "generate.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another generate.py is running")
    prompts = read_jsonl(GEN / "prompts.jsonl")
    if args.categories:
        cats = set(args.categories.split(","))
        prompts = [p for p in prompts if p["category"] in cats]
    g = Generator(prompts)
    if args.recover_capped:
        recover_capped(g, prompts)
        return
    g.state["gen_seconds_base"] = g.state.get("gen_seconds", 0.0)
    # SIGUSR1: finish the documents in flight, start no new ones, exit
    signal.signal(signal.SIGUSR1, lambda *a: (setattr(g, "draining", True), log("draining")))
    stop = threading.Event()

    def reporter():
        while not stop.wait(300):
            g.report()
    threading.Thread(target=reporter, daemon=True).start()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for f in [ex.submit(g.worker) for _ in range(args.workers)]:
            f.result()
    stop.set()
    g.report()


if __name__ == "__main__":
    main()
