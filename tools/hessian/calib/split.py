#!/usr/bin/env python3
# Merges the generated, raw and OpenCode documents, drops the decontamination hits, and writes the kept
# dataset qwen38-calib-v1 to tools/hessian/calib/dataset/: calib.jsonl and heldout.jsonl (split by source unit, which each
# document carries from its prompt), calib-heldout.txt, and manifest.json with sources, token counts, drops,
# generation throughput and the sha256 of every kept file.

import collections
import datetime

import kcpp
from common import *
from decontam import doc_files
from generate import COMPLETIONS, DOCS, key_of, bucket_of, target_of

NAME = "qwen38-calib-v1"
KEEP = ("id", "category", "unit", "templated", "text", "count")
LEN_BINS = [1024, 2048, 4096, 6144, 8192, 16384, 32768, 65536, 131072]


def counted_positions(d):
    spans = d.get("count") or [[0, d["n_tokens"]]]
    return spans


def select_generated(docs):
    """generated documents per bucket in the order they were generated, each taken while it brings its bucket
    closer to the target: what the generator would have kept had capped replies been kept from the start"""
    targets = collections.Counter()
    seen = set()
    for p in read_jsonl(GEN / "prompts.jsonl"):
        if (bucket_of(p), p["heldout"]) not in seen:
            seen.add((bucket_of(p), p["heldout"]))
            targets[key_of(p)] += target_of(p)
    order = {}
    for i, c in enumerate(read_jsonl(COMPLETIONS)):
        order[c["id"]] = i
    filled = collections.Counter()
    chosen, unused = [], []
    for d in sorted(docs, key=lambda d: order.get(d["id"], 1 << 30)):
        k = key_of(d)
        if filled[k] + d["n_tokens"]/2 < targets[k]:
            chosen.append(d)
            filled[k] += d["n_tokens"]
        else:
            unused.append(d)
    return chosen, unused


def thin(docs, target, block=512):
    """count spans thinned to about target tokens: blocks at a fixed stride over each document's counted range; a
    document that counts the shared prefix keeps it whole"""
    kept = {}
    for d in docs:
        p = d.get("prefix_tokens", 0)
        if d["count"] and d["count"][0][0] == 0 and p > 0:
            kept[id(d)] = [[0, p]]
            d["count"] = [[p, d["count"][0][1]]] + d["count"][1:]
    size = lambda d: sum(y - x for x, y in d["count"])
    have = sum(size(d) for d in docs)
    target -= sum(p[0][1] for p in kept.values())
    if have <= target or target <= 0:
        for d in docs:
            d["count"] = kept.get(id(d), []) + d["count"]
        return
    stride = block * have / target
    for d in docs:
        spans = []
        for a, b in d["count"]:
            x = float(a)
            while x < b:
                spans.append([int(x), min(b, int(x) + block)])
                x += stride
        d["count"] = kept.get(id(d), []) + spans


def budget_opencode(docs, long_share=0.4):
    """OpenCode sessions: each session's longest chain, the others dropped (a document costs its whole length in every
    pass, counted or not); then the long sessions' late positions thinned to long_share of the category's target and
    the other sessions' counted tokens to the rest, whichever group falls short leaving its room to the other; the
    tool-call episodes are thinned to their half of the agentic category"""
    unused = []
    for heldout in (False, True):
        oc = [d for d in docs if d["category"] == "opencode" and d["heldout"] == heldout]
        best = {}
        for d in oc:
            sess = d["id"].split("-c")[0] if re.search(r"-c\d+$", d["id"]) else d["id"]
            if sess not in best or d["n_tokens"] > best[sess]["n_tokens"]:
                best[sess] = d
        keep = set(id(d) for d in best.values())
        unused += [d for d in oc if id(d) not in keep]
        target = TARGETS["opencode"] * (HELDOUT if heldout else 1.0)
        size = lambda ds: sum(y - x for d in ds for x, y in d["count"])
        long = [d for d in best.values() if d.get("long")]
        rest = [d for d in best.values() if not d.get("long")]
        t_long = max(long_share * target, target - size(rest))
        thin(long, t_long)
        thin(rest, target - size(long))
        # the tool-call episodes are half of the agentic category
        ep = [d for d in docs if d.get("source") == "opencode_episode" and d["heldout"] == heldout]
        thin(ep, TARGETS["agentic"] * 0.5 * (HELDOUT if heldout else 1.0))
    ids = set(id(d) for d in unused)
    docs[:] = [d for d in docs if id(d) not in ids]
    return unused


def main():
    decontam = json.loads((GEN / "decontam.json").read_text()) if (GEN / "decontam.json").exists() else {}
    docs, seen = [], set()
    for f in doc_files():
        for d in read_jsonl(f):
            if d["id"] in seen:
                raise SystemExit(f"duplicate document id {d['id']}")
            seen.add(d["id"])
            if d["id"] not in decontam:
                docs.append(d)

    for d in docs:
        d["text"] = kcpp.clean(d["text"])
        d["n_tokens"] = len(kcpp.tokenize(d["text"]))
    generated = {d["id"] for d in read_jsonl(DOCS)}
    chosen, unused = select_generated([d for d in docs if d["id"] in generated])
    docs = [d for d in docs if d["id"] not in generated] + chosen
    unused_oc = budget_opencode(docs)
    units = collections.defaultdict(set)
    for d in docs:
        for a, b in counted_positions(d):
            if not (0 <= a <= b <= d["n_tokens"]):
                raise SystemExit(f"{d['id']}: count span [{a}, {b}) outside its {d['n_tokens']} tokens")
        units[d["unit"]].add(d["heldout"])
    split_units = [u for u, s in units.items() if len(s) > 1]
    if split_units:
        raise SystemExit(f"units on both sides of the split: {split_units[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(docs, key=lambda d: (d["category"], d["id"]))
    calib = [d for d in order if not d["heldout"]]
    held = [d for d in order if d["heldout"]]
    write_jsonl(OUT / "calib.jsonl", [{k: d[k] for k in KEEP if k in d} for d in calib])
    write_jsonl(OUT / "heldout.jsonl", [{k: d[k] for k in KEEP if k in d} for d in held])
    # plain text with the chat-template markers as text: tokenize it with special tokens parsed
    (OUT / "calib-heldout.txt").write_text("\n\n".join(d["text"] for d in held) + "\n", encoding="utf-8")

    def tally(ds):
        out = collections.defaultdict(lambda: {"docs": 0, "tokens": 0, "counted": 0, "counted_ge_4096": 0})
        for d in ds:
            t = out[d["category"]]
            t["docs"] += 1
            t["tokens"] += d["n_tokens"]
            for a, b in counted_positions(d):
                t["counted"] += b - a
                t["counted_ge_4096"] += max(0, b - max(a, 4096))
        return dict(sorted(out.items()))

    def histogram(ds):
        h = collections.Counter()
        for d in ds:
            h[next(b for b in LEN_BINS if d["n_tokens"] <= b)] += d["n_tokens"]
        return {f"<={b}": h[b] for b in LEN_BINS}

    gen_state = json.loads((GEN / "state.json").read_text()) if (GEN / "state.json").exists() else {}
    drops = collections.Counter((d["category"], d["reason"]) for d in read_jsonl(GEN / "drops.jsonl"))
    oc_checks = [json.loads(p.read_text()) for p in sorted((WORK / "opencode" / "checks").glob("*.json"))] if (WORK / "opencode" / "checks").exists() else []
    cfg = load_sources()
    files = {p.name: sha256_file(p) for p in [OUT / "calib.jsonl", OUT / "heldout.jsonl", OUT / "calib-heldout.txt", OUT / "eval-opencode.jsonl"] if p.exists()}
    manifest = {
        "name": NAME,
        "dataset_tag": f"{NAME}:{files['calib.jsonl'][:12]}",
        "created": datetime.date.today().isoformat(),
        "generator": {"model": gen_model().name, "server": "koboldcpp (this fork), tools/hessian/calib/serve.sh",
                      "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "max_new_tokens": "min(6144, 8192 - prompt)",
                      "thinking_mix": {"xhigh": 0.45, "medium": 0.15, "low": 0.15, "off": 0.25, "math": "xhigh 0.6, medium 0.2, low 0.2"},
                      "loop_filter": {"span": 64, "min_occurrences": 3, "compression_ratio_threshold": gen_state.get("ratio_threshold")}},
        "targets": TARGETS, "heldout_share": HELDOUT,
        "calib": tally(calib), "heldout": tally(held),
        "calib_total_counted": sum(v["counted"] for v in tally(calib).values()),
        "calib_length_histogram_tokens": histogram(calib),
        "unused_generated": {c: {"docs": sum(1 for d in unused if d["category"] == c), "tokens": sum(d["n_tokens"] for d in unused if d["category"] == c)}
                             for c in sorted({d["category"] for d in unused})},
        "unused_opencode_chains": {"docs": len(unused_oc), "tokens": sum(d["n_tokens"] for d in unused_oc)},
        "truncated_at_cap": {c: sum(d["n_tokens"] for d in calib if d.get("truncated") and d["category"] == c) for c in TARGETS},
        "drops": {"generation": {f"{c}/{r}": n for (c, r), n in sorted(drops.items())},
                  "decontamination": dict(collections.Counter(v["category"] for v in decontam.values())),
                  "opencode_failed": [c["id"] for c in oc_checks if c.get("failed")]},
        "opencode_runs": {c["id"]: {"steps": c["main_requests"], "chains": c.get("chains"), "overflows": c.get("overflows", 0),
                                    "token_mismatches": len(c["token_mismatches"]), "tool_calls": c["tool_calls"],
                                    "documents": [{k: d[k] for k in ("n_tokens", "counted")} for d in c.get("docs", [])]}
                          for c in oc_checks},
        "throughput": {"generated_tokens": gen_state.get("gen_tokens"), "seconds": round(gen_state.get("gen_seconds", 0)),
                       "tokens_per_second": round(gen_state["gen_tokens"] / gen_state["gen_seconds"], 1) if gen_state.get("gen_seconds") else None},
        "sources": {"datasets": {k: {"repo": v["repo"], "revision": v["revision"], "files": v["files"]} for k, v in cfg["datasets"].items()},
                    "pg19": {"repo": cfg["pg19"]["repo"], "revision": cfg["pg19"]["revision"]},
                    "repos": {k: {"commit": v["commit"], "licence": v.get("licence_found")} for k, v in cfg["repos"].items() if not v.get("skipped")}},
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log(f"{len(calib)} calibration documents ({manifest['calib_total_counted']} counted tokens), {len(held)} held out")
    for c, v in manifest["calib"].items():
        log(f"  {c:13s} {v['counted']:8d} counted / {TARGETS.get(c, 0):7d} target, {v['tokens']:8d} tokens, {v['docs']} docs")
    log(f"dataset tag {manifest['dataset_tag']}")


if __name__ == "__main__":
    main()
