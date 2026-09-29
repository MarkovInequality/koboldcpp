#!/usr/bin/env python3
# Phase 1 checks on the kept dataset (after split.py). --after-cleanup checks only what must hold once
# cleanup.sh has run: the work dir is gone and the kept files match the manifest's hashes.

import argparse
import collections

import kcpp
from common import *
from decontam import eval_grams, grams

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
TOOL_TARGET = {"bash": 0.50, "read": 0.25, "grep": 0.11, "edit": 0.08}


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, ok, what):
        print(f"[{'ok' if ok else 'FAIL'}] {what}")
        self.failed += not ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-cleanup", action="store_true")
    args = ap.parse_args()
    R = Report()
    man = json.loads((OUT / "manifest.json").read_text())

    if args.after_cleanup:
        R.check(not WORK.exists(), f"{WORK} is gone")
        for name, h in man["files"].items():
            R.check(sha256_file(OUT / name) == h, f"{name} matches the manifest's sha256")
        sys.exit(1 if R.failed else 0)

    calib = read_jsonl(OUT / "calib.jsonl")
    held = read_jsonl(OUT / "heldout.jsonl")

    total = man["calib_total_counted"]
    for c, t in TARGETS.items():
        got = man["calib"].get(c, {}).get("counted", 0)
        R.check(abs(got - t) <= 0.10*t, f"{c}: {got} counted tokens, target {t} +-10 %")
    R.check(1_000_000 <= total <= 1_100_000, f"total counted {total} within 1.0-1.1M")

    oc_ids = {d["id"] for d in calib + held if d["category"] in ("opencode",) or d["id"].startswith("oc-")}
    n_tok, too_long, bad_spans, specials = {}, [], [], []
    ge4096 = 0
    for d in calib + held:
        ids = kcpp.tokenize(d["text"])
        n = len(ids)
        n_tok[d["id"]] = n
        if n > (MAX_LEN_OPENCODE if d["id"] in oc_ids else MAX_LEN):
            too_long.append(d["id"])
        spans = d.get("count") or [[0, n]]
        prev = 0
        for a, b in spans:
            if a < prev or b < a or b > n:
                bad_spans.append(d["id"])
            prev = b
        if d.get("templated") and (ids.count(248045) != d["text"].count(IM_START) or ids.count(248046) != d["text"].count(IM_END)):
            specials.append(d["id"])
        if d in calib:
            ge4096 += sum(max(0, b - max(a, 4096)) for a, b in spans)
    R.check(not too_long, f"no document over 8192 tokens (131072 for OpenCode): {too_long[:5]}")
    R.check(not bad_spans, f"count spans inside their documents and in order: {bad_spans[:5]}")
    R.check(not specials, f"one <|im_start|>/<|im_end|> token per marker in every templated document: {specials[:5]}")
    R.check(ge4096 >= 0.25*total, f"{ge4096/total:.1%} of counted tokens at positions >= 4096 (>= 25 %)")

    # OpenCode: shared prefix share, tool mix, token-count and reasoning checks
    oc_docs = [d for d in calib if d["id"].startswith("oc-")]
    if oc_docs:
        pref, oc_total = 0, 0
        for d in oc_docs:
            p = len(kcpp.tokenize(d["text"][:d["text"].index(IM_END) + len(IM_END) + 1]))
            pref += sum(min(b, p) - min(a, p) for a, b in d["count"])
            oc_total += sum(b - a for a, b in d["count"])
        R.check(0.15 <= pref/oc_total <= 0.25, f"shared prefix {pref/oc_total:.1%} of OpenCode-related counted tokens (15-25 %)")
        checks = {c["id"]: c for c in (json.loads(p.read_text()) for p in (WORK / "opencode" / "checks").glob("*.json"))}
        tools = collections.Counter()
        for c in checks.values():
            if c["id"].startswith("e"):
                tools.update(c["tool_calls"])
        n_calls = sum(tools.values())
        for t, share in TOOL_TARGET.items():
            got = tools[t] / max(1, n_calls)
            R.check(abs(got - share) <= 0.05, f"episode tool calls: {t} {got:.1%} (target {share:.0%} +-5 points)")
        mism = {k: len(c["token_mismatches"]) for k, c in checks.items() if c.get("token_mismatches")}
        R.check(not mism, f"every captured request's rendered prompt matches koboldcpp's usage.prompt_tokens: {mism}")
        checked = sum(c.get("prompt_tokens_checked", 0) for c in checks.values())
        R.check(checked > 0, f"{checked} captured requests had usage.prompt_tokens to compare")
        first = min((c for c in checks.values() if c["id"].startswith("s") and any(d["assistant_turns"] for d in c.get("docs", []))),
                    key=lambda c: c["id"], default=None)
        R.check(first is not None and not any(d["reasoning_missing"] for d in first["docs"]),
                f"earlier reasoning present in the rendered prompts of the first session ({first and first['id']})")
        missing = [c["id"] for c in checks.values() if any(d["reasoning_missing"] for d in c.get("docs", []))]
        R.check(not missing, f"earlier reasoning present in every rendered OpenCode document: {missing[:5]}")
    else:
        R.check(False, "no OpenCode documents yet")

    # generation: koboldcpp's prompt token counts vs ours
    comp = read_jsonl(GEN / "completions.jsonl")
    mism = [c["id"] for c in comp if c["prompt_tokens"] != c["n_prompt"]]
    R.check(not mism, f"koboldcpp's usage.prompt_tokens equals our count for every generation ({len(comp)} completions): {mism[:5]}")

    ev = eval_grams()
    hits = [d["id"] for d in calib + held if grams(d["text"]) & ev]
    R.check(not hits, f"no 13-gram overlap with the evaluation texts: {hits[:5]}")

    snap = WORK / "opencode" / "snapshot-before.json"
    if snap.exists():
        import subprocess
        r = subprocess.run([sys.executable, str(CALIB / "opencode" / "snapshot.py"), "--compare", str(snap)], capture_output=True, text=True)
        R.check(r.returncode == 0, f"the user's OpenCode data and original repos are unchanged {r.stdout.strip() if r.returncode else ''}")
    R.check(man["throughput"]["tokens_per_second"] is not None and man["drops"] is not None, "generation throughput and drop counts are in the manifest")
    print(f"{R.failed} checks failed")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
