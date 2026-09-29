#!/usr/bin/env python3
# Drops every document that shares a word 13-gram (after lower-casing and collapsing whitespace) with the
# evaluation texts: wiki.test.raw, build-hq/tech-eval.txt, and the GSM8K and MATH test sets (NuminaMath-CoT
# contains problems from both). Reads the document files given (default: every work/gen document file) and
# writes the dropped ids to work/gen/decontam.json.

import argparse
import collections

from common import *

N = 13
EVAL_TEXTS = [ROOT / "build-hq/models/wikitext-2-raw/wiki.test.raw", ROOT / "build-hq/tech-eval.txt"]


def grams(text):
    w = normalize(text).split(" ")
    return {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}


def eval_grams():
    g = set()
    for p in EVAL_TEXTS:
        g |= grams(p.read_text(errors="replace"))
    for row in read_jsonl(SOURCES / "gsm8k_test.jsonl"):
        g |= grams(row["question"]) | grams(row["answer"])
    for row in read_jsonl(SOURCES / "math_test.jsonl"):
        g |= grams(row["problem"]) | grams(row["solution"])
    return g


def doc_files():
    return [p for p in [GEN / "docs.jsonl", GEN / "raw_docs.jsonl", WORK / "opencode" / "docs.jsonl"] if p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    ev = eval_grams()
    log(f"{len(ev)} evaluation 13-grams")
    dropped = {}
    n = 0
    for f in (args.files or doc_files()):
        for d in read_jsonl(f):
            n += 1
            hit = grams(d["text"]) & ev
            if hit:
                dropped[d["id"]] = {"category": d["category"], "n_grams": len(hit)}
    (GEN / "decontam.json").write_text(json.dumps(dropped, indent=1))
    log(f"{len(dropped)} of {n} documents share a 13-gram with the evaluation texts")
    for k, v in sorted(collections.Counter(x["category"] for x in dropped.values()).items()):
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
