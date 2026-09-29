#!/usr/bin/env python3
# The chat evaluation texts, built from the calibration dataset (../calib/dataset/) into OUTDIR:
#   heldout.txt   the first 9000 characters of each held-out document
#   opencode.txt  a 9000-character slice of each of the user's OpenCode sessions, from the first turn start after a
#                 seeded random point
# each joined by <|endoftext|>. The output depends only on the dataset, so the reference caches stay valid.
#
# usage: texts.py OUTDIR

import json
import random
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "calib" / "dataset"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(line) for line in open(DATASET / "heldout.jsonl")]
    (out / "heldout.txt").write_text("<|endoftext|>".join(d["text"][:9000] for d in docs), encoding="utf-8")

    rng = random.Random(1)
    parts = []
    for line in open(DATASET / "eval-opencode.jsonl"):
        t = json.loads(line)["text"]
        a = rng.randrange(len(t) // 4, max(len(t) // 4 + 1, len(t) - 12000))
        a = t.find("<|im_start|>", a)
        parts.append(t[a:a + 9000])
    (out / "opencode.txt").write_text("<|endoftext|>".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
