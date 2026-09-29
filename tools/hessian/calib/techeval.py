#!/usr/bin/env python3
# Builds tech-eval.txt, an out-of-domain evaluation text: this repo's top-level docs (README.md, docs/*.md and
# tools/*/README*.md, in sorted order, joined by newlines) cut at 115,000 bytes, then the first 95,000 bytes of
# src/llama-model.cpp, all read from git at REV. The default REV reproduces, byte for byte, the file that
# qwen38-calib-v1 was decontaminated against and that the earlier HQ measurements scored on.
#
# usage: techeval.py [--rev REV] [OUT]   (default OUT: dataset/tech-eval.txt)

import argparse
import fnmatch
import subprocess
from pathlib import Path

from common import OUT, ROOT

DOCS = ["README.md", "docs/*.md", "tools/*/README*.md"]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="7957c3ad6")
    ap.add_argument("out", nargs="?", default=str(OUT / "tech-eval.txt"))
    a = ap.parse_args()
    files = git("ls-tree", "-r", "--name-only", a.rev).decode().split("\n")
    docs = sorted(f for f in files if any(fnmatch.fnmatchcase(f, p) and f.count("/") == p.count("/") for p in DOCS))
    text = b"\n".join(git("show", f"{a.rev}:{f}") for f in docs)[:115000] + git("show", f"{a.rev}:src/llama-model.cpp")[:95000]
    Path(a.out).write_bytes(text)
    print(f"{a.out}: {len(text)} bytes from {len(docs)} docs and src/llama-model.cpp at {a.rev}")


if __name__ == "__main__":
    main()
