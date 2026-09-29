#!/usr/bin/env python3
# Downloads slices of the calibration sources into build-hq/calib/work/sources/. Parquet files are read by
# row group over HTTP range requests, so multi-GB shards are never downloaded whole. Revisions and commits
# left null in sources.json are pinned to the current ones and written back.

import argparse
import io
import subprocess

import pyarrow.parquet as pq
import requests

from common import *

session = requests.Session()
session.headers["User-Agent"] = "hessian-calib-fetch"


class HttpFile(io.RawIOBase):
    BLOCK = 8 << 20

    def __init__(self, url):
        r = session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            lo, hi = i*self.BLOCK, min(self.size, (i + 1)*self.BLOCK) - 1
            for attempt in range(5):
                try:
                    r = session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[i] = r.content
        return self.cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self._block(self.pos // self.BLOCK)
            o = self.pos % self.BLOCK
            chunk = b[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def hf_revision(repo):
    r = session.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("gated"):
        raise SystemExit(f"{repo} is gated")
    return j["sha"]


def fetch_dataset(name, spec):
    out = SOURCES / f"{name}.jsonl"
    if out.exists():
        log(f"{name}: have {out.name}")
        return
    rows = []
    for path in spec["files"]:
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{path}"
        pf = pq.ParquetFile(HttpFile(url))
        cols = spec["columns"]
        got = 0
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=cols)
            for r in t.to_pylist():
                r["_file"] = path
                rows.append(r)
            got += t.num_rows
            if got >= spec["rows"]:
                break
        log(f"{name}: {path}: {got} rows from {g + 1}/{pf.num_row_groups} row groups")
    write_jsonl(out, rows)


def fetch_pg19(spec):
    out = SOURCES / "pg19"
    out.mkdir(parents=True, exist_ok=True)
    listing = session.get(f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/data/train_files.txt", timeout=60)
    listing.raise_for_status()
    files = sorted(l.strip() for l in listing.text.splitlines() if l.strip())
    picks = rng("pg19").sample(files, spec["books"])
    for f in picks:
        dst = out / f.replace("/", "_")
        if dst.exists():
            continue
        r = session.get(spec["asset_root"] + f, timeout=120)
        r.raise_for_status()
        dst.write_bytes(r.content)
    log(f"pg19: {len(picks)} books")


PERMISSIVE = [
    ("MIT", r"Permission is hereby granted, free of charge"),
    ("Apache-2.0", r"Apache License"),
    ("BSD-3-Clause", r"Neither the name|names of its\s+contributors"),
    ("BSD-2-Clause", r"Redistribution and use in source and binary forms"),
    ("ISC", r"Permission to use, copy, modify, and(/or)? distribute this software for any"),
    ("public-domain", r"public domain|dedicate[sd]? .* to the public domain|blessing in place of a legal notice"),
]


def detect_licence(repo_dir):
    cands = [p for p in repo_dir.iterdir() if p.is_file() and re.match(r"(?i)^(licen[cs]e|copying|copyright)", p.name)]
    for p in sorted(cands):
        text = p.read_text(errors="replace")
        for name, pat in PERMISSIVE:
            if re.search(pat, text, re.I | re.S):
                return name, p.name
    return None, None


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_repo(name, spec):
    dst = SOURCES / "repos" / name.replace("/", "__")
    if not dst.exists():
        dst.mkdir(parents=True)
        git("init", "-q", cwd=dst)
        git("remote", "add", "origin", f"https://github.com/{name}.git", cwd=dst)
        ref = spec["commit"] or "HEAD"
        git("fetch", "-q", "--depth", "1", "origin", ref, cwd=dst)
        git("checkout", "-q", "FETCH_HEAD", cwd=dst)
    spec["commit"] = git("rev-parse", "HEAD", cwd=dst)
    lic, lic_file = detect_licence(dst)
    if spec.get("licence") and lic != spec["licence"]:
        lic = None
    if lic is None:
        log(f"{name}: no permissive licence found, skipped")
        spec["skipped"] = True
        return
    spec["licence_found"] = f"{lic} ({lic_file})"
    spec.pop("skipped", None)
    log(f"{name}: {spec['commit'][:12]} {lic}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    cfg = load_sources()
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["datasets"].items():
        if only and name not in only:
            continue
        if not spec["revision"]:
            spec["revision"] = hf_revision(spec["repo"])
            save_sources(cfg)
        fetch_dataset(name, spec)
    if not only or "pg19" in only:
        if not cfg["pg19"]["revision"]:
            cfg["pg19"]["revision"] = hf_revision(cfg["pg19"]["repo"])
        fetch_pg19(cfg["pg19"])
        save_sources(cfg)
    if not only or "template" in only:
        (WORK / "chat_template.jinja").write_text(gguf_string(GEN_MODEL, "tokenizer.chat_template"))
    if not only or "repos" in only:
        for name, spec in cfg["repos"].items():
            fetch_repo(name, spec)
            save_sources(cfg)


if __name__ == "__main__":
    main()
