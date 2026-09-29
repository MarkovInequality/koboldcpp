import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CALIB = Path(__file__).resolve().parent
WORK = CALIB / "work"
SOURCES = WORK / "sources"
GEN = WORK / "gen"
OUT = CALIB / "dataset"


def gen_model():
    """the generator's GGUF, from $GEN_MODEL (Qwen3.8-27B-HQ8_0.gguf for qwen38-calib-v1)"""
    p = os.environ.get("GEN_MODEL")
    if not p:
        sys.exit("set GEN_MODEL to the generator model's GGUF (Qwen3.8-27B-HQ8_0.gguf for qwen38-calib-v1)")
    return Path(p)


def user_repos():
    """the directory holding the user's own repos that the OpenCode tasks' user: entries name, from $USER_REPOS"""
    p = os.environ.get("USER_REPOS")
    if not p:
        sys.exit("set USER_REPOS to the directory holding the repos the OpenCode tasks' user: entries name")
    return Path(p)


def opencode_data():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "opencode"


def opencode_config():
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"

SEED = 20260927

# counted tokens per category, before the 10 % held out
TARGETS = {
    "chat": 200_000, "coding": 200_000, "agentic": 100_000, "math": 100_000,
    "multilingual": 100_000, "long": 100_000, "raw": 100_000, "opencode": 100_000,
}
HELDOUT = 0.10

MAX_LEN = 8192
MAX_LEN_OPENCODE = 131072


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def rng(tag):
    return random.Random(f"{SEED}:{tag}")


def load_sources():
    with open(CALIB / "sources.json") as f:
        return json.load(f)


def save_sources(cfg):
    with open(CALIB / "sources.json", "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


_ws = re.compile(r"\s+")


def normalize(text):
    return _ws.sub(" ", text.lower()).strip()


def gguf_string(path, key):
    """a string value from a GGUF file's key-value header"""
    import struct
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        def u(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def s():
            return f.read(u("Q")).decode("utf-8")

        def skip(t):
            if t == 8:
                f.seek(u("Q"), 1)
            elif t == 9:
                et, n = u("I"), u("Q")
                for _ in range(n):
                    skip(et)
            else:
                f.seek(sizes[t], 1)
        assert f.read(4) == b"GGUF"
        u("I"), u("Q")
        for _ in range(u("Q")):
            k, t = s(), u("I")
            if k == key and t == 8:
                return s()
            skip(t)
    return None
