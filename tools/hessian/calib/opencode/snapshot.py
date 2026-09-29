#!/usr/bin/env python3
# State of what the OpenCode runs must not touch: the user's OpenCode database (row counts, read from a copy)
# and config, and the original repos (HEAD and git status). With two arguments, compares a snapshot to now.
#
# usage: snapshot.py OUT.json | snapshot.py --compare BEFORE.json

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
REPOS = ["airi", "neovim", "hegel-rust", "voicebox", "xet-core", "remove-refusals-with-transformers"]


def db_state():
    src = HOME / ".local/share/opencode"
    out = {"files": {}}
    for p in sorted(src.glob("opencode.db*")):
        st = p.stat()
        out["files"][p.name] = {"size": st.st_size, "mtime": st.st_mtime}
    with tempfile.TemporaryDirectory() as td:
        for p in src.glob("opencode.db*"):
            shutil.copy2(p, td)
        c = sqlite3.connect(Path(td) / "opencode.db")
        out["rows"] = {t: c.execute(f'select count(*) from "{t}"').fetchone()[0]
                       for (t,) in c.execute("select name from sqlite_master where type='table'")}
        c.close()
    return out


def config_state():
    out = {}
    for p in sorted((HOME / ".config/opencode").rglob("*")):
        if p.is_file() and "node_modules" not in p.parts:
            out[str(p.relative_to(HOME))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def repo_state():
    out = {}
    for r in REPOS:
        d = HOME / "Sandbox" / r
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout
        out[r] = {"head": head, "status": hashlib.sha256(status.encode()).hexdigest(), "n_changed": len(status.splitlines())}
    return out


def snapshot():
    return {"opencode_db": db_state(), "opencode_config": config_state(), "repos": repo_state()}


def main():
    if sys.argv[1] == "--compare":
        before = json.loads(Path(sys.argv[2]).read_text())
        now = snapshot()
        problems = []
        if before["opencode_db"]["rows"] != now["opencode_db"]["rows"]:
            problems.append(f"opencode.db rows changed: {before['opencode_db']['rows']} -> {now['opencode_db']['rows']}")
        if before["opencode_db"]["files"] != now["opencode_db"]["files"]:
            problems.append("opencode.db files changed (size or mtime)")
        if before["opencode_config"] != now["opencode_config"]:
            problems.append("~/.config/opencode changed")
        for r, s in before["repos"].items():
            if now["repos"][r] != s:
                problems.append(f"repo {r} changed: {s} -> {now['repos'][r]}")
        print(json.dumps({"ok": not problems, "problems": problems}, indent=1))
        sys.exit(1 if problems else 0)
    Path(sys.argv[1]).write_text(json.dumps(snapshot(), indent=1))


if __name__ == "__main__":
    main()
