#!/usr/bin/env python3
# Compares the prompt OpenCode sent in two proxy captures (work/opencode/captures/*.jsonl): the tool list, each
# shared tool's schema, and the system prompt as a unified diff. For the run.sh --user-config check: the probe made
# with the user's own config against a calibration session should differ only in the working-directory line.
#
# usage: promptdiff.py A.jsonl B.jsonl

import difflib
import json
import sys


def main_request(path):
    for line in open(path):
        req = json.loads(line).get("request")
        if isinstance(req, dict) and req.get("tools") and req["messages"][0]["role"] == "system":
            return req
    sys.exit(f"no main-session request in {path}")


def text(content):
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: promptdiff.py A.jsonl B.jsonl")
    pa, pb = sys.argv[1:]
    a, b = main_request(pa), main_request(pb)
    ta = {t["function"]["name"]: t for t in a["tools"]}
    tb = {t["function"]["name"]: t for t in b["tools"]}
    for p, only in ((pa, set(ta) - set(tb)), (pb, set(tb) - set(ta))):
        if only:
            print(f"tools only in {p}: {', '.join(sorted(only))}")
    for n in sorted(set(ta) & set(tb)):
        if ta[n] != tb[n]:
            print(f"tool {n}: the schemas differ")
    if ta == tb:
        print(f"tools identical ({len(ta)})")
    diff = difflib.unified_diff(text(a["messages"][0]["content"]).splitlines(), text(b["messages"][0]["content"]).splitlines(),
                                pa, pb, n=0, lineterm="")
    print("\n".join(diff) or "system prompts identical")


if __name__ == "__main__":
    main()
