#!/usr/bin/env python3
# Turns a task's capture into its document: the main session's last request, rendered as koboldcpp renders it,
# followed by that request's response as the template renders it in history, up to its <|im_end|>. OpenCode
# sends every earlier step back with its reasoning, so each request is a prefix of the next and this is the
# whole session as the model processed it. Sets the count spans, and records the checks (rendered prompt vs
# koboldcpp's usage.prompt_tokens for every request, earlier reasoning present, tool calls) in checks/<id>.json.
#
# usage: render.py TASK_ID

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kcpp  # noqa: E402
from common import *  # noqa: E402,F403

OC = WORK / "opencode"
LONG_FROM = 32768
LOG_LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] CtxLimit:(\d+)/\d+,.*Generated:(\d+)/")


def server_prompt_tokens(chats):
    """koboldcpp's prompt size per captured request, from its log (CtxLimit - Generated): its streamed replies to
    tool requests carry no usage. It serves one request at a time, so the i-th log line of the capture's time
    span belongs to the i-th captured request (the proxy records a request when its response ends)."""
    import datetime
    lines = []
    for l in (WORK / "server-opencode.log").read_text(errors="replace").splitlines():
        m = LOG_LINE.match(l)
        if m:
            h, mi, se, ctx, gen = map(int, m.groups())
            lines.append((h*3600 + mi*60 + se, ctx - gen))
    def tod(t):
        d = datetime.datetime.fromtimestamp(t)
        return d.hour*3600 + d.minute*60 + d.second
    lo, hi = tod(min(c["t0"] for c in chats)) - 2, tod(max(c["t1"] for c in chats)) + 2
    span = [n for t, n in lines if lo <= t <= hi]
    return span if len(span) == len(chats) else None


def parse_response(raw):
    """(assistant message, usage) from a streamed or plain chat-completions response"""
    msg = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls, usage, finish = {}, None, None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = j.get("usage") or usage
        for ch in j.get("choices", []):
            d = ch.get("delta") or ch.get("message") or {}
            finish = ch.get("finish_reason") or finish
            msg["content"] += d.get("content") or ""
            msg["reasoning_content"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return msg, usage, finish


def main():
    tid = sys.argv[1]
    task = next(t for t in read_jsonl(Path(__file__).parent / "tasks.jsonl") if t["id"] == tid)
    cap = read_jsonl(OC / "captures" / f"{tid}.jsonl")
    chats = [c for c in cap if c["path"].rstrip("/").endswith("/chat/completions") and isinstance(c.get("request"), dict)
             and c.get("status") == 200]
    main_chain = [c for c in chats if c["request"].get("tools")]
    checks = {"id": tid, "requests": len(chats), "main_requests": len(main_chain), "token_mismatches": [], "tool_calls": {},
              "failed": False, "overflows": sum(1 for c in cap if c.get("overflow"))}
    if not main_chain:
        checks["failed"] = "no main-session requests"
        (OC / "checks").mkdir(exist_ok=True)
        (OC / "checks" / f"{tid}.json").write_text(json.dumps(checks, indent=1))
        log(f"{tid}: no main-session requests captured")
        return

    rendered = []
    chats.sort(key=lambda c: c["t1"])
    from_log = server_prompt_tokens(chats)
    checks["prompt_tokens_source"] = "server log" if from_log else "usage (log lines did not match one to one)"
    n_checked = 0
    for i, c in enumerate(chats):
        prompt = kcpp.render_request(c["request"])
        n = len(kcpp.tokenize(prompt))
        resp, usage, finish = parse_response(c["response"])
        pt = from_log[i] if from_log else (usage or {}).get("prompt_tokens")
        matched = pt is None or pt == n
        if pt is not None:
            n_checked += 1
            if pt != n:
                checks["token_mismatches"].append({"t0": c["t0"], "rendered": n, "prompt_tokens": pt, "tools": bool(c["request"].get("tools"))})
        if c in main_chain:
            for tc in resp.get("tool_calls", []):
                name = tc["function"]["name"]
                checks["tool_calls"][name] = checks["tool_calls"].get(name, 0) + 1
            if matched:
                rendered.append((c, prompt, n, resp, finish))
    checks["prompt_tokens_checked"] = n_checked

    # a compaction restarts the history: each chain of requests that extend one another is a sequence of its own,
    # rendered as its last request that fits, with that request's response
    def key(c):
        return [(m.get("role"), json.dumps(m.get("content"), sort_keys=True)) for m in c["request"]["messages"]]
    chains = []
    for x in rendered:
        k = key(x[0])
        if chains and len(k) >= len(chains[-1][0]) and k[:len(chains[-1][0])] == chains[-1][0]:
            chains[-1] = (k, chains[-1][1] + [x])
        else:
            chains.append((k, [x]))
    checks["chains"] = len(chains)
    repo = task["repo"]
    unit = f"repo:{repo.split(':', 1)[1]}" if repo.startswith("public:") else repo
    category = "opencode" if task["kind"] == "session" else "agentic"
    docs = [x for x in read_jsonl(OC / "docs.jsonl") if x["id"] != f"oc-{tid}" and not x["id"].startswith(f"oc-{tid}-c")]
    checks["docs"] = []
    for ci, (_, chain) in enumerate(chains):
        doc = None
        for c, prompt, n, resp, finish in reversed(chain):
            full = kcpp.render_turn(c["request"], resp)
            if not full.startswith(prompt):
                checks["failed"] = "the rendered response does not extend the rendered prompt"
                break
            text = full.rstrip("\n")
            ids = kcpp.tokenize(text)
            if len(ids) <= MAX_LEN_OPENCODE:
                doc = (c, prompt, text, ids)
                break
        if doc is None:
            continue
        c, prompt, text, ids = doc
        # every earlier assistant turn's reasoning is in the rendered prompt
        missing = [i for i, m in enumerate(c["request"]["messages"]) if m.get("role") == "assistant"
                   and (m.get("reasoning_content") or "").strip() and m["reasoning_content"].strip() not in prompt]
        prefix_end = len(kcpp.tokenize(text[:text.index("<|im_end|>") + len("<|im_end|>\n")]))
        n = len(ids)
        if task.get("long"):
            spans = [[LONG_FROM, n]] if n > LONG_FROM else []
        elif task.get("count_prefix") and ci == 0:
            spans = [[0, n]]
        else:
            spans = [[prefix_end, n]]
        counted = sum(b - a for a, b in spans)
        info = {"chain": ci, "steps": len(chain), "n_tokens": n, "prefix_tokens": prefix_end, "counted": counted,
                "assistant_turns": sum(1 for m in c["request"]["messages"] if m.get("role") == "assistant"), "reasoning_missing": missing}
        checks["docs"].append(info)
        log(f"{tid} chain {ci}: {n} tokens ({counted} counted, prefix {prefix_end}), {len(chain)} steps")
        if counted == 0:
            continue
        docs.append({"id": f"oc-{tid}" + (f"-c{ci}" if ci else ""), "category": category, "unit": unit, "heldout": task["heldout"],
                     "templated": True, "bucket": [category, f"opencode_{task['kind']}", None], "source": f"opencode_{task['kind']}",
                     "n_tokens": n, "prefix_tokens": prefix_end, "count": spans, "counted": counted, "variant": task["variant"],
                     "lang": task.get("lang"), "long": task.get("long", False), "text": text})
    if not checks["docs"]:
        checks["failed"] = checks["failed"] or "no request fits 131072 tokens"
    write_jsonl(OC / "docs.jsonl", docs)
    log(f"{tid}: {len(main_chain)} steps in {len(chains)} chains, {len(checks['token_mismatches'])} token mismatches, "
        f"{checks['overflows']} overflows, tools {checks['tool_calls']}")
    (OC / "checks").mkdir(exist_ok=True)
    (OC / "checks" / f"{tid}.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
