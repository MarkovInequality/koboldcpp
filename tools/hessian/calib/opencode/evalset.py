#!/usr/bin/env python3
# The evaluation set: the user's recorded Qwen3.8 OpenCode sessions, rendered as koboldcpp would render them
# for the model. Messages, reasoning, tool calls and tool results come from a copy of opencode.db (the original
# is only read); the system prompt and tool schemas come from a capture of the current OpenCode version, with the
# session's working directory put in its environment block. Never used for calibration.
#
# usage: evalset.py [CAPTURE.jsonl]   (default: the first session capture)

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kcpp  # noqa: E402
from common import *  # noqa: E402,F403

DB = Path.home() / ".local/share/opencode"


def template_request(capture):
    for c in read_jsonl(capture):
        req = c.get("request")
        if isinstance(req, dict) and req.get("tools") and req["messages"][0]["role"] == "system":
            return req
    raise SystemExit(f"no main-session request in {capture}")


def session_messages(db, sid):
    """OpenAI-style messages for a session: a step per assistant message, tool results after it"""
    out = []
    for mid, mdata in db.execute("select id, data from message where session_id=? order by time_created, id", (sid,)):
        m = json.loads(mdata)
        parts = [json.loads(d) for (d,) in db.execute("select data from part where message_id=? order by time_created, id", (mid,))]
        if m["role"] == "user":
            text = "\n".join(p["text"] for p in parts if p.get("type") == "text" and not p.get("synthetic"))
            if text.strip():
                out.append({"role": "user", "content": text})
            continue
        step = None

        def flush():
            if step and (step["content"] or step["reasoning_content"] or step["tool_calls"]):
                msg = {"role": "assistant", "content": step["content"], "reasoning_content": step["reasoning_content"]}
                if step["tool_calls"]:
                    msg["tool_calls"] = step["tool_calls"]
                out.append(msg)
                out.extend(step["results"])
        for p in parts:
            t = p.get("type")
            if t == "step-start" or step is None:
                flush()
                step = {"content": "", "reasoning_content": "", "tool_calls": [], "results": []}
            if t == "text":
                step["content"] += p.get("text", "")
            elif t == "reasoning":
                step["reasoning_content"] += p.get("text", "")
            elif t == "tool":
                st = p.get("state", {})
                step["tool_calls"].append({"id": p.get("callID", ""), "type": "function",
                                           "function": {"name": p["tool"], "arguments": st.get("input", {})}})
                res = st.get("output") if st.get("status") == "completed" else st.get("error", "")
                step["results"].append({"role": "tool", "tool_call_id": p.get("callID", ""), "content": res or ""})
        flush()
    return out


def main():
    oc = WORK / "opencode"
    capture = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((oc / "captures").glob("s*.jsonl"))[0]
    tmpl = template_request(capture)
    system = tmpl["messages"][0]["content"]
    m = re.search(r"Working directory: (\S+)", system)
    workdir = m.group(1) if m else None

    with tempfile.TemporaryDirectory() as td:
        for p in DB.glob("opencode.db*"):
            shutil.copy2(p, td)
        db = sqlite3.connect(Path(td) / "opencode.db")
        sessions = db.execute("select id, directory, title from session where parent_id is null order by time_created").fetchall()
        rows = []
        for sid, directory, title in sessions:
            models = {json.loads(d).get("modelID") or "" for (d,) in db.execute("select data from message where session_id=?", (sid,))}
            if not any("Qwen3.8" in x for x in models):
                continue
            msgs = session_messages(db, sid)
            if not any(x["role"] == "assistant" for x in msgs):
                continue
            sys_msg = system.replace(workdir, directory) if workdir else system
            req = {"messages": [{"role": "system", "content": sys_msg}] + msgs[:-1], "tools": tmpl["tools"],
                   "reasoning_effort": tmpl.get("reasoning_effort")}
            last = msgs[-1]
            text = kcpp.render_turn(req, last).rstrip("\n") if last["role"] == "assistant" else kcpp.render_request(req)
            rows.append({"id": sid, "title": title, "directory": directory, "messages": len(msgs),
                         "n_tokens": len(kcpp.tokenize(text)), "text": text})
        db.close()
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "eval-opencode.jsonl", rows)
    log(f"{len(rows)} sessions, {sum(r['n_tokens'] for r in rows)} tokens, median {sorted(r['n_tokens'] for r in rows)[len(rows)//2] if rows else 0}")


if __name__ == "__main__":
    main()
