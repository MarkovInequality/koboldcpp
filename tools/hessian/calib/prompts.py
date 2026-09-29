#!/usr/bin/env python3
# Builds the prompt list for every generated category (all but OpenCode) into work/gen/prompts.jsonl, in the
# order generate.py consumes them. Each category gets several times the prompts it needs; generate.py stops
# once a category's counted tokens reach its target, separately for the calibration and held-out sides.

import ast
import collections
import json

from common import *

HELDOUT_REPOS = {"expressjs/express", "encode/httpx", "google/leveldb"}

LANG_SHARE = {"ts": 0.25, "py": 0.25, "rust": 0.20, "c": 0.20, "other": 0.10}
EXTS = {
    "ts": (".ts", ".tsx", ".js", ".mjs", ".cjs"),
    "py": (".py",),
    "rust": (".rs",),
    "c": (".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".cxx"),
    "other": (".go", ".java", ".kt", ".sh", ".sql"),
}
FENCE = {".ts": "ts", ".tsx": "tsx", ".js": "js", ".mjs": "js", ".cjs": "js", ".py": "python", ".rs": "rust",
         ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cxx": "cpp", ".cu": "cuda",
         ".cuh": "cuda", ".go": "go", ".java": "java", ".kt": "kotlin", ".sh": "bash", ".sql": "sql"}


def pick_think(r, math=False):
    x = r.random()
    if math:
        return "xhigh" if x < 0.6 else "medium" if x < 0.8 else "low"
    return "xhigh" if x < 0.45 else "medium" if x < 0.60 else "low" if x < 0.75 else "off"


def is_heldout(unit):
    return stable_hash(unit) % 10 == 0


CODE_KW = re.compile(r"```|\bdef |\bfunction\b|#include|\bclass \w+|\bimport \w+|Traceback|\bconsole\.log|\bpublic static\b|"
                     r"\bfn \w+\(|\bSELECT\b.*\bFROM\b|\b(python|javascript|typescript|java|c\+\+|rust|golang|sql|bash|"
                     r"html|css|react|node\.?js|django|flask|pandas|numpy|regex|api|compile|compiler|segfault|"
                     r"stack overflow|git|docker|kubernetes|code|script|program|function|bug|error)\b", re.I)


def code_lang(text):
    t = text.lower()
    if re.search(r"\b(typescript|javascript|node\.?js|react|vue|angular|npm|\.tsx?\b|\.jsx?\b|console\.log|const \w+ =)", t):
        return "ts"
    if re.search(r"\b(python|django|flask|pandas|numpy|pip|def \w+\(|import \w+$)", t, re.M):
        return "py"
    if re.search(r"\b(rust|cargo|fn \w+\(|impl\b|let mut)", t):
        return "rust"
    if re.search(r"(\bc\+\+|\bcuda\b|#include|\bstd::|\bprintf\(|\bmalloc\(|\bc language|\bin c\b)", t):
        return "c"
    if re.search(r"\b(golang|go\b|java|kotlin|bash|shell|sql|select .* from)", t):
        return "other"
    return None


def is_coding(text):
    t = text[:4000]
    hits = len(CODE_KW.findall(t))
    return "```" in t or hits >= 3 or code_lang(t) is not None and hits >= 1


def weighted_by_lang(items, n, tag):
    """n items, drawn per code language in LANG_SHARE proportions (as far as each language has items)"""
    by = collections.defaultdict(list)
    for it in items:
        by[it.get("code_lang") or "other"].append(it)
    r = rng(tag)
    for v in by.values():
        r.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        lang = r.choices(list(LANG_SHARE), weights=list(LANG_SHARE.values()))[0]
        if by[lang]:
            out.append(by[lang].pop())
    return out


# ---------------------------------------------------------------- chat sources

def wildchat():
    rows = read_jsonl(SOURCES / "wildchat.jsonl")
    seen, per_user, per_prefix = set(), collections.Counter(), collections.Counter()
    out = []
    r = rng("wildchat")
    r.shuffle(rows)
    for row in rows:
        if row["language"] != "English" or row["toxic"] or row["redacted"]:
            continue
        if any(m and m.get("flagged") for m in (row["openai_moderation"] or [])):
            continue
        if any(m and max([v for v in m.values() if isinstance(v, (int, float))], default=0) > 0.1 for m in (row["detoxify_moderation"] or [])):
            continue
        users = [m["content"] for m in row["conversation"] if m["role"] == "user"]
        if not users or not (20 <= len(users[0]) <= 12000) or any(len(u) > 12000 or not u.strip() for u in users[:3]):
            continue
        key = normalize(users[0])
        prefix = key[:80]
        if key in seen or per_user[row["hashed_ip"]] >= 2 or per_prefix[prefix] >= 1:
            continue
        seen.add(key)
        per_user[row["hashed_ip"]] += 1
        per_prefix[prefix] += 1
        out.append({"source": "wildchat", "unit": f"wildchat:{row['conversation_hash']}", "turns": users[:3],
                    "coding": is_coding(users[0]), "code_lang": code_lang(users[0])})
    return out


def oasst2():
    rows = [m for m in read_jsonl(SOURCES / "oasst2.jsonl") if not m["deleted"] and m["review_result"] is not False]
    children = collections.defaultdict(list)
    for m in rows:
        children[m["parent_id"]].append(m)
    out = []
    for root in children[None]:
        if root["role"] != "prompter" or root["lang"] != "en":
            continue
        turns, node = [root["text"]], root
        while len(turns) < 3:
            answers = sorted((c for c in children[node["message_id"]] if c["role"] == "assistant"),
                             key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 0))
            if not answers:
                break
            nxt = [c for c in children[answers[0]["message_id"]] if c["role"] == "prompter"]
            if not nxt:
                break
            node = nxt[0]
            turns.append(node["text"])
        if len(turns[0]) < 10:
            continue
        out.append({"source": "oasst2", "unit": f"oasst2:{root['message_tree_id']}", "turns": turns,
                    "coding": is_coding(turns[0]), "code_lang": code_lang(turns[0])})
    rng("oasst2").shuffle(out)
    return out


def conversations(items, n, multi_share, tag):
    """n prompts, multi_share of them multi-turn; the rest keep only their first turn"""
    multi = [it for it in items if len(it["turns"]) > 1][:int(n*multi_share)]
    taken = {id(it) for it in multi}
    single = [dict(it, turns=it["turns"][:1]) for it in items if id(it) not in taken][:n - len(multi)]
    out = multi + single
    rng(tag).shuffle(out)
    for i, it in enumerate(out):
        it["heldout"] = i % 10 == 9
    return out


# ---------------------------------------------------------------- repos

def repo_files(name, spec, exts, min_chars, max_chars):
    root = SOURCES / "repos" / name.replace("/", "__")
    paths = spec.get("paths") or ["."]
    out = []
    for sub in paths:
        for p in sorted((root / sub).rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or not rel.endswith(exts) or "/." in "/" + rel:
                continue
            if re.search(r"(^|/)(node_modules|vendor|third_party|dist|build|fixtures?|__snapshots__)/|\.min\.|\.d\.ts$|generated", rel):
                continue
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            if not (min_chars <= len(text) <= max_chars) or max(map(len, text.splitlines() or [""])) > 400:
                continue
            out.append((rel, text))
    return out


def lang_of_repo(cfg, name):
    return cfg["repos"][name]["lang"]


MUTATIONS = [(" < ", " <= "), (" <= ", " < "), (" > ", " >= "), (" >= ", " > "), (" == ", " != "), (" != ", " == "),
             (" && ", " || "), (" || ", " && "), ("+ 1", "- 1"), ("- 1", "+ 1")]


def mutate(text, r):
    lines = text.splitlines(keepends=True)
    cands = [(i, a, b) for i, l in enumerate(lines) for a, b in MUTATIONS
             if a in l and not l.lstrip().startswith(("//", "#", "*", "/*"))]
    if not cands:
        return None
    i, a, b = r.choice(cands)
    lines[i] = lines[i].replace(a, b, 1)
    return "".join(lines)


PORT_TO = {"ts": ["Python", "Rust", "Go"], "py": ["TypeScript", "Rust", "Go"], "rust": ["C++", "Python", "Go"],
           "c": ["Rust", "modern C++", "Zig"], "other": ["Python", "Rust", "TypeScript"]}

CODE_TASKS = [
    ("explain", "Explain what this file does, how its main pieces fit together, and anything non-obvious a new contributor should know."),
    ("explain", "Walk me through this code. What is it responsible for, and what are the key functions and data structures?"),
    ("review", "Review this code as you would a pull request. Point out bugs, edge cases, performance problems and style issues, with concrete suggestions."),
    ("review", "What would you change in this file to make it more robust and easier to maintain? Be specific."),
    ("bug", "Something in this file is broken: it compiles, but it misbehaves at runtime. Find the bug, explain it, and show the fix."),
    ("tests", "Write unit tests for the most important functions in this file, using the test framework this project uses. Cover edge cases."),
    ("port", "Port this code to {lang}, keeping its behaviour and public interface as close as possible. Explain any design changes."),
    ("question", "I'm trying to understand the error handling in this file. How are failures reported and propagated, and are there any cases that aren't handled?"),
]


def repo_prompt(repo, rel, text, task, r, lang):
    ext = "." + rel.rsplit(".", 1)[-1]
    kind, instr = task
    if kind == "bug":
        m = mutate(text, r)
        if m is None:
            kind, instr = CODE_TASKS[0]
        else:
            text = m
    if kind == "port":
        instr = instr.format(lang=r.choice(PORT_TO[lang]))
    block = f"```{FENCE.get(ext, '')}\n{text.rstrip()}\n```"
    head = f"This is `{rel}` from the {repo} repository."
    if r.random() < 0.5:
        return f"{head}\n\n{block}\n\n{instr}", kind
    return f"{instr}\n\n{head}\n\n{block}", kind


# ---------------------------------------------------------------- hermes tools

def hermes_messages(row):
    conv = row["conversations"]
    conv = ast.literal_eval(conv) if isinstance(conv, str) else conv
    tools_raw = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    tools = [{"type": "function", "function": t} for t in tools_raw] if tools_raw else None
    msgs = []
    for m in conv:
        role, value = m["from"], m["value"]
        if role == "system":
            continue
        if role == "human":
            msgs.append({"role": "user", "content": value})
        elif role == "gpt":
            reasoning = ""
            mt = re.match(r"\s*<think>(.*?)</think>(.*)", value, re.S)
            if mt:
                reasoning, value = mt.group(1).strip(), mt.group(2)
            calls = []
            for c in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", value, re.S):
                try:
                    j = json.loads(c)
                except json.JSONDecodeError:
                    try:
                        j = ast.literal_eval(c)
                    except Exception:
                        return None
                if not isinstance(j, dict) or "name" not in j:
                    return None
                args = j.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                calls.append({"type": "function", "function": {"name": j["name"], "arguments": args}})
            content = re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.S).strip()
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        elif role == "tool":
            for body in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", value, re.S) or [value]:
                try:
                    j = json.loads(body)
                    body = json.dumps(j.get("content", j), ensure_ascii=False) if isinstance(j, dict) else body
                except json.JSONDecodeError:
                    pass
                msgs.append({"role": "tool", "content": body})
        else:
            return None
    # regenerate the final assistant turn
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    if not msgs or msgs[0]["role"] != "user":
        return None
    return msgs, tools


# ---------------------------------------------------------------- multilingual

ML_LANGS = {  # fineweb-2 subset, aya code, share
    "cmn_Hani": ("zho", 0.40), "jpn_Jpan": ("jpn", 0.0667), "kor_Hang": ("kor", 0.0667), "spa_Latn": ("spa", 0.0667),
    "fra_Latn": ("fra", 0.0667), "deu_Latn": ("deu", 0.0667), "rus_Cyrl": ("rus", 0.0667), "arb_Arab": ("arb", 0.0667),
    "por_Latn": ("por", 0.0667), "vie_Latn": ("vie", 0.0667),
}

ML_TASKS = {
    "cmn_Hani": ["请用中文概括下面这篇文章的主要内容，并列出三个要点。", "请把下面的文字翻译成英文。", "读完下面的文章后，请回答：作者的主要观点是什么？有哪些论据支持它？"],
    "jpn_Jpan": ["次の文章の要点を日本語でまとめてください。", "次の文章を英語に翻訳してください。", "次の文章を読んで、筆者が最も伝えたいことは何か説明してください。"],
    "kor_Hang": ["다음 글의 핵심 내용을 한국어로 요약해 주세요.", "다음 글을 영어로 번역해 주세요.", "다음 글을 읽고 글쓴이의 주장과 그 근거를 설명해 주세요."],
    "spa_Latn": ["Resume en español las ideas principales del siguiente texto.", "Traduce el siguiente texto al inglés.", "Lee el siguiente texto y explica cuál es la tesis del autor y cómo la defiende."],
    "fra_Latn": ["Résume en français les idées principales du texte suivant.", "Traduis le texte suivant en anglais.", "Lis le texte suivant et explique quelle est la thèse de l'auteur et comment il la défend."],
    "deu_Latn": ["Fasse die wichtigsten Punkte des folgenden Textes auf Deutsch zusammen.", "Übersetze den folgenden Text ins Englische.", "Lies den folgenden Text und erkläre, was die Hauptaussage des Autors ist und wie er sie begründet."],
    "rus_Cyrl": ["Кратко перескажи по-русски основные идеи следующего текста.", "Переведи следующий текст на английский язык.", "Прочитай текст и объясни, в чём главная мысль автора и как он её обосновывает."],
    "arb_Arab": ["لخّص الأفكار الرئيسية في النص التالي باللغة العربية.", "ترجم النص التالي إلى اللغة الإنجليزية.", "اقرأ النص التالي واشرح الفكرة الرئيسية للكاتب وكيف يدعمها."],
    "por_Latn": ["Resuma em português as ideias principais do texto a seguir.", "Traduza o texto a seguir para o inglês.", "Leia o texto a seguir e explique qual é a tese do autor e como ele a defende."],
    "vie_Latn": ["Hãy tóm tắt bằng tiếng Việt những ý chính của đoạn văn sau.", "Hãy dịch đoạn văn sau sang tiếng Anh.", "Đọc đoạn văn sau và giải thích quan điểm chính của tác giả cùng các lập luận hỗ trợ."],
}


# ---------------------------------------------------------------- long context

BOOK_TASKS = [
    "Summarize this excerpt: what happens, who the main characters are, and how they relate to each other.",
    "Describe the narrator's voice and the author's style in this excerpt, quoting a few short passages as evidence.",
    "List the events of this excerpt in the order they happen, then explain which one matters most for what follows.",
    "What are the main tensions or conflicts in this excerpt? How does the author build them?",
    "Write a study guide for this excerpt: a short summary, five discussion questions, and a paragraph on its themes.",
]
WIKI_TASKS = [
    "Using only the documents above, write a short summary of each and explain any connections between them.",
    "Which of the documents above mention dates or places? List them per document, then say which document is the most detailed.",
    "Based only on the documents above, write a set of ten quiz questions with answers, drawn from all of them.",
    "Compare the documents above: what kind of subject does each cover, and what does each leave out that a reader might want?",
]
LONG_CODE_TASKS = [
    "Explain the architecture of this file: its main components, how control flows between them, and the key invariants.",
    "Review this file in depth. Rank the problems you find by severity and propose fixes.",
    "Write developer documentation for this file: an overview, then each public function or type with its purpose and caveats.",
]


def wiki_article(row):
    return f"# {row['title']}\n\n{row['text'].strip()}"


# ---------------------------------------------------------------- build

def main():
    cfg = load_sources()
    out = []

    def add(category, n_needed, items):
        out.extend(items)
        log(f"{category}: {len(items)} prompts ({n_needed} wanted)")

    def P(category, unit, source, messages, think, heldout=None, **kw):
        return dict(id=f"{category}-{source}-{stable_hash(unit + json.dumps(messages)[:2000]):012x}", category=category,
                    unit=unit, heldout=is_heldout(unit) if heldout is None else heldout, source=source,
                    messages=messages, think=think, **kw)

    wc, oa = wildchat(), oasst2()
    log(f"wildchat usable: {len(wc)} ({sum(x['coding'] for x in wc)} coding); oasst2 usable: {len(oa)} ({sum(x['coding'] for x in oa)} coding)")

    # chat: WildChat 130k, oasst2 70k; ~30 % multi-turn
    r = rng("chat")
    items = conversations([x for x in wc if not x["coding"]], 400, 0.30, "chat-wc") + \
            conversations([x for x in oa if not x["coding"]], 220, 0.30, "chat-oa")
    chat = []
    for it in items:
        chat.append(P("chat", it["unit"], it["source"], [{"role": "user", "content": it["turns"][0]}], pick_think(r),
                      heldout=it["heldout"], followups=it["turns"][1:], share=0.65 if it["source"] == "wildchat" else 0.35))
    r.shuffle(chat)
    add("chat", "200k", chat)

    # coding: WildChat 40k, oasst2 20k as plain chat, repos 140k
    r = rng("coding")
    coding = []
    for i, it in enumerate(weighted_by_lang([x for x in wc if x["coding"]], 150, "coding-wc") + weighted_by_lang([x for x in oa if x["coding"]], 80, "coding-oa")):
        coding.append(P("coding", it["unit"], it["source"], [{"role": "user", "content": it["turns"][0]}], pick_think(r),
                        heldout=i % 10 == 9, code_lang=it.get("code_lang"), share=0.20 if it["source"] == "wildchat" else 0.10))
    files = collections.defaultdict(list)
    for name, spec in cfg["repos"].items():
        if spec.get("skipped"):
            continue
        for rel, text in repo_files(name, spec, EXTS[spec["lang"]], 1500, 16000):
            files[spec["lang"]].append((name, rel, text))
    for v in files.values():
        r.shuffle(v)
    n_repo = 0
    while n_repo < 360 and any(files.values()):
        lang = r.choices(list(LANG_SHARE), weights=list(LANG_SHARE.values()))[0]
        if not files[lang]:
            continue
        name, rel, text = files[lang].pop()
        prompt, kind = repo_prompt(name, rel, text, r.choice(CODE_TASKS), r, lang)
        coding.append(P("coding", f"repo:{name}", "repo", [{"role": "user", "content": prompt}], pick_think(r),
                        heldout=name in HELDOUT_REPOS, code_lang=lang, task=kind, file=rel, share=0.70))
        n_repo += 1
    r.shuffle(coding)
    add("coding", "200k", coding)

    # agentic: hermes_reasoning_tool_use 50k (OpenCode episodes come from opencode/)
    r = rng("agentic")
    rows = read_jsonl(SOURCES / "hermes_tools.jsonl")
    r.shuffle(rows)
    order = {"multistep": 0, "multiturn": 1, "single": 2, "relevance": 3}
    rows.sort(key=lambda x: order.get(x["scenario_category"], 4) + r.random()*2.5)
    agentic = []
    for i, row in enumerate(rows[:600]):
        conv = hermes_messages(row)
        if conv is None:
            continue
        msgs, tools = conv
        agentic.append(P("agentic", f"hermes:{stable_hash(str(row['conversations'])):x}", "hermes", msgs, pick_think(r),
                         heldout=len(agentic) % 10 == 9, tools=tools, scenario=row["scenario_category"], share=0.5))
    r.shuffle(agentic)
    add("agentic", "50k", agentic)

    # math: NuminaMath-CoT problems, stratified over its sources, thinking on
    r = rng("math")
    by_src = collections.defaultdict(list)
    for row in read_jsonl(SOURCES / "numina.jsonl"):
        if 20 <= len(row["problem"]) <= 3000:
            by_src[row["source"]].append(row)
    for v in by_src.values():
        r.shuffle(v)
    math = []
    while len(math) < 300 and any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src]:
                row = by_src[src].pop()
                math.append(P("math", f"numina:{stable_hash(row['problem']):x}", "numina", [{"role": "user", "content": row["problem"]}],
                              pick_think(r, math=True), heldout=len(math) % 10 == 9, numina_source=src))
    r.shuffle(math)
    add("math", "100k", math)

    # multilingual: aya prompts 50k, fineweb-2 as a templated task 25k (its raw half is in raw.py)
    r = rng("multilingual")
    aya = collections.defaultdict(list)
    for row in read_jsonl(SOURCES / "aya.jsonl"):
        if 10 <= len(row["inputs"]) <= 6000:
            aya[row["language_code"]].append(row)
    fw = collections.defaultdict(list)
    for row in read_jsonl(SOURCES / "fineweb2.jsonl"):
        if 800 <= len(row["text"]) <= 9000:
            fw[row["_file"].split("/")[1]].append(row)
    ml = []
    for sub, (code, share) in ML_LANGS.items():
        seen = set()
        rows = aya[code]
        r.shuffle(rows)
        n_aya = max(6, round(80*share))
        k = 0
        for row in rows:
            key = normalize(row["inputs"])[:100]
            if key in seen:
                continue
            seen.add(key)
            ml.append(P("multilingual", f"aya:{stable_hash(row['inputs']):x}", "aya", [{"role": "user", "content": row["inputs"]}],
                        pick_think(r), heldout=k % 5 == 4, lang=sub, share=0.5*share))
            k += 1
            if k >= n_aya:
                break
        rows = fw[sub]
        r.shuffle(rows)
        for k, row in enumerate(rows[:max(5, round(30*share))]):
            task = r.choice(ML_TASKS[sub])
            ml.append(P("multilingual", f"fineweb2:{row['id']}", "fineweb2", [{"role": "user", "content": f"{task}\n\n{row['text'].strip()}"}],
                        pick_think(r), heldout=k % 5 == 4, lang=sub, share=0.25*share))
    r.shuffle(ml)
    add("multilingual", "75k", ml)

    # long context: 4.5-6k-token prompts from PG-19, repos and Wikipedia
    r = rng("long")
    long = []
    books = sorted((SOURCES / "pg19").iterdir())
    for b in books:
        text = b.read_text(errors="replace")
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        start = len(paras)//5
        for rep in range(3):
            i = r.randrange(start, max(start + 1, len(paras) - 200))
            excerpt, j = [], i
            budget = r.randint(16000, 21000)
            while j < len(paras) and sum(map(len, excerpt)) < budget:
                excerpt.append(paras[j])
                j += 1
            body = "\n\n".join(excerpt)
            long.append(P("long", f"pg19:{b.stem}", "pg19", [{"role": "user", "content": f"{body}\n\n---\n\n{r.choice(BOOK_TASKS)}"}],
                          pick_think(r), share=0.35, max_prompt=5600))
    for name, spec in cfg["repos"].items():
        if spec.get("skipped"):
            continue
        for rel, text in r.sample(big := repo_files(name, spec, EXTS[spec["lang"]], 13000, 18000), min(3, len(big))):
            ext = "." + rel.rsplit(".", 1)[-1]
            long.append(P("long", f"repo:{name}", "repo", [{"role": "user", "content":
                          f"This is `{rel}` from the {name} repository.\n\n```{FENCE.get(ext, '')}\n{text.rstrip()}\n```\n\n{r.choice(LONG_CODE_TASKS)}"}],
                          pick_think(r), heldout=name in HELDOUT_REPOS, code_lang=spec["lang"], share=0.30, max_prompt=5600))
    wk = [row for row in read_jsonl(SOURCES / "wikipedia.jsonl") if 1500 <= len(row["text"]) <= 12000]
    r.shuffle(wk)
    pools = {False: [x for x in wk if not is_heldout(f"wiki:{x['id']}")], True: [x for x in wk if is_heldout(f"wiki:{x['id']}")]}
    for side, pool in pools.items():
        en = [x for x in pool if x["_file"].startswith("20231101.en")]
        zh = [x for x in pool if x["_file"].startswith("20231101.zh")]
        for k in range(60 if not side else 12):
            src = zh if k % 5 == 4 else en
            arts, total = [], 0
            budget = r.randint(16000, 21000) if src is en else r.randint(5000, 7000)
            while src and total < budget:
                a = src.pop()
                arts.append(a)
                total += len(a["text"])
            if not arts:
                break
            docs = "\n\n".join(f"<document id=\"{i + 1}\">\n{wiki_article(a)}\n</document>" for i, a in enumerate(arts))
            long.append(P("long", f"wiki:{arts[0]['id']}", "wikipedia", [{"role": "user", "content": f"{docs}\n\n{r.choice(WIKI_TASKS)}"}],
                          pick_think(r), heldout=side, share=0.35, max_prompt=5600))
    r.shuffle(long)
    add("long", "100k", long)

    write_jsonl(GEN / "prompts.jsonl", out)
    log(f"{len(out)} prompts, {sum(p['heldout'] for p in out)} held out")


if __name__ == "__main__":
    main()
