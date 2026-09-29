#!/usr/bin/env python3
# Untemplated documents: web, encyclopedic, fiction and code (category raw) and fineweb-2 web text (the raw half
# of multilingual), each cut to a length drawn uniformly from 1-8k tokens, concatenating source documents of
# one kind where one is too short. Writes work/gen/raw_docs.jsonl.

import collections

import kcpp
from common import *
from prompts import EXTS, HELDOUT_REPOS, LANG_SHARE, ML_LANGS, is_heldout, repo_files

OUT = GEN / "raw_docs.jsonl"


def cut(text, n):
    """text cut to at most n tokens, at a character boundary"""
    ids = kcpp.tokenize(text)
    if len(ids) <= n:
        return text, len(ids)
    prefix = kcpp.detokenize(ids[:n]).rstrip("�")
    if not text.startswith(prefix):
        prefix = text[:int(len(text) * n / len(ids))]
    while True:
        k = len(kcpp.tokenize(prefix))
        if k <= n:
            return prefix, k
        prefix = prefix[:int(len(prefix)*0.98)]


def build(pool, target, r, unit_of, sep="\n\n", min_len=1024):
    """documents from consecutive texts of pool (list of (unit, text)) until target tokens"""
    docs, total, i = [], 0, 0
    while total < target and i < len(pool):
        want = min(r.randint(min_len, MAX_LEN), max(256, int(target - total)))
        parts, units, n = [], [], 0
        while i < len(pool) and n < want:
            u, t = pool[i]
            i += 1
            parts.append(t.strip())
            units.append(u)
            n += len(t) // 3
        text, k = cut(sep.join(parts), want)
        if k < 256:
            continue
        docs.append({"text": text, "n_tokens": k, "unit": unit_of(units)})
        total += k
    return docs


def main():
    cfg = load_sources()
    prompts = read_jsonl(GEN / "prompts.jsonl")
    used_titles = set()
    used_fw2 = set()
    used_files = set()
    for p in prompts:
        body = p["messages"][0]["content"]
        if p["source"] == "wikipedia":
            used_titles.update(re.findall(r'<document id="\d+">\n# (.*)\n', body))
        elif p["source"] == "fineweb2":
            used_fw2.add(p["unit"])
        elif p["source"] == "repo" and p.get("file"):
            used_files.add((p["unit"], p["file"]))

    out = []

    def emit(category, source, docs, heldout, **kw):
        for k, d in enumerate(docs):
            out.append({"id": f"{category}-{source}-{'h' if heldout else 'c'}{k:03d}-{stable_hash(d['text'][:4000]):08x}",
                        "category": category, "unit": d["unit"], "heldout": heldout, "templated": False,
                        "bucket": [category, source, kw.get("lang")], "source": source, "n_tokens": d["n_tokens"],
                        **kw, "text": d["text"]})

    def sides(rows, unit_fn):
        pools = {False: [], True: []}
        for row in rows:
            pools[is_heldout(unit_fn(row))].append(row)
        return pools

    # fineweb-edu 40k
    r = rng("raw-fineweb-edu")
    rows = read_jsonl(SOURCES / "fineweb_edu.jsonl")
    r.shuffle(rows)
    for h, pool in sides(rows, lambda x: f"fineweb-edu:{x['id']}").items():
        docs = build([(f"fineweb-edu:{x['id']}", x["text"]) for x in pool], TARGETS["raw"]*0.40*(HELDOUT if h else 1), r,
                     lambda us: us[0])
        emit("raw", "fineweb_edu", docs, h)

    # wikipedia 30k, articles not used in long-context prompts
    r = rng("raw-wikipedia")
    rows = [x for x in read_jsonl(SOURCES / "wikipedia.jsonl") if x["title"] not in used_titles]
    r.shuffle(rows)
    for h, pool in sides(rows, lambda x: f"wiki:{x['id']}").items():
        docs = build([(f"wiki:{x['id']}", f"{x['title']}\n\n{x['text']}") for x in pool], TARGETS["raw"]*0.30*(HELDOUT if h else 1), r,
                     lambda us: us[0])
        emit("raw", "wikipedia", docs, h)

    # pg19 15k: excerpts, one book per document
    r = rng("raw-pg19")
    books = sorted((SOURCES / "pg19").iterdir())
    for h in (False, True):
        target = TARGETS["raw"]*0.15*(HELDOUT if h else 1)
        docs, total = [], 0
        mine = [b for b in books if is_heldout(f"pg19:{b.stem}") == h]
        r.shuffle(mine)
        for b in mine * 3:
            if total >= target:
                break
            text = b.read_text(errors="replace")
            want = min(r.randint(1024, MAX_LEN), max(1024, int(target - total)))
            start = r.randrange(len(text)//5, max(len(text)//5 + 1, len(text) - want*6))
            start = text.find("\n\n", start) + 2
            piece, k = cut(text[start:start + want*8], want)
            docs.append({"text": piece, "n_tokens": k, "unit": f"pg19:{b.stem}"})
            total += k
        emit("raw", "pg19", docs, h)

    # code 15k: files of the pinned repos, by language share; a document concatenates files of one repo
    r = rng("raw-code")
    for h in (False, True):
        target = TARGETS["raw"]*0.15*(HELDOUT if h else 1)
        files = collections.defaultdict(lambda: collections.defaultdict(list))
        for name, spec in cfg["repos"].items():
            if spec.get("skipped") or (name in HELDOUT_REPOS) != h:
                continue
            for rel, text in repo_files(name, spec, EXTS[spec["lang"]], 1500, 40000):
                if (f"repo:{name}", rel) not in used_files:
                    files[spec["lang"]][f"repo:{name}"].append(text)
        for by_unit in files.values():
            for v in by_unit.values():
                r.shuffle(v)
        total = 0
        while total < target:
            langs = [l for l in LANG_SHARE if any(files[l].values())]
            if not langs:
                break
            lang = r.choices(langs, weights=[LANG_SHARE[l] for l in langs])[0]
            unit = r.choice(sorted(u for u, v in files[lang].items() if v))
            want, parts, n = min(r.randint(1024, MAX_LEN), max(1024, int(target - total))), [], 0
            while files[lang][unit] and n < want:
                parts.append(files[lang][unit].pop())
                n += len(parts[-1]) // 3
            text, k = cut("\n\n".join(parts), want)
            emit("raw", "code", [{"text": text, "n_tokens": k, "unit": unit}], h, code_lang=lang)
            total += k

    # multilingual raw half: fineweb-2, 25k by language share
    r = rng("raw-fineweb2")
    rows = [x for x in read_jsonl(SOURCES / "fineweb2.jsonl") if f"fineweb2:{x['id']}" not in used_fw2]
    held_total, held_target = 0, TARGETS["multilingual"]*0.25*HELDOUT
    for sub in sorted(ML_LANGS, key=lambda s: stable_hash(s)):
        share = ML_LANGS[sub][1]
        mine = [x for x in rows if x["_file"].split("/")[1] == sub]
        r.shuffle(mine)
        for h, pool in sides(mine, lambda x: f"fineweb2:{x['id']}").items():
            if h and held_total >= held_target:
                continue
            target = held_target - held_total if h else TARGETS["multilingual"]*0.25*share
            docs = build([(f"fineweb2:{x['id']}", x["text"]) for x in pool], min(target, 1024) if h else target, r, lambda us: us[0])
            held_total += sum(d["n_tokens"] for d in docs) if h else 0
            emit("multilingual", "fineweb2_raw", docs, h, lang=sub)

    write_jsonl(OUT, out)
    tot = collections.Counter()
    for d in out:
        tot[(d["category"], d["source"], d["heldout"])] += d["n_tokens"]
    for k, v in sorted(tot.items()):
        log(k, v)


if __name__ == "__main__":
    main()
