#!/usr/bin/env python3
# Checks and diagnostics on hessian-collect output. Grams are read in row slabs through a memory map, so the
# 101 GB 27B file is never loaded whole; the fp64 factorizations hold one Gram at a time.
#
#   analyze.py sanity FILE [--unsloth IMATRIX] [--layers 0,31,63] [--expect-count N]
#       counts, finiteness (streamed over every Gram), fp64 Cholesky of G/mean(diag G) + 0.01 I for the inputs of
#       the given layers and the LM head, per-weight correlation of log mean-square activations with another
#       imatrix, and channel 3994's share of the attention-input energy in the first layers
#   analyze.py epsilon A B [--layers ...]
#       per input: eps = ||(G_B + l I)^-1/2 (G_A - G_B) (G_B + l I)^-1/2||_2 with l = 0.01 mean(diag G_B), both Grams
#       divided by their counts, and the relative Frobenius difference
#   analyze.py subsets CALIB.jsonl OUTDIR
#       document files for the diagnostics: halves A/B split by source unit, and the OpenCode documents with their
#       counted positions limited to 0-8k, 8-32k and 32-64k

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

SLAB = 1024


class Hessian:
    def __init__(self, path):
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        f = self.r.fields
        names = f["hessian.alias.names"].contents() if "hessian.alias.names" in f else []
        owners = f["hessian.alias.owners"].contents() if "hessian.alias.owners" in f else []
        self.owner = {n: o for n, o in zip(names, owners)}
        for k in self.t:
            if k.endswith(".in_gram"):
                self.owner[k[:-8]] = k[:-8]

    def weights(self):
        return sorted(k[:-8] for k in self.t if k.endswith(".in_sum2"))

    def grams(self):
        return sorted(k[:-8] for k in self.t if k.endswith(".in_gram"))

    def n(self, w):
        return self.t[w + ".in_sum2"].data.size

    def count(self, w):
        return float(np.asarray(self.t[w + ".counts"].data).ravel()[0])

    def sum2(self, w):
        return np.asarray(self.t[w + ".in_sum2"].data, dtype=np.float64).ravel()

    def rows(self, w, r0, r1):
        n = self.n(w)
        return np.asarray(self.t[self.owner[w] + ".in_gram"].data).reshape(n, n)[r0:r1]

    def gram64(self, w):
        n = self.n(w)
        g = np.empty((n, n))
        for r0 in range(0, n, SLAB):
            g[r0:r0 + SLAB] = self.rows(w, r0, min(n, r0 + SLAB))
        return g


def layer_of(w):
    return int(w.split(".")[1]) if w.startswith("blk.") else None


def pick(names, layers):
    return [w for w in names if layer_of(w) in layers or (layer_of(w) is None and "out" in layers)]


def parse_layers(s):
    out = set()
    for item in (s.split(",") if s != "none" else []):
        out.add("out" if item in ("out", "output") else int(item))
    return out


def cmd_sanity(a):
    h = Hessian(a.file)
    ok = True
    counts = {w: h.count(w) for w in h.weights()}
    blk = [c for w, c in counts.items() if layer_of(w) is not None]
    out = [c for w, c in counts.items() if layer_of(w) is None]
    print(f"{len(counts)} weights, {len(h.grams())} Grams; block counts {min(blk):.0f}..{max(blk):.0f}, LM head {out}")
    if a.expect_count:
        good = all(c == a.expect_count for c in blk)
        print(f"[{'ok' if good else 'FAIL'}] every block weight counts {a.expect_count} rows; LM head {out[0]/a.expect_count if out else 0:.4f} of it")
        ok &= good
    bad = []
    for g in h.grams():
        n = h.n(g)
        for r0 in range(0, n, SLAB):
            if not np.isfinite(h.rows(g, r0, min(n, r0 + SLAB))).all():
                bad.append(g)
                break
    print(f"[{'ok' if not bad else 'FAIL'}] every Gram entry is finite {bad[:5]}")
    ok &= not bad
    layers = parse_layers(a.layers)
    for g in pick(h.grams(), layers):
        G = h.gram64(g)
        G /= np.mean(np.diag(G))
        G[np.diag_indices_from(G)] += 0.01
        try:
            np.linalg.cholesky(G)
            print(f"[ok]   Cholesky of G/mean(diag G) + 0.01 I: {g} (n = {G.shape[0]})")
        except np.linalg.LinAlgError:
            print(f"[FAIL] Cholesky of G/mean(diag G) + 0.01 I: {g}")
            ok = False
        del G
    if a.unsloth:
        u = Hessian(a.unsloth)
        cors = []
        for w in h.weights():
            if w + ".in_sum2" not in u.t:
                continue
            x = np.log(np.maximum(h.sum2(w) / h.count(w), 1e-30))
            y = np.log(np.maximum(u.sum2(w) / u.count(w), 1e-30))
            cors.append((np.corrcoef(x, y)[0, 1], w))
        cors.sort()
        c = np.array([x for x, _ in cors])
        print(f"log mean-square activations vs {Path(a.unsloth).name}: {len(cors)} weights, correlation median {np.median(c):.4f}, "
              f"min {c.min():.4f} ({cors[0][1]}), 5th percentile {np.quantile(c, 0.05):.4f}")
    for g in h.grams():
        L = layer_of(g)
        if L is not None and L < 4 and h.n(g) > 3994 and ("attn_k" in g or "attn_gate" in g):
            s = h.sum2(g)
            print(f"channel 3994 share of the attention-input energy, layer {L}: {s[3994] / s.sum():.1%}")
    print("sanity:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def epsilon(GA, GB):
    lam = 0.01 * np.mean(np.diag(GB))
    from scipy.linalg import solve_triangular
    GB[np.diag_indices_from(GB)] += lam
    L = np.linalg.cholesky(GB)
    GB[np.diag_indices_from(GB)] -= lam
    D = GA - GB
    X = solve_triangular(L, D, lower=True, overwrite_b=True)
    X = solve_triangular(L, X.T, lower=True, overwrite_b=True).T  # L^-1 D L^-T: the spectrum of (G_B + l I)^-1/2 D (G_B + l I)^-1/2
    X = (X + X.T) / 2
    v = np.random.default_rng(0).standard_normal(X.shape[0])
    lam_max = 0.0
    for _ in range(300):
        w = X @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        v = w / nw
        if abs(nw - lam_max) < 1e-7 * nw:
            break
        lam_max = nw
    return lam_max, np.linalg.norm(GA - GB) / np.linalg.norm(GB)


def cmd_epsilon(a):
    A, B = Hessian(a.a), Hessian(a.b)
    layers = parse_layers(a.layers) if a.layers else None
    names = [g for g in B.grams() if g in A.owner and (layers is None or g in pick([g], layers))]
    print(f"{'input (owner)':34s} {'n':>6s} {'eps':>8s} {'rel. Frobenius':>15s}")
    for g in names:
        GA = A.gram64(g) / A.count(g)
        GB = B.gram64(g) / B.count(g)
        e, f = epsilon(GA, GB)
        print(f"{g:34s} {GB.shape[0]:6d} {e:8.4f} {f:15.4f}", flush=True)
        del GA, GB
    return 0


def cmd_subsets(a):
    docs = [json.loads(l) for l in open(a.calib)]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    half = lambda d: int(hashlib.sha1(d["unit"].encode()).hexdigest(), 16) % 2
    for k, name in ((0, "half-a"), (1, "half-b")):
        with open(out / f"{name}.jsonl", "w") as f:
            for d in docs:
                if half(d) == k:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
    oc = [d for d in docs if d["id"].startswith("oc-")]
    for lo, hi, name in ((0, 8192, "pos-0-8k"), (8192, 32768, "pos-8-32k"), (32768, 65536, "pos-32-64k"), (65536, 131072, "pos-64-128k")):
        with open(out / f"{name}.jsonl", "w") as f:
            for d in oc:
                n = d["count"][-1][1] if d.get("count") else None
                if n is None or n <= lo:
                    continue
                f.write(json.dumps(dict(d, count=[[lo, min(hi, n)]]), ensure_ascii=False) + "\n")
    print(f"halves: {sum(half(d) == 0 for d in docs)} / {sum(half(d) == 1 for d in docs)} documents; OpenCode documents: {len(oc)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("sanity")
    p.add_argument("file")
    p.add_argument("--unsloth")
    p.add_argument("--layers", default="0,31,63,out")
    p.add_argument("--expect-count", type=float)
    p = sp.add_parser("epsilon")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--layers")
    p = sp.add_parser("subsets")
    p.add_argument("calib")
    p.add_argument("outdir")
    a = ap.parse_args()
    sys.exit({"sanity": cmd_sanity, "epsilon": cmd_epsilon, "subsets": cmd_subsets}[a.cmd](a))


if __name__ == "__main__":
    main()
