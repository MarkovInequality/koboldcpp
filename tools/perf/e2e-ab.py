#!/usr/bin/env python3
# Interleaved A/B of kcpp-e2e.py configs: runs A and B alternately for some rounds and reports, per request, the
# median and max of the wall time and of its part outside prompt processing and generation (SmartCache snapshots and
# restores, HTTP), with processed tokens.
#
# usage: e2e-ab.py A B [--configs agentic,checkpoints] [--rounds 3] [--out DIR]
#   A, B: build:DIR (a tree with koboldcpp.py and its library) or lib:FILE (this repo's tree with FILE as the library)

import argparse, json, os, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.realpath(__file__))

def side_args(spec):
    kind, _, path = spec.partition(":")
    if kind not in ("build", "lib") or not path:
        sys.exit(f"bad side {spec!r}: use build:DIR or lib:FILE")
    return ["--build" if kind == "build" else "--lib", os.path.realpath(os.path.expanduser(path))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--configs", default="agentic,checkpoints")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.out or tempfile.mkdtemp(prefix="e2e-ab-")
    os.makedirs(out, exist_ok=True)
    runs = {"A": [], "B": []}
    for rnd in range(args.rounds):
        for side, spec in (("A", args.a), ("B", args.b)):
            path = os.path.join(out, f"{side}{rnd}.json")
            print(f"round {rnd + 1}/{args.rounds}: {side} ({spec})", flush=True)
            with open(os.path.join(out, f"{side}{rnd}.txt"), "w") as log:
                subprocess.run([sys.executable, os.path.join(HERE, "kcpp-e2e.py"), "run", "--configs", args.configs, "--out", path]
                               + side_args(spec), stdout=log, stderr=subprocess.STDOUT)
            if os.path.exists(path):
                runs[side].append(json.load(open(path)))

    def series(side, c, k, f):
        return [r[c][k][f] for r in runs[side] if c in r and k in r[c] and f in r[c][k]]

    print(f"\n{'request':34s} {'wall A med/max':>15s} {'wall B med/max':>15s} {'B/A':>6s}   {'other A':>13s} {'other B':>13s}   processed A/B")
    first = runs["A"][0] if runs["A"] else {}
    for c, d in first.items():
        if c.startswith("_") or not isinstance(d, dict):
            continue
        for k, e in d.items():
            if not isinstance(e, dict) or "wall" not in e:
                continue
            wa, wb = series("A", c, k, "wall"), series("B", c, k, "wall")
            if not wa or not wb:
                continue
            oth = lambda s: [r[c][k]["wall"] - r[c][k].get("pp_s", 0) - r[c][k].get("eval_s", 0) for r in runs[s] if c in r and k in r[c]]
            oa, ob = oth("A"), oth("B")
            pa, pb = series("A", c, k, "n_pp"), series("B", c, k, "n_pp")
            print(f"{c + '/' + k:34s} {statistics.median(wa):7.2f}/{max(wa):6.2f} {statistics.median(wb):7.2f}/{max(wb):6.2f} "
                  f"{statistics.median(wb)/statistics.median(wa):6.3f}   {statistics.median(oa):6.3f}/{max(oa):5.2f} "
                  f"{statistics.median(ob):6.3f}/{max(ob):5.2f}   {pa[0] if pa else '-'}/{pb[0] if pb else '-'}")
    print(f"\nruns in {out}")

if __name__ == "__main__":
    main()
