#!/usr/bin/env python3
# Interleaved llama-bench A/B: runs each binary once per round, in turn, so slow GPU modes hit every binary alike;
# reports per test the median and max of each binary and the per-round (paired) differences against the first.
#
#   lb-ab.py ROUNDS "LLAMA-BENCH ARGS" BIN_A BIN_B [...] [--csv OUT] [--model GGUF]
#
# The binaries get the user's setup (-ngl 99 -fa on -ctk/-ctv q5_1 -ub 1024) before the given arguments, e.g.
#   lb-ab.py 6 "-p 5 -n 32 -nrs 4 -r 5" ./llama-bench-old ./llama-bench-cuda
#   lb-ab.py 3 "-d 88000 -p 5 -n 16 -nrs 4 -r 3" ./llama-bench-old ./llama-bench-cuda

import argparse, csv, io, os, statistics, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("rounds", type=int)
ap.add_argument("args")
ap.add_argument("bins", nargs="+")
ap.add_argument("--csv", help="also append every row here")
ap.add_argument("--model", default=os.path.expanduser("~/AI/qwen3/Qwen3.8-27B-HQ4_K_M.gguf"))
a = ap.parse_args()

base = ["-m", a.model, "-ngl", "99", "-fa", "on", "-ctk", "q5_1", "-ctv", "q5_1", "-ub", "1024", "-o", "csv"]
rows = []
for r in range(a.rounds):
    for b in a.bins:
        out = subprocess.run([b] + base + a.args.split(), capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(f"{b} failed:\n{out.stderr[-2000:]}")
        # keep the CSV: llama-bench's log lines can land on stdout too
        lines = [l for l in out.stdout.splitlines() if l.startswith('"') or l.startswith("build_commit")]
        for row in csv.DictReader(io.StringIO("\n".join(lines))):
            row["bin"], row["round"] = b, r
            rows.append(row)
            test = f"pp{row['n_prompt']} tg{row['n_gen']} d{row['n_depth']}"
            print(f"round {r} {os.path.basename(b):24s} {test:20s} {float(row['avg_ns'])/1e6:9.2f} ms", flush=True)

if a.csv:
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)

tests = sorted({(int(x["n_prompt"]), int(x["n_gen"]), int(x["n_depth"])) for x in rows})
for t in tests:
    name = f"pp{t[0]} tg{t[1]} d{t[2]}"
    ms = {b: [float(x["avg_ns"])/1e6 for x in rows if x["bin"] == b and (int(x["n_prompt"]), int(x["n_gen"]), int(x["n_depth"])) == t]
          for b in a.bins}
    print(f"{name}: " + "  ".join(f"{os.path.basename(b)} med {statistics.median(v):8.2f} ms (max {max(v):8.2f}, n={len(v)})"
                                    for b, v in ms.items()))
    ref = ms[a.bins[0]]
    for b in a.bins[1:]:
        d = [y - x for x, y in zip(ref, ms[b])]
        print(f"  {os.path.basename(b)} - {os.path.basename(a.bins[0])}: " + " ".join(f"{x:+.2f}" for x in d) +
              f"  median {statistics.median(d):+.2f} ms, faster in {sum(x < 0 for x in d)} of {len(d)}")
