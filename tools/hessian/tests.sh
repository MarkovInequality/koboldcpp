#!/usr/bin/env bash
#
# Tests for hessian-collect on Qwen3-0.6B: the GEMM-input hook, the file format and loaders, count spans and
# resume on the CPU (fp64) path, and with a CUDA build the SYRK path against it. Outputs go to
# tools/hessian/hessian-tests/, which is removed when every test passes.
#
# usage: tools/hessian/tests.sh <Qwen3-0.6B-BF16.gguf> <wiki.train.raw> [hook] [cpu] [cuda]   (default: all)
# Needs the make targets hessian-collect test-hessian quantize_gguf (hessian-collect-cuda for cuda).
# THREADS=N limits the CPU threads; M27=<Qwen3.8-27B GGUF> adds the 27B coverage check to hook.

set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
W=$ROOT/tools/hessian/hessian-tests
if [[ $# -lt 2 || ! -f $1 || ! -f $2 ]]; then
    echo "usage: $0 <Qwen3-0.6B-BF16.gguf> <wiki.train.raw> [hook] [cpu] [cuda]" >&2
    exit 1
fi
M=$(realpath "$1")
TEXT=$(realpath "$2")
shift 2
HC=$ROOT/hessian-collect
HCC=$ROOT/hessian-collect-cuda
TH=$ROOT/test-hessian
THREADS=${THREADS:-$(nproc)}
PHASES=${*:-hook cpu cuda}
mkdir -p "$W"
cd "$W"

n_fail=0
check() {
    local what=$1; shift
    if "$@" > "$W/last.log" 2>&1; then
        echo "[ok]   $what"
    else
        echo "[FAIL] $what"; tail -5 "$W/last.log" | sed 's/^/       /'
        n_fail=$((n_fail + 1))
    fi
}
note() { echo "       $*"; }

common=(--threads "$THREADS" --output-stride 1 --dataset wiki20)
docs20=(--text "$TEXT" --chunk 512 --chunks 20)
docs4=(--text "$TEXT" --chunk 512 --chunks 4)

# documents for the count-span tests: one of ~1500 tokens, and two others to pack it with
mkdir -p spans
python3 - "$TEXT" "$ROOT/hessian-tokenize" "$M" <<'PY'
import json, subprocess, sys
t = open(sys.argv[1], encoding="utf-8").read()
a, b, c = t[200000:206000], t[300000:303000], t[400000:404000]
n = len(json.loads(subprocess.run([sys.argv[2], sys.argv[3]], input=json.dumps(a) + "\n", capture_output=True, text=True).stdout))
def w(name, docs):
    with open(f"spans/{name}.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
w("full", [{"id": "d", "text": a}])
w("fullspan", [{"id": "d", "text": a, "count": [[0, n]]}])
w("first", [{"id": "d", "text": a, "count": [[0, 700]]}])
w("rest", [{"id": "d", "text": a, "count": [[700, n]]}])
w("packed", [{"id": "x", "text": b, "count": []}, {"id": "d", "text": a}, {"id": "y", "text": c, "count": []}])
w("tail", [{"id": "d", "text": a, "count": [[60, 100]]}])
PY

if [[ " $PHASES " == *" hook "* ]]; then
    echo "== GEMM-input hook"
    check "logits unchanged with the hook on (CPU)" "$HC" -m "$M" "${docs4[@]}" -o x.gguf --threads "$THREADS" --test-hook
    check "coverage: 196 layer weights + LM head, 113 input groups" bash -c "'$HC' -m '$M' ${docs4[*]} -o x.gguf --threads $THREADS --dry-run > dry.txt && grep -q '197 weights in 113 groups' dry.txt"
    check "groups: q/k/v share an input, gate/up share an input" bash -c "grep -q 'blk.5.attn_k.weight blk.5.attn_q.weight blk.5.attn_v.weight' dry.txt && grep -q 'blk.5.ffn_gate.weight blk.5.ffn_up.weight' dry.txt"
    check "every input is a computed F32 node with uniform rows" bash -c "! grep -q 'non-uniform' dry.txt"
    grep -E '^layer  0 ' dry.txt | sed 's/^/       /'
    if [[ -n ${M27:-} ]]; then
        check "27B coverage: 496 layer weights + output; attn_gate/attn_qkv/ssm_alpha/ssm_beta share an input" bash -c \
            "'$HC' -m '$M27' ${docs4[*]} -o x.gguf --threads $THREADS --dry-run > dry27.txt && grep -q '497 weights' dry27.txt && grep -q 'blk.0.attn_gate.weight blk.0.attn_qkv.weight blk.0.ssm_alpha.weight blk.0.ssm_beta.weight' dry27.txt"
    fi
fi

if [[ " $PHASES " == *" cpu "* ]]; then
    echo "== CPU (fp64) path, format, loaders"
    check "collect 20 chunks (one pass)" "$HC" -m "$M" "${docs20[@]}" "${common[@]}" --group-layers 29 -o h20.gguf --imatrix-out h20-im.gguf
    check "format: Grams and aliases load, G symmetric, in_sum2 = diag G bitwise" "$TH" check h20.gguf --weights 197
    check "imatrix-out = the hessian file's in_sum2/counts" "$TH" diag h20.gguf h20-im.gguf --tol 0
    check "imatrix loader reads a hessian file in < 100 MB" bash -c "'$TH' imatrix-load h20.gguf | tee /dev/stderr | awk '{ exit !(\$(NF-1) < 100) }'"
    note "$(tail -1 last.log)"

    [[ -f hq8.gguf ]] || "$ROOT/quantize_gguf" --hadamard "$M" hq8.gguf Q8_0 > quant-hq8.log 2>&1
    check "collect on the HQ8_0 model" "$HC" -m hq8.gguf "${docs20[@]}" "${common[@]}" --group-layers 29 -o h20-hq8.gguf
    # quantization noise grows with depth (0.02 at layer 0, 0.08 at layer 27); a rotated capture gives 1.4
    check "natural space: HQ8_0 vs BF16 Grams within 0.15 relative Frobenius (a rotated capture: 1.4)" "$TH" compare h20-hq8.gguf h20.gguf --per-count --frob --tol 0.15
    note "$(tail -1 last.log)"

    mkdir -p qa qb
    ln -sf ../h20.gguf qa/im.gguf
    ln -sf ../h20-im.gguf qb/im.gguf
    (cd qa && "$ROOT/quantize_gguf" --hadamard --imatrix im.gguf "$M" out.gguf Q4_K_M > quant.log 2>&1)
    (cd qb && "$ROOT/quantize_gguf" --hadamard --imatrix im.gguf "$M" out.gguf Q4_K_M > quant.log 2>&1)
    check "--imatrix <hessian file> quantizes byte-identically to --imatrix <its imatrix-out>" cmp qa/out.gguf qb/out.gguf
    rm -f qa/out.gguf qb/out.gguf

    check "resume: stop after 2 of 6 passes" "$HC" -m "$M" "${docs4[@]}" "${common[@]}" --group-layers 5 --stop-after-pass 2 -o res.gguf
    check "resume: finish" "$HC" -m "$M" "${docs4[@]}" "${common[@]}" --group-layers 5 --resume -o res.gguf
    check "resume: uninterrupted run" "$HC" -m "$M" "${docs4[@]}" "${common[@]}" --group-layers 5 -o full5.gguf
    check "resumed file = uninterrupted file" cmp res.gguf full5.gguf

    # counted [60, 100) at stride 64 in ubatches of 64: position 60 is the only output; the second ubatch has
    # none, so its last token (99, counted) gets a logit only to make the decode valid
    check "a forced logit: collect (small ubatches)" "$HC" -m "$M" --docs spans/tail.jsonl --trim-docs --ubatch 64 --output-stride 64 --threads "$THREADS" --layers out -o spans/tail.gguf
    check "... the LM head counts only the document's own output" "$TH" count spans/tail.gguf token_embd.weight 1

    for s in full fullspan first rest packed; do
        check "collect spans/$s" "$HC" -m "$M" --docs spans/$s.jsonl "${common[@]}" --group-layers 29 -o spans/$s.gguf
    done
    check "count spans: complementary spans add up to the whole document within 1e-6" "$TH" sum spans/first.gguf spans/rest.gguf spans/full.gguf --tol 1e-6
    note "$(tail -1 last.log)"
    check "count spans: a span over the whole document = no span, bytewise" cmp spans/fullspan.gguf spans/full.gguf
    check "packing: the document's Grams packed with others = alone" "$TH" compare spans/packed.gguf spans/full.gguf --tol 1e-5
    note "$(tail -1 last.log)"
fi

if [[ " $PHASES " == *" cuda "* ]]; then
    echo "== CUDA (SYRK) path"
    if [[ ! -x $HCC ]]; then
        echo "[FAIL] no $HCC"; n_fail=$((n_fail + 1))
    else
        check "logits unchanged with the hook on (CUDA)" "$HCC" -m "$M" -ngl 99 "${docs4[@]}" -o x.gguf --test-hook
        check "GPU accumulation, one pass" "$HCC" -m "$M" -ngl 99 "${docs20[@]}" "${common[@]}" --group-layers 29 -o g20.gguf
        check "CPU accumulation of the same GPU activations" "$HCC" -m "$M" -ngl 99 "${docs20[@]}" "${common[@]}" --group-layers 29 --accum cpu -o g20-cpu.gguf
        # measured: median 1.4e-8, 99.99th percentile 6.7e-7, max 1.8e-5 (11 of 441M entries above 1e-5)
        check "precision: GPU fp32 vs CPU fp64 within 5e-5" "$TH" compare g20.gguf g20-cpu.gguf --tol 5e-5
        note "$(tail -1 last.log)"
        check "GPU accumulation, 6 passes" "$HCC" -m "$M" -ngl 99 "${docs20[@]}" "${common[@]}" --group-layers 5 -o g20-p6.gguf
        check "pass invariance: 1 pass = 6 passes, bytewise" cmp g20.gguf g20-p6.gguf
        check "host inputs: a layer's norm on the CPU" "$HCC" -m "$M" -ngl 99 -ot 'blk\.3\.attn_norm\.weight=CPU' --no-op-offload "${docs20[@]}" "${common[@]}" --group-layers 29 -o g20-host.gguf
        check "host inputs: same Grams within 1e-5" "$TH" compare g20-host.gguf g20.gguf --tol 1e-5
        note "$(tail -1 last.log)"
        check "all inputs on the host (-ngl 0), SYRK from staging" "$HCC" -m "$M" -ngl 0 "${docs4[@]}" "${common[@]}" --group-layers 29 -o h4-cuda.gguf
        check "all inputs on the host, CPU fp64" "$HCC" -m "$M" -ngl 0 "${docs4[@]}" "${common[@]}" --group-layers 29 --accum cpu -o h4-cpu.gguf
        check "host staging path = fp64 within 5e-5" "$TH" compare h4-cuda.gguf h4-cpu.gguf --tol 5e-5
        note "$(tail -1 last.log)"
    fi
fi

if [[ $n_fail -eq 0 ]]; then
    echo "all tests passed"
    cd "$ROOT" && rm -rf "$W"
else
    echo "$n_fail tests failed; outputs kept in $W"
    exit 1
fi
