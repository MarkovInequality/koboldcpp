#!/bin/bash
# iter.sh NAME ALPHA DAMP [NOTE]: quantizes REF with TYPES' tensor types as HQ types and full-Hessian GPTQ from HESSIAN
# (--hessian-alpha ALPHA, LLAMA_HQ_GPTQ_DAMP=DAMP; FTYPE, default Q4_K_M, covers the tensors TYPES doesn't list),
# scores it with eval.sh, appends a row to LOG (default WORK/tuning.md) and WORK/results.tsv, then deletes it.
# iter.sh --baseline MODEL NAME: scores an existing model and logs it; nothing is deleted.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need TYPES "a tensor-type file for --tensor-type-file"
need HESSIAN "the hessian-collect file"
LOG=${LOG:-$WORK/tuning.md}
TSV=$WORK/results.tsv

if [[ ! -f $LOG ]]; then
    cat > "$LOG" <<HDR
# Full-Hessian GPTQ tuning: $(basename "$REF"), $(basename "$TYPES") as HQ types

Each run quantizes \`$(basename "$REF")\` with \`--tensor-type-file $(basename "$TYPES") --hadamard --hessian
$(basename "$HESSIAN") --hessian-alpha α\` (\`LLAMA_HQ_GPTQ_DAMP=damp\`); GPTQ uses
H = R·((1−α)·Ĝ + α·diag(Ĝ) + damp·I)·Rᵀ with Ĝ = G / mean(diag G), so α = 1 is the diagonal (imatrix) GPTQ.
Scores are KL(reference ‖ model) and top-1 agreement with the reference, over the second half of each window
(\`tools/hessian/eval/README.md\`):

- **wiki**: wikitext-2 test, 80 windows × 512 tokens
- **heldout**: the calibration set's held-out documents, 64 windows × 1024, with \`--chat-mask\`
- **opencode**: the user's own OpenCode sessions (never calibrated on), 36 windows × 2048, with \`--chat-mask\`

The quantized file is deleted after each row is written.

| run | α | damp | quant min | bpw | wiki KL | wiki top-1 % | heldout KL | heldout top-1 % | opencode KL | opencode top-1 % | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
HDR
fi

row() { # row NAME ALPHA DAMP MINUTES EVAL_OUTPUT NOTE
    echo "| $1 | $2 | $3 | $4 | $(cells "$5") | $6 |" >> "$LOG"
    echo "$5" | awk -v n="$1" -v a="$2" -v d="$3" -v m="$4" '{ print n "\t" a "\t" d "\t" m "\t" $0 }' | tr ' ' '\t' >> "$TSV"
}

if [[ ${1:-} == --baseline ]]; then
    [[ $# -eq 3 ]] || { echo "usage: $0 --baseline MODEL NAME" >&2; exit 1; }
    res=$("$HERE/eval.sh" "$2")
    echo "$res"
    row "$3" - - - "$res" "baseline, not deleted"
    exit 0
fi
[[ $# -ge 3 ]] || { echo "usage: $0 NAME ALPHA DAMP [NOTE] | $0 --baseline MODEL NAME" >&2; exit 1; }

name=$1 alpha=$2 damp=$3 note=${4:-}
out=${QUANT_DIR:-$WORK}/tune-$name.gguf
t0=$SECONDS
LLAMA_HQ_GPTQ_DAMP=$damp "$QUANTIZE" --hadamard --tensor-type-file "$TYPES" --hessian "$HESSIAN" --hessian-alpha "$alpha" \
    "$REF" "$out" "${FTYPE:-Q4_K_M}" > "$WORK/quant-$name.log" 2>&1
mins=$(( (SECONDS - t0 + 30)/60 ))
res=$("$HERE/eval.sh" "$out")
echo "$res"
row "$name" "$alpha" "$damp" "$mins" "$res" "$note"
rm -f "$out"
