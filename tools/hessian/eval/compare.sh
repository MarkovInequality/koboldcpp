#!/bin/bash
# compare.sh ROTATED.gguf: quantizes REF with TYPES' tensor types twice from the same calibration data, rotated (HQ
# types, full-Hessian GPTQ from HESSIAN at the quantizer's defaults; kept as ROTATED.gguf) and unrotated (the plain
# types with IMATRIX; deleted after scoring), scores both (and BASELINE, if set) and appends the comparison to LOG
# (default WORK/tuning.md). FTYPE (default Q4_K_M) covers the tensors TYPES doesn't list.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need TYPES "a tensor-type file for --tensor-type-file"
need HESSIAN "the hessian-collect file"
need IMATRIX "an imatrix from the same calibration data (hessian-collect --imatrix-out)"
[[ $# -eq 1 ]] || { echo "usage: $0 ROTATED.gguf" >&2; exit 1; }
LOG=${LOG:-$WORK/tuning.md}
ROT=$1
PLAIN=${QUANT_DIR:-$WORK}/unrotated.gguf

t0=$SECONDS
"$QUANTIZE" --hadamard --tensor-type-file "$TYPES" --hessian "$HESSIAN" "$REF" "$ROT" "${FTYPE:-Q4_K_M}" > "$WORK/quant-rot.log" 2>&1
m_rot=$(( (SECONDS - t0 + 30)/60 ))
t0=$SECONDS
"$QUANTIZE" --tensor-type-file "$TYPES" --imatrix "$IMATRIX" "$REF" "$PLAIN" "${FTYPE:-Q4_K_M}" > "$WORK/quant-plain.log" 2>&1
m_plain=$(( (SECONDS - t0 + 30)/60 ))

r_rot=$("$HERE/eval.sh" "$ROT")
r_plain=$("$HERE/eval.sh" "$PLAIN")
echo "rotated:"; echo "$r_rot"; echo "unrotated:"; echo "$r_plain"
rows=
if [[ -n ${BASELINE:-} ]]; then
    r_base=$("$HERE/eval.sh" "$BASELINE")
    echo "baseline:"; echo "$r_base"
    rows="| $(basename "$BASELINE") (baseline) | - | $(cells "$r_base") |"$'\n'
fi
rows+="| unrotated, \`--imatrix\` | $m_plain | $(cells "$r_plain") |"$'\n'
rows+="| **rotated, full-Hessian GPTQ** | $m_rot | $(cells "$r_rot") |"

cat >> "$LOG" <<SEC

## Rotated vs unrotated, same calibration data

Both use \`$(basename "$TYPES")\`. **Rotated:** \`--hadamard --hessian $(basename "$HESSIAN")\` at the quantizer's defaults,
kept as \`$(basename "$ROT")\`. **Unrotated:** the plain types with \`--imatrix $(basename "$IMATRIX")\`, deleted after
scoring.

| quantization | quant min | bpw | wiki KL | wiki top-1 % | heldout KL | heldout top-1 % | opencode KL | opencode top-1 % |
|---|---|---|---|---|---|---|---|---|
$rows
SEC
rm -f "$PLAIN"
echo "done"
