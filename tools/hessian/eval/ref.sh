#!/bin/bash
# ref.sh: the reference's log-probs for every eval set, cached in WORK (see common.sh); a set whose cache is already
# valid is skipped by test-hadamard-ppl. REF_NGL places the reference (default 99; 33 for Qwen3.8-27B BF16 on 32 GB).
set -euo pipefail
source "$(dirname "$0")/common.sh"
for s in $SETS; do
    eval_set "$s"
    flags=()
    (( CHAT )) && flags=(--special)
    t0=$SECONDS
    "$PPL" --ref-ngl "${REF_NGL:-99}" -c $CTX --chunks $CHUNKS --parallel $PAR --cache "$WORK/$s.kl-cache" "${flags[@]}" \
        "$TEXT" "$REF" > "$WORK/ref-$s.log" 2>&1
    echo "$s: $((SECONDS - t0)) s"
done
