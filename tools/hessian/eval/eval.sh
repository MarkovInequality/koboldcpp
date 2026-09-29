#!/bin/bash
# eval.sh MODEL: one line per eval set, "set bpw NLL PPL meanKL p99KL top1", against the cached reference log-probs
# (run ref.sh first). Chat sets are scored with --chat-mask.
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL=$1
for s in $SETS; do
    eval_set "$s"
    flags=()
    (( CHAT )) && flags=(--special --chat-mask)
    row=$("$PPL" -ngl 99 -c $CTX --chunks $CHUNKS --parallel $PAR --cache "$WORK/$s.kl-cache" "${flags[@]}" \
          "$TEXT" "$REF" "$MODEL" 2>/dev/null | awk -v m="$(basename "$MODEL")" '$1 == m { $1 = ""; print }') || true
    echo "$s${row:- FAILED}"
done
