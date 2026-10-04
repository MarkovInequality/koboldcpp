#!/bin/bash
# the rollback suite on CUDA (fused), CUDA with fusion off, and the CPU
# usage: tests/test-recurrent-state-rollback.sh MODEL [extra args]
set -u
cd "$(dirname "$0")/.."
model=$1; shift
args=(-m "$model" -c 4096 -p "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away there lived" "$@")
fail=0
run() {
    local label=$1; shift
    if "$@" > /tmp/rollback-$$.log 2>&1; then echo "$label: PASS"; else echo "$label: FAIL"; fail=1; fi
    grep -a "^test_\|^  \|^fork" /tmp/rollback-$$.log | sed 's/^/    /'
}
run "CUDA"            ./test-recurrent-state-rollback-cuda -ngl 99 "${args[@]}"
run "CUDA, no fusion" env GGML_CUDA_DISABLE_FUSION=1 ./test-recurrent-state-rollback-cuda -ngl 99 "${args[@]}"
run "CPU"             ./test-recurrent-state-rollback -ngl 0 -ub 16 "${args[@]}"
rm -f /tmp/rollback-$$.log
exit $fail
