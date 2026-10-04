#!/bin/bash
# The correctness half of the standing suite: run before and after every change, fully at every phase end.
# Builds the CUDA test binaries first. Speed checks are separate (llama-bench, kcpp-e2e, test-backend-ops perf).
#
# usage: tools/perf/standing-suite.sh [--quick] [--golden FILE]
#   --quick   skip the CPU rollback leg and the e2e run
#   --golden  kcpp-e2e golden to check against (default: tools/perf/golden/kcpp-e2e-27b.json)
set -u
cd "$(dirname "$0")/../.."
model=${MODEL:-$HOME/AI/qwen3/Qwen3.8-27B-HQ4_K_M.gguf}
small=${SMALL_MODEL:-tools/models/Qwen3-0.6B-BF16.gguf}
golden=tools/perf/golden/kcpp-e2e-27b.json
quick=0
while [ $# -gt 0 ]; do
    case $1 in
        --quick) quick=1 ;;
        --golden) golden=$2; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
    shift
done

log=$(mktemp -d /tmp/standing-suite-XXXXXX)
fail=()
step() {
    local name=$1; shift
    printf "%-44s " "$name"
    if "$@" > "$log/${name//[^a-zA-Z0-9]/_}.log" 2>&1; then echo PASS; else echo "FAIL (see $log)"; fail+=("$name"); fi
}

make LLAMA_CUBLAS=1 -j8 test-backend-ops test-hadamard-archs-cuda test-hadamard-archs-cuda-novmm \
    test-save-load-state-cuda test-recurrent-state-rollback-cuda test-recurrent-state-rollback test-kcpp-sampler koboldcpp_cublas \
    > "$log/build.log" 2>&1 || { echo "build failed, see $log/build.log"; exit 1; }

# cases that assert the global fusion/RHT counters run on one thread
ops="FLASH_ATTN_EXT,MUL_MAT,MUL_MAT_ID,GATED_DELTA_NET,RMS_NORM,RMS_NORM_SCALE,NORM_SCALE,SSM_CONV,GLU,SET_ROWS"
counted="RHT,RHT_FUSED,ADD_ADD,ADD_RMS_NORM,GATED_DELTA_NET_CACHE_FUSION,MUL_MAT_ID_FUSION,MUL_MAT_VEC_FUSION,RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE,ROPE_SET_ROWS,SSM_CONV_BIAS_SILU,SSM_SCAN_ROLLBACK,TOPK_MOE,MUL_MAT_HADAMARD"
step "test-backend-ops ops"       ./test-backend-ops -b CUDA0 -j 4 -o "$ops"
step "test-backend-ops RHT and fusion cases" ./test-backend-ops -b CUDA0 -o "$counted"
step "test-kcpp-sampler"          ./test-kcpp-sampler
step "test-hadamard-archs-cuda"   ./test-hadamard-archs-cuda -d CUDA0
step "test-hadamard-archs-cuda -g" ./test-hadamard-archs-cuda -g
step "-g, sync uploads"           env GGML_SCHED_SYNC_UPLOADS=1 ./test-hadamard-archs-cuda -g
step "-g, NO_VMM"                 ./test-hadamard-archs-cuda-novmm -g -a '^(llama|qwen35|qwen3moe)$'
step "-g, -ngl 1"                 ./test-hadamard-archs-cuda -g -ngl 1 -a '^(llama|qwen35|qwen35moe|qwen3moe|gemma3|mamba|deepseek2)$'
step "-g, 2 devices, pp"          env GGML_CUDA_DEVICES=2 ./test-hadamard-archs-cuda -g -pp -a '^(llama|qwen35|qwen35moe|qwen3moe|gemma3|mamba|deepseek2)$'
step "test-save-load-state-cuda (0.6B)" ./test-save-load-state-cuda -m "$small" -ngl 99 -c 4096
step "test-save-load-state-cuda (27B)"  ./test-save-load-state-cuda -m "$model" -ngl 99 -c 4096
if [ $quick = 1 ]; then
    step "rollback suite (CUDA)"  ./test-recurrent-state-rollback-cuda -m "$model" -ngl 99 -c 4096 -p "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away there lived"
else
    step "rollback suite (CUDA, no fusion, CPU)" tests/test-recurrent-state-rollback.sh "$model"
    step "kcpp-e2e check"         venv/bin/python tools/perf/kcpp-e2e.py check "$golden" --configs mtp,nomtp,guidance,grammar,media
fi

echo
if [ ${#fail[@]} -eq 0 ]; then echo "standing suite: PASS"; rm -rf "$log"; else echo "standing suite: FAIL (${fail[*]}), logs in $log"; exit 1; fi
