#!/usr/bin/env bash
#
# Integration tests for the Hadamard-rotated (HQ) types, on a small model (Qwen3-0.6B).
#
# usage:   tests-hq.sh <repo-root> <model-bf16.gguf> <wiki.test.raw> [work-dir]
# example: tests-hq.sh . models/Qwen3-0.6B-BF16.gguf wikitext-2-raw/wiki.test.raw /tmp/hq
#
# Needs the make targets quantize_gguf main test-hadamard test-hadamard-quants test-hadamard-quantize
# test-hadamard-llama test-hadamard-ppl. With a CUDA build (make LLAMA_CUBLAS=1 maincuda
# test-hadamard-ppl-cuda) set CUDA=1 to run the generation, quality and CUDA-graph steps on the GPU.
# BIN=<dir> takes the binaries from another directory than the repo root.

set -eu

if [ $# -lt 3 ]; then
    sed -n '3,11p' "$0"
    exit 1
fi

ROOT=$(realpath "$1")
SRC=$(realpath "$2")
TEXT=$(realpath "$3")
WORK=${4:-/tmp/hq}
CUDA=${CUDA:-0}
CHUNKS=${CHUNKS:-20}

mkdir -p "$WORK"

BIN=$(realpath "${BIN:-$ROOT}")
QUANTIZE=$BIN/quantize_gguf
MAIN=$BIN/main
PPL=$BIN/test-hadamard-ppl
NGL=0
if [ "$CUDA" = 1 ]; then
    MAIN=$BIN/maincuda
    PPL=$BIN/test-hadamard-ppl-cuda
    NGL=99
fi

PROMPT="The capital of France is"

gen() {
    "$MAIN" -no-cnv -m "$1" -p "$PROMPT" -n 24 --temp 0 -ngl $NGL 2>/dev/null
}

# the ppl/KL line of one model: bpw NLL PPL meanKL p99KL top1
quality() {
    local cache=$1; shift
    "$PPL" -ngl $NGL --chunks $CHUNKS --cache "$cache" "$@" 2>/dev/null
}

field() { # field <table> <model-basename> <column>
    echo "$1" | awk -v m="$2" -v c="$3" '$1 == m { print $c }'
}

ratio_le() { # ratio_le a b r: a <= b*r
    awk -v a="$1" -v b="$2" -v r="$3" 'BEGIN { exit !(a <= b*r) }'
}

echo "== backends have no rotated-type remaps"
if grep -rn 'ggml_get_base_type\|ggml_is_rotated' "$ROOT"/ggml/src/ggml-*/; then
    echo "FAIL: a backend still references the rotated types"
    exit 1
fi
echo PASS

echo "== unit tests"
"$BIN"/test-hadamard > "$WORK"/unit.log
"$BIN"/test-hadamard-quants >> "$WORK"/unit.log
"$BIN"/test-hadamard-quantize "$WORK" "$SRC" >> "$WORK"/unit.log 2>&1
"$QUANTIZE" --pure "$SRC" "$WORK"/pure-q4k.gguf Q4_K > /dev/null 2>&1
"$QUANTIZE" --pure "$SRC" "$WORK"/pure-iq4nl.gguf IQ4_NL > /dev/null 2>&1
"$BIN"/test-hadamard-llama "$WORK" "$WORK"/pure-q4k.gguf "$WORK"/pure-iq4nl.gguf >> "$WORK"/unit.log 2>&1
grep -c PASS "$WORK"/unit.log | sed 's/^/checks passed: /'
echo PASS

echo "== quantize with --hadamard and generate"
for FT in Q4_K_M Q5_K_M IQ4_XS IQ3_S IQ3_XS IQ2_XXS; do
    "$QUANTIZE" --hadamard "$SRC" "$WORK"/hq-$FT.gguf $FT > "$WORK"/q-$FT.log 2>&1
    OUT=$(gen "$WORK"/hq-$FT.gguf)
    echo "$FT: $OUT"
    case $FT in
        Q4_K_M|Q5_K_M|IQ4_XS)
            echo "$OUT" | grep -q Paris || { echo "FAIL: $FT generation"; exit 1; } ;;
    esac
done
echo PASS

echo "== requantize HQ5_K -> HQ4_XS"
"$QUANTIZE" --hadamard --pure "$SRC" "$WORK"/pure-hq5k.gguf Q5_K > /dev/null 2>&1
"$QUANTIZE" --hadamard --pure --allow-requantize "$WORK"/pure-hq5k.gguf "$WORK"/req-hq4xs.gguf IQ4_XS > /dev/null 2>&1
"$QUANTIZE" --hadamard --pure "$SRC" "$WORK"/pure-hq4xs.gguf IQ4_XS > /dev/null 2>&1
T=$(quality "$WORK"/ref.kl "$TEXT" "$SRC" "$WORK"/pure-hq5k.gguf "$WORK"/pure-hq4xs.gguf "$WORK"/req-hq4xs.gguf)
echo "$T"
# quantizing twice adds the two errors; re-rotating the already rotated data would be far worse
K_SUM=$(awk -v a="$(field "$T" pure-hq5k.gguf 5)" -v b="$(field "$T" pure-hq4xs.gguf 5)" 'BEGIN { print a + b }')
K_REQ=$(field "$T" req-hq4xs.gguf 5)
ratio_le "$K_REQ" "$K_SUM" 1.5 || { echo "FAIL: requantized KL $K_REQ vs HQ5_K + direct HQ4_XS $K_SUM"; exit 1; }
echo PASS

echo "== runtime LoRA and --lora merge"
python3 - "$SRC" "$WORK"/lora.gguf "$ROOT"/gguf-py <<'EOF'
import sys
sys.path.insert(0, sys.argv[3])
import numpy as np
import gguf

reader = gguf.GGUFReader(sys.argv[1])
arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
shapes = {t.name: [int(x) for x in t.shape] for t in reader.tensors}

w = gguf.GGUFWriter(sys.argv[2], arch)
w.add_string("general.type", "adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", 16.0)
rng = np.random.default_rng(0)
r = 16
for name, shape in shapes.items():
    if len(shape) != 2 or not any(k in name for k in ("attn_q.weight", "attn_v.weight", "ffn_down.weight", "ffn_up.weight")):
        continue
    n_in, n_out = shape
    w.add_tensor(name + ".lora_a", (rng.standard_normal((r, n_in)) * 0.2 / np.sqrt(n_in)).astype(np.float32))
    w.add_tensor(name + ".lora_b", (rng.standard_normal((n_out, r)) * 0.2 / np.sqrt(r)).astype(np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
EOF
"$QUANTIZE" --hadamard --pure "$SRC" "$WORK"/pure-hq4k.gguf Q4_K > /dev/null 2>&1
"$QUANTIZE" --hadamard --pure --lora "$WORK"/lora.gguf "$SRC" "$WORK"/merged-hq4k.gguf Q4_K > /dev/null 2>&1
"$QUANTIZE" --hadamard --pure --allow-requantize --lora "$WORK"/lora.gguf "$WORK"/pure-hq5k.gguf "$WORK"/merged-from-hq5k.gguf Q4_K > /dev/null 2>&1
"$QUANTIZE" --pure --lora "$WORK"/lora.gguf "$SRC" "$WORK"/merged-q8.gguf Q8_0 > /dev/null 2>&1

T0=$(quality "$WORK"/ref.kl "$TEXT" "$SRC" "$WORK"/pure-hq4k.gguf "$WORK"/pure-hq4xs.gguf)
TL=$(quality "$WORK"/ref-lora.kl --lora "$WORK"/lora.gguf "$TEXT" "$SRC" "$WORK"/pure-hq4k.gguf "$WORK"/pure-hq4xs.gguf)
# no --lora here: the cache still holds the runtime-LoRA reference written by the run above
TMERGED=$(quality "$WORK"/ref-lora.kl "$TEXT" "$SRC" "$WORK"/merged-hq4k.gguf "$WORK"/merged-from-hq5k.gguf "$WORK"/merged-q8.gguf)
echo "without LoRA:";            echo "$T0"
echo "runtime LoRA:";            echo "$TL"
echo "merged (vs runtime-LoRA reference):"; echo "$TMERGED"

for M in pure-hq4k.gguf pure-hq4xs.gguf; do
    K0=$(field "$T0" $M 5); KL=$(field "$TL" $M 5)
    ratio_le "$KL" "$K0" 1.5 || { echo "FAIL: $M KL with runtime LoRA $KL vs $K0 without"; exit 1; }
done
K_RT=$(field "$TL" pure-hq4k.gguf 5)
K_MERGED=$(field "$TMERGED" merged-hq4k.gguf 5)
K_FROM_HQ=$(field "$TMERGED" merged-from-hq5k.gguf 5)
K_Q8=$(field "$TMERGED" merged-q8.gguf 5)
ratio_le "$K_MERGED" "$K_RT" 1.5  || { echo "FAIL: merged HQ4_K KL $K_MERGED vs runtime LoRA $K_RT"; exit 1; }
ratio_le "$K_FROM_HQ" "$K_MERGED" 1.5 || { echo "FAIL: merge into an HQ source KL $K_FROM_HQ vs $K_MERGED"; exit 1; }
ratio_le "$K_Q8" 0.01 1 || { echo "FAIL: merge into a non-HQ target KL $K_Q8"; exit 1; }
echo PASS

if [ "$CUDA" = 1 ]; then
    echo "== CUDA graphs on/off"
    # skipped for builds without CUDA graphs (LLAMA_CUDA_NO_GRAPHS=1, or CUDA 11)
    if "$MAIN" -no-cnv -m "$WORK"/hq-Q4_K_M.gguf -p hi -n 1 -ngl $NGL 2>&1 | grep -q "USE_GRAPHS = 1"; then
        A=$(gen "$WORK"/hq-Q4_K_M.gguf)
        B=$(GGML_CUDA_DISABLE_GRAPHS=1 gen "$WORK"/hq-Q4_K_M.gguf)
        [ "$A" = "$B" ] || { echo "FAIL: CUDA graphs change the output"; echo "$A"; echo "$B"; exit 1; }
        echo PASS
    else
        echo "SKIPPED: this build has no CUDA graphs"
    fi
fi

echo "ALL PASS"
