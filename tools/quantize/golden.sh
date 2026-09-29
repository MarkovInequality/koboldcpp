#!/usr/bin/env bash
#
# Golden outputs for refactoring the quantizer and the Hessian collector: quantizes a model in fixed
# configurations and runs hessian-collect once, then records the files' sha256 (record) or redoes them and
# compares (check). A change that isn't meant to change any output must pass check.
#
# usage: tools/quantize/golden.sh record|check <model-bf16.gguf> <imatrix.gguf> <hessian.gguf> <text> <dir>
# e.g.:  tools/quantize/golden.sh record tools/models/Qwen3-0.6B-BF16.gguf im06.gguf h06.gguf tools/models/wiki.train.raw /tmp/golden
# BIN=<dir> takes the binaries (quantize_gguf, hessian-collect) from another directory than the repo root.

set -u

if [ $# -ne 6 ] || { [ "$1" != record ] && [ "$1" != check ]; }; then
    sed -n '3,9p' "$0"
    exit 1
fi

MODE=$1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$(realpath "${BIN:-$ROOT}")
SRC=$(realpath "$2")
IM=$(realpath "$3")
HESS=$(realpath "$4")
TEXT=$(realpath "$5")
DIR=$6
mkdir -p "$DIR"
SUMS=$DIR/golden.sha256

CONFIGS=(
    "hq-q4km        --hadamard Q4_K_M"
    "hq-q4km-im     --hadamard --imatrix $IM Q4_K_M"
    "hq-iq3m-im     --hadamard --imatrix $IM IQ3_M"
    "hq-q4_0        --hadamard Q4_0"
    "hq-iq4xs-im    --hadamard --imatrix $IM IQ4_XS"
    "plain-q4km-im  --imatrix $IM Q4_K_M"
    "hq-q4km-hess   --hadamard --hessian $HESS Q4_K_M"
)

out=$DIR/current.sha256
: > "$out"
for c in "${CONFIGS[@]}"; do
    read -r name args <<< "$c"
    ft=${args##* }
    args=${args% *}
    # shellcheck disable=SC2086
    if ! "$BIN"/quantize_gguf $args "$SRC" "$DIR/$name.gguf" "$ft" > "$DIR/$name.log" 2>&1; then
        echo "FAIL: $name: quantize_gguf failed, see $DIR/$name.log"
        exit 1
    fi
    (cd "$DIR" && sha256sum "$name.gguf") >> "$out"
    rm -f "$DIR/$name.gguf"
done

if ! "$BIN"/hessian-collect -m "$SRC" --text "$TEXT" --chunk 512 --chunks 4 --output-stride 1 --group-layers 29 \
        --threads "$(nproc)" -o "$DIR/hc4.gguf" > "$DIR/hc4.log" 2>&1; then
    echo "FAIL: hessian-collect failed, see $DIR/hc4.log"
    exit 1
fi
(cd "$DIR" && sha256sum hc4.gguf) >> "$out"
rm -f "$DIR/hc4.gguf"

if [ "$MODE" = record ]; then
    mv "$out" "$SUMS"
    echo "recorded $(wc -l < "$SUMS") outputs in $SUMS"
    exit 0
fi

if [ ! -f "$SUMS" ]; then
    echo "FAIL: nothing recorded in $DIR"
    exit 1
fi
fail=0
while read -r sum name; do
    want=$(awk -v n="$name" '$2 == n { print $1 }' "$SUMS")
    if [ "$sum" = "$want" ]; then
        echo "[ok]   $name"
    else
        echo "[FAIL] $name differs from the recorded output"
        fail=1
    fi
done < "$out"
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
