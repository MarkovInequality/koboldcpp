#!/usr/bin/env bash

set -eu

if [ $# -lt 1 ]
then
    echo "usage:   $0 path_to_build_binary [path_to_temp_folder]"
    echo "example: $0 ../.. ../../tmp"
    exit 1
fi

if [ $# -gt 1 ]
then
    TMP_DIR=$2
else
    TMP_DIR=/tmp
fi

set -x

SPLIT=$1/gguf-split
QUANTIZE=$1/quantize_gguf
MAIN=$1/main
WORK_PATH=$TMP_DIR/quantize
ROOT_DIR=$(realpath $(dirname $0)/../../)

mkdir -p "$WORK_PATH"

# Clean up in case of previously failed test
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf

# 1. Get a model
if [ ! -f $WORK_PATH/Qwen3-0.6B-Q8_0.gguf ]
then
    curl -fL -o $WORK_PATH/Qwen3-0.6B-Q8_0.gguf.part https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
    mv $WORK_PATH/Qwen3-0.6B-Q8_0.gguf.part $WORK_PATH/Qwen3-0.6B-Q8_0.gguf
fi
echo PASS

# 2. Split model
$SPLIT --split-max-tensors 28  $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-split
echo PASS
echo

# 3. Requant model with '--keep-split'
$QUANTIZE --allow-requantize --keep-split $WORK_PATH/ggml-model-split-00001-of-00012.gguf $WORK_PATH/ggml-model-requant.gguf Q4_K
echo PASS
echo

# 3a. Test the requanted model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-requant-00001-of-00012.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 4. Requant mode without '--keep-split'
$QUANTIZE --allow-requantize $WORK_PATH/ggml-model-split-00001-of-00012.gguf $WORK_PATH/ggml-model-requant-merge.gguf Q4_K
echo PASS
echo

# 4b. Test the requanted model is loading properly
$MAIN -no-cnv --model $WORK_PATH/ggml-model-requant-merge.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 5. Hadamard-rotated (HQ) quantization; tools/quantize/tests-hq.sh has the full HQ suite
grep -rn 'ggml_get_base_type\|ggml_is_rotated' "$ROOT_DIR"/ggml/src/ggml-*/ && { echo "FAIL: a backend references the rotated types"; exit 1; }
$QUANTIZE --allow-requantize --hadamard $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-rot.gguf Q4_K
$MAIN -no-cnv --model $WORK_PATH/ggml-model-rot.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 5a. Rotated token embeddings are refused
if $QUANTIZE --allow-requantize --token-embedding-type hq4_K $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-rot-bad.gguf Q4_K
then
    echo "FAIL: rotated token embeddings should have been rejected"
    exit 1
fi
echo PASS
echo

# Clean up
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf $WORK_PATH/ggml-model-rot*.gguf
