#!/usr/bin/env bash

set -eu

if [ $# -lt 1 ]
then
    echo "usage:   $0 path_to_build_binary [path_to_temp_folder]"
    echo "example: $0 ../../build/bin ../../tmp"
    exit 1
fi

if [ $# -gt 1 ]
then
    TMP_DIR=$2
else
    TMP_DIR=/tmp
fi

set -x

SPLIT=$1/llama-gguf-split
QUANTIZE=$1/llama-quantize
MAIN=$1/llama-completion
WORK_PATH=$TMP_DIR/quantize
ROOT_DIR=$(realpath $(dirname $0)/../../)

mkdir -p "$WORK_PATH"

# Clean up in case of previously failed test
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf

# 1. Get a model
(
cd $WORK_PATH
"$ROOT_DIR"/scripts/hf.sh --repo ggml-org/Qwen3-0.6B-GGUF --file Qwen3-0.6B-Q8_0.gguf
)
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

# 5. Hadamard-rotated quantization (ConvRot): --hadamard converts Q4_K -> Q4R_K
$QUANTIZE --allow-requantize --hadamard $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-rot.gguf Q4_K
echo PASS
echo

# 5a. The rotated model must load and generate. A missing or wrong inference-side rotation does
#     not crash - it produces garbage - so this step only catches the gross failures (the graph
#     guard throwing, a backend having no kernel for the rotated type). Use the perplexity tool
#     for the quality check.
$MAIN -no-cnv --model $WORK_PATH/ggml-model-rot.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 5b. A rotated type can also be selected per-tensor, without --hadamard
$QUANTIZE --allow-requantize --tensor-type 'blk\..*ffn_down'=q4r_K $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-rot-partial.gguf Q4_K
echo PASS
echo

$MAIN -no-cnv --model $WORK_PATH/ggml-model-rot-partial.gguf -p "I believe the meaning of life is" --n-predict 32
echo PASS
echo

# 5c. Rotating the token embeddings is not representable by the engine - must fail loud
if $QUANTIZE --allow-requantize --token-embedding-type q4r_K $WORK_PATH/Qwen3-0.6B-Q8_0.gguf $WORK_PATH/ggml-model-rot-bad.gguf Q4_K
then
    echo "FAIL: rotated token embeddings should have been rejected"
    exit 1
fi
echo PASS
echo

# Clean up
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf $WORK_PATH/ggml-model-rot*.gguf
