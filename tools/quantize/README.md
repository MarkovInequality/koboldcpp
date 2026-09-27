# quantize

This tool takes a GGUF input model file, typically in a high-precision format like F32 or BF16, and converts it to a quantized format.
Quantization reduces the precision of model weights (e.g., from 32-bit floats to 4-bit integers), which shrinks the model's size and can speed up inference.
This process however, may introduce some accuracy loss which is usually measured in [Perplexity](https://huggingface.co/docs/transformers/en/perplexity) (ppl) and/or [Kullback–Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (kld).
This can be minimized by using a suitable imatrix file.

You can also use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to build your own quants without any setup. It syncs from llama.cpp `main` every 6 hours.

## Overview

Quantization is done in two phases:
- Convert the original model to GGUF format.
- Quantize the converted GGUF file.

If the model supports multimodal inputs (images or audio), you also need to convert and quantize the multimodal encoders and projectors.

To perform these tasks, you need to install the Python requirements:

```bash
python3 -m pip install -r requirements.txt
```

Or if you use `uv`:

```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

## Prepare the input GGUF file

To convert a model from a Hugging Face repo, you can use a command like the following:

```
python convert_hf_to_gguf.py --outfile gemma-4-E2B-it-bf16.gguf --outtype bf16 --remote google/gemma-4-E2B-it
```

Notes:
- In the usual case where the model is distributed in 16-bit format, `--outtype auto` (or omitting `--outtype` entirely) also works well.
- If you have previously downloaded the model locally, specify the directory and remove the `--remote` flag.
- For compatibility reasons, the Python requirements install transformers 4, but more and more models (like Gemma 4) require transformers 5. You can safely `pip install -U transformers` to get the latest version.

## Quantize the GGUF

After you have created a high-quality GGUF version of the model, you use `quantize_gguf` (built with `make quantize_gguf`) to apply quantization. For example, quantize to `Q4_K_M` using a command like the following:

```bash
./quantize_gguf gemma-4-E2B-it-bf16.gguf gemma-4-E2B-it-Q4_K_M.gguf Q4_K_M
```

Various quantization methods are described [later in this document](#quantize).

Options:
* `--allow-requantize` allow requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
* `--leave-output-tensor` leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
* `--pure` disable k-quant mixtures and quantizes all tensors to the same type
* `--hadamard` rotate each weight row by a randomized Hadamard transform and store the tensor as its [HQ type](#hadamard-rotated-hq-quantization), improving quality at the same file size
* `--hadamard-seed N` seed of the rotation's random signs (default: a fixed seed, so builds are reproducible)
* `--lora FILE[:scale]` merge a LoRA adapter into the weights while quantizing (may be repeated)
* `--imatrix file_name` use data in file_name as importance matrix for quant optimizations
* `--include-weights tensor_name` use importance matrix for this tensor (can be specified multiple times)
* `--exclude-weights tensor_name` use importance matrix for the tensors **not** specified (include/exclude cannot be mixed)
* `--output-tensor-type` use a specific quant type for the output.weight tensor
* `--token-embedding-type` use a specific quant type for the token embeddings tensor
* `--keep-split` generate the quantized model in the same shards as the input file instead of a single quantized file

Advanced options:
* `--tensor-type` quantize specific tensor(s) to specific quant types. Supports regex syntax. May be specified multiple times.
* `--prune-layers` prune (remove) the layers in the list
* `--override-kv` option to override model metadata by key in the quantized model. May be specified multiple times.

## (Optional) Convert the multimodal components

llama.cpp will convert the LLM portion of the source model, which is enough for conversational applications. If the model accepts multimodal inputs and you wish to take advantage of them, you need to create a separate GGUF file. This file is generically known as `mmproj`, for "multimedia projector"; however, it may contain various components such as vision or audio encoders in addition to projections.

Multimodal components are usually much smaller than the LLMs they come with. In addition, their quality has a direct impact on the quality of LLM generations, because these components are in charge of preparing the inputs for the LLM: the closer inputs are to data seen during training, the better LLM results will be.

For these reasons, multimodal components are usually kept in a high-quality format such as bf16 or q8. The impact on speed and memory from using a smaller quant is negligible, but overall quality could be impacted.

```bash
python convert_hf_to_gguf.py --mmproj --outfile mmproj-gemma-4-E2B-it-Q8_0.gguf --outtype q8_0 --remote google/gemma-4-E2B-it
```

## Run the quantized model


```bash
./build/bin/llama cli -m ./gemma-4-E2B-it-Q4_K_M.gguf --mmproj ./mmproj-gemma-4-E2B-it-Q8_0.gguf --image <input_image> --prompt "Describe this image"
```

## Quantization Examples

```bash
# naive Q4_K_M quantization using default settings and 8 CPU threads. Output will be "ggml-model-Q4_K_M.gguf"
./quantize_gguf input-model-f32.gguf q4_k_m 8
```

```bash
#  quantize model enabling re-quantization, leaving the output tensor unquantized and all others quantized at the same level (Q4_K)
./quantize_gguf --allow-requantize --leave-output-tensor --pure input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model using an importance matrix for specified tensors only (attn_v and ffn_down)
./quantize_gguf --imatrix imatrix.gguf --include-weights attn_v --include-weights ffn_down input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model setting output tensor to Q5_K_M, token embeddings to Q3_K_M, and keeping the input file's shards
./quantize_gguf --imatrix imatrix.gguf --output-tensor-type q5_k --token-embedding-type q3_k --keep-split input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model using a regex to quantize attn_k tensors in odd layers to Q5_K_M and attn_q tensors in even layers to Q3_K_M
./quantize_gguf --imatrix imatrix.gguf --tensor-type "\.(\d*[13579])\.attn_k=q5_k" --tensor-type "\.(\d*[02468])\.attn_q=q3_k" input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model setting tensors attn_v and ffn_down to Q5_K_M and pruning layers 20, 21, and 22
./quantize_gguf --imatrix imatrix.gguf --tensor-type attn_v=q5_k --tensor-type ffn_down=q5_k --prune-layers 20,21,22 input-model-f32.gguf q4_k_m 8
```

```bash
# override expert used count metadata to 16, prune layers 20, 21, and 22 without quantizing the model (copy tensors) and use specified name for the output file
./quantize_gguf --imatrix imatrix.gguf --override-kv qwen3moe.expert_used_count=int:16 --prune-layers 20,21,22 input-model-f32.gguf pruned-model-f32.gguf copy 8
```

```bash
# Hadamard-rotated (HQ) Q4_K_M: every Q4_K/Q5_K/IQ* tensor of the mix becomes its HQ variant. Same size, lower KL
./quantize_gguf --hadamard input-model-f16.gguf q4_k_m 8
```

```bash
# the same with an importance matrix: the HQ tensors are quantized with GPTQ error feedback
./quantize_gguf --hadamard --imatrix imatrix.gguf input-model-f16.gguf q4_k_m 8
```

```bash
# rotate only the ffn_down tensors, leaving everything else unrotated
./quantize_gguf --tensor-type "ffn_down=hq4_K" input-model-f16.gguf q4_k_m 8
```

```bash
# merge a LoRA adapter at half strength while quantizing
./quantize_gguf --hadamard --lora adapter.gguf:0.5 input-model-f16.gguf q4_k_m 8
```

## Memory/Disk Requirements

When running the larger models, make sure you have enough disk space to store all the intermediate files.
As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same. For example (Llama 3.1):

| Model | Original size | Quantized size (Q4_K_M) |
| ----: | ------------: | ----------------------: |
|    8B |       32.1 GB |                  4.9 GB |
|   70B |      280.9 GB |                 43.1 GB |
|  405B |    1,625.1 GB |                249.1 GB |


## Quantization

Several quantization methods are supported. They differ in the resulting model disk size and inference speed. For example,

### [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

| Measure                     | IQ1_S        | IQ1_M        | IQ2_XXS      | IQ2_XS        | IQ2_S         | IQ2_M        |
| --------------------------- | ------------ | ------------ | ------------ | ------------- | ------------- | ------------ |
| bits/weight                 |       2.0042 |       2.1460 |       2.3824 |        2.5882 |        2.7403 |       2.9294 |
| size (GiB)                  |       1.87   |       2.01   |       2.23   |        2.42   |        2.56   |       2.74   |
| prompt processing t/s @ 512 | 858.88 ±1.22 | 847.99 ±0.47 | 852.39 ±0.85 | 826.99 ±12.51 | 783.55 ±13.73 | 787.68 ±7.00 |
| text generation t/s @ 128   |  79.73 ±0.79 |  72.92 ±0.14 |  79.86 ±0.22 |  78.04 ±0.46  |  77.30 ±2.47  |  74.44 ±0.15 |

| Measure                     | IQ3_XXS      | IQ3_XS       | IQ3_S        | IQ3_M         | IQ4_XS        | IQ4_NL       |
| --------------------------- | ------------ | ------------ | ------------ | ------------- | ------------- | ------------ |
| bits/weight                 |       3.2548 |       3.4977 |       3.6606 |        3.7628 |        4.4597 |       4.6818 |
| size (GiB)                  |       3.04   |       3.27   |       3.42   |        3.52   |        4.17   |       4.38   |
| prompt processing t/s @ 512 | 813.88 ±6.53 | 708.71 ±1.26 | 798.78 ±8.81 | 768.70 ±13.73 | 771.80 ±11.38 | 806.03 ±7.07 |
| text generation t/s @ 128   |  73.95 ±0.20 |  71.67 ±0.54 |  69.31 ±0.63 |  70.15 ±0.33  |  77.51 ±0.20  |  76.63 ±0.28 |


| Measure                     | Q2_K_S       | Q2_K         | Q3_K_S       | Q3_K_M       | Q3_K_L       | Q4_K_S       |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| bits/weight                 |       2.9697 |       3.1593 |       3.6429 |       3.9960 |       4.2979 |       4.6672 |
| size (GiB)                  |       2.78   |       2.95   |       3.41   |       3.74   |       4.02   |       4.36   |
| prompt processing t/s @ 512 | 798.91 ±6.40 | 784.45 ±7.85 | 752.17 ±7.94 | 783.44 ±9.92 | 761.17 ±7.55 | 818.55 ±9.58 |
| text generation t/s @ 128   |  90.01 ±0.12 |  79.85 ±0.20 |  69.84 ±0.18 |  71.68 ±0.22 |  69.38 ±0.49 |  76.71 ±0.20 |

| Measure                     | Q4_K_S       | Q4_K_M        | Q5_K_S       | Q5_K_M       | Q6_K          | Q8_0         |
| --------------------------- | ------------ | ------------- | ------------ | ------------ | ------------- | ------------ |
| bits/weight                 |       4.6672 |        4.8944 |       5.5704 |       5.7036 |        6.5633 |       8.5008 |
| size (GiB)                  |       4.36   |        4.58   |       5.21   |       5.33   |        6.14   |       7.95   |
| prompt processing t/s @ 512 | 818.55 ±9.58 | 821.81 ±21.44 | 752.52 ±0.99 | 758.69 ±7.43 | 812.01 ±10.82 | 865.09 ±8.30 |
| text generation t/s @ 128   |  76.71 ±0.20 |  71.93 ±1.52  |  69.53 ±0.18 |  67.23 ±1.08 |  58.67 ±3.13  |  50.93 ±0.08 |

| Measure                     | F16          |
| --------------------------- | ------------ |
| bits/weight                 |      16.0005 |
| size (GiB)                  |      14.96   |
| prompt processing t/s @ 512 | 923.49 ±0.53 |
| text generation t/s @ 128   |  29.17 ±0.04 |

## Hadamard-rotated (HQ) quantization

`--hadamard` stores each weight row `w` as `R·w`, where `R` is a randomized Hadamard transform over the row's **full** input dimension, and quantizes that. At inference the engine rotates the activation feeding the GEMM by the same `R`, so the result is unchanged:

```
y = (W·Rᵀ)·(R·x) = W·x
```

`R` is orthogonal, so it moves nothing but the basis. What it buys is well-behaved stored values: the rotation spreads every outlier across the whole row, so each quant block sees near-Gaussian values and no single value forces a block's scale. Quantization error drops **at identical file size**, since an HQ type uses its base type's block layout exactly.

`R = (1/√n)·(H_K ⊗ H_P)·diag(s)`: `s` are random signs from the seed, `H_P` is a Walsh-Hadamard transform of a power-of-2 size `P`, and `H_K` is a fixed Hadamard matrix of order `K` (a multiple of 4, from N. J. A. Sloane's library) that covers the odd part of `n = K·P`. Every row width divisible by 4 whose odd part is at most 63 is supported: that is every hidden and FFN width of the common model families (Llama, Mistral, Qwen2.5/3, Gemma 2/3, Phi-3, ...). Tensors of other widths keep their unrotated type, with a warning.

### Types

| HQ type | base type |
| ------- | --------- |
| `HQ4_0` / `HQ4_1` / `HQ5_0` / `HQ5_1` / `HQ8_0` | `Q4_0` / `Q4_1` / `Q5_0` / `Q5_1` / `Q8_0` |
| `HQ2_K` / `HQ3_K` / `HQ4_K` / `HQ5_K` / `HQ6_K` | `Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` |
| `HQ2_XXS` / `HQ2_XS` / `HQ2_S` | `IQ2_XXS` / `IQ2_XS` / `IQ2_S` |
| `HQ3_XXS` / `HQ3_S` | `IQ3_XXS` / `IQ3_S` |
| `HQ4_NL` / `HQ4_XS` | `IQ4_NL` / `IQ4_XS` |

There are no HQ file types: `--hadamard` maps each tensor type the chosen mix picks to its HQ variant, so `--hadamard Q4_K_M` gives HQ4_K / HQ6_K tensors and `--hadamard IQ3_XS` gives HQ3_S / HQ3_XXS / HQ4_K tensors. Types without an HQ variant (the float types, IQ1_S/IQ1_M, TQ1_0/TQ2_0, MXFP4/NVFP4) stay as they are. Individual tensors can be selected with `--tensor-type`, e.g. `--tensor-type "ffn_down=hq4_K"`.

Each HQ quantizer is its base quantizer's search with uniform weights. The HQ variants of the legacy types (Q4_0 ... Q8_0) differ in one more way: they re-pick every code at the stored scale, as the K-quants already do, which is never worse.

### With an imatrix: GPTQ

**With `--imatrix`, HQ tensors are quantized with GPTQ error feedback** in the rotated space; without one, with nearest-neighbour rounding (uniform weights). The IQ2 HQ types don't require an imatrix, but they need one to be useful.

GPTQ uses the imatrix `v` as a diagonal Hessian of the unrotated inputs, rotated along with the rows: `H = R·(diag(v/mean(v)) + 0.01·I)·Rᵀ`. The rotation spreads the few loud input channels over every coordinate, so no per-coordinate weight can protect them, but `H` keeps them in its off-diagonal. The quantizer picks each value in turn (for the IQ2/IQ3 grids, each group of 8) with the type's usual scales and rounding rules, and feeds its error into the values not yet quantized, so that the errors cancel along the loud directions. The file format and inference are unchanged.

* **Getting an imatrix:** this fork has no imatrix tool. Use upstream llama.cpp's `llama-imatrix` on the unrotated model; the same file serves the unrotated and the HQ types.
* **Cost:** one Cholesky factorization of `H⁻¹` per distinct input width and imatrix entry (tensors that share an input, such as q/k/v and gate/up, reuse it), in fp64 on the CPU: `n³/3` flops, about 8 s at `n = 17408` on 8 cores. Encoding adds about `n²` multiply-adds per weight row. The factorization needs `10·n²` bytes (3 GB at `n = 17408`); a tensor whose factor would exceed `--max-buffer-size` (default 8 GiB, enough up to `n ≈ 29 000`) keeps uniform weights, with a warning. The factors and their cache are capped by `--max-buffer-size` separately from the tensor buffers, so peak memory can reach about twice it.
* **Tensors without an imatrix entry** use uniform weights, with one warning that counts them, and so do the experts of a 3D tensor whose imatrix slice is all zero. The unrotated types in the same mix use the imatrix as usual.
* **Metadata:** `quantize.hq.gptq_damp` records the damping when GPTQ ran, and is kept when HQ tensors are copied from a file that has it; nothing reads it.
* **Developer switches:** `LLAMA_HQ_GPTQ=0` turns GPTQ off (HQ tensors then ignore the imatrix, byte-identical to a run without one), and `LLAMA_HQ_GPTQ_DAMP=<x>` sets the damping (default 0.01, at least 0.001).

### Measured quality

KL is the mean KL divergence to the BF16 model's next-token distribution (lower is better), over 80 windows of 512 tokens with the second half of each scored. The imatrix comes from upstream `llama-imatrix` on wikitext-2's training split (100 × 512 tokens). Single types use `--pure --token-embedding-type q6_K`. "—": the type needs an imatrix.

**Qwen3-0.6B, with GPTQ.** Each cell is KL on wikitext-2 test / on `tech-eval`, 210 KB of this repo's docs and C++ that lies outside the imatrix's domain. The embedding is 26 % of this model's parameters, which is why the bpw is high.

| type | bpw | plain | imatrix | HQ | HQ + GPTQ (`--hadamard --imatrix`) |
| ---- | --: | ----: | ------: | -: | ---------------------------------: |
| Q8_0 | 8.00 | 0.00363 / 0.00266 | 0.00363 / 0.00266 | 0.00305 / 0.00212 | **0.00238 / 0.00166** |
| Q6_K | 6.57 | 0.0134 / 0.0094 | 0.0099 / 0.0075 | 0.0117 / 0.0088 | **0.0051 / 0.0042** |
| Q5_1 | 6.15 | 0.0548 / 0.0407 | 0.0276 / 0.0222 | 0.0418 / 0.0300 | **0.0156 / 0.0131** |
| Q5_0 | 5.78 | 0.0538 / 0.0411 | 0.0438 / 0.0336 | 0.0532 / 0.0389 | **0.0177 / 0.0155** |
| Q5_K | 5.78 | 0.0499 / 0.0364 | 0.0301 / 0.0239 | 0.0471 / 0.0347 | **0.0175 / 0.0141** |
| Q4_1 | 5.41 | 0.2297 / 0.1578 | 0.0875 / 0.0691 | 0.1656 / 0.1134 | **0.0519 / 0.0432** |
| Q4_0 | 5.04 | 0.2124 / 0.1445 | 0.1506 / 0.1186 | 0.2656 / 0.1890 | **0.0642 / 0.0559** |
| Q4_K | 5.04 | 0.1886 / 0.1375 | 0.0863 / 0.0723 | 0.1685 / 0.1176 | **0.0541 / 0.0455** |
| IQ4_NL | 5.04 | 0.1433 / 0.1048 | 0.0943 / 0.0760 | 0.1772 / 0.1385 | **0.0542 / 0.0471** |
| IQ4_XS | 4.86 | 0.1486 / 0.1062 | 0.0972 / 0.0778 | 0.1870 / 0.1427 | **0.0579 / 0.0489** |
| Q3_K | 4.26 | 0.5854 / 0.4533 | 0.3992 / 0.3234 | 0.9008 / 0.7100 | **0.2061 / 0.1814** |
| IQ3_S | 4.26 | 2.668 / 1.806 | 0.3299 / 0.2490 | 0.7997 / 0.4844 | **0.1805 / 0.1523** |
| IQ3_XXS | 3.98 | — | 0.5249 / 0.4327 | 1.541 / 1.005 | **0.3325 / 0.2832** |
| Q2_K | 3.66 | 8.257 / 9.973 | 1.191 / 1.163 | 6.346 / 5.542 | **0.830 / 0.718** |
| IQ2_S | 3.61 | — | 1.880 / 1.425 | 3.951 / 3.628 | **0.6471 / 0.5829** |
| IQ2_XS | 3.43 | — | 2.117 / 2.192 | 7.339 / 6.701 | **1.063 / 0.933** |
| IQ2_XXS | 3.24 | — | 3.243 / 4.029 | 9.144 / 10.73 | **1.555 / 1.401** |

(Q8_0 ignores an imatrix, so its two columns are the same file. Q3_K is the `Q3_K_M` ftype with `--pure`, IQ2_S the `IQ2_M` one.)

| mix | bpw | plain | imatrix | `--hadamard` | `--hadamard --imatrix` |
| --- | --: | ----: | ------: | -----------: | ---------------------: |
| Q8_0 | 8.50 | 0.00263 / 0.00194 | 0.00263 / 0.00194 | 0.00192 / 0.00140 | **0.00125 / 0.00097** |
| Q6_K | 6.57 | 0.0134 / 0.0094 | 0.0099 / 0.0075 | 0.0117 / 0.0088 | **0.0051 / 0.0042** |
| Q5_K_M | 5.89 | 0.0371 / 0.0262 | 0.0257 / 0.0206 | 0.0334 / 0.0245 | **0.0141 / 0.0116** |
| Q4_K_M | 5.25 | 0.1105 / 0.0818 | 0.0672 / 0.0572 | 0.1073 / 0.0784 | **0.0427 / 0.0353** |
| Q4_0 | 5.04–5.05 | 0.2124 / 0.1445 | 0.1440 / 0.1096 | 0.2656 / 0.1890 | **0.0641 / 0.0556** |
| IQ4_XS | 4.86–4.88 | 0.1453 / 0.1053 | 0.0972 / 0.0778 | 0.1505 / 0.1156 | **0.0579 / 0.0489** |
| Q3_K_M | 4.58 | 0.4286 / 0.3284 | 0.2362 / 0.1895 | 0.3794 / 0.2725 | **0.1270 / 0.1008** |
| IQ3_XS | 4.12 | — | 0.4180 / 0.3240 | 0.9310 / 0.6595 | **0.2301 / 0.1938** |
| Q2_K | 3.90 | 2.477 / 2.058 | 0.7625 / 0.6658 | 2.009 / 1.576 | **0.4614 / 0.4117** |
| IQ2_XXS | 3.00 | — | 3.367 / 3.973 | 7.351 / 7.147 | **1.296 / 1.270** |

GPTQ adds a roughly fixed 3–7 s per run on this model: 1.3–1.6× the time of uniform HQ for the IQ types, up to 5× for the fastest uniform quantizers (HQ3_K: 1.5 s → 8.3 s; Q4_K_M: 6.0 s → 8.2 s on 8 threads).

**Qwen3-4B, without GPTQ** (wikitext-2 test, 20 400 tokens scored; measured before GPTQ went in, so the HQ column is uniform weights, and before `--hadamard` rotated the mixes' Q6_K tensors):

| type | bpw | KL plain | KL imatrix | KL HQ | PPL plain / imatrix / HQ |
| ---- | --: | -------: | ---------: | ----: | ------------------------ |
| Q5_K | 5.60 | 0.0286 | **0.0170** | 0.0240 | 14.14 / 14.48 / 13.85 |
| Q4_K | 4.70 | 0.0966 | **0.0523** | 0.0844 | 14.96 / 15.08 / 15.81 |
| IQ4_NL | 4.70 | 0.0812 | **0.0615** | 0.1036 | 15.62 / 15.56 / 15.58 |
| IQ4_XS | 4.48 | 0.0814 | **0.0611** | 0.1101 | 15.64 / 15.44 / 16.33 |
| IQ3_S | 3.74 | 0.6182 | **0.1663** | 0.2649 | 19.74 / 14.88 / 16.09 |
| IQ3_XXS | 3.40 | — | **0.2689** | 0.5145 | — / 15.51 / 18.44 |
| IQ2_S | 2.95 | — | **0.7201** | 1.1914 | — / 19.46 / 30.96 |
| IQ2_XS | 2.73 | — | **0.9122** | 3.0025 | — / 22.15 / 215.3 |
| IQ2_XXS | 2.50 | — | **1.4076** | 3.3513 | — / 33.31 / 198.6 |

| mix | bpw | KL plain | KL imatrix | KL `--hadamard` |
| --- | --: | -------: | ---------: | --------------: |
| Q4_K_M | 4.96 | 0.0690 | **0.0451** | 0.0561 |
| IQ4_XS | 4.50–4.54 | 0.0739 | **0.0578** | 0.0940 |
| IQ3_XS | 3.60 | — | **0.2092** | 0.3168 |
| IQ2_XXS | 2.47 | — | **1.1819** | 2.5960 |

(BF16: PPL 14.25. "—": the type needs an imatrix. Wikitext perplexity alone is a poor guide at these differences, which is why KL is reported.)

**What this means:**

* **With an imatrix, use `--hadamard --imatrix`.** On Qwen3-0.6B, HQ + GPTQ beats the unrotated types with the same imatrix at every type and mix: KL is 34–60 % lower from 3 to 8 bits and 30–68 % lower at 2 bits, on in-domain and out-of-domain text alike. A prototype of the same quantizer measured 32–38 % on Qwen3-4B.
* **Without an imatrix, `--hadamard` helps Q4_1, Q5_1, Q8_0, the K-quants except Q3_K, and IQ3_S:** HQ4_1 and HQ5_1 cut KL by 24–28 % against plain Q4_1 and Q5_1, and HQ3_S more than halves IQ3_S's KL. `--hadamard Q4_K_M` is 3–4 % better than plain Q4_K_M on Qwen3-0.6B (it cut KL by 19 % on Qwen3-4B before its Q6_K tensors were rotated too).
* **Without an imatrix, don't use HQ4_0, HQ3_K, HQ4_XS or HQ4_NL** (worse than their plain types) or **the HQ2 types** (unusable). With an imatrix, all of them beat their unrotated counterparts.

Why uniform HQ needs GPTQ: a few input channels of an LLM carry most of the activation energy (attention-sink / massive activations, e.g. four channels at ±1000–5000 in a couple of `ffn_down` inputs). Plain quantization happens to be accurate in exactly those weight columns — for IQ4 especially — while the rotation spreads each of them over the whole row, where nearest-neighbour rounding gives them only average accuracy. An imatrix can't be applied per rotated column to fix this; error feedback in the rotated space, with the imatrix as a diagonal Hessian, does (see "With an imatrix: GPTQ" above).

### Speed

Each rotated GEMM input costs one extra kernel (two for decode-sized batches), shared by every weight that reads the same activation (QKV share one rotation, FFN up/gate another). On CUDA the rotation writes the 8-bit activation layout the quantized GEMMs need, so it replaces their own conversion. RTX 5090, 9285-token prompt + 512 generated tokens, HQ vs the same model unrotated:

| model | prompt processing | generation |
| ----- | ----------------- | ---------- |
| Qwen3-4B Q4_K_M | −9 % | −10 % |
| Qwen3-4B Q4_K_M, HQ4_K/HQ5_K only (before 2026-09-26) | −6 % | −7 % (−14 % without CUDA graphs) |
| Qwen3.8-27B Q4_K_M, HQ4_K/HQ5_K only (before 2026-09-26) | −3 % | −5 % |

`--hadamard Q4_K_M` now also rotates the mix's Q6_K tensors, which costs about 3 % more prompt throughput and 2 % more generation on Qwen3-4B. Their `ffn_down` inputs need rotations of their own, and where `attn_v` is HQ6_K it shares the Q/K rotation but needs a different 8-bit layout than HQ4_K, so on CUDA that rotation writes F32 and each GEMM converts it itself.

On the CPU (8 threads, Qwen3-4B pure Q4_K), both prompt processing and generation are about 5 % slower.

### Seed and metadata

The random signs come from a seed stored in the file as `hadamard.seed` (uint64), the only HQ metadata key. The default seed is fixed, so builds are reproducible; `--hadamard-seed N` picks another. A file has one seed: requantizing a file that already has rotated tensors keeps its seed (and `--hadamard-seed` with a different value is an error).

### Requantizing

* HQ → the same HQ type is a plain copy.
* HQ → a different HQ type (e.g. HQ5_K → HQ4_XS) dequantizes and quantizes again, without re-rotating.
* HQ → an unrotated type is refused: the stored values are in the rotated space. Requantize from the original model instead.

### Merging LoRA adapters

`--lora FILE[:scale]` (repeatable) merges adapters into the weights while quantizing, for any target type. Deltas are added before the rotation; for an HQ source they are rotated with the file's seed first. The merged adapters are recorded in `general.merged_loras`. Merging into an f16/bf16 source quantizes once; **merging into an already quantized source quantizes twice** and costs quality accordingly. Runtime LoRA adapters work on HQ models unchanged.

### Supported devices

HQ models run on the **CPU** and the **CUDA backend** (CUDA, plus ROCm and MUSA, which build the same code; ROCm and MUSA are **unverified**, since they haven't been run on real hardware). A model with HQ tensors is refused at load if any other device is in use (Vulkan, Metal, SYCL, OpenCL, RPC, ...): the rotation would silently fall back to the CPU with a device round trip for every rotated activation. On a Vulkan or Metal build, run HQ models CPU-only (koboldcpp: `--usecpu`; llama tools: `--device none`).

### Things to know

* **These files only load in this fork**, and not in its older ConvRot builds: the HQ types use ggml type indices 150–166, which upstream llama.cpp does not have. Builds of this fork from before 2026-09-26 only know 150–158, so they can't load files with the HQ variants of the legacy types or of Q2_K/Q3_K/Q6_K. Do not distribute HQ files.
* **ConvRot files (`Q4R_K`/`Q5R_K`) no longer load.** They are refused with a hint; requantize them from the original model.
* **Token embeddings are never rotated**; asking for it is an error. Neither is `output.weight` in a file without `token_embd` (CodeShell reads its embeddings from it).
* **Speculative-decoding drafts that use the target's output projection** (EAGLE3, DFlash) rotate their input with the target's seed when that projection is rotated.
* **Some architectures are not supported yet.** Models whose graphs call the matmul directly rather than through the common wrapper (deepseek2, glm-dsa, plm and a few others) refuse to run with a rotated weight, naming the offending tensor.

### Per-weight sensitivity: `tensor-kl`

`tensor-kl` measures how much each weight of a model matters to its quantization, at the model's operating point. It takes a model to perturb (usually a quantized mix, e.g. an Unsloth UD file or a `--hadamard` one) and the float model it came from. One weight at a time, it replaces the weight by the same weight quantized to each of a list of types, and measures how much the KL divergence and top-token agreement against the float model change from the unperturbed model's. The result is a `--tensor-type-file` with the measured changes in place of the types, which shows where a mix would gain most from more bits, or lose least from fewer.

Build it with `make LLAMA_CUBLAS=1 tensor-kl-cuda` (CPU only: `make tensor-kl`).

```bash
./tensor-kl-cuda [options] <text-file> <reference.gguf> <model.gguf>
```

* `<text-file>`: the text to score, e.g. wikitext-2's `wiki.test.raw`. It's cut into windows of `-c` tokens, and the last half of each window is scored.
* `<reference.gguf>`: the model the weights come from, in one file: the float model, or a near-lossless quantization of it such as an HQ8_0 file, whose weights are dequantized (and rotated back). It supplies each weight's values for quantizing, and the log-probabilities the model is measured against.
* `<model.gguf>`: the model to perturb, with any types. Its weights must have the reference's names and shapes. It can be the reference itself, to measure each weight against an otherwise float model.

**How it works**

1. The reference's log-probabilities over the windows are computed once and saved to a cache file; later runs with the same settings reuse it, and the reference isn't loaded at all.
2. The model is loaded and scored once, unperturbed: its KL and top-1 agreement are the baseline for every change.
3. For each weight, the reference's values are read once and quantized to all the types in one step, exactly as `quantize_gguf` would (see below), and the variants are kept in RAM.
4. For each type, the weight is replaced by its variant, a tensor of that type in a buffer of its own on the weight's device, and the model is scored. The variant runs on its type's own kernels, with the input rotated for an HQ type, so it behaves exactly as the weight would in a file quantized that way. After the last type, the weight is put back.

The variants are quantized with `quantize_gguf`'s own code: plain types with the imatrix, HQ types rotated with the model's seed (or `quantize_gguf`'s default one for a model without HQ weights) and quantized with GPTQ where the imatrix has an entry. A variant of a weight's own type therefore reproduces the model exactly, and measures 0, when the model was quantized by this fork with the same imatrix. Files quantized elsewhere differ slightly in the imatrix-weighted types (Unsloth's Qwen3.8-27B UD-Q4_K_M: about 1e-5 KL), and that difference is part of each change.

**Options**

| option | default | meaning |
|---|---|---|
| `-ngl N` | 0 | GPU layers of the reference. Only used to compute its log-probabilities when they aren't cached. |
| `-perturbngl N` | `-ngl` | GPU layers of the model to perturb. |
| `-ot REGEX=DEVICE` | none | Places the weights whose name matches on a device, like llama.cpp's `--override-tensor`, for both models: `DEVICE` is a buffer type name such as `CUDA0` or `CPU`. Repeatable. |
| `-c N` | 512 | Tokens per window. The last `N/2 - 1` of each window are scored, so each has at least `N/2` tokens of context. |
| `--chunks N` | 10 | Number of windows. More windows lower the noise and cost proportionally more time. |
| `--parallel N` | 1 | Windows per batch. With a model larger than VRAM, the weights in RAM cross to the GPU once per batch, so a larger batch is much faster, up to what VRAM and RAM hold. |
| `--cache FILE` | `<reference>.ngl<N>.p<P>[.ot<hash>].kl-cache` | The reference's log-probabilities (FP16). The default name records the settings that change them: `-ngl`, `--parallel` and the `-ot` patterns. A cache is reused when it holds at least `--chunks` windows of the same text, window size and vocabulary. The files are shared with `test-hadamard-ppl`. |
| `--imatrix FILE` | none | Importance matrix: plain types are quantized with it, HQ types with GPTQ. Weights without an entry are quantized without it, with a warning that names them. |
| `--types T,T,...` | `q4_K,hq4_K` | The quantized types to try, by ggml type name, in any case. Types that need an imatrix (`iq2_xxs`, `iq2_xs`, `iq1_s`) are skipped for weights without one. |
| `--variants-mb N` | 4096 | RAM for one weight's prepared variants. A weight whose variants would take more (`output` and `token_embd` of a large model) is prepared one type at a time. |
| `--activations A` | `f32` | How the GEMMs round, on CUDA. `f32`: every weight runs as a dequantized F32 cuBLAS GEMM, which rounds nothing in between. `f16`: F16 cuBLAS GEMMs. `q8`: the quantized kernels, as in inference, which round each GEMM's input to 8 bits. See below for why `f32` is the default; it's recorded in the default cache name. |
| `--group G` | `tensor` | What is perturbed at once: `tensor`, one weight; `layer`, all the weights of one block (`^blk\.N\.`); `kind`, one kind of weight in every selected block (`^blk\.\d+\.ffn_down\.weight$`). |
| `--tensors REGEX` | all | Only the weights whose name matches. |
| `--layers A[-B]` | all | Only the weights of blocks A to B; this leaves out `output` and `token_embd`. |

`LLAMA_HQ_GPTQ=0` and `LLAMA_HQ_GPTQ_DAMP=<x>` work as in `quantize_gguf`.

The weights perturbed are the ones `quantize_gguf` would quantize (2D and 3D weights, not norms, `ssm_conv1d` and the like) that the model has loaded; llama.cpp doesn't load the MTP layer's weights for normal inference, so those are left out. `token_embd` gets no HQ types, and neither does `output` in a model without `token_embd`. When `output` is tied to `token_embd`, both uses change together.

**Output**

On stdout, a header, a line per weight (or group) and a ranking per type; everything but the weight lines is a `#` comment. Progress and timings go to stderr.

```
# reference Qwen3.8-27B-bf16.gguf, perturbing Qwen3.8-27B-UD-Q4_K_M.gguf by tensor
# 10 chunks of 512 tokens, 255 scored per chunk, 5 per batch; -ngl 32, -perturbngl 99; imatrix imatrix_unsloth.gguf (GPTQ for HQ)
# unperturbed model: KL 0.009007, top-1 94.784 %
# pattern=[(type, KL change, top-1 change in percentage points), ...]  # the weight's type in the model
^output\.weight$=[(q4_k, 4.028e-03, -1.373), (hq4_k, 3.987e-03, -2.157), (hq5_k, 8.582e-04, -0.824), (q5_k, 9.871e-04, -0.275)]  # q6_k
^blk\.63\.ffn_down\.weight$=[(q4_k, 2.465e-04, -0.314), (hq4_k, 1.252e-04, -0.039), (hq5_k, 2.440e-05, +0.118), (q5_k, 1.325e-04, +0.000)]  # q6_k
...
#
# ranking for q4_k, largest KL change first
# rank  pattern                       KL change    top-1  Mparams  now
#    1  ^output\.weight$              4.028e-03   -1.373   1271.4  q6_k
#    2  ^blk\.63\.ffn_down\.weight$   2.465e-04   -0.314     89.1  q6_k
```

A KL change is the model's mean KL divergence from the reference with the variant, minus the unperturbed model's; positive is worse. A top-1 change is in percentage points of tokens whose most likely token agrees with the reference's; negative is worse.

**Reading the results**

* **Rounding.** A change to one weight changes every activation after it, and any rounding of intermediate values then comes out differently in every later layer. Against a reference computed with the same rounding, that re-roll costs about the same whatever the size of the change, a floor under every result: with the quantized kernels' 8-bit activations (`--activations q8`), about 3e-4 KL for a layer-0 weight of Qwen3.8-27B and 9e-4 for Qwen3-0.6B, with F16 GEMMs 1.6e-4 on the 0.6B, with F32 GEMMs (`f32`, the default) under 1e-5. Only `f32` measures the weights themselves. The quantized kernels' own 8-bit rounding of a weight's input is a real inference cost, the same for every type of that weight; it doesn't depend on the weight's precision.
* **A perturbed model at its minimum.** When the model is the reference itself, or a near-lossless stand-in for it (an HQ8_0 file as both), its KL is 0 and every change adds to it, more for fewer bits. With a quantized mix as the model (a UD file), the change of one weight also interacts with the other weights' errors and can come out either way, so small changes there are noise.
* When the model is the reference itself, compute the reference with the same `-ngl`, `--parallel`, `-ot` and `--activations` as the model; the tool warns when the unperturbed model differs from its own reference.

**Time and memory**

Each (weight, type) costs one evaluation. On an RTX 5090, Qwen3.8-27B UD-Q4_K_M (16 GB) fits in VRAM with `-perturbngl 99`: an evaluation of 10 windows takes 2.8 s with `--parallel 5`, and preparing a weight takes up to about 20 s for `ffn_down` with 7 types (GPTQ). The whole model, 498 weights with 4 types, takes an estimated 2.2 hours. The BF16 model itself doesn't fit (54.7 GB). An HQ8_0 file of it (KL 0.0004 from BF16) serves as both reference and model: with `-ngl 48 -ot "output\.weight=CUDA0" --parallel 5` and F32 GEMMs, an evaluation of 30 windows takes 16 s (the F32 path dequantizes `output.weight` into a 5 GB buffer, hence the layers left in RAM).

```bash
# Qwen3.8-27B UD-Q4_K_M against its BF16 source; -ngl 32 --parallel 5 -ot ... name the cached reference
./tensor-kl-cuda -ngl 32 -ot 'output\.weight=CUDA0' --parallel 5 -perturbngl 99 --chunks 10 \
    --imatrix imatrix_unsloth.gguf --types q4_K,hq4_K,hq5_K,q5_K \
    wiki.test.raw Qwen3.8-27B-bf16.gguf Qwen3.8-27B-UD-Q4_K_M.gguf > Qwen3.8-27B-UD-tensor-kl.txt
```

### Testing

The HQ tests are separate programs, built with make. `maincuda`, `test-backend-ops` and the `-cuda` variants need `LLAMA_CUBLAS=1`; `bench-rht` calls `nvcc` directly (`-arch=native`); the rest are CPU-only:

```bash
make LLAMA_CUBLAS=1 -j$(nproc) quantize_gguf main maincuda test-hadamard test-hadamard-quants test-hadamard-quantize \
    test-hq-gptq test-hadamard-llama test-hadamard-llama-cuda test-hadamard-ppl test-hadamard-ppl-cuda tensor-kl tensor-kl-cuda \
    test-backend-ops bench-rht
```

Some take a small model and a text file: a BF16 GGUF of Qwen3-0.6B (e.g. `Qwen3-0.6B-BF16.gguf` from `unsloth/Qwen3-0.6B-GGUF`) and wikitext-2's `wiki.test.raw` (in `wikitext-2-raw-v1.zip` from `ggml-org/ci` on Hugging Face). Every program exits non-zero on a failure; the `test-hadamard*` ones print PASS/FAIL per check.

| program | what it checks | how to run |
| ------- | -------------- | ---------- |
| `test-hadamard` | The rotation itself: every Hadamard matrix (`H·Hᵀ = K·I`, the set of orders, row orientation), the width rule, the random signs (golden vectors), the transform against a dense `R` built from its definition, thread safety, that every way of splitting the work gives the same bytes, the HQ types' traits, pairing and names, row validation, and the CPU op with 1, 4 and 16 threads. | `./test-hadamard` |
| `test-hadamard-quants` | The HQ quantizers: the base quantizers are unchanged (output hashes), HQ output decodes with the base type's kernels, no imatrix is needed or used, first-use initialization, determinism across threads, and a quantize → dequantize → un-rotate round trip. Also reports MSE against the base quantizers and how often the IQ2/IQ3 neighbour search differs from an exhaustive one. About a minute. | `./test-hadamard-quants` |
| `test-hadamard-quantize` | The quantizer tool, through `llama_model_quantize` on a two-layer copy of the model: stored rows are `R·w` with the file's seed; the same seed gives identical files, any thread count gives identical files, and `--hadamard-seed` changes only the HQ tensors and the key; `hadamard.seed` is the only HQ metadata without an imatrix; with a synthetic imatrix, GPTQ runs by default (one info line, `quantize.hq.gptq_damp`), tensors without an entry get the uniform bytes with one warning, `LLAMA_HQ_GPTQ=0` gives the uniform bytes, `LLAMA_HQ_GPTQ_DAMP` is checked and recorded, files are identical across thread counts, and HQ → the same HQ stays a copy; unsupported widths keep their type (and an explicit HQ type for one is refused); the refusals (HQ token embeddings, a missing seed, a conflicting seed, HQ → unrotated); the requantize rules, including GPTQ in the source's seed; the `--lora` merge math, with and without an imatrix, and its refusals; and per-expert factors for a 3D tensor. About a minute. | `./test-hadamard-quantize <workdir> <Qwen3-0.6B-BF16.gguf>` |
| `test-hq-gptq` | GPTQ for the HQ types: the fp64 factor of `H⁻¹` (`U·H·Uᵀ = I` for widths up to 17 408 and several imatrix shapes, the worst case at the damping floor, the refusals, the same bytes for any thread count, and its timing), the factor cache (hits, LRU eviction, the memory cap), `quantize_hq`'s output unchanged for all seventeen types (hashes recorded before GPTQ went in, or when a type was added), a scaled identity factor giving the uniform bytes, and for every type: the output error under `H` against uniform rounding, valid output that decodes with the base type's kernels, sign parity, and the same bytes for any row grouping and thread count. About a minute; `--quick` skips the factor sweep. | `./test-hq-gptq [--quick]` |
| `test-hadamard-llama` | Loading and running HQ models. It takes plain quantizations and relabels their eligible tensors as HQ (the data isn't rotated; that doesn't matter here). Checks: HQ tensors load as their base type with the same bytes and buffer types (including the CPU repack), the loader still reports the HQ types, a saved model round-trips with its types and seed, one rotation per distinct rotated activation, no Hadamard graph inputs, the LoRA branch uses the unrotated input, the graph guard refuses every misuse of a rotated weight (wrong input, GET_ROWS, ADD, writes, views), a missing seed and a rotated `token_embd` are refused, and a model with an RPC device (an in-process server on 127.0.0.1:50931) is refused. The `-cuda` build also loads a real HQ model on the GPU and compares its logits with the CPU's. | `./quantize_gguf --pure <model-bf16.gguf> pure-q4k.gguf Q4_K`, then `./test-hadamard-llama <workdir> pure-q4k.gguf [more plain quantizations]` (same for `-cuda`) |
| `test-backend-ops` | Upstream's backend test harness plus this fork's cases (`make_test_cases_fork`): the RHT op on CUDA against the CPU for every kernel, Hadamard order and width class, row counts, strides and 3D inputs; that CUDA accepts every width the rule accepts; and the graphs where the rotation writes the matmuls' 8-bit input directly, including every case that must fall back. | `./test-backend-ops -o RHT -b CUDA0`, also `-o MUL_MAT` and `-o MUL_MAT_ID`; repeat with `GGML_CUDA_DISABLE_FUSION=1` (all fusion off) and `GGML_CUDA_RHT_F32=1` (only the direct 8-bit output off) |
| `bench-rht` | The CUDA kernels against the reference transform for every width class, the 8-bit outputs byte for byte against `quantize.cu`, and the device sign generator; then timings as CSV: µs per rotation in a CUDA graph for decode, and bandwidth against a copy for 512 rows. | `./bench-rht`; `--sweep` adds the kernel variants, `--once` launches each once for `ncu`, `-n N[,N...]`, `-r ROWS`, `-k A\|B` restrict the shapes |
| `test-hadamard-ppl` | Quality against a reference model: perplexity, mean and 99th-percentile KL divergence, top-token agreement and bits per weight, over windows of a text file. The reference's log-probabilities are cached to a file (FP16, about 6 GB for 80 windows with a 152k vocabulary) and reused, also for runs with fewer windows. `--lora` applies an adapter to every model. | `./test-hadamard-ppl-cuda -ngl 99 --chunks 80 [--cache FILE] [--lora FILE] wiki.test.raw <reference-bf16.gguf> <model.gguf>...` (CPU: `./test-hadamard-ppl` without `-ngl`) |
| `tensor-kl` | Per-weight quantization sensitivity at a model's operating point; see [Per-weight sensitivity](#per-weight-sensitivity-tensor-kl). | `./tensor-kl-cuda [options] wiki.test.raw <reference-bf16.gguf> <model.gguf>` |

`tools/quantize/tests-hq.sh` runs the whole set end to end on Qwen3-0.6B. It checks that no backend references the HQ types, runs the unit tests above, and quantizes and generates with `--hadamard` for Q8_0, Q4_K_M, Q5_K_M, Q4_0, IQ4_XS, Q3_K_M, IQ3_S, IQ3_XS, Q2_K and IQ2_XXS (the 4–8 bit ones must answer "Paris"). It then checks HQ5_K → HQ4_XS requantizing, runtime LoRA and `--lora` merges by KL (it writes a synthetic adapter with Python, so it needs `python3` and `numpy`), and, with `CUDA=1` on a build with CUDA graphs, that graphs on and off give identical output. About 10–15 minutes on the CPU, 5 on the GPU:

```bash
tools/quantize/tests-hq.sh . Qwen3-0.6B-BF16.gguf wikitext-2-raw/wiki.test.raw /tmp/hq          # CPU
CUDA=1 tools/quantize/tests-hq.sh . Qwen3-0.6B-BF16.gguf wikitext-2-raw/wiki.test.raw /tmp/hq   # GPU
```

`BIN=<dir>` takes the binaries from another directory than the repo root, and `CHUNKS=N` sets the number of windows for the KL checks (default 20). `IMATRIX=<imatrix.gguf>` adds the GPTQ step: Q8_0, Q4_K_M, Q5_K_M, Q4_0, IQ4_XS, Q3_K_M, IQ3_S, Q2_K and IQ2_XXS with `--hadamard --imatrix` must beat uniform HQ by KL, and all but IQ3_S, Q2_K and IQ2_XXS must answer "Paris" and come within 10 % of the unrotated mix with the same imatrix.

## Background information on quantize_gguf

- [k-quants](https://github.com/ggml-org/llama.cpp/pull/1684)
- k-quants improvements and i-quants
  - [#2707](https://github.com/ggml-org/llama.cpp/pull/2707)
  - [#2807](https://github.com/ggml-org/llama.cpp/pull/2807)
  - [#4773 - 2-bit i-quants (inference)](https://github.com/ggml-org/llama.cpp/pull/4773)
  - [#4856 - 2-bit i-quants (inference)](https://github.com/ggml-org/llama.cpp/pull/4856)
  - [#4861 - importance matrix](https://github.com/ggml-org/llama.cpp/pull/4861)
  - [#4872 - MoE models](https://github.com/ggml-org/llama.cpp/pull/4872)
  - [#4897 - 2-bit quantization](https://github.com/ggml-org/llama.cpp/pull/4897)
  - [#4930 - imatrix for all k-quants](https://github.com/ggml-org/llama.cpp/pull/4930)
  - [#4951 - imatrix on the GPU](https://github.com/ggml-org/llama.cpp/pull/4957)
  - [#4969 - imatrix for legacy quants](https://github.com/ggml-org/llama.cpp/pull/4969)
  - [#4996 - k-quants tuning](https://github.com/ggml-org/llama.cpp/pull/4996)
  - [#5060 - Q3_K_XS](https://github.com/ggml-org/llama.cpp/pull/5060)
  - [#5196 - 3-bit i-quants](https://github.com/ggml-org/llama.cpp/pull/5196)
  - [quantization tuning](https://github.com/ggml-org/llama.cpp/pull/5320), [another one](https://github.com/ggml-org/llama.cpp/pull/5334), and [another one](https://github.com/ggml-org/llama.cpp/pull/5361)
