# hessian-collect

*The end-to-end walkthrough, from building a calibration dataset to quantizing with the Hessian, is `GUIDE.md`.*

Collects, for every distinct input `x` of a model's weight GEMMs, the full Gram matrix `G = Σ x·xᵀ` over a set
of documents: GPTQ's Hessian (`H = 2·XXᵀ`) up to scale, taken **before** any Hadamard rotation, so one file
serves every HQ seed and every plain type. The design and the measurements behind it are in
`plans/hessian_collection_plan.md`.

## Build

```
make LLAMA_CUBLAS=1 hessian-collect-cuda     # cuBLAS SYRK accumulation (use this for real models)
make hessian-collect                         # CPU only, fp64 accumulation (tests, small models)
make hessian-tokenize test-hessian           # the calibration scripts' tokenizer; the test/compare tool
```

## Usage

```
hessian-collect-cuda -m MODEL.gguf --docs DOCS.jsonl -o OUT.gguf [--imatrix-out IMATRIX.gguf] \
    [-ngl N] [-ot REGEX=DEVICE] [--no-mmap] [-ctk q8_0 -ctv q8_0] [--ubatch 8192] [--ctx 65536] \
    [--max-seqs 8] [--output-stride 8] [--group-layers N|auto] [--layers 0,31,63,out|A-B] [--resume]
```

- **Documents** (`DOCS.jsonl`): one per line, `{"id", "text", "count"}`. Text is tokenized with special tokens
  parsed and the vocab's own BOS rule, and each document is its own sequence from position 0. `count` is an
  optional list of `[start, end)` token spans: only those rows enter the Grams; the other tokens are processed as
  context. `--text FILE --chunk N [--chunks K]` instead takes upstream `llama-imatrix`'s chunks of one text.
- **The LM head** only sees rows where logits are requested: every `--output-stride`-th counted position
  (default 8). Models that select their output rows inside the last layer (Qwen3) need `--output-stride 1` for
  the last layer to see every counted row.
- **Passes:** the documents run once per pass, each pass accumulating the Grams of a group of layers.
  `--group-layers auto` sizes the group from the free VRAM after the model and context are loaded (with 1 GB of
  margin); under WSL the CUDA runtime doesn't see other processes' allocations, so run nothing else on the GPU.
  Pass order and batch order are fixed, so any grouping gives the same bytes.
- **Resume:** each pass rewrites the header's `hessian.layers_done`; `--resume` skips the finished layers of an
  interrupted run (same model, documents and options, or it refuses).
- `--imatrix-out` also writes the plain imatrix (the diagonals) once the file is complete. `--layers` collects a
  subset of the layers (`out` is the LM head).

## Output format

An imatrix GGUF (`general.type = imatrix`), so it also works as `--imatrix` for `quantize_gguf`, whose loader
reads only the `.in_sum2`/`.counts` tensors; `quantize_gguf --hadamard --hessian FILE [--hessian-alpha A]` (A 0.1 by default) uses the
Grams themselves in GPTQ (`tools/quantize/README.md`, "With a hessian file"):

| tensor | shape | meaning |
|---|---|---|
| `<w>.in_sum2` | F32 `[n, 1]` | `diag(G)`, for every weight `w` |
| `<w>.counts` | F32 `[1, 1]` | rows summed |
| `<owner>.in_gram` | F32 `[n, n]` | `G`, once per distinct input, both triangles, a raw sum (`E[xxᵀ] = G / count`) |

The owner of an input is the first weight reading it in the model file's tensor order; the other weights name it
in `hessian.alias.names` / `hessian.alias.owners`. Tensors are ordered by (layer, name). Other keys:
`imatrix.datasets`, `imatrix.chunk_count` (documents), `imatrix.chunk_size` (longest document),
`hessian.version`, `hessian.source_model`, `hessian.output_stride`, `hessian.layers_done` (u8 per layer, the LM
head last), `hessian.complete`. `src/llama-hessian.h` reads Grams in row slabs; `tools/hessian/analyze.py`
does the same from Python.

For Qwen3.8-27B one pass group is:

| input | width | weights | owner |
|---|---|---|---|
| `attn_norm` output (DeltaNet layers) | 5120 | `attn_gate`, `attn_qkv`, `ssm_alpha`, `ssm_beta` | `attn_gate` |
| DeltaNet output | 6144 | `ssm_out` | `ssm_out` |
| `attn_norm` output (attention layers) | 5120 | `attn_k`, `attn_q`, `attn_v` | `attn_k` |
| gated attention output | 6144 | `attn_output` | `attn_output` |
| `post_attention_norm` output | 5120 | `ffn_gate`, `ffn_up` | `ffn_gate` |
| SwiGLU output | 17408 | `ffn_down` | `ffn_down` |
| `result_norm` | 5120 | `output` | `output` |

1.573 GB of Grams per layer, 100.8 GB for the model.

## Placement for Qwen3.8-27B on a 32 GB GPU

```
hessian-collect-cuda -m Qwen3.8-27B-HQ8_0.gguf --docs calib.jsonl -o qwen38-27b-hessian-v1.gguf \
    --imatrix-out qwen38-27b-imatrix-v1.gguf --dataset qwen38-calib-v1:<sha256[0:12]> \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(gate|up|down)\.weight=CPU' --no-mmap -ctk q8_0 -ctv q8_0
```

The FFN weights stay in RAM and stream to the GPU once per ubatch; attention, DeltaNet weights, the KV cache and
the LM head stay on the GPU, which leaves room for 8 layers of Grams per pass. `--no-mmap` keeps the page cache
from evicting the weights.

## How it works

`llama_cparams::collect_mm_inputs` (set through `llama_context::set_collect_mm_inputs`) makes
`rotate_input_if_rotated`, which every weight GEMM goes through, record `(x, w)` in the graph result; the graph
itself is unchanged. An eval callback asks for every recorded `x` in every pass, so the graph is cut at the same
places whatever the pass accumulates, and observes `x` right after it is computed (after the RHT the input may be
freed and reused). The rows of `x` are mapped to (sequence, position) through the ubatch the context is
computing, which the tool checks token by token against the batch it built. Documents that fit a ubatch are
packed with similar lengths and padded after their end to the batch's longest, so the hybrid memory's
equal-length split gives one ubatch per batch; padding never reaches a counted row.

## Tests

`tools/hessian/tests.sh <Qwen3-0.6B-BF16.gguf> <wiki.train.raw> [hook] [cpu] [cuda]` (default: all three), e.g.
`tools/hessian/tests.sh tools/models/Qwen3-0.6B-BF16.gguf tools/models/wiki.train.raw`, on Qwen3-0.6B: the hook (coverage, groups, layouts, logits
unchanged), natural-space capture on an HQ8_0 model, the format and
loaders, `--imatrix` byte-identity, resume, count spans and packing, and on CUDA fp32 SYRK against fp64, pass
invariance and host inputs. `tools/hessian/hessian-tests/` (git-ignored) is removed when everything passes.

## Scoring quantizations (`tools/hessian/eval/`)

Scripts that score quantized models against their BF16 model by KL and top-1 agreement on wikitext and two chat
sets, and that ran the full-Hessian GPTQ tuning and the rotated-vs-unrotated comparison. See `eval/README.md`.

## The calibration set (`tools/hessian/calib/`)

`qwen38-calib-v1` and the scripts that build it are documented in `calib/README.md`. The scripts write the dataset
to `calib/dataset/` (git-ignored: third-party text and the user's own OpenCode sessions).
