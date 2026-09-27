# Plan: Full input Hessians for Qwen3.8-27B (calibration set and collector)

Status: proposed (2026-09-26). Follows up `plans/gptq_rotated_quantizer_plan.md`: its GPTQ uses
`H = R·diag(v)·Rᵀ` built from an imatrix, and its Future work names the richer Hessians (full
covariance, low-rank). This plan produces the data for them. Using the full `H` in the quantizer is
the next plan (see "What comes next").

## Summary

This plan builds a calibration set for Qwen3.8-27B that looks like the model's real inputs. It
then runs that set through `Qwen3.8-27B-HQ8_0.gguf` and stores, for every distinct GEMM input, the
full Gram matrix `G = Σ x·xᵀ` of the unrotated input. That is GPTQ's Hessian up to scale. The file
is an imatrix GGUF with one more tensor kind. Every weight keeps its `<name>.in_sum2` / `<name>.counts`,
so the file also works as an ordinary imatrix. Each distinct input adds a `<owner>.in_gram` F32
`[n, n]` tensor. The Grams total 101 GB.

The collector is a new tool, `hessian-collect`. A small graph hook records each weight GEMM's input
node. The tool accumulates `G` on the GPU with cuBLAS SYRK, over several passes of about 9 layers
each, because the Grams fit in neither VRAM nor RAM at once.

## Goal

1. **Calibration set `qwen38-calib-v1`**, about 1.0M tokens for the Hessians plus 0.1M held out.
   - Most documents are chat-templated with Qwen3.8's own template.
   - Most assistant turns are the model's own generations, thinking included.
   - Each document is its own sequence of at most 8192 tokens, starting at position 0.
2. **`hessian-collect`** runs the set through a GGUF model and writes one file with, for every
   weight the forward pass uses:
   - `<w>.in_sum2` (F32 `[n, 1]`) and `<w>.counts` (F32 `[1, 1]`), with the same meaning and
     layout as upstream `llama-imatrix`'s GGUF output
   - the full Gram of its input, stored once per distinct input and shared through aliases
3. **The Grams are in the unrotated input space**, even when the model is an HQ file. So they
   serve any HQ seed and any non-HQ type.
4. **`--imatrix <hessian file>` works as today.** The imatrix loader stops reading whole files
   into RAM, and the diagonal-GPTQ output stays byte-identical to that from an imatrix with the same
   `in_sum2`.
5. **The 27B run** fits this box (RTX 5090 32 GB, 30 GB RAM, 8 cores, 288 GB free disk) in
   roughly 1.5–2 hours, and can resume after an interruption.

Out of scope:
- using the full `H` in the quantizer (next plan)
- MoE / `MUL_MAT_ID` Grams (the tool refuses them; Qwen3.8-27B is dense)
- the MTP layer `blk.64`, which a normal forward pass doesn't run (Unsloth's imatrix has no
  entries for it either)
- vision inputs (image tokens do reach the LLM layers; a later version could add mtmd documents)
- sequential GPTQ, which collects each layer's inputs through the already-quantized earlier layers

## Facts this plan rests on (checked 2026-09-26)

**Model** (`Qwen3.8-27B-bf16.gguf` metadata; the HQ8_0 file has the same tensors):
- arch `qwen35`, 64 layers plus the MTP layer `blk.64`
- `n_embd` 5120, `n_ff` 17408, 24 heads × 256, 4 KV heads, SSM inner size 6144
- `full_attention_interval` 4: layers 3, 7, …, 63 are attention (16), the other 48 are Gated DeltaNet

| layer kind | input (width) | weights reading it | owner (loader order) |
|---|---|---|---|
| SSM | `attn_norm` output (5120) | `attn_gate`, `attn_qkv`, `ssm_alpha`, `ssm_beta` | `attn_gate` |
| SSM | DeltaNet output (6144) | `ssm_out` | `ssm_out` |
| attention | `attn_norm` output (5120) | `attn_k`, `attn_q`, `attn_v` | `attn_k` |
| attention | gated attention output (6144) | `attn_output` | `attn_output` |
| both | `post_attention_norm` output (5120) | `ffn_gate`, `ffn_up` | `ffn_gate` |
| both | SwiGLU output (17408) | `ffn_down` | `ffn_down` |
| — | `result_norm` (5120) | `output` | `output` |

- **Weight count:** 48·8 + 16·7 = 496 weights in the layers, which matches Unsloth's imatrix
  (992 tensors), plus `output.weight`.
- **Gram size per layer:** 2·5120² + 6144² + 17408² = 393.2M floats, which is 1.573 GB in F32.
  - 64 layers + `output`: **100.8 GB**. `ffn_down` is 77 % of that.
  - A copy per weight instead of per input would add 25.2 GB.
  - Packed triangles would take 50.4 GB.
- **`imatrix_unsloth.gguf`:** `imatrix.datasets = ['unsloth_calibration_dataset']`,
  `chunk_size` 8192, counts 1,149,614 tokens. It has no `output` entry and no `blk.64` entries.
- **Unsloth's description of that set:** "high-quality, hand-curated and cleaned data", refined for
  "agentic coding, chat, and multilingual performance". Their docs also say that "using text only
  calibration datasets is not effective for instruct models" [Unsloth].
- **Chat template** (from the GGUF, Unsloth-patched):
  - thinking is on by default, with `reasoning_effort` ∈ {xhigh (default), medium, low}, which
    injects a system instruction
  - `enable_thinking=false` emits an empty think block
  - `preserve_thinking` keeps earlier turns' reasoning by default
  - tool calls render as `<tool_call><function=…><parameter=…>` XML
  - GGUF sampling defaults: temp 1.0, top_p 0.95, top_k 20

**Engine** (this fork):
- **HQ inputs:** a rotated weight's GEMM reads `RHT(x)` (`rotate_input_if_rotated`,
  `src/llama-graph.cpp:1513`). After the RHT runs, `x` can be freed and its memory reused by the
  GEMM's own output. So a callback on the GEMM can't read `x`, and it can't tell which other weights
  share that input either, especially once the scheduler copies inputs between splits. That is why
  §4 adds a graph hook.
- **The eval callback can't stop a graph.** Returning `false` ends only the current split
  (`ggml/src/ggml-backend.cpp:1782`), so a pass can't stop early after its last layer.
- **Output rows:** Qwen3.8 selects them after the last layer (`src/models/qwen35.cpp:215`), so
  every layer sees every token whatever the logits flags. Qwen3 (the 0.6B test model) selects them
  inside the last layer (`src/models/qwen3.cpp:114`), so its tests need logits at every position.
- **Op offload:** with a batch of at least 32 (`GGML_OP_OFFLOAD_MIN_BATCH`), a GEMM whose weight is
  in RAM runs on the GPU, and the weight is copied over once per ubatch.
- **Available tools:** upstream `llama-imatrix` at 53ed051 is built in
  `build-hq/upstream-llama/build/bin/`. The fork has no imatrix tool. `common_imatrix_load` reads
  every tensor of the file into RAM (`no_alloc = false`).

**HQ8_0 placement:**
- FFN weights: 64 × 284 MB = 18.2 GB
- other layer weights: 7.7 GB
- `output`: 1.35 GB
- `token_embd`: 1.35 GB, CPU (it's a `get_rows`)

## Design

### 1. The calibration set

**Size: 1.0M tokens for the Hessians, plus 0.1M held out.**
- Unsloth used 1.15M tokens for this model.
- 1.0M tokens is at least 57 tokens per dimension for the widest input (17408).
- The full-covariance GPTQ that overfit on Qwen3-0.6B had 6.5 per dimension (20k tokens, width
  3072: in-sample output error 0.207 against 0.409 held out).
- Phase 5's split-half check reports whether 1.0M is enough.

**Mix** (shares of the 1.0M; "self" means the assistant turns are generated by the model):

| category | share | content | assistant turns |
|---|--:|---|---|
| chat | 25 % | open prompt sets (e.g. OpenAssistant oasst2, WildChat), ~30 % multi-turn; writing, advice, explanation, role-play, summarizing a supplied text | self |
| coding | 20 % | code questions, review/explain/fix a real source file supplied in the prompt, several languages | self |
| agentic / tool use | 10 % | tool schemas + multi-step tool conversations (e.g. `interstellarninja/hermes_reasoning_tool_use`, which Bartowski's imatrix set uses), converted to Qwen's XML tool format | as in the source; final turn self |
| math / STEM reasoning | 10 % | train splits only (e.g. GSM8K, MATH), science QA | self, thinking on |
| multilingual | 10 % | Chinese heavily, then ja, ko, es, fr, de, ru, ar, pt, vi…: prose and chat | self, same language |
| long context | 10 % | 6–8k-token documents: papers, long code files, RAG prompts with several documents | self (short answer) |
| raw text | 15 % | no template: encyclopedic, fiction, news, science, code files; Bartowski's `calibration_datav3` fits here | — |

The sources above are candidates. Check each one's availability and licence when downloading. The
Hessian file contains no text, but the calibration set on disk does.

**Why most of it is templated and self-generated:**
- An instruct model's activations depend on its template tokens and on its own style of answer
  [Unsloth].
- Bartowski found that a 2:3 ratio of prose to templated conversations worked best for imatrix
  calibration [Bartowski].
- Williams et al. show that calibration data generated by the model itself works as well as or
  better than real data for quantization and pruning [Williams 2025].
- Prose and code still make up about a third of the tokens, but mostly inside a templated task
  (summarize, review, answer from documents), which is how a chat model usually sees them.

**Generation:**
- **Server:** koboldcpp (this fork) with `--parallelrequests 8`, serving `Qwen3.8-27B-HQ8_0.gguf`
  for fidelity (KL 0.0004 against BF16). If that's too slow, `Qwen3.8-27B-UD-HQ-Q4_K_M.gguf`
  runs fully on the GPU.
- **Prompts:** rendered in Python (jinja2 3.1.2 on the system) from the template string in the
  GGUF, with `add_generation_prompt=True`. They're sent to the raw completion endpoint.
- **The document is prompt + completion + `<|im_end|>`,** exactly the text the model processed. No
  re-parsing of reasoning or tool calls happens.
- **Thinking mix:** on for 75 % (xhigh 45 %, medium 15 %, low 15 %), off for 25 %.
- **Sampling:** the GGUF defaults. At most 6144 new tokens, and the whole document is capped at
  8192.
- **Multi-turn:** conversations are generated turn by turn and rendered with the template's
  defaults (`preserve_thinking` on).
- **Volume:** about 450k generated tokens. Phase 1 measures the throughput. At 150 tokens/s
  aggregate that takes under an hour.

**Lengths:**
- Chat 0.5–6k tokens, long context 6–8k.
- Raw text is cut into documents whose lengths are spread over 1–8k.
- **Target:** at least 25 % of tokens at positions ≥ 4096, so late-position activations (attention
  sinks, DeltaNet state growth) are represented.

**Keeping the evaluation texts clean:**
- Drop any document that shares a 13-gram with `wiki.test.raw` or `build-hq/tech-eval.txt`,
  after lower-casing and collapsing whitespace.
- Take no text from this repository, since tech-eval is built from it.
- Use train splits only for math sets.

**Held out:** 10 % of the documents in each category are never used for Hessians. They're written
as `calib-heldout.txt` for later KL runs on in-distribution text, next to wiki.test and tech-eval.

**Files** (data in `build-hq/calib/`, which is git-ignored; scripts committed in
`tools/hessian/calib/`):
- `qwen38-calib-v1.jsonl`: one document per line, `{"id", "category", "templated", "text"}`
- `qwen38-calib-v1-heldout.jsonl`
- `manifest.json`: sources, per-category tokens, the length histogram, and the jsonl's sha256

### 2. What is collected

- **The Gram:** for each distinct GEMM input `x` of width `n`, `G = Σ_t x_t·x_tᵀ` over every row
  the GEMM processes. It's a raw sum, like `in_sum2`: `E[xxᵀ] = G / count`, and it's GPTQ's
  `H = 2·XXᵀ` up to scale.
- **The input is taken before rotation** (the `x` passed to `build_lora_mm`, not `RHT(x)`), so `G`
  is seed-independent. The HQ quantizer forms `R·G·Rᵀ` with the file's seed, as it does for
  `diag(v)` today.
- **Uncentred,** as GPTQ uses it. `in_sum2 = diag(G)` by construction.
- **Every token of every document counts.** The only exception is `output.weight`: its input exists
  only at positions that request logits. The tool requests every 8th position
  (`--output-stride 8`), which gives about 125k rows (24 per dimension) and a logits buffer of
  1 GB instead of 8 GB per 8192-token ubatch.

### 3. File format

A GGUF written the way `llama-imatrix` writes its own. Tensors are ordered by (layer, name), as in
`weight_name_comparer`, so each pass writes one contiguous byte range.

```
general.type                = "imatrix"
imatrix.datasets            = ["qwen38-calib-v1:<sha256[0:12]>"]
imatrix.chunk_count         = <documents>
imatrix.chunk_size          = 8192                      # longest sequence
hessian.version             = 1
hessian.source_model        = "Qwen3.8-27B-HQ8_0.gguf"
hessian.alias.names         = ["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight", ...]
hessian.alias.owners        = ["blk.0.attn_gate.weight", "blk.0.attn_gate.weight", ...]
hessian.layers_done         = u8[65]                    # 64 layers + output; resume bookkeeping
hessian.complete            = bool

blk.0.attn_gate.weight.in_sum2   F32 [5120, 1]
blk.0.attn_gate.weight.counts    F32 [1, 1]
blk.0.attn_gate.weight.in_gram   F32 [5120, 5120]      # owner: attn_gate, attn_qkv, ssm_alpha, ssm_beta
blk.0.attn_qkv.weight.in_sum2    F32 [5120, 1]
blk.0.attn_qkv.weight.counts     F32 [1, 1]
...
blk.0.ffn_down.weight.in_gram    F32 [17408, 17408]
...
output.weight.in_gram            F32 [5120, 5120]
```

- **Lookup:** a weight's Gram is `<w>.in_gram`. Failing that, it's `<owner>.in_gram` through the
  alias arrays.
- **The owner** is the first weight of the group in loader order, so it doesn't depend on graph build
  order.
- **Both triangles are stored,** since the request is for the entire matrix. Packed storage (50 GB)
  is noted in "Decisions".
- **Every KV has a fixed size,** so the header is written once before any data and rewritten in
  place at the end of each pass.

### 4. Capturing GEMM inputs (graph hook)

- **What's recorded:** `llama_cparams::collect_mm_inputs` (default off) makes `build_lora_mm` and
  the direct `rotate_input_if_rotated` call sites record `(x, w)` in the graph result. Here `x` is
  the unrotated input node and `w` the weight. Records with the same `x` form one group.
- **How the tool turns it on and reads it:** through an internal setter and accessor on
  `llama_context`. The tool links the internal headers, as tensor-kl does, so the public API
  doesn't change.
- **With the flag off, nothing changes.** With it on, the graph is identical too: only a side
  table is filled.
- **The callback:**
  - `ask` returns true for every recorded `x`, in every pass. Graph views, fusion boundaries and
    so the activations are then identical whichever layers a pass accumulates, which makes passes
    bitwise-invariant (Phase 4).
  - `observe` runs right after `x` is computed, so `x` is intact. It accumulates only if `x`'s
    group belongs to the current pass.
- **A recorded `x` must be a computed node** (not a leaf), F32, with `nb[0] = 4` and a uniform row
  stride. Otherwise the tool copies it to a contiguous buffer, or refuses a leaf with a message
  naming the weight.
- **Coverage check:** at the end, every 2D `blk.*` weight and `output` of the model must have a
  record, `blk.<nextn>` excepted. A weight that some architecture multiplies outside
  `build_lora_mm` is reported by name.

### 5. Accumulation

- **CUDA path:** one F32 `n × n` accumulator per group of the current pass.
  - `observe` calls `cublasSsyrk(UPPER, N, n, k = rows, 1, x, lda = nb[1]/4, 1, G, n)` on
    `x->data` directly: the scheduler has already synchronized the backend.
  - The tool then synchronizes its own stream before returning, because ggml may overwrite `x`
    afterwards.
  - If `x` is in host memory, it's copied to a device staging buffer first.
- **CPU path:** fp64, multithreaded blocked loops. It's used for tests and for CPU-only builds on
  small models, and it's the reference that Phase 4 compares the GPU against.
- **Precision:**
  - Within a SYRK, fp32 sums over up to 8192 rows; then about 125 ubatch adds.
  - The expected error is about 10⁻⁶ of `√(G_ii·G_jj)` per entry. Sampling noise at 1M tokens is
    about 10⁻³, a thousand times more.
  - Phase 4 measures it against the fp64 path.
- **At the end of a pass,** each accumulator is:
  - downloaded (at most 1.21 GB of host memory, for `ffn_down`)
  - mirrored upper → lower in tiles
  - checked finite, and its diagonal checked non-negative
  - written with `pwrite` at the tensor's offset
  - `in_sum2 = diag(G)` and `counts` are written with it, for the group's owner and its aliases
- **Cost:** the SYRKs total `Σ n²/2` multiply-adds per token = 25 GFLOP per token, 25 PFLOP for
  1.0M tokens. At an assumed 50 TFLOP/s sustained F32 that's about 8 minutes, whatever the number
  of passes.

### 6. Passes, placement and memory

**Placement for the 27B run:** `-ngl 99 -ot 'blk\.[0-9]+\.ffn_(gate|up|down)\.weight=CPU' --mlock`
- The FFN weights (18.2 GB) stay in RAM, and op offload streams them to the GPU once per
  8192-token ubatch.
- Attention and DeltaNet weights, the KV cache, the recurrent states and `output` stay on the
  GPU. Attention over 8k tokens stays off the CPU.

**VRAM budget** (Phase 4 replaces the estimates with measurements):

| item | GB |
|---|--:|
| free on the 5090 (the desktop uses 2.6) | 29.9 |
| non-FFN weights + `output` | −9.0 |
| compute buffer, KV (16 layers × 8192), 8 recurrent states, 1 GB logits (estimate) | −5 |
| left for accumulators | ≈ 15 → **9 layers × 1.573 GB** |

- **Passes:** ⌈64 / 9⌉ = 8. `output` joins the last pass.
- **Sizing:** the tool picks the group size from `cudaMemGetInfo` after creating the context, with
  a 1 GB margin. `--group-layers N` overrides it.
- **Host RAM:** 18.2 GB of FFN weights + 1.35 GB `token_embd` + at most 1.2 GB for downloads, which
  is about 21 of 30 GB. Analysis scripts on the 101 GB file must read row slabs (a 27B broadcast has
  OOM-killed a session before).

**Ubatches:**
- Documents are packed, whole, into ubatches of at most 8192 tokens and at most 8 sequences. Each
  document is its own `seq_id` with positions from 0.
- The tool calls `llama_decode` once per ubatch (`n_batch = n_ubatch = n_ctx = 8192`,
  `n_seq_max = 8`) and clears memory between ubatches.
- Pack order is fixed (sorted by id), so passes and resumed runs see identical ubatches.

**Time** (estimate):
- If a pass runs at about 2,000 tokens/s (18 GB of weight streaming plus 27B of compute per
  8192-token ubatch), it takes about 8 minutes, so 8 passes take about 70 minutes.
- Add about 8 minutes of SYRK, about 2 minutes of writes (101 GB) and model loads.
- If the forward pass dominates, "Future work" lists two ways to halve it.

### 7. The tool

```
hessian-collect -m MODEL.gguf --docs CALIB.jsonl -o OUT.gguf
    [-ngl N] [-ot REGEX=DEV] [--mlock] [--ubatch 8192] [--max-seqs 8] [--output-stride 8]
    [--group-layers N|auto] [--layers A-B] [--resume] [--imatrix-out FILE]
    [--text FILE --chunk N]          # upstream llama-imatrix-style chunks of one text, for tests
```

- **Tokenization:** `parse_special = true`, so template tokens stay single tokens, and the vocab's
  own BOS rule applies (Qwen adds none).
- **Mapping:** a 64-token dry decode records the groups and shapes. The tool then writes the
  header and allocates the file (`fallocate`).
- **`--resume`** reads `hessian.layers_done` and skips finished layers. The data order is
  deterministic, so a resumed run writes the same bytes.
- **`--layers A-B`** collects a subset (used by Phase 5's split-half check).
- **`--imatrix-out`** also writes a diag-only imatrix from the same run.
- **Progress:** per pass, tokens/s, SYRK time and write time; at the end, peak VRAM and RSS.
- **Build:** Makefile targets `hessian-collect` and `hessian-collect-cuda`, linked like
  `tensor-kl` / `tensor-kl-cuda`.

### 8. Loaders

- **`common_imatrix_load`** opens GGUFs with `no_alloc = true` and reads only the `.in_sum2` and
  `.counts` data, from their file offsets. Behaviour on ordinary imatrix files is unchanged.
- **New `common/hessian-loader.{h,cpp}`:** open a file; `has(w)`, `n(w)`, `owner(w)`, and
  `read(w, float * dst)`. The read goes through `pread` into the caller's buffer, and the rows can
  come in slabs. The next plan's quantizer uses it.

## Decisions to confirm

1. **One Gram per distinct input, with aliases (101 GB),** rather than one per weight (126 GB).
   Recommended: the input is what the Hessian belongs to, and GPTQ factors each input once.
2. **Full `[n, n]` storage (101 GB) as requested.** Packed upper triangles (1D, 50 GB) would halve
   disk and write time, but break the "a plain `[n, n]` tensor" simplicity.
3. **Generation model:** HQ8_0 (fidelity) by default, UD-HQ-Q4_K_M if Phase 1's throughput is too
   low.
4. **Size and mix:** 1.0M + 0.1M tokens, 8192-token cap, the mix in §1.
5. **`output.weight`** included, from every 8th position.

## Implementation conventions

As in the earlier plans:
- Keep comments minimal.
- Match the surrounding style.
- Every feature that can fail gets a test in its phase, and a phase is done only when its tests
  pass.
- Quantizer output doesn't change in this plan, except where Goal 4 says it must stay identical.

---

## Phase 1 — Calibration set (`tools/hessian/calib/`, data in `build-hq/calib/`)

1. **`fetch.py`** downloads the sources into `build-hq/calib/sources/`, pinned by revision.
2. **`prompts.py`** builds the prompt list per category, with categories, languages and the
   thinking mix sampled from a fixed seed.
3. **`generate.py`** renders each prompt with the GGUF's template, posts it to koboldcpp's raw
   completion endpoint (8 parallel requests) and appends the finished document. It can be resumed.
   Multi-turn conversations loop back through it.
4. **`raw.py`** cuts raw-text documents to the length distribution.
5. **`decontam.py`** does the 13-gram filter.
6. **`split.py`** makes the 10 % held-out split per category, writes the jsonl files and the
   manifest, and exports `calib-heldout.txt`.
7. **`stats.py`** reports the checks below.

**Checks:**
- Per-category tokens are within ±10 % of target, and the total is within 1.0–1.1M.
- No document exceeds 8192 tokens.
- At least 25 % of tokens sit at positions ≥ 4096.
- Tokenizing with the model's vocab gives one `<|im_start|>` / `<|im_end|>` token per turn
  marker, so the specials aren't split.
- No 13-gram overlap with the evaluation texts.
- Generation throughput and the finished-response rate are recorded in the manifest.

## Phase 2 — GEMM-input hook (`src/llama-graph.{h,cpp}`, `src/llama-cparams.h`, `src/llama-context.{h,cpp}`)

- `collect_mm_inputs`, the recording in `build_lora_mm` and the direct rotate call sites, the
  per-graph table, and the internal setter and accessor. Setting the flag forces a graph rebuild.
- **Tests** (Qwen3-0.6B BF16 and an HQ8_0 of it; Qwen3.8-27B HQ8_0 for coverage only):
  - **Coverage:** all 196 0.6B layer weights plus `output`, and all 496 + 1 on the 27B.
  - **Groups:** q/k/v share an input, and so do gate/up. On the 27B also
    `attn_gate`/`attn_qkv`/`ssm_alpha`/`ssm_beta`.
  - Every recorded `x` is a computed node, F32, with regular rows. The test prints the layouts.
  - Logits are bitwise identical with the flag off and on, on CPU and CUDA.

## Phase 3 — Collector: CPU path, format, loaders (`tools/hessian/hessian-collect.cpp`, `common/`)

The tool on its CPU (fp64) path: documents, packing, `--text/--chunk`, the header written first,
per-pass `pwrite`, aliases, `--resume`, `--imatrix-out`. Plus the loader changes (§8).

**Tests** (Qwen3-0.6B, `--output-stride 1`, since Qwen3 selects output rows inside its last layer):
- **Diagonal vs upstream:** `hessian-collect --text wiki.train.raw --chunk 512` on 20 chunks
  against upstream `llama-imatrix` with the same chunks. `in_sum2 / counts` agree per weight
  within 10⁻³ relative, and the counts are equal.
- **Natural space:** Grams from the HQ8_0 0.6B and from BF16 on the same documents differ by at
  most 10⁻² in `max |ΔG_ij| / √(G_ii·G_jj)`. A rotated capture would differ by O(1).
- **Format:**
  - the loader reads back every Gram, and the aliases resolve
  - `in_sum2 = diag(in_gram)` bitwise
  - `G` is symmetric bitwise
  - the header parses with `gguf_init_from_file(no_alloc = true)`
- **Imatrix compatibility:** `quantize_gguf --hadamard --imatrix <hessian file> Q4_K_M` gives
  output byte-identical to `--imatrix <--imatrix-out file>`. The imatrix loader's peak RSS stays
  below 100 MB on a 1 GB hessian file.
- **Resume:** a run stopped after pass 2 (a test-only `--stop-after-pass`) and resumed gives a file
  identical to an uninterrupted run.

## Phase 4 — CUDA accumulation and passes

The cuBLAS SYRK path, device or host inputs, VRAM auto-sizing, the end-of-pass download, mirror and
write.

**Tests:**
- **Precision:** on 0.6B, GPU (fp32) against CPU (fp64) Grams: `max |ΔG_ij| / √(G_ii·G_jj)`
  ≤ 10⁻⁵ over all groups. The measured value is recorded.
- **Pass invariance:** on 0.6B, one pass against `--group-layers 5` (6 passes) gives identical
  bytes.
- **Host inputs:** `-ot` placing a layer's norm on the CPU gives the same Gram (to 10⁻⁵).
- **27B trial on a 50k-token subset,** in the placement of §6:
  - measure tokens/s, SYRK time, compute-buffer size, peak VRAM and RSS, and the chosen group size
  - fix §6's budget and time estimate from these
  - if any buffer doesn't fit, revisit the placement before Phase 5

## Phase 5 — The 27B run

- **Run:** the full 1.0M-token set on `Qwen3.8-27B-HQ8_0.gguf`, writing
  `~/Sandbox/unquantized/Qwen3.8-27B-gguf/qwen38-27b-hessian-v1.gguf`. Record wall time per pass,
  peak VRAM and RSS, and the file size (expected 100.8 GB).
- **Sanity checks,** streamed over the file:
  - All counts equal the token total; `output` has about 1/8 of it.
  - Everything is finite.
  - The fp64 Cholesky of `G/mean(diag G) + 0.01·I` succeeds for every input of layers 0, 31 and 63
    and for `output`.
  - Per weight, the correlation of `log v̄` against `imatrix_unsloth.gguf` is recorded. Expect high
    agreement: same model, different text.
  - Channel 3994's share of the attention-input energy in early layers is recorded. The GPTQ plan
    measured 93–98 % from Unsloth's imatrix.
- **Split-half check:**
  - Collect layers 0, 31 and 63 (`--layers`) from two disjoint halves A and B of the set.
  - For each input, report `ε = ‖(G_B + λI)^{-1/2}(G_A − G_B)(G_B + λI)^{-1/2}‖₂`, with
    `λ = 0.01·mean(diag G_B)`, both Grams normalized by their counts. Also report the relative
    Frobenius difference.
  - These numbers go to the next plan. They're a diagnostic, not a gate: whether 1.0M tokens is
    enough is decided by held-out KL there.

## Phase 6 — Docs

- **`tools/hessian/README.md`:** usage, the format of §3, placement advice, and how to rebuild the
  calibration set.
- **`tools/quantize/README.md`:** a line saying a hessian file can be passed as `--imatrix`.
- Update this plan's Implementation record and the project memory.

## Test list

| test | phase |
|---|---|
| calibration stats (targets, lengths, specials, decontamination) | 1 |
| hook coverage, groups, layouts, flag-off/on logits identical | 2 |
| diagonal vs upstream `llama-imatrix` | 3 |
| HQ vs BF16 natural-space Gram | 3 |
| format round trip, aliases, symmetry, `in_sum2 = diag` | 3 |
| `--imatrix <hessian>` byte-identical quantize, lazy loader RSS | 3 |
| resume identical | 3 |
| GPU fp32 vs CPU fp64 precision | 4 |
| pass invariance, host inputs | 4 |
| 27B subset trial (budget) | 4 |
| 27B sanity checks and split-half diagnostic | 5 |

`tools/hessian/tests.sh` runs Phases 2–4 on 0.6B, like `tools/quantize/tests-hq.sh`.

## What comes next (separate plan)

- **`quantize_gguf --hessian FILE`:**
  - the HQ types use GPTQ with `H = R·(G/mean(diag G) + damp·I)·Rᵀ`, which needs a real Cholesky
    in place of §2's closed-form inverse, roughly 3× the factor time
  - the plain types use GPTQ in natural order, to test the finding that Unsloth's UD files look like
    plain quantization + full-covariance GPTQ
- **A/B,** all from this one file (its `in_sum2` is the diagonal baseline): KL on wiki.test,
  tech-eval and `calib-heldout` for:
  - full-`H` GPTQ
  - diag GPTQ
  - UD-Q4_K_M
- The GPTQ plan's low-rank variant can be computed offline from the same Grams.

## Future work

- **Early exit:** a scheduler abort (the callback's `false` stopping the whole graph, not one
  split) would skip the layers after the pass's group, about 45 % of 8 passes' forward cost.
- **Tile-packed accumulators:** upper-triangle 1024² tiles via batched GEMM halve accumulator VRAM,
  which means 4 passes instead of 8.
- MoE (`MUL_MAT_ID`) Grams per expert, MTP layers, vision documents.

## Risks

- **RAM pressure:** the FFN weights (18.2 GB) must stay resident. Evicted pages would be re-read
  from disk for every ubatch. Mitigations: `--mlock`, nothing else large running, and Phase 4's RSS
  measurement.
- **Op-offload throughput:** mmap'ed pageable memory copies more slowly than pinned memory. If
  Phase 4 shows the stream dominating, try `--no-mmap` with host-buffer registration, or a larger
  ubatch.
- **The size of the output (101 GB, 288 GB free):**
  - a failed run can resume (`--resume`)
  - nothing may read the whole file: the loader and the analysis scripts work in slabs
- **Calibration bias:** everything the model generates comes from one model at its own sampling
  settings. Real usage has user-written text too, which the prompt sources, raw text and tool
  traces supply.
- **Silent wrong-data hazards, each with a test:**
  - capturing the rotated input (natural-space test)
  - an overwritten `x` (fp64 reference and upstream diagonal agreement)
  - a missed weight (coverage)
  - pass-dependent activations (pass invariance)
  - alias or offset mistakes in the file (round trip)

## Key file index

| item | location |
|---|---|
| model files | `~/Sandbox/unquantized/Qwen3.8-27B-gguf/` (`Qwen3.8-27B-HQ8_0.gguf`, `-bf16`, `-UD-HQ-Q4_K_M`, `imatrix_unsloth.gguf`) |
| GEMM input and rotation | `src/llama-graph.cpp:1513` (`rotate_input_if_rotated`), `build_lora_mm` `:1586` |
| Qwen3.8 graph | `src/models/qwen35.cpp` (output rows `:215`, qkvz `:237–241`, alpha/beta `:362–369`, `ssm_out` `:464`, FFN `:477`) |
| scheduler callback | `ggml/src/ggml-backend.cpp:1751–1787` |
| imatrix loader | `common/imatrix-loader.{h,cpp}` |
| upstream collector (reference) | `git show 511f9c137:tools/imatrix/imatrix.cpp`; binary `build-hq/upstream-llama/build/bin/llama-imatrix` (53ed051) |
| tool pattern to follow (internal headers, `-ot`, CUDA link) | `tests/tensor-kl.cpp`, `tests/kl-eval.h`, Makefile `tensor-kl-cuda` |
| 0.6B test assets | `build-hq/models/Qwen3-0.6B-BF16.gguf`, `build-hq/models/wikitext-2-raw/`, `build-hq/imatrix-06.gguf` |

## References

Checked on 2026-09-26.

- **[Unsloth]** Unsloth Dynamic 2.0 and 3.0 GGUF docs:
  https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs,
  https://unsloth.ai/docs/basics/dynamic-3.0-ggufs. The calibration set's size and curation, "text
  only calibration datasets is not effective for instruct models".
- **[Bartowski]** "New imatrix dataset", https://huggingface.co/blog/bartowski/imatrix-dataset:
  prose + templated `hermes_reasoning_tool_use` at 2:3, rendered through the model's template.
  `calibration_datav3`: https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8.
- **[Williams 2025]** M. Williams, G. Chrysostomou, N. Aletras. *Self-calibration for Language
  Model Quantization and Pruning.* NAACL 2025, pp. 10149–10167, arXiv:2410.17170.
- **[GPTQ]** E. Frantar et al., ICLR 2023, arXiv:2210.17323. `H = 2XXᵀ`, 1 % damping.
