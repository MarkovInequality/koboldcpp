# Plan: fixes from the audit of the HQ / GPTQ / Hessian work

## Summary

An audit of `4a708cd34..d1bc7756a` (the Hadamard-rotated types, GPTQ, full-Hessian collection and GPTQ)
found five correctness problems, some architectural debt, a few optimizations and a fair amount of
duplicated code. This plan fixes them in an order that keeps a byte-level oracle valid: first record
what the current code produces, then fix correctness, then refactor under "the bytes don't change",
then make the changes that are allowed to change numbers, judged by KL.

## Findings addressed

| id | finding | phase |
|---|---|---|
| C1 | `src/llama-hessian.cpp` uses `<unistd.h>`/`pread`; it is unity-built into `llama.o` and the CMake `gpttype_adapter`, so MSVC (and MinGW, which has no `pread`) can't build koboldcpp | 2 |
| C2 | ~17 architectures multiply weights with `ggml_mul_mat` directly, bypassing the rotation hook: `--hadamard` makes files of them that refuse to load | 2 |
| C3 | the graph check compares the input width with a *view's* width: a view narrower than the stored row passes, though rows are rotated over their full width | 2 |
| C4 | hessian-collect's forced last logit adds an off-stride row to the LM-head Gram | 2 |
| C5 | `quantize.hq.gptq_hessian*` keys are dropped when HQ tensors are copied from a full-H source, while `gptq_damp` is kept | 2 |
| A2 | GPTQ on/off and damping are env vars read inside libllama (and again in tensor-kl); alpha and the Hessian are params | 3 |
| A3 | `--max-buffer-size` bounds the row buffers and, separately, the GPTQ factors | 3 (docs) |
| A4 | `llama_model_quantize_impl` grew ~250 lines of inline HQ/GPTQ logic | 3 |
| A5 | the KV-cache refactor into `src/llama-hadamard.h` is churn against upstream; `llama_hadamard_inplace` is dead | 3 |
| R1 | `ggml-quants-hq.c` copies `make_qkx3_quants` / `make_qp_quants` as uniform-weight specializations | 3 |
| R2 | in `ggml-quants-hq.c`: HQ4_K/HQ5_K pack codes by hand, HQ4_0/HQ5_0 and HQ4_1/HQ5_1 are pairwise identical, HQ2_XS/HQ2_S share their search | 3 |
| R3 | `ggml.c`: 17 hand-written HQ type traits, 17 + 5 case labels in `ggml_quantize_chunk` / `ggml_quantize_init` | 3 |
| R4 | `tensor-kl` copies the seed default, `tensor_allows_quantization` and the imatrix normalization | 3 |
| R5 | tokenize-with-retry written three times (hessian-collect, hessian-tokenize, `kl-eval.h`) | 3 |
| R7 | eight Makefile tool targets repeat the same link list | 3 |
| R9 | `ggml-cuda.cu` forward-declares the RHT entry points because the kernel header can't be included | 3 |
| O4 | the codebook decode grids are rebuilt on every `quantize_hq` call | 3 |
| O5 | the LoRA merge allocates a vector per row | 3 |
| O1 | `llama_gptq_factor_full` holds A as n×2n doubles plus an n×n float Gram (22·n² bytes) | 4 |
| O2 | `llama_parallel_rows` spawns and joins threads for every panel of the factorizations | 4 |
| O3 | hessian-collect's eval callback asks for the inputs of inactive layers, forcing a split and a sync at each | 4 |
| O6 | IQ4 codes are chosen with the unrounded `1/(d·l)`, unlike every other HQ type (quality experiment) | 4 |

Not changed, on purpose: `nearest_int`, `best_index_int8` and `get_scale_min_k4` stay as local copies in
`ggml-quants-hq.c` (a few lines each, hot, and upstream itself copies them per file); hessian-collect and
tensor-kl keep their POSIX I/O (Linux-only tools, like their `sysconf`/`posix_fadvise` use).

## Implementation conventions

As in the earlier plans:
- Keep comments minimal; match the surrounding style.
- Every change that can fail gets a test in its phase, and a phase is done only when its tests pass.
- The base quantizers in `ggml-quants.c` aren't modified (thin exported wrappers are fine, as
  `ggml_iq2_entry` already is).
- Phase 3 is refactoring only: every recorded hash and every golden file must stay byte-identical.
- Quantizer changes that change bytes (Phase 4) are judged by KL against the BF16 model, not weight MSE.

Assets: `tools/models/{Qwen3-0.6B-BF16.gguf, wiki.train.raw, wiki.test.raw}`. Work files go to a git-ignored
directory (`$WORK` below).

---

## Phase 1 — Oracles and baselines (no product change)

1. **GPTQ output hashes** (`test-hq-gptq`): the uniform quantizer's output is already pinned by recorded
   hashes; GPTQ's isn't. Add `test_gptq_hashes`: for all 17 types and their three widths, `hash_rows`
   rows quantized with the diagonal factor of `spiky_v`, recorded from the current code. This pins the
   element-wise and group steps and `llama_gptq_factor` (whose output doesn't depend on threads).
2. **Architecture sweep** (new `tests/test-hadamard-archs.cpp`, targets `test-hadamard-archs` and
   `-cuda`): the random-model fixture of upstream's `tests/test-llama-archs.cpp` (at `511f9c137`, the
   fork's last upstream merge) builds a tiny F32 model of every architecture, dense and MoE. For each:
   logits of the model as built; then every tensor the quantizer would rotate (the predicate of
   Phase 2.2) is rotated in place (`w ← R·w` per row) and put in the model's rotated set, and the logits
   must match (NMSE < 1e-6) with no guard error. It also reports how many tensors were rotated. Run it
   on the current code: it must fail for the architectures of C2, which shows that it detects them.
3. **Calibration files for 0.6B**: `hessian-collect-cuda` on 100 chunks × 512 of `wiki.train.raw`, with
   `--imatrix-out`, gives `h06.gguf` (Hessian) and `im06.gguf` (imatrix).
4. **Golden quantizations** (new `tools/quantize/golden.sh record|check <model> <imatrix> <hessian> <dir>`):
   quantizes a fixed set of configurations and records their sha256; `check` requantizes and compares.
   Configurations: `--hadamard Q4_K_M`; `--hadamard --imatrix Q4_K_M`; `--hadamard --imatrix IQ3_M`;
   `--hadamard Q4_0`; `--hadamard --imatrix IQ4_XS`; `--imatrix Q4_K_M` (plain); `--hadamard --hessian
   Q4_K_M`. Also recorded: `hessian-collect` (CPU) on 4 chunks, for C4/O3/R5.
5. Run every existing test suite (`test-hadamard`, `test-hadamard-quants`, `test-hadamard-quantize`,
   `test-hadamard-llama`, `test-hq-gptq`, `test-hessian`, `tools/hessian/tests.sh`, `test-backend-ops -o RHT`)
   to know they pass before any change.

## Phase 2 — Correctness

1. **C1, portable Hessian reader.** `llama_hessian` reads through `llama_file` (`src/llama-mmap.h`: fopen
   with 64-bit seeks on every platform) instead of an fd with `pread`. Reads are single-threaded
   (the factor build), so one file object suffices; it is `mutable` because the read methods are const.
   *Tests:* `test-hessian` and the cpu phase of `tools/hessian/tests.sh` pass; a check in
   `tools/quantize/tests-hq.sh` fails if `src/*.cpp` includes `unistd.h`/`fcntl.h` or calls
   `pread`/`pwrite` outside `llama-mmap.cpp`. (No MSVC or MinGW here, so this can't be compile-tested.)
2. **C2, route every weight GEMM.**
   - `llm_graph_context::build_mm(w, cur)`: `ggml_mul_mat(w, rotate_input_if_rotated(w, cur))`, no LoRA
     (so LoRA behaviour of those architectures doesn't change), and `build_mm_id` for `ggml_mul_mat_id`.
   - Every model file that multiplies a model weight directly calls these instead.
   - The predicate "the quantizer would rotate this tensor" becomes one function in `llama-quant.cpp`,
     used by the quantizer itself and exported through `llama-ext.h` as `llama_quant_tensor_rotatable`
     for the sweep and tensor-kl: quantizable, not an embedding (token embeddings, or `output.weight`
     without `token_embd`), and a width the RHT accepts.
   - *Tests:* `test-hadamard-archs` passes for every architecture the fixture supports. Any tensor the
     quantizer rotates but a graph uses outside a GEMM is either routed or excluded from rotation.
3. **C3, full-row views only.** `rotate_input_if_rotated` and the graph check require a rotated weight's
   view to keep whole stored rows (`ne[0]` and `nb[0]` of its root, row-aligned offset and stride). *Tests*
   (`test-hadamard-llama`, guard): refuses a view narrower than the stored row with an RHT input of the view's
   width, and one starting mid-row; still accepts row-range views and a view of every other row.
4. **C4, LM-head rows.** Where a GEMM input's rows are the ubatch's outputs, `counted_runs` counts a row only
   if it is one of the document's stride outputs, not merely counted. *Tests* (`tools/hessian/tests.sh`):
   a 100-token document counting [60, 100) at stride 64 in ubatches of 64, whose second ubatch has no output of
   its own: the LM head counts 1 row (it was 2); a new `test-hessian count FILE WEIGHT [N]` checks a weight's
   count. Layer Grams stay byte-identical to Phase 1.4's.
5. **C5, one rule for the three GPTQ keys.** Each of `quantize.hq.gptq_damp`, `quantize.hq.gptq_hessian_alpha`
   and `quantize.hq.gptq_hessian` is written when this run used it, otherwise kept only if HQ tensors were
   copied from the source. *Tests* (`test-hadamard-quantize`, new full-Hessian section with a synthetic
   Hessian file): `--hessian` writes all three keys and uses GPTQ with the full Gram (lower output error
   than uniform under the full H); copying that file keeps all three; a run without HQ copies and
   without `--hessian` has no Hessian keys.

## Phase 3 — Architecture and reuse, byte-identical

Gate for every step: `test-hq-gptq` (both hash sets), `golden.sh check`, and the suites of Phase 1.5 pass.

1. **R3, `ggml.c`.** The HQ traits come from one macro line per type; `ggml_quantize_init` maps HQ types to
   their base; `ggml_quantize_chunk` sends every rotated type to `quantize_hq` before its switch.
2. **R1, R2, O4, `ggml-quants-hq.c`.**
   - `ggml-quants.c` exports `ggml_make_qkx3_quants` / `ggml_make_qp_quants` wrappers; the HQ file calls them
     with all-one weights (bitwise identical: every weight enters as a multiplication by 1) and drops its copies.
   - HQ4_K/HQ5_K pack through `hq_set_codes`; one `_0` and one `_1` implementation serve both legacy widths;
     HQ2_XS/HQ2_S share one 16-value search.
   - The codebook decode grids are built once per type (under the critical section), not per call.
3. **A2, GPTQ settings as params.** `llama_model_quantize_params` gains `hq_gptq` (default true) and
   `hq_gptq_damp` (default `LLAMA_GPTQ_DAMP_DEFAULT`); libllama reads no environment. `quantize_gguf` and
   `tensor-kl` get `--no-gptq` and `--gptq-damp D`; the eval harness passes `--gptq-damp`.
   *Tests:* `test-hadamard-quantize` sets the params instead of the environment (same checks: off gives
   the uniform bytes, damp 0.0001 refused, 0.1 used and recorded); `golden.sh check` (defaults unchanged).
4. **A4, quantizer structure.** From `llama_model_quantize_impl`: the per-tensor HQ decision (target type,
   GPTQ plan) with its counters in a struct, the metadata writing (with C5's rule), and the per-expert
   factor lookup become functions.
5. **A3, memory cap (docs).** `--max-buffer-size`'s help and README say that GPTQ factors (the cache and
   the one being built) are held within the same size again, so the peak can reach about twice it.
   Bounding both together would halve the factor cap and break the 27B full-H defaults, so it stays.
6. **R4, tensor-kl.** Takes the seed default from `llama_model_quantize_default_params()`, quantizability
   and rotatability from `llama-ext.h`, and the imatrix means from a new `common_imatrix_means` in
   `common/imatrix-loader`, which `quantize.cpp` uses too.
7. **R5.** hessian-collect, hessian-tokenize and `kl-eval.h` use `common_tokenize` / `common_detokenize`.
   *Tests:* hessian-collect output = Phase 1.4's; an existing `.kl-cache` is still accepted (its token
   hash is unchanged).
8. **O5.** The LoRA merge reuses one delta buffer per worker thread.
9. **A5.** `src/llama-kv-cache.cpp` goes back to upstream's code; `src/llama-hadamard.h` is deleted.
10. **R9.** The kernels move to `ggml-cuda/rht-impl.cuh`; `rht.cuh` declares `ggml_cuda_op_rht` and
    `ggml_cuda_rht_write_q8_1` like every other op header. *Tests:* CUDA build, `test-backend-ops -o RHT`,
    `bench-rht` builds.
11. **R7.** The Makefile's tool targets share `LLAMA_TOOL_OBJS` / `LLAMA_TOOL_OBJS_CUDA`. *Test:* every
    target builds.

## Phase 4 — Optimizations (numbers may change)

1. **O1, full-Gram factor in n² doubles.**
   - The Gram is read in row slabs straight into A (a new `read_gram(row0, nrows, dst)` callback;
     `llama_hessian::read` already takes row ranges), and normalized after, from A's diagonal.
   - Cholesky of `J·H·J = Wᵀ·W` in place (not augmented), then W⁻¹ in place by a blocked upper-triangular
     inverse (per block column: `X12 ← X11·W12`, then `X12 ← −X12·W22⁻¹`, then invert `W22`), all
     row-parallel with a fixed operation order. `U[i][j] = W⁻¹[n−1−j][n−1−i]`.
   - Peak: 8·n² + 2·n² bytes + a slab (from 22·n²): at n = 17408, 6.7 GB → 3.0 GB; the largest width under
     the 8 GiB default rises from ~19.8k to ~29k (70B-class `ffn_down`).
   - *Tests* (`test-hq-gptq`): all existing full-Gram checks (diagonal Gram = imatrix factor, α = 1,
     `|U·H·Uᵀ·y − y|` bounds, 1 vs 8 threads bitwise, refusals, cache); new: `llama_gptq_build_bytes_full`
     is the new formula, a failed slab read is refused. End to end: KL of the golden `--hessian` quant
     within noise of Phase 1's (and its factor time).
2. **O2, a persistent worker pool** inside the factorizations (Cholesky panels and the new inverse):
   threads live for one factorization; tasks keep their fixed split, so U stays bitwise independent of
   the thread count. *Tests:* the same, plus factor timings at n = 5120/17408 before and after.
3. **O3.** hessian-collect's eval callback answers "ask" with the group's `active`, so only the current
   pass's layers split the graph. *Tests:* layer Grams = Phase 1.4's; `tools/hessian/tests.sh` passes.
4. **O6, IQ4 codes at the stored scale** (experiment): choose HQ4_NL/HQ4_XS codes with the inverse of the
   stored (fp16-rounded) scale, as the other HQ types do, in both the uniform encoder and GPTQ. Keep only
   if KL vs BF16 improves for `--hadamard IQ4_XS` and `--hadamard IQ4_NL`, with and without the imatrix
   (0.6B, `wiki.test.raw`); then re-record the affected hashes and golden files. Otherwise revert and
   record the result.

## Phase 5 — Docs and records

`tools/quantize/README.md` (new flags, no env vars, memory note, the architecture sweep), the eval
harness README, this plan's implementation record, memory notes.

## Test list

| area | test | phase |
|---|---|---|
| GPTQ bytes of all 17 types pinned | `test-hq-gptq` (GPTQ hashes) | 1 |
| every architecture: rotated = unrotated logits, no guard error | `test-hadamard-archs` | 1, 2 |
| quantized files unchanged by refactors | `tools/quantize/golden.sh check` | 1, 3 |
| no POSIX I/O in `src/` | `tests-hq.sh` | 2 |
| column and transposed views refused | `test-hadamard-llama` | 2 |
| LM-head rows follow the stride | `tools/hessian/tests.sh` + `test-hessian count` | 2 |
| full-Hessian quantization and the three keys | `test-hadamard-quantize` | 2 |
| GPTQ params instead of env | `test-hadamard-quantize` | 3 |
| full-Gram factor accuracy, determinism, memory | `test-hq-gptq` | 4 |
| KL of the full-H quant, IQ4 experiment | `test-hadamard-ppl` | 4 |

## Risks

- **The fixture may not build every architecture** in this fork; it skips what upstream's skips, and
  any other failure unrelated to rotation (it fails without rotation too) is skipped and listed.
- **Routing changes 17 upstream model files.** `build_mm` keeps each call's shape, so the diff is one
  token per call site; upstream merges will conflict on those lines, which is the price of correctness.
- **In-place inverse accuracy.** Triangular inversion is less stable than solving with the identity for
  ill-conditioned W; the damping floor bounds the condition number and the U·H·Uᵀ tests measure it.

## Implementation record

Implemented 2026-09-29; uncommitted. All measurements on this box (8 cores, RTX 5090, WSL2) with Qwen3-0.6B.

### Phase 1

- **GPTQ hashes:** `hq_gptq_hashes` (51 entries) in `test-hq-gptq`, and `--record`, which prints both tables; it
  reproduced every recorded uniform hash.
- **`test-hadamard-archs`:** the fixture is upstream's `tests/test-llama-archs.cpp` at `511f9c137`, lines `nmse` to
  `arch_supported`, unchanged. Deviation: upstream skips architectures the model saver can't write only for its
  save/reload roundtrip, so this test doesn't skip them. Every variant runs with flash attention on (auto) and off (see
  Phase 2). On the code before the fixes: 117 variants pass (NMSE ~3e-14), 10 fail with a guard error (minicpm3,
  deepseek2, deepseek32, deepseek4, glm-dsa, bailingmoe3, dots3note, mistral4, kimi-linear, kimi-k3), 3 are broken without
  rotation too (eagle3 and dflash don't build; qwen3tts gives non-finite logits, upstream's `n_vocab_out` TODO). Not
  built by the fixture: clip, gptj, the BERT family, gemma4, gemma4-assistant, gemma-embedding, deepseek2-ocr,
  t5encoder, rwkv6, rwkv6qwen2, rwkv7, arwkv7, graniteswitch, chameleon, wavtokenizer-dec, plm, llama-embed. 4 s.
- **Calibration files:** `hessian-collect-cuda`, 100 × 512 of `wiki.train.raw`: `h06.gguf` (1.77 GB) and `im06.gguf`, 9 s.
- **Golden outputs:** `golden.sh` takes 2 minutes and reproduces its own record.
- **Baselines** (KL to BF16, 40 × 512 of `wiki.test.raw`): `--hadamard --hessian Q4_K_M` 0.027308; `--hadamard
  IQ4_XS` 0.145369, with the imatrix 0.058425; `--hadamard IQ4_NL` 0.137891, with the imatrix 0.054590.

**After the upstream merge (2026-10-04):**
- The fixture is now upstream's `tests/test-llama-archs.cpp` at `53ed051ce`, which covers the new maple, hy_v4,
  hrm_text and spark2_5 architectures, and fixes bailingmoe3's rope sections.
- Upstream's fixture now defaults to a weight stdev of 0.1. This test keeps 0.01, because its 1e-6 NMSE threshold
  assumes that scale: at 0.1, CUDA reaches ~1.5e-6 in a few architectures.
- Result: 264/264 on CPU and CUDA.
- The weight GEMMs of hy-v4, the nemotron-h-moe latent projections and the deepseek4 MTP head went through
  `build_mm` as part of the merge.

### Phase 2

- **C1:** `llama_hessian` reads through `llama_file`. Its buffered reads return short past the end instead of failing,
  so `read_at` checks the range first. The new `tests-hq.sh` check flags the old code.
- **C2:** `build_mm` / `build_mm_id`: 130 call sites in 17 model files; the static conv helpers of bailingmoe3,
  kimi-linear and kimi-k3 get their input from `rotate_input_if_rotated` at the call site; dflash's static helper calls
  `g.build_mm`; deepseek4's `hc_fn` and granite-switch's switched LoRA go through the helpers too. The sweep found two
  more problems beyond the audit's list:
  - the non-flash-attention MLA path of `build_attn_mha` (`kqv = ggml_mul_mat(v_mla, kqv)`), which the CUDA sweep hit
    because flash attention falls back off there. Fixed, and the sweep now runs both paths;
  - deepseek4's compressor position tables (`attn_compressor_ape`, `indexer_compressor_ape`) are read with GET_ROWS
    but were rotatable. The rotation rule (`tensor_is_rotatable`) now requires the op that `llama-arch.cpp`'s tensor
    table gives for the name (new `llm_tensor_op_for_name`) to be MUL_MAT or MUL_MAT_ID, plus the old exception for
    `output.weight` without `token_embd`. Of all the fixture's variants, only deepseek4's count changed (65 → 62).
  - `llama-model.o` now depends on `src/models/*.cpp`, which it includes; edits there didn't rebuild it.

  Result: 254 of 254 variants (127 × flash attention on and off) pass on the CPU and on CUDA.
- **C3:** a view of a rotated weight must keep its stored rows: the same `ne[0]` and `nb[0]`, a row-aligned offset and
  row stride (skipping whole rows is fine). Checked in `rotate_input_if_rotated` and in the graph guard. The planned
  transposed-view test was dropped: `ggml_mul_mat` asserts on a transposed src0 before the guard could see it; the
  guard tests are a narrower view (refused), a view starting mid-row (refused), every other row (accepted).
- **C4:** correction to the audit and the plan: a logit is forced only for a decode with no output of its own (a packed
  batch, or a ubatch-sized chunk of a long document), so full 512-token chunks never trigger it and the planned
  "256, was 260" check is wrong. The test is a 100-token document counting [60, 100) at stride 64 in ubatches of 64: the
  LM head counted 2 rows, now 1. Writing it turned up a crash: the discovery decode asked for 64 logits, more than
  `n_outputs_max` allows below `--ubatch` ≈ 512; it now asks for one.
- **C5:** the rule is as planned. In the new full-Hessian test (G = diag(v) + u·uᵀ, u of the diagonal's size) the
  attn_q output error under G is 0.00541 uniform, 0.00416 diagonal GPTQ, 0.00175 full-H GPTQ; the copy check fails on
  the old rule.

### Phase 3

- **Gate:** all 102 recorded hashes, 8 of 8 golden files byte-identical (with O1 set aside), `test-hadamard-quantize`,
  `test-hadamard-llama`, `tools/hessian/tests.sh` (36 checks), `test-backend-ops -o RHT` (426 on CUDA).
- **R1:** the exported `ggml_make_qkx3_quants` / `ggml_make_qp_quants` with all-one weights give the same bytes, as both
  objects build with the same flags (ISO C, so no FMA contraction, no fast-math).
- **A2:** `hq_gptq` and `hq_gptq_damp` in `llama_model_quantize_params`; `--no-gptq` and `--gptq-damp` in
  `quantize_gguf` and `tensor-kl`; the eval harness passes `--gptq-damp`. libllama reads no environment for GPTQ.
- **A4:** `hq_plan` (`target_type`, `plan_gptq`, `factor`, `finish`).
- **R5:** `common/common.h` brought two clashes: its `die` macro (hessian-collect's `die` is now `fatal`) and its
  `regex_escape`, which `tensor-kl` had copied (removed). Tokenization is unchanged: existing `.kl-cache` files are
  accepted and every KL row reproduces exactly.
- **R4** also moved the imatrix normalization into `common_imatrix_means`, which `quantize.cpp` uses too.
- Not done, as planned: hessian-collect and tensor-kl keep their POSIX I/O. There is no MSVC or MinGW here, so the
  Windows build itself is untested.

### Phase 4

- **O1:** peak RSS of the n = 17408 full-Gram factor 5.33 → 3.00 GB (build bytes 6.67 → 3.06 GB); time 10.4–10.9 s →
  10.6–11.6 s (Gram read and rotations 2.6 s, Cholesky 3.8 s, inverse 4.4 s, U 0.5 s; tiles of 32–128 rows and 256–512
  reductions made no difference). The U entries differ only below fp32 rounding at these widths: the golden full-H
  quantization stays byte-identical and so its KL (0.027308). `test-hq-gptq` bounds hold (n = 17408: 1.8e-7).
- **O2:** starting and joining threads costs ~450 µs per parallel loop here: 0.25 s of an n = 17408 factor (~550
  loops), ~70 ms of an n = 5120 one. With the pool, n = 5120 takes 0.46 s (0.50 s without). The pool lives in
  `llama-quant-gptq.cpp`; the factor and cache functions no longer take a `workers` vector. Output unchanged (hashes).
- **O3:** the 6-pass collection's output is byte-identical; no measurable time change on 0.6B (5.7–5.8 s of passes),
  where a sync is cheap. Kept for the simpler callback.
- **O6, reverted** (codes at the stored scale; KL to BF16):

  | | uniform | with the imatrix |
  |---|---|---|
  | `--hadamard IQ4_XS` | 0.145369 → 0.145227 | 0.058425 → 0.056522 |
  | `--hadamard IQ4_NL` | 0.137891 → 0.137895 | 0.054590 → 0.055313 |

  Not an improvement for all four, so the change is reverted: the scale's fp16 rounding moves few codes, and the
  differences look like noise in both directions.
