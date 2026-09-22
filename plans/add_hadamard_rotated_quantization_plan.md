# Plan: Add Hadamard-Rotated Quantization (ConvRot) to the LLM Path

## Goal
Add "rotated" variants of the Q4/Q5 quantizations — `Q4R_0`, `Q4R_1`, `Q4R_K`, `Q5R_0`, `Q5R_1`, `Q5R_K` — that store weights Hadamard-rotated *before* quantization (Strategy A). The inference engine rotates the corresponding input activations so the model output is unchanged. This suppresses weight outliers, improving low-bit quantization quality.

## Locked design decisions
1. **LLM path only** (sdcpp skipped — it has no quantized GEMM).
2. **Pure block-Hadamard** (deterministic Walsh-Hadamard, self-describing — no random signs).
3. **New ggml types** (not a metadata list): the rotation is a property of the type. All 6 variants.
4. **Rotation group = base type's block size** (baked into the type): 32 for `_0`/`_1`, 256 for `_K`. No group-size metadata.
5. **Strategy A**: weights stay quantized (rotated); the GEMM consumes them as the base type; inference rotates the activation.
6. **Shared Hadamard code**: extract `ggml_gen_hadamard`'s core so the KV-cache and ConvRot paths share it.

## Core math / invariant
For weight `W` (input dim `D`, rotation group `g = blck_size(base_type)`), store `W' = W·H_g` quantized as the base type. `H_g` is the orthonormal Walsh-Hadamard of size `g`, applied block-diagonally over `D/g` groups. At inference, rotate the input activation: `y = W'·(H_g·x) = W·H_g·H_g·x = W·x` (since `H_g·H_g = I`). Output unchanged. `H_g` is symmetric (`H_g = H_gᵀ`), so the same matrix is used to rotate the weight (quantize) and the activation (inference).

## Architecture (end-to-end)
- **Quantize:** user selects a rotated type (or `--hadamard` + base ftype). The quantizer dequantizes to f32, rotates by `H_g`, quantizes **as the base type**, and stores the tensor with the **rotated type**.
- **Store:** GGUF type index = rotated type (e.g. `Q4R_K`); bytes = base-type layout (`Q4_K`).
- **Load:** new engine reads the rotated type (valid, `< GGML_TYPE_COUNT`). Old engine hits `type >= GGML_TYPE_COUNT` → hard error (fail-loud).
- **Infer:** graph builder sees a rotated-type weight → rotates the input activation by `H_g` → the GEMM consumes the weight as the base type.

---

## Implementation notes for the implementing agent
Cross-cutting caveats that apply across all phases:
1. **Line numbers will drift.** The file:line references in this plan (and the index below) are a snapshot. Re-verify each against the current code as you start a phase — the *symbols* (function names, the `type_traits`/`type_traits_cpu` entries, `ggml_quantize_chunk`, the `build_qkv`/`build_ffn`/`build_attn` injection points) are far more stable than the line numbers.
2. **Two subtle gotchas:**
   - The quantizer must call `ggml_quantize_chunk` with the **base type** (e.g. `Q4_K`), *not* the rotated type (`Q4R_K`) — `ggml_quantize_chunk` has its own type switch with no rotated cases. The rotated type is used only for the stored tensor type (`gguf_set_tensor_type`).
   - Metal's `get_pipeline_mul_mm` builds the kernel name from `ggml_type_name(type)` — remap via `ggml_get_base_type` so it reuses the existing `q4_K`/`q4_0`/… kernels.
3. **Build after each phase.** The sequencing (0→1→2→3→4) is set up so each phase is independently verifiable: a rotated tensor *loads* after Phase 2, a rotated *artifact* after Phase 3, correct *perplexity* after Phase 4.

---

## Phase 0 — Shared Hadamard utilities
New header-only file **`src/llama-hadamard.h`**:
- `void llama_gen_hadamard_matrix(float * out, int n)` — fills an `n×n` orthonormal Walsh-Hadamard (Sylvester construction, scaled 1/√n). Extract the core from `ggml_gen_hadamard` (`src/llama-kv-cache.cpp:24`).
- `void llama_hadamard_inplace(float * vec, int n)` — in-place FWHT butterfly (O(n log n)), scaled 1/√n. (Algorithm from `ggml_compute_forward_fwht_f32`, `ggml/src/ggml-cpu/ops.cpp:11847`.)
- Refactor `ggml_gen_hadamard` (`src/llama-kv-cache.cpp:24`) to call `llama_gen_hadamard_matrix` (no behavior change to the KV-cache path).

CMake: header-only, no new source target.

## Phase 1 — New ggml types (`Q4R_*`, `Q5R_*`)
**`ggml/include/ggml.h:395-440`** — add at the end of the enum (before `GGML_TYPE_COUNT`), bumping the count from 43 → 49:
```
GGML_TYPE_Q4R_0 = 43,  GGML_TYPE_Q4R_1 = 44,  GGML_TYPE_Q4R_K = 45,
GGML_TYPE_Q5R_0 = 46,  GGML_TYPE_Q5R_1 = 47,  GGML_TYPE_Q5R_K = 48,
GGML_TYPE_COUNT = 49,
```
Use **distinct `type_name`s** (`"q4r_0"`, …, `"q5r_k"`) for clear logs; the Metal name-based pipeline is handled via the `ggml_get_base_type` remap (Phase 2).

**`ggml/src/ggml.c:635`** (`type_traits`) — add 6 entries, each copying the base type's row (same `type_size`, `blck_size`, `to_float`, `from_float_ref`):
- `Q4R_0`←`Q4_0`(`:696`), `Q4R_1`←`Q4_1`, `Q4R_K`←`Q4_K`(`:787`), `Q5R_0`←`Q5_0`(`:724`), `Q5R_1`←`Q5_1`, `Q5R_K`←`Q5_K`(`:795`).

**`ggml/src/ggml-cpu/ggml-cpu.c:215`** (`type_traits_cpu`) — add 6 entries, each copying the base row (same `vec_dot`, `vec_dot_type`, `from_float`):
- `Q4R_K`←`Q4_K`(`:311`): `vec_dot=ggml_vec_dot_q4_K_q8_K`, `vec_dot_type=GGML_TYPE_Q8_K`, `from_float=quantize_row_q4_K`. Others similarly (`Q4R_0`←`Q4_0` `:240`, `Q5R_0`←`Q5_0` `:260`, …).

This makes CPU GEMM, dequant, quantize, Blas, and GGUF byte-size all work identically — **no kernel duplication**.

## Phase 2 — `ggml_get_base_type` helper + backend dispatch
New helpers in **`ggml/src/ggml.c`** (declared in `ggml/include/ggml.h`):
```
ggml_type ggml_get_rotated_type(ggml_type t) // Q4_0→Q4R_0, …, Q5_K→Q5R_K, else identity  (single source of truth for the pairing)
ggml_type ggml_get_base_type(ggml_type t)    // Q4R_0→Q4_0, …, Q5R_K→Q5_K, else identity  (inverse of ggml_get_rotated_type)
bool      ggml_is_rotated(ggml_type t)       // ggml_get_base_type(t) != t
```
**DEVIATION (dedup, added in Phase 3):** both `ggml_get_rotated_type` and `ggml_get_base_type` are plain case statements (kept in sync). The rotation-group / divisibility logic is centralized in two quantizer helpers (`llama_hadamard_rot_group`, `llama_hadamard_can_rotate` in `src/llama-quant.cpp`), shared by `llama_hadamard_target_type` and `llama_model_quantize_impl` — no duplicated group math.
Rotation group for a rotated type = `ggml_blck_size(ggml_get_base_type(t))` (32 or 256).

**Backend dispatch** (route rotated types to the base type's kernel; no kernel duplication):
- **CPU:** table-driven (`ggml-cpu.c:1183-1184`) — handled by Phase 1's `type_traits_cpu`. No change.
- **Blas:** table-driven (`ggml-blas.cpp:69-70`) — handled by `type_traits`. No change.
- **CUDA:** `ggml/src/ggml-cuda/mmq.cu:8` (`ggml_cuda_mul_mat_q_switch_type`) and `mmvq.cu:1096` (`mul_mat_vec_q_switch_type`) — add cases (or dispatch on `ggml_get_base_type`). Kernels use `ggml_cuda_type_traits<base>` (`common.cuh:1059`) — correct since layout is identical.
- **Metal:** `ggml-metal-ops.cpp:2375-2397` (small-batch if-chain) and `ggml-metal-device.cpp:842` (`get_pipeline_mul_mv`) — add cases. **Name gotcha:** `get_pipeline_mul_mm` (`:785`) builds the kernel name from `ggml_type_name(type)` — remap via `ggml_type_name(ggml_get_base_type(type))` so it reuses the existing `q4_K`/`q4_0`/… kernels.
- **Vulkan:** `ggml-vulkan.cpp:4803+` — add `CREATE_MM2(Q4R_K, pipeline_…[Q4R_K], matmul_q4_k_f32, …)` per variant (reuse the base shader). Pipeline arrays are indexed by `GGML_TYPE_COUNT` (`:921`) → auto-size.
- **Spacemit-CPU:** `ggml-cpu/spacemit/ime.cpp:190` — add cases.
- **Optional:** ftype label switch (`src/llama-model-loader.cpp:749-775`).

## Phase 3 — Quantizer (create rotated models)
**`include/llama.h:436-452`** — add `bool hadamard;` to `llama_model_quantize_params` (the `--hadamard` convenience flag: auto-convert base Q4/Q5 types to their rotated variants where divisible).

**`tools/quantize/quantize.cpp`:**
- ~~`QUANT_OPTIONS` — add the 6 rotated types.~~ **DEVIATION:** did NOT add 6 rotated types / new `llama_ftype`s (that would require new ftype entries + mix logic). Direct per-tensor selection of a rotated variant instead uses the **existing** `--tensor-type name=ggml_type` option — a rotated type is a valid `ggml_type` there, e.g. `--tensor-type 'blk.0..*'=q4r_K` (pattern is a regex). `--hadamard` covers all layers.
- Arg loop — add `--hadamard` (sets `params.hadamard`).
- `usage()` — document `--hadamard` (and point at `--tensor-type` for per-tensor rotation).

**`src/llama-quant.cpp`:**
- `#include "llama-hadamard.h"`.
- New helper `llama_hadamard_target_type(t, ne0)` — returns `ggml_get_rotated_type(t)` **only** when `ne0 % blck_size(base) == 0`; otherwise returns `t` unchanged. (Reuses `ggml_get_rotated_type` for the pairing — **no duplicate switch**; see Phase 2 dedup.)
- `llama_model_quantize_default_params` — init `hadamard = false`.
- In `llama_model_quantize_impl`, per tensor:
  - **Target-type resolution (preliminary pass):** if `params.hadamard`, apply `llama_hadamard_target_type` to the resolved type (Q4_K → Q4R_K where divisible). A rotated type can also arrive directly via `--tensor-type`.
  - **Resolve quant/store types (outer scope):** `rotated_target = ggml_is_rotated(new_type)`; `rot_group = blck_size(base)`; `rotate = rotated_target && (ne[0] % rot_group == 0)`; `quant_type = rotated_target ? base : new_type`; `store_type = rotate ? new_type : quant_type`. **DEVIATION / safety:** if a rotated target is *not* divisible (only reachable via `--tensor-type`, since `--hadamard` never picks non-divisible), fall back to the base type — `store_type` becomes the base type and no rotation is done (prevents an out-of-bounds in the rotation butterfly).
  - `quantize = cur_type != store_type` (**DEVIATION:** was `!= new_type`; using `store_type` makes the non-divisible fallback a no-op copy rather than a wasteful/possibly-forbidden requantize).
  - **Between dequantize and re-quantize**, if `rotate`:
    - Copy the slab into `f32_conv_buf` (the F32+mmap branch is read-only — must not rotate in place); dequant non-F32 into it.
    - `llama_hadamard_inplace` per row, per group of `g = rot_group`.
    - **Imatrix:** copy + rotate the per-column imatrix vector (length `ne[0]` per expert) with the same block-Hadamard, keeping it in the rotated space.
    - **`bytes_per_row`** now includes the f32-buffer size when rotating an F32 tensor (the copy buffer), so `nrows_slab` stays within `max_buf_size` (quantizer-only; inference never slabs).
  - **Quantize as the BASE type:** `llama_tensor_quantize_impl(quant_type, …)` — so `ggml_quantize_chunk` (its own type switch) uses the base type's quantizer. Output bytes = base-type layout.
  - **Store with `store_type`:** `gguf_set_tensor_type(…, store_type)` — the rotated type when actually rotating, else the base type.

## Phase 4 — Inference (apply activation rotation) — DONE
**DEVIATIONS (implemented, verified):**
1. **No standalone `fwht` op exists** in this codebase — the FWHT compute path is only reachable via the `GGML_HINT_SRC0_IS_HADAMARD` hint on a `mul_mat` node (the same mechanism the KV-cache rotation uses; `llama_mul_mat_hadamard` in `src/llama-impl.h`). Reused that mechanism: each rotated GEMM gets a `mul_mat(H_g, x)` node with the hint set.
2. **Materialized F32 H_g marker matrix is required** (plan assumed none): on backends without a fast FWHT path for the group size (g=32 is not in CUDA/Metal/Vulkan's {64,128,256,512} fast set), the hinted node falls back to a *regular* mul_mat — which is only correct if src0 actually contains H_g. So H_g is materialized (F32) via `llama_gen_hadamard_matrix` and cached in a small real-memory ggml context owned by `llm_graph_result` (`ctx_hadamard` — the graph's compute context is no-alloc, so the matrix data cannot live there; it persists across graph resets since it is a constant). Fast FWHT when supported, exact F32 GEMM fallback otherwise — correct on all backends.
3. **Injection centralized in `build_lora_mm` / `build_lora_mm_id`** (the common GEMM wrappers, `src/llama-graph.cpp`) instead of explicit edits to `build_qkv`/`build_attn`/`build_ffn`: this covers every model that uses the wrappers, **including lm_head and model-specific projections** that the plan's injection points would have missed. Cost: the input is rotated per GEMM rather than once per (input, group) — negligible (O(D·log g) vs O(D·n_ff) per GEMM). The rotate block is factored into a shared helper `llm_graph_context::rotate_input_if_rotated(w, cur)` (returns `cur` unchanged for non-rotated weights) so the two wrappers do not duplicate it.
4. **Token embeddings are excluded from rotation** (quantizer, `src/llama-quant.cpp`): inference assumes every GEMM input activation is in the unrotated space. A rotated embedding table would put the first layer's input in the rotated space, which the engine cannot detect — so `--hadamard` skips `TOKEN_EMBD` tensors and an explicit `--tensor-type token_embd.weight=q4r_*` fails loud. (No quality loss: the benefit is in the rotated *weight* space of the first layer's GEMMs, which is still rotated.)
5. **Load-time guard (removed after audit).** An initial version added a guard in `llama_model_loader::create_tensor` (both branches) throwing when a rotated tensor's input dim is not divisible by the rotation group. The audit found it unreachable: the GGUF reader already rejects `ne[0] % blck_size(type) != 0` for *all* types at file parse (`ggml/src/gguf.cpp`, `gguf_init_from_file`/`_buffer`), and rotated types share the base type's blck (in this fork `QK_K = 256`, so the `_K` rotation group is 256). The file-less path (`llama_model_init_from_user`) is also covered: the gguf API asserts the same invariant when metadata is built (`gguf_set_tensor_type`), and graph construction aborts loud via `GGML_ASSERT` in `build_hadamard_rotate` (in this codebase `GGML_ASSERT` maps to `GGML_ABORT`, active in release). The guard was deleted as dead code.
6. **GEMMs that bypass the wrappers — now covered or fail-loud (gap fixed).** GEMMs that do not go through `build_lora_mm` / `build_lora_mm_id` would consume a rotated weight as `W'·x` instead of `W'·(H_g·x)` — silently wrong. Two classes, both handled:
    - **(b) Graph-level GEMMs in `src/llama-graph.cpp` — fixed:** the MLA `v_mla` projection (`build_attn`), the pooled-embedding `dense_2`/`dense_3` heads (`build_dense_out`), and the `cls`/`cls_out` classification heads (`build_pooling`) now rotate their input via `rotate_input_if_rotated(w, cur)`. This is always correct for a direct `mul_mat(w, cur)`: GEMM validity forces `w->ne[1] == cur->ne[0]`, and the weight is rotated along `w->ne[1]`, so dim 0 of `cur` is exactly the rotated dim. (The mean-pooling `mul_mat(transpose(embd), inp_mean)` site needs no fix — its src0 is the token embedding, which the quantizer never rotates.)
    - **(a) Model graphs with direct `ggml_mul_mat` calls (e.g. deepseek2, glm-dsa, plm, and ~18 other model files, ~126 sites) — fail loud:** rather than hand-fixing every exotic layout, a post-build guard `llm_graph_check_hadamard_rotation(gf)` (called once per graph build from `llama_model::build_graph`) walks the graph and throws if any `mul_mat`/`mul_mat_id` node has a rotated-type src0 whose src1 is not the hinted block-Hadamard rotation node. Covered GEMMs (the wrappers + the fixed graph-level sites) feed the rotation node directly as the GEMM input, so they pass; anything else now errors with the offending tensor name instead of silently corrupting output. Consequence: `--hadamard` on a model whose graph routes a rotated weight through an uncovered direct GEMM now refuses to run (loud) where it previously produced garbage (silent). Individual model sites can be fixed incrementally by routing them through `rotate_input_if_rotated`.
    `print_info` logs `n_hadamard_rotated` when present.

**Post-implementation audit (phase 4 vs phases 0-3):**
- **Bug fixed — LoRA + rotated weights (silent wrong output):** the initial injection rotated `cur` *in place* before the LoRA branch, so the adapter delta computed `B·(A·(H_g·x))` instead of `B·A·x` — adapters are trained in the unrotated space. Fixed in both `build_lora_mm` and `build_lora_mm_id`: only the main-GEMM input is rotated (`cur_rot = rotate_input_if_rotated(w, cur)`); the LoRA branch keeps the unrotated `cur`. (Correct alternative would be to rotate the adapter, `A' = A·H_g`, but keeping the delta in the unrotated space is simpler and exact.)
- **Deduplicated:** the 4-line rotate block that was copy-pasted in both wrappers is now the shared `rotate_input_if_rotated` helper (see deviation 3).
- **Removed dead code:** the redundant load-time guard (see deviation 5).
- **Gap fixed — uncovered GEMMs (silent wrong output):** the audit found GEMMs that bypass the wrappers would consume a rotated weight un-rotated. Fixed in two parts (see deviation 6): (1) the graph-level projections in `src/llama-graph.cpp` (`v_mla`, `dense_2`/`dense_3`, `cls`/`cls_out`) now rotate their input via `rotate_input_if_rotated`; (2) a post-build guard `llm_graph_check_hadamard_rotation` (called from `llama_model::build_graph`, the single choke point all graph builds pass through) throws if any rotated-type weight is consumed by a GEMM whose input is not the hinted rotation node — turning the remaining model-level direct-GEMM paths (deepseek2, glm-dsa, plm, …) from silent corruption into a loud error naming the offending tensor.
- **Verified:** the unity TU `src/llama.cpp` (which `#include`s `llama-graph.cpp`, `llama-quant.cpp`, `llama-model-loader.cpp`, …) passes a clean `-fsyntax-only` compile; group sizes, the Sylvester-matrix vs FWHT-butterfly equivalence, the backend hint fallbacks (CPU always-FWHT; CUDA/Metal/Vulkan fast set {64,128,256,512} with exact regular-GEMM fallback for g=32), the tied-embedding exclusion, and the `mul_mat`/`mul_mat_id` src ordering (src[0]=weight, src[1]=input) were all checked against the phase 0-3 code.

**Verification (small random-weight LLaMA test models, `--hadamard` + Q4_K ftype):** rotated-model embeddings match f32 within the same error as the non-rotated Q4_K baseline (max|F32-Q4R_K| = 0.00417 vs max|F32-Q4K| = 0.00408; Q4R_0 and Q5R_0 variants also pass; stable across repeated runs). A wrong/missing inference rotation would diverge by O(1) relative to the embedding norm.

---
(original plan text below)
Rotate the input activation with the existing `fwht` ggml op (available on CPU/CUDA/Metal/Vulkan) — it applies H_N along dim 0 for every row. No new ggml op, no materialized H matrix, no cached H tensor. (`llama_hadamard_inplace` is **not** used here — it stays in the offline quantizer only, so it is not in the inference hot path.)

**Rotation helper** (in the graph context): for an activation `x` with `ne[0] = D` and group `g = ggml_blck_size(ggml_get_base_type(weight->type))` (g ∈ {32, 256}, both powers of 2):
```
x_g    = reshape x to [g, D/g, batch]                          // each group of g -> a row (view, no copy)
x_rot  = ggml_fwht(ggml_ones(ctx, GGML_TYPE_F32, 0,0,0), x_g)  // applies H_g to every group
x_rot  = reshape x_rot back to [D, batch]
```
The fwht op requires F32 input and power-of-2 `ne[0]` — both satisfied. Cost is O(D log g) per unique input (~1/4096 of a GEMM).

**Optimization — rotate once, reuse across weights:** each unique input activation is rotated ONCE per group and shared by every weight that consumes it. Never rotate the same (input, group) pair twice:
- QKV: rotate `cur` once → feed `wq`/`wk`/`wv` (3 GEMMs, 1 rotation).
- FFN: rotate `cur` once → feed `up` + `gate`; rotate the gated output once → feed `down`.
- Attn-out: rotate the attention output once → feed `wo`.

**Injection** (`src/llama-graph.cpp`): for each rotated weight, rotate its input activation by `H_g`:
- QKV — `build_qkv` (`:1617-1691`): rotate `cur` before `wq`/`wk`/`wv` (`:1651/1661/1671`; fused `:1631`).
- Attn-out (wo) — `build_attn` (`:2787-2860`): rotate the attention-output `cur` before `wo` (`:2845/2851`).
- FFN — `build_ffn` (`:1694-1896`): rotate `cur` before `up` (`:1732`) + `gate` (`:1749/1754`); rotate the gated `cur` before `down` (`:1875`).

The base graph methods receive the weight tensors as parameters, so the type check is self-contained (no model/side-table needed).

**Load-time guard:** at model load, if a rotated-type tensor is present but the build lacks ConvRot inference → `throw` (safe invariant; a no-op if always compiled in).

## Phase 5 — Tests (in the style of the existing test suite)
The repo's test styles: `tools/quantize/tests.sh` (bash integration: download model → run CLI tools → `echo PASS` per step, `set -eu`); `gguf-py/tests/test_quants.py` (Python (de)quant must exactly match the C implementation via ctypes on libggml); `tests/test-chat-analysis.cpp` (C++ plain `main()` tool in `tests/`); `tests/test_koboldcpp.py` (Python doctest via `doctest.testmod()`).

**5a. Quantizer + inference integration — extend `tools/quantize/tests.sh`:** follow the existing pattern (download `Qwen3-0.6B-Q8_0`, `set -eu`, `echo PASS` per step):
- Requant with a rotated type: `$QUANTIZE ... $WORK_PATH/ggml-model-rot.gguf Q4R_K` (and/or `--hadamard` + `Q4_K`).
- Verify it loads and generates (mirrors existing steps 3a/4b): `$MAIN -no-cnv --model $WORK_PATH/ggml-model-rot.gguf -p "I believe the meaning of life is" --n-predict 32` → `echo PASS`.

**5b. Perplexity quality check — run the `perplexity` tool (`tools/perplexity/`):** quantize the same model as `Q4_K` and `Q4R_K`; run `perplexity` on both. The rotated model's perplexity must be *sane* (garbage ⇒ inference rotation wrong) and ≤ the non-rotated `Q4_K` perplexity, especially at low bits. (f16 is the upper bound.)

**5c. Hadamard correctness + round-trip — C++ test tool in `tests/` (style of `test-chat-analysis.cpp`):** a plain `main()` tool that:
- Verifies `llama_gen_hadamard_matrix` is orthonormal (`H·Hᵀ = I`).
- Verifies `llama_hadamard_inplace(v) == H @ v`.
- Round-trip: rotate f32 → `ggml_quantize_chunk` as the base type → `to_float` (dequant) → inverse-rotate ≈ original (within quant error).
- Quantizer↔inference consistency: the same rotation applied both ways (catches a silent mismatch that would corrupt output).

**5d. Python↔C parity — extend `gguf-py/tests/test_quants.py` (only if Python support is added):** if a Python implementation of the rotated types / Hadamard is added to `gguf-py`, extend `test_quants.py` to verify it exactly matches the C implementation (via ctypes on libggml), following the existing per-type pattern.

**5e. Backwards-compat:** non-rotated file loads unchanged; rotated file loads correctly in the new engine; a simulated old engine (built with `GGML_TYPE_COUNT=43`) refuses a rotated file (hard error at `ggml/src/gguf.cpp:758`).

## Phase 6 — Docs
- Document the feature, the 6 types, and the backwards-compat behavior (new types ⇒ fail-loud on old engines).
- Update the quantizer usage / README.

---

## Backwards compatibility
- **New engine + old (non-rotated) file:** safe (rotated types absent; no behavior change).
- **Non-rotated output:** byte-identical (uses `Q4_K`, not `Q4R_K`).
- **Old engine + rotated file:** **hard error** — old engine's `GGML_TYPE_COUNT` (43) < rotated type index → per-tensor `type >= GGML_TYPE_COUNT` check at `ggml/src/gguf.cpp:758` refuses the file. Fail-loud, not silent.

## Key file:line index
| Item | Location |
|---|---|
| `ggml_gen_hadamard` (refactor) | `src/llama-kv-cache.cpp:24` |
| FWHT butterfly (reuse) | `ggml/src/ggml-cpu/ops.cpp:11847` |
| `ggml_type` enum / `GGML_TYPE_COUNT` | `ggml/include/ggml.h:395-440` |
| `type_traits` (Q4_0/Q5_0/Q4_K/Q5_K) | `ggml/src/ggml.c:696,724,787,795` |
| `type_traits_cpu` (Q4_0/Q5_0/Q4_K/Q5_K) | `ggml/src/ggml-cpu/ggml-cpu.c:240,260,311,321` |
| Block sizes (QK_K/QK4_0/QK5_0/QK4_1/QK5_1) | `ggml/src/ggml-common.h:89,194,229,201,237` |
| CUDA GEMM switches | `ggml/src/ggml-cuda/mmq.cu:8`, `mmvq.cu:1096` |
| Metal dispatch + name gotcha | `ggml-metal-ops.cpp:2375`, `ggml-metal-device.cpp:842,785` |
| Vulkan `CREATE_MM2` + pipeline arrays | `ggml-vulkan.cpp:4803+`, `:921` |
| Spacemit switch | `ggml-cpu/spacemit/ime.cpp:190` |
| `fwht` ggml op (reuse for inference rotation) | `ggml_fwht` / `ggml_compute_forward_fwht_f32` `ggml/src/ggml-cpu/ops.cpp:11847` (also CUDA `fwht.cu`, Metal `ggml-metal-ops.cpp:2235`, Vulkan `ggml-vulkan.cpp`) |
| `GGML_HINT_SRC0_IS_HADAMARD` | `ggml/include/ggml.h:451` |
| `build_qkv` / `build_ffn` / `build_attn` | `src/llama-graph.cpp:1617 / 1694 / 2787` |
| `llama_model_quantize_impl` (dequant/quant) | `src/llama-quant.cpp:896` (dequant `:1295`, quant `:1304`) |
| `llama_tensor_quantize_impl` / `ggml_quantize_chunk` | `src/llama-quant.cpp:747` / `ggml/src/ggml.c:7973` |
| imatrix lookup / `tensor_requires_imatrix` | `src/llama-quant.cpp:1278` / `:805` |
| `llama_model_quantize_params` / defaults | `include/llama.h:436` / `src/llama-quant.cpp:1351` |
| Quantizer CLI (`QUANT_OPTIONS`/args/usage) | `tools/quantize/quantize.cpp:33-74 / 409-485 / 121-180` |
| GGUF tensor-type range check | `ggml/src/gguf.cpp:758` |
| ftype label switch | `src/llama-model-loader.cpp:749-775` |

## Risks & sequencing
- **Riskiest:** Phase 2 (backend dispatch) + Phase 4 (graph plumbing). Mitigate with the `ggml_get_base_type` helper (centralizes aliasing, resolves the Metal name gotcha) and the end-to-end perplexity test.
- **Order:** Phase 0 → 1 → 2 → 3 → 4 → 5 → 6. Types + dispatch first (so a rotated tensor can be loaded), then quantizer (produce an artifact), then inference, then validate with perplexity.
- **Scope note:** the `_0`/`_1` variants (block 32) give weaker mixing than the `_K` variants (block 256); the `_K` variants are where ConvRot pays off most.
- **New-context start:** begin at Phase 0; each phase is independently verifiable (types load after Phase 2; a rotated artifact after Phase 3; correct perplexity after Phase 4).
