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
1. **Line numbers will drift.** The file:line references in this plan (and the index below) are a snapshot. Re-verify each against the current code as you start a phase — the *symbols* (function names, the `type_traits`/`type_traits_cpu` entries, `ggml_quantize_chunk`, the `build_lora_mm` / `build_lora_mm_id` injection points) are far more stable than the line numbers.
2. **Two subtle gotchas:**
   - The quantizer must call `ggml_quantize_chunk` with the **base type** (e.g. `Q4_K`), *not* the rotated type (`Q4R_K`) — `ggml_quantize_chunk` has its own type switch with no rotated cases. The rotated type is used only for the stored tensor type (`gguf_set_tensor_type`).
   - Metal's pipeline getters build the kernel name from `ggml_type_name(type)` — remap via `ggml_get_base_type` in **all five** of them (`mul_mm`, `mul_mv`, `mul_mv_ext`, `mul_mm_id`, `mul_mv_id`) so they reuse the existing `q4_K`/`q4_0`/… kernels. A missed one aborts on a nonexistent kernel name.
   - Vulkan's `supports_op` switches on the raw `src0_type`; without a remap there, rotated weights never reach the GPU at all and every other Vulkan remap is unreachable.
   - The rotation node is a `mul_mat` wrapped in a `ggml_reshape_4d` — anything inspecting the graph for it must walk the reshape chain first.
3. **Build after each phase.** The sequencing (0→1→2→3→4) is set up so each phase is independently verifiable: a rotated tensor *loads* after Phase 2, a rotated *artifact* after Phase 3, correct *perplexity* after Phase 4.
4. **`-fsyntax-only` on the unity TU is not verification.** Two of the four blocker bugs found in the Phase 0-4 audit (the reshape-op guard and the bufferless H_g matrix) compile perfectly and fail only at runtime, one of them only on GPU backends. Do not record a phase as verified without actually quantizing a model and running it — on each backend the change touches.

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

**Backend dispatch** (route rotated types to the base type's kernel; no kernel duplication). The
implemented approach is **not** "add cases" but "remap the type once at each dispatch point via
`ggml_get_base_type`" — no new pipeline/kernel table entries anywhere.

| Backend | Sites remapped |
|---|---|
| **CPU** | none needed — table-driven via Phase 1's `type_traits_cpu` |
| **Blas** | none needed — table-driven via `type_traits`; `supports_op` gates on `to_float != NULL`, which Phase 1 populates |
| **CUDA** | `mmq.cu`: `ggml_cuda_mul_mat_q` (one `type_x` local used for the q8_1 layout, tile config, `J_max` and `mmq_args`), `ggml_cuda_mul_mat_q_switch_type`, `ggml_cuda_should_use_mmq`; `mmvq.cu`: `mul_mat_vec_q_switch_type`; `convert.cu`: all six `ggml_get_to_*_cuda`; `ggml-cuda.cu`: `supports_op` |
| **Metal** | `ggml-metal-device.cpp`: **all five** pipeline getters — `mul_mm`, `mul_mv`, `mul_mv_ext`, `mul_mm_id`, `mul_mv_id` (the name is built from `ggml_type_name`, so a missed one looks up a kernel that does not exist and aborts); `ggml-metal-ops.cpp`: the small-batch mv-ext if-chain |
| **Vulkan** | `supports_op` (**required** — it switches on the raw `src0_type` for both `MUL_MAT` and `MUL_MAT_ID`; without the remap every rotated GEMM silently falls back to CPU and every other Vulkan remap is dead code), `ggml_vk_get_to_fp16`, `ggml_vk_get_mul_mat_mat_pipeline`, `…_mat_id_pipeline`, `ggml_vk_get_dequantize_mul_mat_vec`, `…_vec_id`, `ggml_vk_guess_matmul_pipeline`, `…_id_pipeline`, `ggml_vk_should_use_mmvq` |
| **Spacemit-CPU** | `ime.cpp`: the two op switches, the repack-type switch, and `nbytes` |
| **RPC** | none needed — type-agnostic; it forwards the type index to the remote backend (which must be the same build) |
| **ftype label** | `src/llama-model-loader.cpp` — rotated types report their base type's ftype |

**Not applicable in this fork:** SYCL and OpenCL are not vendored here (`ggml/src/ggml-sycl/` and
`ggml/src/ggml-opencl/` hold only koboldcpp-specific add-on files, not the backends).

**Rule of thumb for any future backend:** a rotated type must be remapped at *every* point that
switches on, indexes by, or string-formats `src0->type`. Points that only call `ggml_type_size` /
`ggml_blck_size` / `ggml_row_size` / `ggml_is_quantized` need no change, since Phase 1 makes those
already correct.

## Phase 3 — Quantizer (create rotated models)
**`include/llama.h:436-452`** — add `bool hadamard;` to `llama_model_quantize_params` (the `--hadamard` convenience flag: auto-convert base Q4/Q5 types to their rotated variants where divisible).

**`tools/quantize/quantize.cpp`:**
- ~~`QUANT_OPTIONS` — add the 6 rotated types.~~ **DEVIATION:** did NOT add 6 rotated types / new `llama_ftype`s (that would require new ftype entries + mix logic). Direct per-tensor selection of a rotated variant instead uses the **existing** `--tensor-type name=ggml_type` option — a rotated type is a valid `ggml_type` there, e.g. `--tensor-type 'blk.0..*'=q4r_K` (pattern is a regex). `--hadamard` covers all layers.
- Arg loop — add `--hadamard` (sets `params.hadamard`).
- `usage()` — document `--hadamard` (and point at `--tensor-type` for per-tensor rotation).

**`src/llama-quant.cpp`:**
- `#include "llama-hadamard.h"`.
- Helper `llama_hadamard_rot_group(t)` — the rotation group for a type (`ggml_blck_size`, which is
  already correct for either spelling since a rotated type shares its base's block size).
- Helper `llama_hadamard_rotate_imatrix(v, n, g)` — moves the imatrix into the rotated space (see below).
- `llama_model_quantize_default_params` — init `hadamard = false`.
- In `llama_model_quantize_impl`, per tensor:
  - **Target-type resolution (preliminary pass):** if `params.hadamard`, `target_type = ggml_get_rotated_type(target_type)`, skipping `TOKEN_EMBD` (see Phase 4 deviation 4). Counts how many tensors were converted and warns if `--hadamard` selected nothing (e.g. a Q6_K/IQ ftype, which has no rotated variant). A rotated type can also arrive directly via `--tensor-type`.
  - **Resolve quant/store types (outer scope):** `base_type = ggml_get_base_type(new_type)`; `rotate = ggml_is_rotated(new_type) && !ggml_is_rotated(cur_type)`; `rot_group_size = llama_hadamard_rot_group(new_type)`. `quantize` and `gguf_set_tensor_type` both use `new_type` unchanged; only the *quantizer call* uses `base_type`.
  - **Fail loud, do not fall back** (**DEVIATION vs the earlier draft**, which silently fell back to the base type):
    - a rotated type on a `TOKEN_EMBD` tensor → throw;
    - `ne[0] % rot_group_size != 0` → throw (`ggml_quantize_chunk` and `gguf_set_tensor_type` would assert on the same condition anyway, just less legibly);
    - requantizing a rotated source to a **non**-rotated type → throw: dequantizing yields values in the rotated space, and storing them under a base type drops the marker that tells inference to rotate — silent corruption;
    - requantizing between two rotated types with **different** groups → throw. Same-group rotated→rotated *is* allowed and skips the rotation (the data is already in the right space).
  - **Between dequantize and re-quantize**, if `rotate`:
    - Copy the slab into `f32_conv_buf` (the F32+mmap branch is read-only — must not rotate in place); dequant non-F32 into it.
    - `llama_hadamard_inplace` per row, per group of `g = rot_group_size`.
    - **Imatrix — do NOT apply `H` to it.** The imatrix holds `E[x_i²]` per input column, i.e. the diagonal of the activation second-moment matrix, and it is non-negative by construction. Under the rotation the right quantity is `E[(H·x)_i²] = Σ_jk H_ij H_ik E[x_j x_k]`; with the same diagonal approximation the imatrix itself relies on, that collapses to `Σ_j H_ij² E[x_j²] = (1/g)·Σ_j E[x_j²]` — **the mean over the group**, not `H·v`. Applying `H` directly makes roughly half the entries negative (every row of `H` but the first has balanced ± signs) and `quantize_row_q4_K_impl` multiplies its weights by these values, so negative importances produce garbage scales. Implemented as `llama_hadamard_rotate_imatrix` (group mean), applied once per expert. **Known limitation:** a block rotation destroys all intra-group imatrix resolution; only the relative importance *between* groups survives. That is inherent to ConvRot, not an implementation shortcut.
    - **`bytes_per_row`** includes the f32-buffer size when rotating an F32 tensor (the copy buffer), so `nrows_slab` stays within `max_buf_size` (quantizer-only; inference never slabs).
  - **Quantize as the BASE type:** `llama_tensor_quantize_impl(base_type, …)` — so `ggml_quantize_chunk` (its own type switch) uses the base type's quantizer. Output bytes = base-type layout.
  - **Store with `new_type`:** `gguf_set_tensor_type(…, new_type)` — the rotated type index is the marker inference keys off.

## Phase 4 — Inference (apply activation rotation) — DONE (revised after audit)
**DEVIATIONS (implemented):**
1. **No standalone `fwht` op exists** in this codebase — the FWHT compute path is only reachable via the `GGML_HINT_SRC0_IS_HADAMARD` hint on a `mul_mat` node (the same mechanism the KV-cache rotation uses; `llama_mul_mat_hadamard` in `src/llama-impl.h`). Reused that mechanism: each rotated activation gets a `mul_mat(H_g, x)` node with the hint set.
2. **A materialized F32 H_g matrix is required, and it must be a graph INPUT** (plan assumed no matrix at all). On backends without a fast FWHT path for the group size (g=32 is not in CUDA/Metal/Vulkan's {64,128,256,512} fast set) the hinted node falls back to a *regular* mul_mat, which is only correct if src0 actually contains H_g. So H_g is materialized (F32) via `llama_gen_hadamard_matrix`.
   **It is allocated in the graph's own context (`ctx0`) with `ggml_set_input` and filled at `set_input` time** by `llm_graph_input_hadamard` (`src/llama-graph.h` / `.cpp`) — exactly the pattern the KV-cache rotation matrices (`self_k_rot`) already use.
   **Do not put it in a private host-memory `ggml_init(no_alloc=false)` context.** Such a tensor has `data != NULL` but `buffer == NULL`: `ggml_backend_sched` assigns it to the *consumer's* backend (pass 4, `4.cur`), `ggml_gallocr` treats `data != NULL` as already-allocated, and no host→device copy is ever issued. On CUDA that is a NULL deref in `ggml_cuda_mul_mat`'s `bad_padding_clear` (`ggml_backend_buffer_get_usage(src0->buffer)`) and in `ggml_cuda_should_fuse_mul_mat_vec_q`, plus a host pointer handed to a device kernel on the fallback path. This was the shipped bug; see the audit below.
3. **Injection centralized in `build_lora_mm` / `build_lora_mm_id`** (the common GEMM wrappers, `src/llama-graph.cpp`) instead of explicit edits to `build_qkv`/`build_attn`/`build_ffn`: this covers every model that uses the wrappers, **including lm_head and model-specific projections** that the plan's injection points would have missed. Factored into the shared helper `llm_graph_context::rotate_input_if_rotated(w, cur)` (returns `cur` unchanged for non-rotated weights).
   **The original "rotate once, reuse" optimization is preserved**, just moved: `rotate_input_if_rotated` memoizes on `(activation tensor, g)` in `hadamard_rot_cache`, so QKV is 3 GEMMs off 1 rotation and FFN up/gate is 2 GEMMs off 1, exactly as the original plan required. (Without the memo the cost is *not* `O(D·log g)` as an earlier revision of this plan claimed — that only holds on the fast FWHT path; the materialized-matrix fallback is `O(D·g)` per rotation.)
4. **Token embeddings are excluded from rotation** (quantizer, `src/llama-quant.cpp`): inference assumes every GEMM input activation is in the unrotated space. A rotated embedding table would put the first layer's input in the rotated space, which the engine cannot detect — so `--hadamard` skips `TOKEN_EMBD` tensors and an explicit `--tensor-type token_embd.weight=q4r_*` fails loud. (No quality loss: the benefit is in the rotated *weight* space of the first layer's GEMMs, which is still rotated.)
5. **Load-time guard (removed).** An initial version added a guard in `llama_model_loader::create_tensor` throwing when a rotated tensor's input dim is not divisible by the rotation group. It is unreachable: the GGUF reader already rejects `ne[0] % blck_size(type) != 0` for *all* types at file parse (`ggml/src/gguf.cpp`, `gguf_init_from_file`/`_buffer`), and rotated types share the base type's blck. The file-less path (`llama_model_init_from_user`) is covered by the same assert inside `gguf_set_tensor_type`, and by `GGML_ASSERT` in `build_hadamard_rotate`. Deleted as dead code.
6. **GEMMs that bypass the wrappers — covered or fail-loud.** GEMMs that do not go through `build_lora_mm` / `build_lora_mm_id` would consume a rotated weight as `W'·x` instead of `W'·(H_g·x)` — silently wrong. Two classes:
    - **(b) Graph-level GEMMs in `src/llama-graph.cpp` — fixed:** the MLA `v_mla` projection (`build_attn_mha`), the pooled-embedding `dense_2`/`dense_3` heads (`build_dense_out`), and the `cls`/`cls_out` classification heads (`build_pooling`) rotate their input via `rotate_input_if_rotated(w, cur)`. This is always correct for a direct `mul_mat(w, cur)`: GEMM validity forces **`w->ne[0] == cur->ne[0]`**, and the quantizer rotates the weight along **`ne[0]`**, so dim 0 of `cur` is exactly the rotated dim. (An earlier revision of this plan said `w->ne[1] == cur->ne[0]` and "rotated along `w->ne[1]`" — both wrong; the code was always right.) The mean-pooling `mul_mat(transpose(embd), inp_mean)` site needs no fix — its src0 is the token embedding, which the quantizer never rotates.
    - **(a) Model graphs with direct `ggml_mul_mat` calls (deepseek2, glm-dsa, plm, and ~14 other model files, ~126 sites) — fail loud:** rather than hand-fixing every exotic layout, the post-build guard `llm_graph_check_hadamard_rotation(gf)` (called once per graph build from `llama_model::build_graph`) walks the graph and throws if any `mul_mat`/`mul_mat_id` node has a rotated-type src0 whose src1 was not rotated by the *matching* H_g. Consequence: `--hadamard` on such a model refuses to run (loud) instead of producing garbage (silent). Individual model sites can be fixed incrementally by routing them through `rotate_input_if_rotated`.
      **Guard implementation gotcha:** `llama_mul_mat_hadamard` returns `ggml_reshape_4d(...)` — the tensor the GEMM consumes has `op == GGML_OP_RESHAPE`, **not** `GGML_OP_MUL_MAT`. The guard must walk `src[0]` through the reshape chain before testing the op and the hint (`llm_graph_hadamard_rot_group`). A guard that tests `x->op == GGML_OP_MUL_MAT` directly rejects *every* correctly-rotated GEMM and makes the feature unloadable. This was the shipped bug; see the audit below.
    `print_info` logs `n_hadamard_rotated` when present.
7. **LoRA:** only the main-GEMM input is rotated (`cur_rot = rotate_input_if_rotated(w, cur)`); the LoRA branch keeps the unrotated `cur`, because adapters are trained in the unrotated space. Rotating `cur` in place would make the delta compute `B·(A·(H_g·x))` instead of `B·A·x`. (The correct alternative — rotating the adapter, `A' = A·H_g` — is equivalent but needless work.)

### Audit (2026-09-22) — findings against the committed Phase 0-4 code
Four correctness bugs and a set of backend gaps were found and fixed. Recorded here so the same
mistakes are not reintroduced.

**Fixed — blockers:**
1. **The fail-loud guard rejected every correctly-rotated GEMM** (`llm_graph_check_hadamard_rotation`). It tested `x->op == GGML_OP_MUL_MAT`, but the rotation node is wrapped in a reshape (see deviation 6). Verified empirically against `ggml.c`: the node's op is `RESHAPE`. Every rotated model threw at graph build, so it never loaded on any backend — which also means the "verified end-to-end" claim in the previous revision of this plan could not have been true of the committed tree. Fixed by walking the reshape chain, and the guard now also checks that the rotation group *matches* the weight's group.
2. **H_g had no backend buffer** (see deviation 2). Fixed by making it a graph input.
3. **The imatrix rotation was mathematically wrong** — it applied `H·v` to a vector of non-negative second moments, producing negative quantizer weights (see Phase 3). Fixed to the group mean.
4. **Requantizing a rotated model to a non-rotated type silently corrupted it.** Fixed with an explicit throw (see Phase 3).

**Fixed — backend gaps** (all listed in the Phase 2 table above):
- **Vulkan `supports_op` rejected rotated types outright** for both `MUL_MAT` and `MUL_MAT_ID`, so every rotated GEMM fell back to CPU and the weights never reached VRAM — which made the other Vulkan remaps dead code.
- **Metal `get_pipeline_mul_mm_id` / `mul_mv_id` / `mul_mv_ext` used the raw type**, so a MoE model built the kernel name `kernel_mul_mm_id_q4r_K_f32`, which does not exist → abort.
- **CUDA `ggml_cuda_should_use_mmq` used the raw type**, silently disabling MMQ for rotated weights (prompt processing fell back to dequant + cuBLAS). Remapping it exposed further raw-type uses inside `ggml_cuda_mul_mat_q` (`mmq_get_q8_1_ds_layout`, `ggml_cuda_mmq_get_J_max`, `mmq_args`), now all routed through one `type_x` local.

**Other corrections:** `llama-hadamard.h`'s helpers are now `inline` (they were non-inline in a header, working only because all three includers land in the unity TU `src/llama.cpp`); `--hadamard` now reports how many tensors it rotated and warns when it rotated none; rotated types report their base type's ftype in `llama-model-loader.cpp`.

**Status: NOT yet verified end-to-end.** The unity TU and the CUDA/Metal/Vulkan edits compile, and
the guard's reshape-unwrapping was verified empirically against `ggml.c`, but no rotated model has
been quantized and evaluated since these fixes. Phase 5 is the gate.

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

### ⚠ Unresolved design risk — type-index squatting (decide before publishing any rotated GGUF)
Taking enum slots 43-48 claims indices that **upstream llama.cpp will allocate to real new types**.
The analysis above only covers *older* engines. A *newer* upstream build reading a `Q4R_K` tensor
sees whatever type 45 has become by then — and if that type's `blck_size` and `type_size` happen to
match `Q4_K`'s, the file passes every validity check in `gguf_init_from_file` and is **silently
mis-decoded**, which is exactly the failure mode this design set out to avoid. Rotated files are
also not portable back into upstream in any form.

Options, in rough order of preference:
1. **Reserve a high fork-private range** (e.g. start at 1000) so upstream growth can never collide.
   Costs: every `[GGML_TYPE_COUNT]`-sized table becomes 1000+ entries — check the per-backend
   pipeline arrays before committing to this.
2. **Drop the new types entirely** and mark rotation with a GGUF KV list of rotated tensor names
   plus the group. Back-compat becomes *silent* on old engines (they would run the rotated weights
   unrotated), so it would need a companion "required feature" KV that old engines reject.
3. **Keep 43-48** and accept the risk, documenting that rotated files are fork-only and must never
   be distributed. Acceptable only while this stays a local experiment.

Nothing downstream of this decision is expensive to change — it is one constant in `ggml.h` plus
the `ggml_get_rotated_type` / `ggml_get_base_type` pair — but it gets much harder once files exist
in the wild.

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
| FWHT compute path (reached only via the mul_mat hint, no standalone op) | `ggml_fwht` / `ggml_compute_forward_fwht_f32` `ggml/src/ggml-cpu/ops.cpp:11847` (also CUDA `fwht.cu`, Metal `ggml-metal-ops.cpp:2235`, Vulkan `ggml-vulkan.cpp`) |
| `GGML_HINT_SRC0_IS_HADAMARD` | `ggml/include/ggml.h:451` |
| `build_lora_mm` / `build_lora_mm_id` (rotation injection) | `src/llama-graph.cpp` |
| `llm_graph_input_hadamard` / `rotate_input_if_rotated` / `llm_graph_check_hadamard_rotation` | `src/llama-graph.h`, `src/llama-graph.cpp` |
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
