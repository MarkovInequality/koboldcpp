# Plan: Add Hadamard-Rotated Quantization (ConvRot) to the LLM Path

## Goal
Add "rotated" variants of the Q4/Q5 K-quants — `Q4R_K` and `Q5R_K` — that store weights Hadamard-rotated *before* quantization (Strategy A). The inference engine rotates the corresponding input activations so the model output is unchanged. This suppresses weight outliers, improving quantization quality.

## Design decisions
1. **LLM path only** (sdcpp skipped — it has no quantized GEMM).
2. **Pure block-Hadamard** (deterministic Walsh-Hadamard, self-describing — no random signs).
3. **New ggml types** (not a metadata list): the rotation is a property of the type.
4. **`_K` variants only.** The rotation group is the base type's block size, so `_K` gives 256 and the `_0`/`_1` types would give 32. A 32-wide rotation is not worth having: it loses on quality *and* on speed (Phase 5 measures +1.71 perplexity and −28% decode). Only 256-wide groups ship.
5. **Rotation groups must be FWHT-able.** Every backend implements a fast Walsh-Hadamard transform for `{64,128,256,512}`. A group outside that set still works — the hinted node falls back to a materialized-matrix GEMM — but it is slower, so the quantizer and the model loader each warn once.
6. **Strategy A**: weights stay quantized (rotated); the GEMM consumes them as the base type; inference rotates the activation.
7. **Shared Hadamard code**: extract `ggml_gen_hadamard`'s core so the KV-cache and ConvRot paths share it.

## Core math / invariant
For weight `W` (input dim `D`, rotation group `g = blck_size(base_type)`), store `W' = W·H_g` quantized as the base type. `H_g` is the orthonormal Walsh-Hadamard of size `g`, applied block-diagonally over `D/g` groups. At inference, rotate the input activation: `y = W'·(H_g·x) = W·H_g·H_g·x = W·x` (since `H_g·H_g = I`). Output unchanged. `H_g` is symmetric (`H_g = H_gᵀ`), so the same matrix rotates the weight (quantize) and the activation (inference).

## Architecture (end-to-end)
- **Quantize:** user selects a rotated type (or `--hadamard` + base ftype). The quantizer dequantizes to f32, rotates by `H_g`, quantizes **as the base type**, and stores the tensor with the **rotated type**.
- **Store:** GGUF type index = rotated type (e.g. `Q4R_K`); bytes = base-type layout (`Q4_K`).
- **Load:** new engine reads the rotated type (valid, `< GGML_TYPE_COUNT`). Old engine hits `type >= GGML_TYPE_COUNT` → hard error (fail-loud).
- **Infer:** graph builder sees a rotated-type weight → rotates the input activation by `H_g` → the GEMM consumes the weight as the base type.

---

## Implementation notes
Cross-cutting constraints that apply across all phases.

1. **Line numbers drift.** The file:line references here are a snapshot. Re-verify each against the current code — the *symbols* (`type_traits` / `type_traits_cpu` entries, `ggml_quantize_chunk`, the `build_lora_mm` / `build_lora_mm_id` injection points) are far more stable.
2. **Non-obvious constraints, each of which will silently break the feature if missed:**
   - The quantizer must call `ggml_quantize_chunk` with the **base type**, not the rotated type — it has its own type switch with no rotated cases. The rotated type is used only for the stored tensor type (`gguf_set_tensor_type`).
   - Metal's pipeline getters build the kernel name from `ggml_type_name(type)`. Remap via `ggml_get_base_type` in **all five** (`mul_mm`, `mul_mv`, `mul_mv_ext`, `mul_mm_id`, `mul_mv_id`); a missed one aborts on a kernel name that does not exist.
   - Vulkan's `supports_op` switches on the raw `src0_type`. Without a remap there, rotated weights never reach the GPU at all, and every other Vulkan remap is unreachable.
   - The rotation node is a `mul_mat` wrapped in a `ggml_reshape_4d`. Anything inspecting the graph for it must walk the reshape chain first — matching on `op == GGML_OP_MUL_MAT` directly will never match.
   - `H_g` must be a graph **input**, not a host-memory tensor. See Phase 4.
3. **Build after each phase.** The sequencing (0→1→2→3→4) is set up so each phase is independently verifiable: a rotated tensor *loads* after Phase 2, a rotated *artifact* after Phase 3, correct *perplexity* after Phase 4.
4. **`-fsyntax-only` is not verification.** The two hardest defects in this design (a graph guard that matches the wrong op, and an `H_g` with no backend buffer) compile perfectly and fail only at runtime — one of them only on GPU backends. Do not record a phase as verified without quantizing a model and running it, on each backend the change touches.

---

## Phase 0 — Shared Hadamard utilities
New header-only file **`src/llama-hadamard.h`** (functions `inline`, since three TUs include it):
- `llama_gen_hadamard_matrix(float * out, int n)` — fills an `n×n` orthonormal Walsh-Hadamard (Sylvester construction, scaled 1/√n). Extract the core from `ggml_gen_hadamard` (`src/llama-kv-cache.cpp:24`).
- `llama_hadamard_inplace(float * vec, int n)` — in-place FWHT butterfly (O(n log n)), scaled 1/√n. (Algorithm from `ggml_compute_forward_fwht_f32`, `ggml/src/ggml-cpu/ops.cpp:11847`.)
- Refactor `ggml_gen_hadamard` to call `llama_gen_hadamard_matrix` (no behavior change to the KV-cache path).

The butterfly and the materialized matrix must agree exactly — inference picks between them depending on the backend, so a mismatch is a silent wrong-output bug. Phase 5c asserts it.

CMake: header-only, no new source target.

## Phase 1 — New ggml types (`Q4R_K`, `Q5R_K`)
**`ggml/include/ggml.h`** — add the two types at **150**, not at the end of the dense range:
```
GGML_TYPE_Q4R_K = 150,
GGML_TYPE_Q5R_K = 151,
GGML_TYPE_COUNT = 152,
```
**The base of 150 is the whole point** — see "Type-index range" below. Appending at 43/44 would claim
the next indices upstream llama.cpp hands out, so a future upstream build could read a rotated file
as some unrelated type and, if the block layout happened to match, mis-decode it silently.

This leaves 45..149 as a **sparse gap**. Those entries are absent from the `type_traits` designated
initializer, so they have `type_name == NULL` and `blck_size == 0`. That is safe, and deliberately so:
- The GGUF reader rejects `blck_size == 0` per tensor (`ggml/src/gguf.cpp`), so a file naming a gap
  type fails to load rather than dividing by zero in `ggml_row_size`.
- `ggml_type_name` returns `"(unused)"` rather than `NULL` for an unnamed slot, so the handful of
  loops that walk `0..GGML_TYPE_COUNT` and print or compare the name cannot deref NULL.
- The four such loops were checked: two in `ggml-vulkan.cpp` (one sets bools, one calls
  `ggml_vk_matmul_shmem_support`, whose `default:` branch reads no type traits), `parse_ggml_type`
  in `tools/quantize/quantize.cpp`, and `tests/test-hadamard.cpp`. None index type traits unguarded.

Use **distinct `type_name`s** (`"q4r_K"`, `"q5r_K"`) for clear logs; Metal's name-based pipeline lookup is handled by the `ggml_get_base_type` remap in Phase 2.

**Fix the Makefile dependencies first.** 22 object rules list their `.cpp`/`.c` but not
`ggml/include/ggml.h`, even though they include it — among them `gguf.o` (which range-checks tensor
types against `GGML_TYPE_COUNT`) and `ggml-cpu.o` (which *defines* `type_traits_cpu[GGML_TYPE_COUNT]`).
Changing the enum without fixing these leaves those objects stale against the rest of the build: the
loader rejects valid files with "invalid ggml type 150, should be in [0, 49)", and a stale
`type_traits_cpu` is an out-of-bounds read waiting to happen. Add `ggml/include/ggml.h` as a
prerequisite to every rule whose source includes it.

**`ggml/src/ggml.c`** (`type_traits`) — 2 entries, each copying the base type's row verbatim (same `type_size`, `blck_size`, `to_float`, `from_float_ref`): `Q4R_K`←`Q4_K`, `Q5R_K`←`Q5_K`.

**`ggml/src/ggml-cpu/ggml-cpu.c`** (`type_traits_cpu`) — 2 entries, each copying the base row (same `vec_dot`, `vec_dot_type`, `from_float`, `nrows`).

This makes CPU GEMM, dequant, quantize, Blas and GGUF byte-size all work identically — **no kernel duplication**.

## Phase 2 — Type helpers + backend dispatch
New helpers in **`ggml/src/ggml.c`** (declared in `ggml/include/ggml.h`):
```
ggml_type ggml_get_rotated_type(ggml_type t)  // Q4_K→Q4R_K, Q5_K→Q5R_K, else identity
ggml_type ggml_get_base_type(ggml_type t)     // the inverse
bool      ggml_is_rotated(ggml_type t)        // ggml_get_base_type(t) != t
bool      ggml_fwht_supports_group(int64_t n) // n in {64,128,256,512}
```
The two direction functions are driven by **one** `GGML_ROTATED_TYPE_PAIRS` X-macro list, so the pairing cannot drift out of sync. The rotation group for a rotated type is just `ggml_blck_size(t)` — rotated types share their base type's block size, so no separate group table is needed.

**Backend dispatch.** The approach is *not* "add cases" but "remap the type once at each dispatch point via `ggml_get_base_type`" — no new pipeline or kernel table entries anywhere.

| Backend | Sites remapped |
|---|---|
| **CPU** | none needed — table-driven via Phase 1's `type_traits_cpu` |
| **Blas** | none needed — table-driven; `supports_op` gates on `to_float != NULL`, which Phase 1 populates |
| **CUDA** | `mmq.cu`: `ggml_cuda_mul_mat_q` (one `type_x` local covering the q8_1 layout, tile config, `J_max` and `mmq_args`), `ggml_cuda_mul_mat_q_switch_type`, `ggml_cuda_should_use_mmq`; `mmvq.cu`: `mul_mat_vec_q_switch_type`; `convert.cu`: all six `ggml_get_to_*_cuda`; `ggml-cuda.cu`: `supports_op` |
| **Metal** | `ggml-metal-device.cpp`: all five pipeline getters; `ggml-metal-ops.cpp`: the small-batch mv-ext if-chain |
| **Vulkan** | `supports_op`, `ggml_vk_get_to_fp16`, `ggml_vk_get_mul_mat_mat_pipeline`, `…_mat_id_pipeline`, `ggml_vk_get_dequantize_mul_mat_vec`, `…_vec_id`, `ggml_vk_guess_matmul_pipeline`, `…_id_pipeline`, `ggml_vk_should_use_mmvq` |
| **Spacemit-CPU** | `ime.cpp`: the two op switches, the repack-type switch, and `nbytes` |
| **Validation** | `ggml-quants.c`: `ggml_validate_row_data` — reached by `--check-tensors` and by requantizing *from* a rotated model |
| **RPC** | none needed — type-agnostic; it forwards the type index to the remote backend (which must be the same build) |
| **ftype label** | `src/llama-model-loader.cpp` — rotated types report their base type's ftype |

**Not applicable in this fork:** SYCL and OpenCL are not vendored (`ggml/src/ggml-sycl/` and `ggml/src/ggml-opencl/` hold only koboldcpp add-on files, not the backends).

**Rule of thumb for any future backend or helper:** a rotated type needs the remap at *every* point that switches on, indexes by, or string-formats a tensor's type — including validation, not just GEMM dispatch. Points that only call `ggml_type_size` / `ggml_blck_size` / `ggml_row_size` / `ggml_is_quantized` need no change, since Phase 1 makes those already correct.

## Phase 3 — Quantizer (create rotated models)
**`include/llama.h`** — add `bool hadamard;` to `llama_model_quantize_params`.

**`tools/quantize/quantize.cpp`:**
- Arg loop — add `--hadamard` (sets `params.hadamard`).
- `usage()` — document `--hadamard`, and point at `--tensor-type` for per-tensor rotation.
- **No new `llama_ftype` entries.** New ftypes would need their own mix logic; per-tensor selection instead reuses the existing `--tensor-type name=ggml_type` option, since a rotated type is a valid `ggml_type` there — e.g. `--tensor-type 'blk\..*ffn_down'=q4r_K` (the pattern is a regex). `--hadamard` covers all layers.

**`src/llama-quant.cpp`:**
- `#include "llama-hadamard.h"`.
- `llama_hadamard_rotate_rows(data, nrows, n_per_row, g)` — rotate every row of an f32 block in place.
- `llama_hadamard_rotate_imatrix(v, n, g)` — move an imatrix vector into the rotated space (see below).
- `llama_hadamard_check_retype(...)` — all the validity checks in one place, so the main loop stays readable.
- `llama_model_quantize_default_params` — init `hadamard = false`.
- In `llama_model_quantize_impl`, per tensor:
  - **Target-type resolution (preliminary pass):** if `params.hadamard`, `target_type = ggml_get_rotated_type(target_type)`, skipping `TOKEN_EMBD` (see Phase 4). Count the conversions; warn if `--hadamard` selected nothing (e.g. a Q6_K or IQ ftype, which has no rotated variant), and warn once per rotation group that has no fast FWHT. A rotated type can also arrive directly via `--tensor-type`.
  - **Resolve types:** `base_type = ggml_get_base_type(new_type)`; `rotate = ggml_is_rotated(new_type) && !ggml_is_rotated(cur_type)`; `rot_group_size = ggml_blck_size(new_type)`. `quantize` and `gguf_set_tensor_type` both use `new_type` unchanged; only the *quantizer call* uses `base_type`.
  - **Reject what the engine cannot represent** (these are wrong-output conditions, so they throw — unlike the FWHT-group warning above, which is only about speed):
    - a rotated type on a `TOKEN_EMBD` tensor;
    - `ne[0] % rot_group_size != 0` (`ggml_quantize_chunk` and `gguf_set_tensor_type` would assert on the same condition anyway, just less legibly);
    - requantizing a rotated source to a **non**-rotated type — dequantizing yields values in the rotated space, and storing them under a base type drops the marker that tells inference to rotate;
    - requantizing between two rotated types with **different** groups. Same-group rotated→rotated *is* allowed and skips the rotation, since the data is already in the right space.
  - **Between dequantize and re-quantize**, if `rotate`:
    - Fill `f32_conv_buf` from the source (the F32+mmap branch is read-only — must not rotate in place), then `llama_hadamard_rotate_rows`.
    - **Imatrix — do NOT apply `H` to it.** The imatrix holds `E[x_i²]` per input column, i.e. the diagonal of the activation second-moment matrix, and is non-negative by construction. Under the rotation the right quantity is `E[(H·x)_i²] = Σ_jk H_ij H_ik E[x_j x_k]`; with the same diagonal approximation the imatrix itself relies on, that collapses to `Σ_j H_ij² E[x_j²] = (1/g)·Σ_j E[x_j²]` — **the mean over the group**, not `H·v`. Applying `H` directly makes roughly half the entries negative (every row of `H` but the first has balanced ± signs), and `quantize_row_q4_K_impl` multiplies its weights by these values, so negative importances produce garbage scales. Applied once per expert. **Known limitation:** a block rotation destroys all intra-group imatrix resolution; only the relative importance *between* groups survives. That is inherent to ConvRot.
    - **`bytes_per_row`** must include the f32-buffer size when rotating an F32 tensor, so `nrows_slab` stays within `max_buf_size` (quantizer-only; inference never slabs).
  - **Quantize as the BASE type**, so `ggml_quantize_chunk` uses the base type's quantizer and the output bytes are the base-type layout.
  - **Store with the rotated type** — that index is the marker inference keys off.

## Phase 4 — Inference (apply activation rotation)
**There is no standalone `fwht` op in this codebase.** The FWHT compute path is reachable only via the `GGML_HINT_SRC0_IS_HADAMARD` hint on a `mul_mat` node — the same mechanism the KV-cache rotation uses (`llama_mul_mat_hadamard`, `src/llama-impl.h`). Each rotated activation gets a `mul_mat(H_g, x)` node with that hint set.

**1. `H_g` is materialized, and it must be a graph input.**
A backend that declines the fast FWHT falls back to a *regular* `mul_mat`, which is only correct if src0 actually contains `H_g`. So `H_g` is materialized as F32 by `llm_graph_input_hadamard` (`src/llama-graph.h` / `.cpp`): allocated in the graph's own context with `ggml_set_input`, filled at `set_input` time — exactly the pattern the KV-cache rotation matrices (`self_k_rot`) already use.

**Do not put it in a private host-memory `ggml_init(no_alloc=false)` context.** Such a tensor has `data != NULL` but `buffer == NULL`: `ggml_backend_sched` assigns it to the *consumer's* backend (pass 4, `4.cur`), `ggml_gallocr` treats `data != NULL` as already allocated, and no host→device copy is ever issued. On CUDA that is a NULL deref in `ggml_cuda_mul_mat`'s `bad_padding_clear` and in `ggml_cuda_should_fuse_mul_mat_vec_q`, plus a host pointer handed to a device kernel on the fallback path.

**2. Injection lives in `build_lora_mm` / `build_lora_mm_id`** (`src/llama-graph.cpp`), the common GEMM wrappers, rather than in `build_qkv` / `build_attn` / `build_ffn` individually. This covers every model that uses the wrappers, including lm_head and model-specific projections. Factored into `llm_graph_context::rotate_input_if_rotated(w, cur)`, which returns `cur` unchanged for non-rotated weights.

**Rotate once, reuse across weights.** `rotate_input_if_rotated` memoizes on `(activation, g)`, so QKV is 3 GEMMs off one rotation and FFN up/gate is 2 off one. Without the memo the per-token cost roughly triples on the decode path, which is launch-bound (Phase 5).

**3. Token embeddings are never rotated.** Inference assumes every GEMM input activation starts in the unrotated space. A rotated embedding table would put the first layer's input in the rotated space, which the engine cannot detect — so `--hadamard` skips `TOKEN_EMBD`, and an explicit `--tensor-type token_embd.weight=q4r_K` is rejected. No quality is lost: the first layer's GEMM *weights* are still rotated.

**4. LoRA: rotate only the main-GEMM input.** The adapter branch keeps the unrotated `cur`, because adapters are trained in the unrotated space; rotating in place would make the delta compute `B·(A·(H_g·x))` instead of `B·A·x`. (Rotating the adapter instead, `A' = A·H_g`, is equivalent but needless work.)

**5. GEMMs that bypass the wrappers — covered, or fail loud.** A GEMM that does not go through the wrappers would consume a rotated weight as `W'·x` instead of `W'·(H_g·x)` — silently wrong. Two classes:
- **Graph-level GEMMs in `src/llama-graph.cpp` — covered directly:** the MLA `v_mla` projection (`build_attn_mha`), the pooled-embedding `dense_2`/`dense_3` heads (`build_dense_out`), and the `cls`/`cls_out` classification heads (`build_pooling`) call `rotate_input_if_rotated(w, cur)`. This is always correct for a direct `mul_mat(w, cur)`: GEMM validity forces `w->ne[0] == cur->ne[0]`, and the quantizer rotates the weight along `ne[0]`, so dim 0 of `cur` is exactly the rotated dim. The mean-pooling `mul_mat(transpose(embd), inp_mean)` site needs nothing — its src0 is the token embedding, which is never rotated.
- **Model graphs calling `ggml_mul_mat` directly** (deepseek2, glm-dsa, plm and ~14 other model files, ~126 sites) — rather than hand-fixing every exotic layout, the post-build guard `llm_graph_check_hadamard_rotation(gf)` (called once per graph build from `llama_model::build_graph`) walks the graph and throws if any `mul_mat`/`mul_mat_id` node has a rotated-type src0 whose src1 was not rotated by the *matching* `H_g`. So `--hadamard` on such a model refuses to run rather than producing garbage, and individual sites can be fixed incrementally by routing them through `rotate_input_if_rotated`.

  **The guard must walk the reshape chain.** `llama_mul_mat_hadamard` returns a `ggml_reshape_4d`, so the tensor the GEMM consumes has `op == GGML_OP_RESHAPE`, not `GGML_OP_MUL_MAT`. `llm_graph_hadamard_rot_group` follows `src[0]` through the reshapes before testing the op and the hint. A guard that tests for `GGML_OP_MUL_MAT` directly rejects *every* correctly-rotated GEMM and makes the feature unloadable.

**6. No load-time divisibility guard is needed.** The GGUF reader already rejects `ne[0] % blck_size(type) != 0` for *all* types at file parse (`ggml/src/gguf.cpp`), and rotated types share the base type's blck. The file-less path (`llama_model_init_from_user`) is covered by the same assert inside `gguf_set_tensor_type`.

**7. Report the rotated tensor count from `load_tensors`, not `print_info`.** `llama_model::print_info()` runs *before* `load_tensors()` populates `tensors_by_name`, so a counter placed there always sees an empty vector.

## Phase 5 — Tests
Repo test styles: `tools/quantize/tests.sh` (bash integration, `set -eu`, `echo PASS` per step); `tests/test-chat-analysis.cpp` (plain `main()` C++ tool in `tests/`); `tests/test_koboldcpp.py` (Python doctest).

**5a. Quantizer + inference integration — extend `tools/quantize/tests.sh`:** `--hadamard` requant + generate; per-tensor `--tensor-type '...'=q4r_K` + generate; and a negative case asserting `--token-embedding-type q4r_K` is rejected.

**5b. Perplexity — `tests/test-hadamard-ppl.cpp`, `make test-hadamard-ppl`.**
This fork has no perplexity tool: `tools/perplexity/main.cpp` is a two-line stub calling `llama_perplexity`, which is not defined anywhere in the tree, so it cannot be built. The test tool computes the same quantity directly — mean NLL over non-overlapping 512-token windows, scoring only the second half of each window so every scored token has ≥256 tokens of context. Takes `-ngl N`, because a CPU-only run does not exercise the GPU paths (see below).

**5c. Hadamard correctness + round-trip — `tests/test-hadamard.cpp`, `make test-hadamard`.**
ggml-only (does not link llama). Covers: `H` orthonormal / symmetric / self-inverse; butterfly == materialized matrix (the two paths inference picks between); the rotated↔base pairing round-trips with identical `blck_size`/`type_size`/`is_quantized` and an FWHT-able group; rotate → quantize-as-base → dequantize → inverse-rotate; `<dequant(W'), H·x> == <W, x>` in the exact form the GEMM computes it; and the imatrix group-mean identity `Σ_j H_ij² m_j == mean(m)`.

**5d. Python↔C parity — N/A**, no `gguf-py` support for the rotated types.

**5e. Backwards-compat** — the old-engine case needs a build with `GGML_TYPE_COUNT=43`. Not run.

### Results (Qwen3-0.6B, wikitext-2 test split, 35190 scored tokens)

| model | CPU | CUDA (RTX 5090, `-ngl 99`) | vs. unrotated |
|---|---|---|---|
| f16 (reference) | 21.56 | 21.57 | — |
| Q4_K  | 23.48 | 23.43 | |
| **Q4R_K** | **23.37** | **23.30** | **−0.11 / −0.13** |
| Q5_K  | 22.36 | 22.30 | |
| **Q5R_K** | **21.95** | **21.92** | **−0.41 / −0.38** |

CPU and CUDA agree within 0.07 ppl on every row (different kernels, fp16 accumulation), and the rotated-vs-unrotated relationship is identical on both. Q5R_K recovers most of the Q5_K→f16 gap. Rotated and unrotated files are byte-for-byte the same size.

These numbers are also the real proof that the quantizer and the engine agree: a missing or mismatched inference-side rotation does not crash, it produces plausible text with perplexity in the hundreds or thousands.

Buffer placement was checked separately: a rotated model puts exactly as many bytes in the CUDA0 buffer as its unrotated twin (372.65 MiB for Q4_K/Q4R_K), confirming `supports_op` accepts the rotated types and the weights are not silently falling back to host.

### Throughput (CUDA, RTX 5090, 9275-token prompt + 512 decoded, median of 5)

| pair | prompt eval | decode | ms/token added |
|---|---|---|---|
| Q4_K → Q4R_K | −2.1% | −13.1% | +0.69 |
| Q5_K → Q5R_K | −1.5% | −8.5% | +0.45 |

Run-to-run spread was 2–14% on prompt eval and 3–8% on decode, so the prompt-eval figures are **inside the noise floor** — read them as "no measurable cost".

Prompt processing is free because the rotation amortizes over the batch. Decode pays a fixed per-token cost whose size points at kernel-launch overhead rather than arithmetic: with `attn_v` and `ffn_down` left at Q6_K, the distinct rotated activations are ~3 per layer, so ~84 extra nodes per token across 28 layers, which at a few µs per launch lands at 0.4–0.7 ms. That is arithmetic consistent with the measurement, not a measured node count — `LOG_DISABLE_LOGS` strips the scheduler debug output. Since the cost is launch-bound, CUDA graphs would largely absorb it.

### Why 32-wide groups are not offered

Evaluated and rejected. A 32-wide rotation measured **+1.71 perplexity (24.86 → 26.57) and −28.3% decode** against the same model at Q4_0.

A 32-wide Hadamard spreads a block's energy evenly across all 32 slots, so weights that were near zero — and quantized exactly — come back carrying ~σ of magnitude and a full quantization step of error, while absmax barely drops. Q4_K's 256-wide rotation instead spreads outliers across eight sub-blocks that each get their own scale and min, so the scale allocation improves. `tests/test-hadamard.cpp` shows the same sign on synthetic Gaussian weights, where rotation cannot help by construction.

The throughput half compounds it: 32 is outside every backend's fast-FWHT set, so it runs as a materialized-matrix GEMM.

Not measured: whether an imatrix would change the 32-wide picture, and whether the 256-wide gains hold on larger models.

### The FWHT fallback

The hinted node falls back to a regular GEMM against `H_g` whenever a backend declines the fast FWHT. That path is **kept reachable on purpose** — no assert guards the group size in `rotate_input_if_rotated`. The gates warn rather than refuse, so a model with an unusual group still runs correctly, just slower; aborting would trade a correct-but-slower path for a crash.

Every way a backend can decline, and what the design does about it:

| trigger | status |
|---|---|
| group size outside {64,128,256,512} | warns at quantize and at load, then falls back correctly |
| non-contiguous src1 or dst (CUDA, Metal, Vulkan) | **cannot happen** |
| Vulkan op fusion (`num_additional_fused_ops != 0`) | **prevented** — `ggml_vk_can_fuse` refuses to fuse a `mul_mat` carrying the ConvRot hint |
| Vulkan pipeline missing | **real** — see below |

**Non-contiguous input cannot occur.** `llama_mul_mat_hadamard` routes a non-contiguous activation through `ggml_cont_2d` before the mul_mat, so src1 is contiguous either way, and the mul_mat output is freshly allocated and therefore contiguous too. Verified against ggml for plain 2D, 3D (`mul_mat_id` shape), permuted (the `v_mla` case), strided-view and transposed inputs: src1 and dst came back contiguous and same-shape in all five.

**The Vulkan pipeline-missing case is real**, and is the reason `H_g` must stay correct rather than be optimized away: `ggml-vulkan.cpp` disables *every* FWHT pipeline on Intel Windows drivers in `[32.0.101.8509, 32.0.101.8860)` because those drivers crash on the fwht kernels. On such a device the materialized-matrix GEMM is the only correct path there is. (The other branch only creates a pipeline when `subgroup_size <= n`; at n=256 every real device satisfies that.)

**The fallback is verified, not assumed.** Forcing `ggml_cuda_op_fwht` to decline at n=256 and re-running the perplexity harness on GPU gave 23.2907 for Q4R_K against 23.3026 through the FWHT kernel, and 21.9079 vs 21.9160 for Q5R_K — differences of ~0.01 ppl, i.e. floating-point accumulation order. Its throughput cost at g=256 is negligible: −11.6% decode through the fallback vs −13.0% through the FWHT kernel, a gap inside the run-to-run spread, since the rotation is launch-bound either way.

### Why GPU verification is mandatory
`ggml_compute_forward_mul_mat` takes the FWHT branch unconditionally on the hint and never reads src0, so a CPU run exercises neither the materialized `H_g` nor any of the Phase 2 backend dispatch. Only a GPU run with the FWHT path disabled reads `H_g` from device memory — which is precisely what a host-memory `H_g` would break.

## Phase 6 — Docs
- Document the feature, the two types, and the backwards-compat behavior (new types ⇒ fail-loud on old engines).
- Update the quantizer usage / README.

---

## Backwards compatibility
- **New engine + old (non-rotated) file:** safe (rotated types absent; no behavior change).
- **Non-rotated output:** byte-identical (uses `Q4_K`, not `Q4R_K`).
- **Old engine + rotated file:** **hard error** — an engine built before these types has
  `GGML_TYPE_COUNT` well below 150, so the per-tensor `type >= GGML_TYPE_COUNT` check at
  `ggml/src/gguf.cpp:758` refuses the file. Fail-loud, not silent.

### Type-index range
The rotated types are based at **150** specifically to stay clear of upstream llama.cpp's
allocation. Appending them at the end of the dense range (43, 44) would claim the very indices
upstream hands out next, and the failure mode there is the bad one: a *newer* upstream build
reading a rotated tensor would see whatever type 43 had become, and if that type's `blck_size` and
`type_size` happened to match `Q4_K`'s, the file would pass every check in `gguf_init_from_file`
and be **silently mis-decoded**. Basing at 150 makes that collision require upstream to add ~105
new types, at which point the mismatch is loud rather than silent.

The cost is a sparse enum: 45..149 are unused, which grows every `[GGML_TYPE_COUNT]`-sized table.
Measured against the actual arrays in this tree that is roughly **250 KB of additional memory and
~1000 extra small heap allocations at Vulkan device init** — the Vulkan pipeline tables dominate,
since `vk_matmul_pipeline2` allocates two `shared_ptr`s per slot in its default constructor. The
`type_traits` / `type_traits_cpu` statics add ~7 KB each. `vk_device_struct` is heap-allocated
(`make_shared`), so nothing lands on the stack. Negligible for a GPU backend, and the gap-slot
hazards are handled in Phase 1.

Rotated files remain fork-only and are not portable into upstream in any form. If they are ever to
be distributed, the alternative worth revisiting is dropping the new types entirely and marking
rotation with a GGUF KV list of rotated tensor names plus the group — but that needs a companion
"required feature" KV, because an engine that ignored the list would run the rotated weights
unrotated and produce garbage silently.

## Key file index
| Item | Location |
|---|---|
| `ggml_gen_hadamard` (refactor) | `src/llama-kv-cache.cpp:24` |
| FWHT butterfly (reuse) | `ggml/src/ggml-cpu/ops.cpp:11847` |
| `ggml_type` enum / `GGML_TYPE_COUNT` | `ggml/include/ggml.h` |
| `type_traits` (Q4_K/Q5_K) | `ggml/src/ggml.c` |
| `type_traits_cpu` (Q4_K/Q5_K) | `ggml/src/ggml-cpu/ggml-cpu.c` |
| `GGML_ROTATED_TYPE_PAIRS` / `ggml_fwht_supports_group` | `ggml/src/ggml.c` |
| Block size `QK_K` | `ggml/src/ggml-common.h:89` |
| CUDA GEMM dispatch | `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `convert.cu`, `ggml-cuda.cu` |
| CUDA FWHT kernel + supported sizes | `ggml/src/ggml-cuda/fwht.cu` |
| Metal pipeline getters (name gotcha) | `ggml/src/ggml-metal/ggml-metal-device.cpp`, `ggml-metal-ops.cpp` |
| Vulkan dispatch + `ggml_vk_can_use_fwht` + fusion guard | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| Spacemit switches | `ggml/src/ggml-cpu/spacemit/ime.cpp` |
| `ggml_validate_row_data` | `ggml/src/ggml-quants.c` |
| `GGML_HINT_SRC0_IS_HADAMARD` / `llama_mul_mat_hadamard` | `ggml/include/ggml.h` / `src/llama-impl.h` |
| `llm_graph_input_hadamard` / `rotate_input_if_rotated` / `llm_graph_check_hadamard_rotation` | `src/llama-graph.h`, `src/llama-graph.cpp` |
| `llama_model_quantize_impl` / `llama_hadamard_*` helpers | `src/llama-quant.cpp` |
| `llama_model_quantize_params` / defaults | `include/llama.h` / `src/llama-quant.cpp` |
| Quantizer CLI (args / usage) | `tools/quantize/quantize.cpp` |
| GGUF tensor-type range check | `ggml/src/gguf.cpp:758` |
| ftype label switch | `src/llama-model-loader.cpp` |
| Tests | `tests/test-hadamard.cpp`, `tests/test-hadamard-ppl.cpp`, `tools/quantize/tests.sh` |

## Risks & sequencing
- **Riskiest:** Phase 2 (backend dispatch) and Phase 4 (graph plumbing). Mitigated by centralizing the aliasing in `ggml_get_base_type`, by the post-build graph guard, and by the end-to-end perplexity test on each backend.
- **Order:** Phase 0 → 1 → 2 → 3 → 4 → 5 → 6. Types and dispatch first (so a rotated tensor can be loaded), then the quantizer (produce an artifact), then inference, then validate with perplexity.
- **Resolved:** the type-index collision risk is closed by basing the enum at 150 (see "Type-index range"); the remaining cost is a sparse enum, quantified there.
- **Verification status:** CPU and CUDA verified end-to-end, including the fallback path. Vulkan verified only to the extent that rotated and unrotated models produce byte-identical device buffer splits — no numerical run, since this machine has no Vulkan GPU driver (WSL2 without an NVIDIA ICD). Metal untested, no hardware.
- **New-context start:** begin at Phase 0; each phase is independently verifiable (types load after Phase 2; a rotated artifact after Phase 3; correct perplexity after Phase 4).
