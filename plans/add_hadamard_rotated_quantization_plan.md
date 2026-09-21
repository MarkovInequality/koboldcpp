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
ggml_type ggml_get_base_type(ggml_type t)  // Q4R_0→Q4_0, …, Q5R_K→Q5_K, else identity
bool      ggml_is_rotated(ggml_type t)     // ggml_get_base_type(t) != t
```
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
- `QUANT_OPTIONS` (`:33-74`) — add the 6 rotated types.
- Arg loop (`:409-485`) — add `--hadamard` (sets `params.hadamard`).
- `usage()` (`:121-180`) — document.

**`src/llama-quant.cpp`:**
- `llama_model_quantize_default_params` (`:1351`) — init `hadamard = false`.
- In `llama_model_quantize_impl` (`:896`), per tensor:
  - Resolve the target type. If `params.hadamard` and the base type is a Q4/Q5 type with `ne[0] % blck_size(base) == 0`, use the rotated variant (Q4_K → Q4R_K). (Or the user selected the rotated type directly.)
  - **Between dequantize (`:1295`) and re-quantize (`:1304`)**, if the target is rotated:
    - Copy the slab into `f32_conv_buf` (the F32+mmap branch at `:1288` is read-only — must not rotate in place).
    - `llama_hadamard_inplace` per row, per group of `g = blck_size(base_type)`.
    - **Imatrix:** rotate the per-column imatrix vector (`:1278`) with the same block-Hadamard (it's a 1-D array of length `ne[0]`), keeping it in the rotated space.
  - **Quantize as the BASE type:** `llama_tensor_quantize_impl(ggml_get_base_type(target), rotated_f32, …)` — so `ggml_quantize_chunk` (`ggml.c:7973`, which has its own type switch) uses the base type's quantizer. Output bytes = base-type layout.
  - **Store with the ROTATED type:** `gguf_set_tensor_type(…, target)` at `:1316`.
- Divisibility: only rotate where `ne[0] % g == 0`; otherwise use the plain base type.

## Phase 4 — Inference (apply activation rotation)
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
