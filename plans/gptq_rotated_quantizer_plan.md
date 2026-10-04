# Plan: GPTQ error feedback for the HQ types

Status: implemented (2026-09-26), phases 1–6; see the Implementation record. Builds on
`plans/full_row_rht_quantization_plan.md` (the HQ types) and supersedes its §8.1 ("no imatrix for HQ
types"). A research prototype (`build-hq/investigate/requant3.cpp`, see "The prototype") produced the
measurements quoted before the Implementation record. *(`build-hq/investigate/` was deleted on
2026-09-27. Its measurements survive only as quoted here.)*

## Summary

The HQ types quantize rows rotated by a randomized Hadamard transform `R`, using nearest-level
rounding, and ignore `--imatrix`. As a result they lose to the unrotated imatrix quants. The
rotation hides the few input channels that carry most of the activation energy from any
per-coordinate importance weight.

This plan makes `--imatrix` drive GPTQ error feedback in the rotated space, with the Hessian
`H = R·diag(v)·Rᵀ` built from the ordinary imatrix `v`. On Qwen3-0.6B, HQ4_K + GPTQ reaches KL
0.0549 against 0.0863 for Q4_K + imatrix. The file format and inference don't change.

## Goal

1. **`--imatrix` with an HQ target type quantizes with GPTQ/LDLQ error feedback** in the rotated
   space. It's **on by default** and applies to **all nine HQ types**:
   - element-wise grids: HQ4_K, HQ5_K, HQ4_XS, HQ4_NL
   - codebook grids: HQ3_S, HQ3_XXS, HQ2_S, HQ2_XS, HQ2_XXS
   - *(Added 2026-09-26: the element-wise HQ4_0, HQ4_1, HQ5_0, HQ5_1, HQ8_0, HQ2_K, HQ3_K and
     HQ6_K; see "The new HQ types" in the Implementation record.)*
2. **The file format, inference, the types and `hadamard.seed` stay as they are.** Only the choice
   of codes changes, so GPTQ-made files load in any build that loads HQ files.
3. **The uniform-weight HQ quantizer runs bit for bit as today** in three cases: without
   `--imatrix`, for an HQ tensor with no imatrix entry, and with `LLAMA_HQ_GPTQ=0`.
4. **Benchmark:** Qwen3-0.6B with its existing imatrix, `build-hq/imatrix-06.gguf` (Phase 5).
   Large widths are covered by Phase 1's factorization timing at `n = 17408`. For Qwen3.8-27B
   `--hadamard --imatrix Q4_K_M`, §5 estimates under an hour on an 8-core CPU and about 3 GB of
   extra transient memory. That run is Future work.

Out of scope:
- collecting an imatrix (the fork has no imatrix tool; use upstream `llama-imatrix`)
- richer Hessians: low-rank, mean term, full covariance (Future work)
- GPU factorization (Future work)
- non-HQ types

## Measured motivation

KL divergence against BF16 (lower is better), 80 × 512-token windows with the second half
scored. The imatrix comes from upstream `llama-imatrix` on wikitext-2 train (100 × 512 tokens).
Single types use `--pure --token-embedding-type q6_K`.

**Qwen3-0.6B, HQ4_K (2026-09-25):**

| model (5.04 bpw) | KL wiki.test | top-1 % | KL tech-eval | top-1 % |
|---|---|---|---|---|
| Q4_K | 0.1886 | 78.2 | 0.1375 | 83.6 |
| HQ4_K | 0.1685 | 79.7 | 0.1176 | 85.2 |
| Q4_K + imatrix | 0.0863 | 84.8 | 0.0723 | 88.2 |
| HQ4_K with `--imatrix` today (ignored; tensor data byte-identical to HQ4_K) | 0.1685 | 79.7 | 0.1176 | 85.2 |
| **HQ4_K + GPTQ (prototype)** | **0.0549** | **88.0** | **0.0457** | **90.7** |

`tech-eval` (`tools/hessian/calib/dataset/tech-eval.txt`) is 115 KB of the repo's docs plus 95 KB of C++ (`tools/hessian/calib/techeval.py` rebuilds it).
It lies outside the calibration domain, and the GPTQ gain holds there: 37 % below Q4_K + imatrix,
against 36 % on wiki.test.

**Qwen3-4B, wiki.test (2026-09-24):**

| type | plain | imatrix | HQ (uniform) | HQ + rotated-space imatrix | **HQ + GPTQ** |
|---|---|---|---|---|---|
| Q4_K | 0.0966 | 0.0523 | 0.0844 | – | **0.0339** |
| Q5_K | 0.0286 | 0.0170 | 0.0240 | – | **0.0111** |
| IQ4_XS | 0.0814 | 0.0611 | 0.1101 | 0.1015 | **0.0383** |
| IQ4_NL | 0.0812 | 0.0615 | 0.1036 | – | **0.0384** |
| IQ3_S | 0.6182 | 0.1663 | 0.2649 | 0.2590 | not prototyped |

- GPTQ with the diagonal Hessian is **32–38 % below base + imatrix** for every type tried.
- **Calibration:** HQ4_XS used the upstream imatrix. The other GPTQ rows took `E[x²]` from 20 k
  captured wiki.train activations, which is the same quantity.
- **Richer Hessians go further:** with the full activation covariance, HQ4_XS reaches 0.0278, and
  plain IQ4_XS + GPTQ reaches 0.0341.
- **Outside the calibration domain (tech-eval),** Q4_K rows: imatrix 0.0327, HQ 0.0513,
  HQ + GPTQ 0.0219.

**Why uniform rounding loses** (investigation in `build-hq/investigate/`, deleted 2026-09-27):
- **It's not activation quantization.** Dequantizing to F32 gives the same KLs.
- **A few input channels carry most of the activation energy** (massive activations
  [Sun et al. 2024] / attention sinks). In Qwen3-4B `blk.6.ffn_down`, 4 channels hold 99.8 % of
  the input energy. Plain
  quantization is accurate in exactly those columns.
- **The rotation hides them from any per-coordinate weight.** It spreads each such channel over
  every coordinate, as `R·e_k` with magnitude `1/√n` everywhere. The rotated imatrix
  `diag(R·diag(v)·Rᵀ)` equals `mean(v)` in every coordinate, so a per-coordinate weight has
  nothing to work with.
- **The information sits in the off-diagonal of `R·diag(v)·Rᵀ`,** and only error feedback can use
  it: GPTQ correlates the rounding errors so that they cancel along the loud directions.
- **KL follows layer-output error, not weight MSE.** Across 17 Qwen3-4B variants the Spearman
  correlation is 0.94. GPTQ *raises* weight MSE (HQ4_XS 0.0057 → 0.0076 relative) while cutting
  KL by two thirds.

**Expected on Qwen3.8-27B** (from its Unsloth imatrix, `imatrix_unsloth.gguf`; not measured end to
end):
- **One residual channel, 3994, dominates the attention inputs on almost every token.** In the
  normalized residual it holds 93–98 % of the energy in early layers and about 50–70 % in mid
  layers. The FFN norm scales it down.
- **Estimated gain.** The proxy is the output error of nearest rounding over GPTQ's
  (`Σv / Σ D_jj`, 1 % damping). Its geometric mean over tensors is 1.88×, against 1.92× for
  Qwen3-4B.
- **The gain sits in the attention inputs:** `attn_q` up to 18×, `attn_qkv` 2.8× typical. The
  4B's gain sits in a few sink `ffn_down` tensors instead.

## Design

### 1. Hessian and damping

- **Scope:** per HQ tensor, and per expert for 3D tensors, with an imatrix entry `v` of length
  `n = ne[0]`.
- **Normalization:** `v̄ = v / mean(v)`, so the channels average 1. GPTQ's choices don't depend on
  the scale of `H`, so the result is the same as with raw `v`.
  - The imatrix's own normalization (counts, chunk sizes) can't matter.
  - The damping is an absolute number.
  - Every later quantity has a known bound.
- **Hessian:** `H = R·(diag(v̄) + damp·I)·Rᵀ`, with `damp = 0.01` by default.
  - `R` is the RHT of the file's seed, the same one `ggml_rht_ref` applies to the rows.
  - `R` is orthogonal, so the damping commutes with it.
- **What damping does:** it adds `damp` to every channel's normalized importance. No direction
  counts as less important than 1 % of the average channel.
  - **Why it's needed:** GPTQ pushes rounding error toward the directions the Hessian says are
    cheap. Without a floor, a channel with `v = 0` would look free and could absorb unlimited
    error. Such channels exist: in Qwen3.8-27B layer 7, the FFN norm's scale zeroes channel
    3994, so its imatrix entry is exactly 0. The same holds for any channel the calibration text
    didn't exercise. The floor also bounds the conditioning (§2).
  - **The trade-off:** a larger damp pulls `H` toward `c·I`, and the output toward today's
    uniform HQ output. A smaller damp trusts the imatrix more and shifts error more aggressively,
    but depends more on the calibration text. Phase 5 sweeps damp ∈ {0.001, 0.01, 0.1}.
  - **`LLAMA_HQ_GPTQ_DAMP=<x>`** sets `damp` for experiments. It's read once per run and
    recorded in `quantize.hq.gptq_damp`. A value below 0.001 is refused at startup, because
    0.001 is the smallest damp at which §2's factorization is guaranteed to complete.
- **An all-zero `v`:** the tensor keeps uniform weights and is counted in the "no imatrix entry"
  warning. The quantizer already rejects non-finite values.
- **Column order:** GPTQ's act-order option quantizes "columns in order of decreasing activation
  size" [GPTQ repo], that is, by the diagonal of `H`. The rotated `H` has a flat diagonal,
  `1 + damp`, so there's nothing to sort by. Columns go in natural order, which also keeps
  super-blocks whole.
- **The damping convention is GPTQ's:** λ is 1 % of the average diagonal of `H` [GPTQ, §4
  step 3]. After normalization that average is 1, so `damp = 0.01` is exactly GPTQ's default.

### 2. Factorization: `U = chol(H⁻¹)` from the closed-form inverse

GPTQ needs `U`, the upper Cholesky factor of `H⁻¹` (`H⁻¹ = UᵀU`).

- **Closed-form inverse:** `H⁻¹ = R·diag(g)·Rᵀ` with `gₖ = 1/(v̄ₖ + damp)`, which lies in
  `[1/(n + damp), 1/damp]`.
  - Column `i` of `H⁻¹` is `R(g ⊙ Rᵀeᵢ)`: two fp64 transforms per column, `O(n² log n)` in all,
    parallel over columns.
  - That leaves one `O(n³/3)` step, the Cholesky of `H⁻¹`. Factoring `H` and inverting the
    triangle would take twice as long.
- **Conditioning, and why fp64:** `κ(H) = (max v̄ + damp)/(min v̄ + damp) ≤ 1 + n/damp`, since
  `max v̄ ≤ n`.
  - At the default damp and `n = 17408` that's 1.7·10⁶. The loud directions are exactly the
    smallest eigenvalues of `H⁻¹`, so fp32 errors (u·κ ≈ 0.1) would land on the directions that
    matter most.
  - In fp64, Wilkinson's sufficient condition for a Cholesky to run to completion,
    `20·n^{3/2}·u·κ₂ < 1` with unit roundoff `u = 2⁻⁵³` [Wilkinson 1968, as stated in
    Higham 1990], holds for every `damp ≥ 0.001` up to `n ≈ 30 000`. At the default it's below
    0.01.
  - So the factorization always completes. Phase 1 tests the worst case.
  - Building `H⁻¹` with fp64 transforms perturbs it by about `u·‖H⁻¹‖` per entry. That's far
    below its smallest eigenvalue, `‖H⁻¹‖/κ ≥ ‖H⁻¹‖/(3·10⁷)`, so the computed matrix stays
    positive definite with the same bound.
  - `U` is stored in fp32.
- **Blocked right-looking Cholesky:**
  - panel width 128 (tunable)
  - the trailing symmetric update is split into tiles across the quantizer's worker threads
  - the tile partition is fixed and independent of the thread count, so `U` is bitwise
    reproducible
  - the tile kernel is plain loops written for auto-vectorization, since koboldcpp has no
    LAPACK/BLAS dependency
- **Storage:** packed upper triangle in fp32. Row `j` holds `U[j][j..n−1]` at offset
  `j·n − j(j−1)/2`, which is `n(n+1)/2` floats: 606 MB at `n = 17408`.
- **Memory cap:** building a factor needs `10·n²` bytes: the fp64 matrix plus the packed fp32
  output.
  - The cap is `--max-buffer-size` (default 8 GiB, so `n` up to about 29 000), shared with the
    factor cache.
  - A tensor whose factor can't fit even after emptying the cache keeps uniform weights, with one
    warning that names it.
- **Reuse:** tensors that share a GEMM input have identical imatrix vectors: q/k/v, gate/up, and
  in Qwen3.8 `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta`.
  - Factors go in a 4-entry LRU cache. The key is (`v̄` compared exactly, `n`, seed, damp).
  - **The visit order:** the quantizer visits tensors in the loader's order.
    `weight_name_comparer` (`src/llama-model-loader.h:54`) sorts by layer index, then by name,
    whatever the file's own order. So within a layer the tensors are alphabetical, and shared
    inputs sit within 4 distinct inputs of each other.
  - **Example:** a Qwen3.8 layer (`attn_gate`, `attn_qkv`, `ffn_down`, `ffn_gate`, `ffn_up`,
    `ssm_alpha`, `ssm_beta`, `ssm_out`) has 4 distinct inputs, and every repeat hits.
  - **The exception:** `--keep-split` re-sorts by file offset, where a miss can happen. A miss
    only costs time.

### 3. The encoder: one HQ driver with an optional factor

All HQ encoding goes through one driver. It takes an optional factor: without one it is the
uniform quantizer, and with one it adds error feedback between the same scale search and the same
rounding rules. The two paths can't drift apart.

```c
// rows: nrows rotated rows, modified in place by the error feedback
// U:    packed upper Cholesky factor of H^-1 (row j holds U[j][j..n-1])
GGML_API size_t ggml_quantize_rows_gptq(enum ggml_type type, float * rows, void * dst,
                                        int64_t nrows, int64_t n_per_row, const float * U);
```

- **Two entry points into the driver:**
  - `quantize_hq` (called by `ggml_quantize_chunk`) passes `U = NULL` and never writes the rows.
  - `ggml_quantize_rows_gptq` passes a factor.
  - Both call `ggml_quantize_init(type)` first, for the IQ2/IQ3 grids.
- **The driver walks each row in lazy blocks of 256 columns.** This is GPTQ's lazy batch update
  [GPTQ, §4 step 2]: the paper uses 128 columns, and here the block matches the super-block.
  HQ4_NL's last block may be shorter.
  1. At each scale unit (the 256-value super-block; the 32-value block for HQ4_NL), just before
     its first element, run the type's HQ `_impl` on the unit's current values. That writes the
     unit's scales and codes, using the uniform-weight search.
  2. Without a factor, the codebook types run their `*_repick` pass on the unit, and the unit is
     done.
  3. With a factor, re-pick each element (§3.1) or group (§3.2) in order. Apply its error at once
     to the rest of the block, and keep the block's errors `E` (rows × 256).
  4. With a factor, update all later columns in one pass, `X[:, end:] −= E·U[block, end:]`. The
     pass is tiled over columns, so a tile of `X` stays in cache while the block's 256 rows of
     `U` stream through it once.
- **One call, many rows:** rows are independent. A call processes its `nrows` together, so each
  `U` row is read once per call. The tool passes groups of 32 rows.
- **Deterministic:** the per-row arithmetic doesn't depend on `nrows`, so the output is identical
  for any grouping and thread count.
- **Per-unit calls:** the `_impl`s and `*_repick`s handle their units independently. Calling them
  one unit at a time gives the same bytes as whole-row calls, which Phase 2 checks against hashes
  of the current output.

#### 3.1 Element-wise types (HQ4_K, HQ5_K, HQ4_XS, HQ4_NL)

For each element:
- choose the code with the type's final-assignment rule, the same helper the `_impl` uses:
  `nearest_int((x + dm)/d)` clamped for K-quants, `best_index_int8(16, values, x/d)` for IQ4
- `err = (x_j − x̂_j)/U_jj`
- `x_k −= err·U_jk` for the rest of the block
- store `E[j] = err`

#### 3.2 Codebook types (HQ2_XXS, HQ2_XS, HQ2_S, HQ3_XXS, HQ3_S): BlockLDLQ

Groups are 8 values, the sign unit (for IQ3, two 4-value grid points). Scales come from the
`_impl`: `d` and the 4-bit sub-block scales. For each group:
- **Signs** follow the type's rule applied to the current values: the sign of each value, and for
  the parity types (HQ2_XXS/XS, HQ3_XXS) a flip of the smallest `|x|`. The rule is one helper
  that the `_impl`s and the group step share.
- **Codes** come from the exact branch-and-bound `hq_repick` at the sub-block scale `db`, the
  search `*_repick` uses.
- **Propagate:** `δ = x_g − x̂_g`; solve `U_ggᵀ·e = δ` (an 8×8 triangular solve);
  `x_k −= Σ_{j∈g} e_j·U_jk` for the later columns of the block; store `E[g] = e`.

**This is QuIP#'s g-block LDLQ [QuIP#, §4.1].** QuIP# rounds block `k` as
`Ŵ_k = Q(W_k + (W − Ŵ)_{<k}·A_k)`, with `A` from a g-block LDL of `H` and `Q` an unweighted
vector quantizer. The GPTQ form here (the triangular solve against the diagonal block of `U`) is
the same algorithm. On a random 16 × 16 `H` with `g = 4`, both give identical codes and the same
proxy loss. Choosing each group in plain L2 is therefore QuIP#'s choice too. Weighting by
`diag((U_ggᵀU_gg)⁻¹)` is an A/B test in Phase 5.

#### 3.3 Equivalence with the uniform quantizer

With `U = c·I` there's no feedback, and every code comes from the same helper the `_impl` used.
So the output is **bitwise identical** to the no-factor path. The tests build an exact `c·I`; the
factor of a constant `v` is identity only up to rounding. This catches indexing errors in the
feedback, which would otherwise only show up as worse quality.

### 4. The quantizer tool

- **When GPTQ runs:** all four of these hold:
  - the target type is HQ
  - `--imatrix` has an entry for the tensor
  - the tensor is really quantized (an HQ → same-HQ copy stays a copy)
  - `LLAMA_HQ_GPTQ` isn't `0`
- **`LLAMA_HQ_GPTQ`, default on.** Unset, or any value other than `0`, leaves GPTQ on.
  `LLAMA_HQ_GPTQ=0` turns it off:
  - HQ tensors ignore the imatrix: uniform weights, tensor data byte-identical to a run without
    `--imatrix`, and a warning that the imatrix is ignored for N HQ tensors.
  - Non-HQ tensors in the same mix still use the imatrix.
  - No `quantize.hq.gptq_damp` key is written.
  - **Uses:** A/B comparisons (the same command with and without GPTQ), a fallback if GPTQ is
    too slow or needs too much memory on some machine, and the Phase 3 test that the uniform path
    is untouched.
- **The imatrix lookup** (`llama-quant.cpp:1504`) skips HQ targets today
  (`!ggml_is_rotated(new_type)`). The condition is removed, so HQ targets get the same lookup,
  size check and per-expert slice as every other type.
- **Imatrix requirements:** `tensor_requires_imatrix` stays false for HQ types, so
  `--hadamard IQ2_XXS` runs without an imatrix, with uniform weights.
- **Logging:**
  - the pre-pass (`:1306`) counts GPTQ tensors: "N HQ tensors use the imatrix through GPTQ (damp
    0.01)"
  - HQ tensors without a usable entry get one warning with their count, and use uniform weights
  - each tensor's line says `converting to hq4_K (gptq) ..`, with the factor and encode times
- **Rows:** the existing slab loop. For HQ targets the F32 buffer is always a private copy (from
  rotation or dequantization), so the encoder may modify it. GPTQ tensors go through
  `llama_tensor_quantize_gptq`, which runs `llama_parallel_rows` over groups of 32 rows.
- **3D experts:** per expert `i03`, `v` is that expert's imatrix slice, with one factor per
  expert.
- **Requantization:**
  - **HQ → a different HQ type:** the dequantized rows are already in the `R` space of the
    inherited seed, and `H` uses the same seed.
  - **Unrotated source:** rotate, then GPTQ.
  - **LoRA merges** happen before GPTQ.
- **Metadata:** `quantize.hq.gptq_damp` (f32) when any tensor used GPTQ. It's provenance only; the
  loader ignores it. The imatrix keys are written as usual.

### 5. Cost

**The benchmark model, Qwen3-0.6B:** 196 HQ tensors. Their inputs are 1024 wide (140 tensors),
2048 (28, `attn_output`) and 3072 (28, `ffn_down`).
- A factorization is at most `3072³/3` ≈ 10¹⁰ flop, which is seconds for the whole model.
- The prototype's HQ4_K + GPTQ run took 24.4 s (GPU factor, no lazy batching), against 6.6 s
  for plain HQ4_K.
- The target is at most 3× plain HQ's time on this model.

**Estimates for Qwen3.8-27B:** 64 layers. Each layer has distinct inputs of width 5120
(attention, FFN), 6144 (`ssm_out` / `attn_output`) and 17408 (`ffn_down`).

| step | work | estimate on 8 cores |
|---|---|---|
| factorization | `n³/3` per distinct input: 1.9·10¹² flop per layer, 1.2·10¹⁴ in all (fp64) | 10–20 min at 100–200 GFLOP/s |
| encoding | `nrows·n²/2` multiply-adds per tensor, about 10¹⁴ in all (fp32) | 5–15 min |
| scale searches | the same as uniform HQ | unchanged |
| memory | at `n = 17408`: 2.4 GB fp64 work + 0.6 GB packed `U` + the cache (about 0.2 GB for the layer's other inputs) | about 3.2 GB transient |

For reference, the prototype took 436 s for Qwen3-4B IQ4_XS with a GPU factor and no lazy
batching.

## Implementation conventions

As in `plans/full_row_rht_quantization_plan.md`:
- Keep comments minimal.
- Match the surrounding style.
- Every feature that can fail gets a test in the same phase, and a phase is done only when its
  tests pass.
- The base quantizers in `ggml-quants.c` aren't modified.

---

## Phase 1 — fp64 RHT and factorization

- **fp64 RHT** (`ggml/src/ggml-hadamard.c`, declared next to `ggml_rht_ref` in
  `ggml/include/ggml.h`): `ggml_rht_ref_f64(double * x, int64_t n, uint64_t seed)` and
  `ggml_rht_inv_f64` (`Rᵀ`). They follow the definition of `ggml_rht_ref` exactly, so `R` stays
  defined in one place:
  - signs first, then `H_P`, then the `Ĥ_K` mix, then `1/√n`
  - the same `ggml_rht_plan` and sign words
  - `Ĥ_K`'s entries are ±1, so the existing tables serve both precisions
- **The factor** (`src/llama-quant-gptq.{h,cpp}`):
  `llama_gptq_factor(const float * v, int64_t n, uint64_t seed, float damp, std::vector<float> & U, std::vector<std::thread> & workers, int nthread)`,
  as in §2: normalization, the closed-form `H⁻¹` columns in parallel, the blocked fp64 Cholesky,
  and the packed fp32 output.
- **The LRU cache and the memory cap** (§2).
- **Build:** add the file to `src/llama.cpp`'s `#include` list and to the `llama.o`
  prerequisites in the `Makefile`. CMake compiles the unity file `src/llama.cpp`, so it needs no
  change.
- **Tests:**
  - **`test-hadamard`:**
    - f64 agrees with `ggml_rht_ref` to fp32 rounding (relative 1e-6) for every `K` in the table
      and the widths 1024, 2048, 3072, 2560, 4096, 5120, 6144, 9728, 17408
    - `inv(ref(x)) = x` to 1e-12
    - the norm is preserved
  - **The new `tests/test-hq-gptq.cpp`** (a `Makefile` target that links like
    `test-hadamard-quantize`):
    - **`U·H·Uᵀ = I`,** checked on random vectors `y` by applying `H` with the fast transforms.
      The error `‖U·H·Uᵀ·y − y‖/‖y‖` must stay below `50·2⁻²⁴·√κ`, the fp32 rounding of `U`
      amplified by its conditioning.
      - widths `n` ∈ {256, 1024, 2048, 3072, 5120, 6144, 9728, 17408}
      - `v` constant, one spike ×10⁴, log-normal, and with zeros
    - **The worst case completes:** `v` with all energy in one channel, at `damp = 0.001` and
      `n = 17408`.
    - **Constant `v`:** `U = (1 + damp)^(−1/2)·I` within 1e-6.
    - **Refusals:** a damp below 0.001 is refused. An all-zero `v` is refused without factoring.
    - **Determinism:** 1 and 8 threads give bitwise-identical `U`.
    - **The cache:** hits for an equal `v̄`, eviction at the 5th distinct input, and the
      memory-cap fallback with a small `--max-buffer-size`.
    - **Timing** for `n = 17408` is printed. It fails only past a generous bound, and it's the
      early warning for §5.

## Phase 2 — The HQ driver and the element-wise types (`ggml/src/ggml-quants-hq.c`, `ggml/include/ggml.h`)

- **The driver (§3),** with `quantize_hq` built on it (no factor) and `ggml_quantize_rows_gptq`.
- **The element-wise step (§3.1)** for HQ4_K, HQ5_K, HQ4_XS and HQ4_NL. The final-assignment
  rules are small helpers that the `_impl`s and the step share.
- **Until Phase 4,** the driver rejects a factor for the codebook types (`GGML_ABORT`), and the
  tool (Phase 3) doesn't pass them one.
- **Tests** (`test-hq-gptq`):
  - **`quantize_hq`'s output is unchanged** for all nine types. It's compared against hashes of
    the current output, recorded before the driver goes in, over several widths and seeds.
  - **§3.3 equivalence:** `U = c·I` gives the same bytes as no factor, for the four types.
  - **Error reduction:** on synthetic rows with a spiky `v`, the in-sample output error
    `tr(ΔW·H·ΔWᵀ)` is at least 30 % below uniform HQ for every type. The prototype's
    `blk.6.ffn_down` went from 0.00212 to 0.00001.
  - **Widths:** including HQ4_NL rows that aren't a multiple of 256.
  - **Decoding:** the output decodes with the base type's `to_float`.
  - **Determinism:** identical output for group sizes 1, 7 and 32, and for any thread count.

## Phase 3 — Quantizer tool (`src/llama-quant.cpp`)

- Everything in §4.
- **Until Phase 4,** GPTQ covers the four element-wise types. The codebook types keep uniform
  weights and are counted in the no-entry warning.
- **Tests** (`test-hadamard-quantize`, in place of its "imatrix ignored" checks):
  - **The imatrix is used:** with a synthetic imatrix, the HQ tensors differ from a run without
    one, and there's one info line with the count.
  - **Missing entries:** tensors without an imatrix entry are byte-identical to the no-imatrix
    run, with one warning.
  - **Default on:** with `LLAMA_HQ_GPTQ` unset, GPTQ runs.
  - **Opt-out:** `LLAMA_HQ_GPTQ=0` reproduces the no-imatrix HQ bytes exactly, with the
    imatrix-ignored warning and no damp key.
  - **Damp:** `LLAMA_HQ_GPTQ_DAMP=0.0001` is refused, and `=0.1` is recorded in the key.
  - **Reproducible:** the same seed and imatrix give identical files at 1 and 8 threads.
  - **Copies stay copies:** HQ → same HQ with an imatrix is a byte copy.
  - **Requantization:** HQ5_K → HQ4_XS with an imatrix uses GPTQ in the inherited seed's space.
    Its in-sample output error beats the uniform requantization. A second seed must not appear.
  - **LoRA:** `--lora` merges with an imatrix work.
  - **Metadata:** `quantize.hq.gptq_damp` is written only when GPTQ ran.
  - **3D experts:** a synthetic two-expert 3D tensor with different per-expert `v` gets
    per-expert factors. Its output equals quantizing each expert slice alone.
- **Integration** (`tools/quantize/tests-hq.sh`, with an optional `IMATRIX=<file>`; here
  `IMATRIX=build-hq/imatrix-06.gguf`): for Q4_K_M, Q5_K_M and IQ4_XS with
  `--hadamard --imatrix` on Qwen3-0.6B:
  - generation says "Paris"
  - KL(HQ + GPTQ) < KL(HQ uniform)
  - KL(HQ + GPTQ) ≤ 1.1 × KL(base + imatrix)

## Phase 4 — Codebook types (`ggml/src/ggml-quants-hq.c`)

- **The sign rules:** one helper per rule, shared by `quantize_row_hq2_xxs_impl` / `hq2_xs` /
  `hq2_s` / `hq3_xxs` / `hq3_s` and the group step.
- **The group step (§3.2):** `hq_repick` at the `db` of the `*_repick` functions, the 8×8
  triangular solve and in-block propagation, in the shared driver.
- **Lift the limits:** remove the driver's `GGML_ABORT` and the tool's element-wise-only limit.
- **Tests:**
  - the Phase 2 set for the five types, with a 15 % error-reduction threshold (grid types gain
    less per element)
  - decoded groups are valid grid points with valid sign parity (HQ2_XXS/XS, HQ3_XXS)
  - in `tests-hq.sh`, the IQ3_S and IQ2_XXS mixes join the integration run: generation, and
    KL(HQ + GPTQ) < KL(HQ uniform)

## Phase 5 — Quality and cost measurement on Qwen3-0.6B

**The benchmark setup.** Everything already exists:

| item | path |
|---|---|
| model | `tools/models/Qwen3-0.6B-BF16.gguf` |
| imatrix | `build-hq/imatrix-06.gguf`: upstream `llama-imatrix` on wikitext-2 train, 100 × 512 tokens, 196 entries |
| evaluation, in domain | `tools/models/wiki.test.raw`, reference cache `build-hq/q06-80.kl` |
| evaluation, out of domain | `tools/hessian/calib/dataset/tech-eval.txt`, reference cache `build-hq/q06-tech.kl` |
| driver to extend | `build-hq/q06cmp.sh` (Q4_K only so far) |
| Q4_K baselines | `build-hq/q06/`: `Q4_K-plain`, `Q4_K-im`, `Q4_K-hq`, and the prototype's `Q4_K-hq-gptqim` |

- **Measurement:** `./test-hadamard-ppl-cuda -ngl 99 --chunks 80 --cache <cache> <text> <model-bf16> <models...>`
  reports KL and top-1 against BF16. Single types use `--pure --token-embedding-type q6_K`.
- **Tables:**
  - all nine HQ types as single types, and the four mixes (`Q4_K_M`, `IQ4_XS`, `IQ3_XS`,
    `IQ2_XXS`)
  - columns: plain, imatrix, HQ uniform, HQ + GPTQ
  - both texts
- **Prototype parity:** HQ4_K + GPTQ against the prototype file `q06/Q4_K-hq-gptqim.gguf` (0.0549
  wiki.test, 0.0457 tech-eval), within ±5 %. The two differ only in rounding: summation order,
  and the route to `U` (fp64 closed form here, fp32 `H` plus cuSOLVER in the prototype).
- **Acceptance:** HQ + GPTQ beats base + imatrix on both texts for HQ4_K, HQ5_K, HQ4_XS and
  HQ4_NL. For comparison, Q4_K's margin here is 36–37 %, and the prototype's 32–38 % on the 4B.
- **A/B tests:**
  - damp ∈ {0.001, 0.01, 0.1}
  - the codebook group metric (§3.2)
  - re-searching scales after in-block feedback
- **The HQ2 types** are unusable with uniform weights (KL about 3 on the 4B). See whether GPTQ
  makes them competitive with IQ2 + imatrix.
- **Cost:**
  - wall time and peak RSS of every run, against plain HQ and against the prototype's 24.4 s for
    HQ4_K
  - Phase 1's `n = 17408` factor timing, as the check on the large-model estimate in §5
- **The outcome sets the README recommendations.**

## Phase 6 — Docs

- **`tools/quantize/README.md`, the HQ section:**
  - `--imatrix` applies to HQ types through GPTQ, on by default
  - where to get an imatrix (upstream `llama-imatrix`)
  - time and memory cost, including the `--max-buffer-size` cap on factors
  - the Phase 5 tables and recommendations
  - `LLAMA_HQ_GPTQ=0` and `LLAMA_HQ_GPTQ_DAMP` in a developer note
  - the tests table: `test-hq-gptq`, and the updated `test-hadamard-quantize` description
- **`plans/full_row_rht_quantization_plan.md`:** §8.1 and its Future work entry point to this plan.

## Test list

| area | test | phase |
|---|---|---|
| fp64 RHT = fp32 RHT, inverse, norm | `test-hadamard` | 1 |
| `U·H·Uᵀ = I`, worst case at the damp floor, constant `v`, refusals, threads, cache and memory cap, 17408 timing | `test-hq-gptq` | 1 |
| `quantize_hq` output hashes, equivalence at `U = c·I`, error reduction, decode, grouping and threads (4 element-wise types) | `test-hq-gptq` | 2 |
| routing, default on, opt-out, damp, warnings, copies, requantization seed, LoRA, metadata, 3D experts, reproducibility | `test-hadamard-quantize` | 3 |
| end-to-end generation and KL thresholds with a real imatrix (element-wise mixes) | `tests-hq.sh` (`IMATRIX=build-hq/imatrix-06.gguf`) | 3 |
| the Phase 2 set plus sign parity (5 codebook types); IQ3_S / IQ2_XXS mixes end to end | `test-hq-gptq`, `tests-hq.sh` | 4 |

Everything runs on the CPU, so there is nothing this machine can't test.

## Implementation record

Implemented 2026-09-26, phases 1–6 in order, except that Phase 4 followed Phase 3 at once, so the
tool's element-wise-only limit never shipped. All tests in the Test list pass, including
`tests-hq.sh` with `CUDA=1 IMATRIX=build-hq/imatrix-06.gguf` (`build-hq/int-gptq.log`).

### Deviations and additions

- **`llama_parallel_rows`** moved from `llama-quant.cpp` into `llama-quant-gptq.h`, which the
  factorization, the row-group dispatch and the tool share.
- **AVX2+FMA clones, picked at run time.** `llama.o` isn't built with `-march` (only the C objects
  get `-march=native`, and not in `LLAMA_PORTABLE` builds), so the fp64 Cholesky kernels and the
  encoder's trailing update also get a `target("avx2,fma")` clone, chosen with
  `__builtin_cpu_supports`, on x86 builds whose baseline lacks AVX2. The kernel bodies must be
  force-inlined into the clone: the first version wasn't, and the Cholesky ran at 10 GFLOP/s instead
  of 350. The result depends on which clone runs, so `U` is reproducible per machine, not across
  machines.
- **The IQ4 `_impl` exports its inverse scales.** HQ4_XS/NL choose codes with the unrounded
  super-block scale but decode with the fp16 one, so the element step can't recompute the choice from
  the block. `quantize_row_hq4_nl_impl` now writes the `idl` each 32-block's codes were chosen with.
- **`hq_k_code`** (the K-quant rule) clamps `nearest_int`'s argument to its valid range. In-range
  values are unchanged, but feedback can push a value far outside its unit's range.
- **The codebook `_impl`s export a sign mode per sub-block** (`HQ_SIGNS_RULE`, `_INVERT`, `_KEEP`).
  Bitwise equivalence at `U = c·I` needs them: the searches complement all signs when a sub-block's
  scale comes out negative, and leave a skipped (near-zero) sub-block's signs at 0.
- **HQ3_S and an all-zero 32-value sub-block.** The IQ3_S search (upstream's too) doesn't advance
  its output after skipping an exactly-zero sub-block inside a nonzero super-block, so it misplaces
  that super-block's later codes. The uniform path keeps this bit for bit; the group step writes every
  group in place, so the §3.3 test avoids that input for HQ3_S. A rotated row never has one unless
  the whole row is zero.
- **The pre-pass decides GPTQ per tensor,** including the all-zero-`v` and memory-cap checks,
  because the header (with `quantize.hq.gptq_damp`) is sized before any tensor is written.
  - An imatrix entry of the wrong size aborts the run, as for every type (only with GPTQ on).
  - All-zero expert slices of a 3D tensor use uniform weights and get their own warning count.
  - A factorization that fails, which the damp floor rules out, gets a warning naming the tensor.
- **`quantize.hq.gptq_damp` is kept** when no GPTQ runs but HQ tensors are copied from a source that
  has it, since it still describes them.
- **Memory:** the factor cap is `--max-buffer-size`, separate from the slab buffers, so peak memory
  can reach about twice it. That keeps the pre-pass's fit test independent of tensor order.
- **Tests:** `test-hq-gptq --quick` skips the factor sweep. The `quantize_hq` hashes were recorded on
  x86-64 with FMA before the driver went in; builds that round differently would need new ones.

### Measurements

**Factorization** (Ryzen 7 9800X3D, 8 threads, `test-hq-gptq`): `n = 17408` in 7.7 s (0.9 s for
`H⁻¹`, 5.0 s of Cholesky at 350 GFLOP/s), peak 3.0 GB. `‖U·H·Uᵀ·y − y‖/‖y‖` is at most 2.2·10⁻⁷ for
every width and `v` shape, and 1.4·10⁻⁶ for the worst case (κ = 1.7·10⁷, damp 0.001), 3–4 orders
below the bounds.

**Encoding** (8 threads, synthetic rows): HQ4_K at `n = 17408` runs at 252 GFLOP/s, 1.2 ms per row;
HQ3_S at 2.3 ms per row; HQ4_K at `n = 5120` at 0.19 ms per row, including the scale searches.

**The §5 estimate for Qwen3.8-27B Q4_K_M**, from these rates:
- factorization: about 9 s per layer (one 17408, one 6144 and two 5120 inputs), about 10 minutes in
  all
- encoding: about 17 s per layer, about 18 minutes in all
- extra memory: 3.0 GB transient, plus the cache

**In-sample output error** (`test-hq-gptq`, 64 Gaussian rows, `v` log-normal plus 4 channels at 10³):
GPTQ reaches 0.14–0.22× the uniform error for the element-wise types and 0.20–0.24× for the codebook
types.

**Qwen3-0.6B** (`build-hq/q06gptq.sh`, `build-hq/q06ab.sh`, outputs in `build-hq/q06g/`,
`build-hq/q06ab/`): KL against BF16 as **wiki.test / tech-eval**, 80 × 512 tokens, the imatrix
`build-hq/imatrix-06.gguf`. Single types use `--pure --token-embedding-type q6_K`; that embedding is
26 % of the parameters, so bpw is high. "—": the type needs an imatrix. The last column is HQ + GPTQ
against base + imatrix.

| type | bpw | plain | imatrix | HQ | HQ + GPTQ | vs imatrix |
|---|--:|--:|--:|--:|--:|--:|
| Q5_K | 5.78 | 0.0499 / 0.0364 | 0.0301 / 0.0239 | 0.0471 / 0.0347 | **0.0175 / 0.0141** | −42 % / −41 % |
| Q4_K | 5.04 | 0.1886 / 0.1375 | 0.0863 / 0.0723 | 0.1685 / 0.1176 | **0.0541 / 0.0455** | −37 % / −37 % |
| IQ4_NL | 5.04 | 0.1433 / 0.1048 | 0.0943 / 0.0760 | 0.1772 / 0.1385 | **0.0542 / 0.0471** | −43 % / −38 % |
| IQ4_XS | 4.86 | 0.1486 / 0.1062 | 0.0972 / 0.0778 | 0.1870 / 0.1427 | **0.0579 / 0.0489** | −40 % / −37 % |
| IQ3_S | 4.26 | 2.668 / 1.806 | 0.3299 / 0.2490 | 0.7997 / 0.4844 | **0.1805 / 0.1523** | −45 % / −39 % |
| IQ3_XXS | 3.98 | — | 0.5249 / 0.4327 | 1.541 / 1.005 | **0.3325 / 0.2832** | −37 % / −35 % |
| IQ2_S | 3.61 | — | 1.880 / 1.425 | 3.951 / 3.628 | **0.6471 / 0.5829** | −66 % / −59 % |
| IQ2_XS | 3.43 | — | 2.117 / 2.192 | 7.339 / 6.701 | **1.063 / 0.933** | −50 % / −57 % |
| IQ2_XXS | 3.24 | — | 3.243 / 4.029 | 9.144 / 10.73 | **1.555 / 1.401** | −52 % / −65 % |

(IQ2_S tensors come from the `IQ2_M` ftype; the `IQ2_S` ftype makes IQ2_XS tensors.)

| mix | bpw | plain | imatrix | `--hadamard` | `--hadamard --imatrix` | vs imatrix |
|---|--:|--:|--:|--:|--:|--:|
| Q4_K_M | 5.25 | 0.1105 / 0.0818 | 0.0672 / 0.0572 | 0.1067 / 0.0781 | **0.0440 / 0.0363** | −35 % / −37 % |
| IQ4_XS | 4.86–4.88 | 0.1453 / 0.1053 | 0.0972 / 0.0778 | 0.1505 / 0.1156 | **0.0579 / 0.0489** | −40 % / −37 % |
| IQ3_XS | 4.12 | — | 0.4180 / 0.3240 | 0.9310 / 0.6595 | **0.2301 / 0.1938** | −45 % / −40 % |
| IQ2_XXS | 3.00 | — | 3.367 / 3.973 | 10.96 / 11.32 | **1.394 / 1.344** | −59 % / −66 % |

- **Acceptance:** HQ + GPTQ beats base + imatrix on both texts for all nine types and the four mixes,
  not just the four element-wise types the criterion names.
- **Prototype parity:** HQ4_K + GPTQ gives 0.0541 / 0.0455 against the prototype's 0.0549 / 0.0457,
  within ±5 %.
- **The HQ2 types become usable-ish:** with GPTQ, HQ2_S / HQ2_XS / HQ2_XXS halve or better
  IQ2 + imatrix's KL. At 0.6B their absolute quality is still poor (top-1 44–62 % on wiki.test).

**Damp** (wiki.test / tech-eval): no value wins everywhere, so the default stays 0.01.

| type | 0.001 | 0.01 | 0.1 |
|---|--:|--:|--:|
| HQ4_K | 0.0537 / 0.0460 | 0.0541 / 0.0455 | 0.0569 / 0.0444 |
| HQ4_XS | 0.0551 / 0.0494 | 0.0579 / 0.0489 | 0.0558 / 0.0481 |
| HQ3_S | 0.1828 / 0.1547 | 0.1805 / 0.1523 | 0.1734 / 0.1442 |
| HQ2_XXS | 1.536 / 1.527 | 1.555 / 1.401 | 1.897 / 1.649 |

**A/B tests with code variants** (`build-hq/quantize-ab`, built from the patched copy
`build-hq/investigate/ggml-quants-hq-ab.c` with `link-quantize-ab.sh`; byte-identical to
`quantize_gguf` with the switches `GGML_HQ_AB_METRIC` and `GGML_HQ_AB_RESEARCH` off). *(The patched
source and the link script were deleted with `build-hq/investigate/` on 2026-09-27; only the
`quantize-ab` binary remains.)*
- **Codebook group metric:** picking each group in `diag((U_ggᵀU_gg)⁻¹)` instead of plain L2 changes
  nothing measurable. For all five codebook types, every KL is within 0.1 % (e.g. HQ3_S 0.1804 /
  0.1523, HQ2_XXS 1.554 / 1.400). The rotated `H` has a flat diagonal, so those weights are nearly
  equal. Plain L2 stays.
- **Re-searching scales after in-block feedback** (a first pass records the values each element is
  picked at; the scale search re-runs on them): HQ4_K 0.0560 / 0.0457, HQ4_XS 0.0541 / 0.0493,
  HQ3_S 0.1715 / 0.1506, HQ2_XXS 1.562 / 1.398. That's mixed, and within the ±5 % the damp sweep
  already moves, so it isn't adopted.

**Cost** (quantize wall time on 8 threads; peak RSS 1.06 GB in every run, the same as uniform HQ):

| single type | HQ | HQ + GPTQ |
|---|--:|--:|
| Q4_K | 7.6 s | 11.0 s |
| Q5_K | 7.3 s | 11.0 s |
| IQ4_XS | 17.9 s | 23.1 s |
| IQ4_NL | 17.7 s | 22.7 s |
| IQ3_S | 15.9 s | 26.0 s |
| IQ3_XXS | 22.3 s | 29.4 s |
| IQ2_XS | 28.8 s | 36.6 s |
| IQ2_XXS | 16.2 s | 24.0 s |
| Q4_K_M (mix) | 6.7 s | 8.8 s |

GPTQ costs 1.3–1.6× uniform HQ, well inside the 3× target; the prototype took 24.4 s for HQ4_K.

### The new HQ types (2026-09-26)

HQ variants of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K and Q6_K were added, following
`plans/full_row_rht_quantization_plan.md` ("Addendum"). All eight are element-wise, so GPTQ covers
them with the §3.1 step. Changes to the driver:
- **`hq_get_levels`** describes a unit's decode per scale of 16 or 32 values, as symmetric
  (`d·(code + lmin)`) or affine (`d·code + m`); the IQ4 types keep their table lookup.
  - The step picks each code with the helper the type's `_impl` uses in its final pass:
    `hq_s_code` for Q4_0/Q5_0/Q8_0/Q3_K/Q6_K, `hq_k_code` for Q4_1/Q5_1/Q2_K/Q4_K/Q5_K.
  - Where a scale is 0, it keeps the `_impl`'s code.
- **`hq_get_codes` / `hq_set_codes`** cover all twelve element-wise layouts.
- **The unit is the type's block** (`ggml_blck_size`): 32 values for the legacy types, as for HQ4_NL.
- The §3.3 equivalence holds bitwise for all seventeen types, and the pre-existing types' uniform
  output is unchanged (the recorded hashes).

**Results** (tables in the addendum's Implementation record):
- On Qwen3-0.6B, HQ + GPTQ beats base + imatrix for every new type and mix by 30–68 % KL, on both
  texts.
- GPTQ in-sample output error is 0.13–0.25× uniform.
- Cost is a roughly fixed 3–7 s per 0.6B run, up to 5× the fastest uniform quantizers (HQ3_K
  1.5 → 8.3 s).

### After the upstream merge (2026-10-04)

- Upstream's quantizer now streams row slabs across all experts of a 3D tensor in one pass, to keep the threads
  busy (#27830). It used to loop over experts, then over slabs.
- The merge keeps that structure. Inside a slab, LoRA merging and GPTQ run per expert segment, and each expert is
  still factored once with its own imatrix slice.
- GPTQ's output doesn't depend on how rows are grouped into calls, so the files are unchanged:
  - `golden.sh check` is byte-identical against the pre-merge build;
  - `test-hadamard-quantize`'s 3D-expert checks pass ("each expert equals its slice quantized alone").

## Future work

### Low-rank Hessian from the top-k eigenpairs

**Idea.**
- **What it adds:** the imatrix is only the diagonal of `E[xxᵀ]`, so correlations between input
  channels are lost. Keep the top-k eigenpairs of the unrotated `E[xxᵀ]` as well, and the rest of
  the diagonal:
  ```
  Ĥ = U_k·Λ_k·U_kᵀ + diag(v − diag(U_k·Λ_k·U_kᵀ))
  ```
- **What it keeps:** `diag(Ĥ) = v` exactly, so it keeps everything the diagonal exploits. It
  drops only correlations inside the low-energy tail, and stays positive semidefinite.
- **It survives the rotation:** `R·Ĥ·Rᵀ = (R·U_k)·Λ_k·(R·U_k)ᵀ + R·diag(t)·Rᵀ`. Rotating `k`
  eigenvectors costs `k·n log n`.
- **Factorization:** the Woodbury identity gives the inverse on top of §2's closed form:
  ```
  (R·(T + λI + U_kΛ_kU_kᵀ)·Rᵀ)⁻¹ = R·[G − G·U_k·(Λ_k⁻¹ + U_kᵀ·G·U_k)⁻¹·U_kᵀ·G]·Rᵀ,   G = (T + λI)⁻¹
  ```
  This matches the direct inverse to 2·10⁻¹⁴ in a numerical check (n = 64, k = 5). It costs
  `O(n²k)` extra. The Cholesky and the encoder are unchanged.

**Measured in-sample** (prototype mode `gptq-lr<k>`; Qwen3-4B HQ4_K; `Ĥ` from 20 k tokens of
wiki.train; output error `Σ tr(ΔW·H·ΔWᵀ)/tr(W·H·Wᵀ)` over all tensors, under the true covariance
of the same activations):

| Hessian | output error | vs diag |
|---|---|---|
| uniform (no GPTQ) | 0.992 | |
| diag (the imatrix, this plan) | 0.535 | |
| diag + mean `μμᵀ` (`gptq-diagmu`) | 0.500 | −7 % |
| k = 1 | 0.499 | −7 % |
| k = 8 | 0.466 | −13 % |
| k = 64 | 0.394 | −26 % |

- **Rank 1 is the activation mean direction:** k = 1 matches diag + `μμᵀ`.
- **The gain varies by tensor** (k = 64 vs diag): `attn_output` −41 %, `ffn_down` −16 %.
- **It follows how spread out each input's correlations are.** The share of off-diagonal structure
  left out at k = 64 is: attention input 3 %, FFN input 5 %, attention output 9 %,
  `ffn_down` input 23 %.

**Held-out KL isn't measured yet,** and it decides whether to go ahead.
- In-sample error favours richer Hessians: full covariance on Qwen3-0.6B had output error 0.207
  in-sample against 0.409 held out.
- The five models were in `build-hq/investigate/q4b/Q4_K-hq-gptq-*.gguf`, with the evaluation
  commands in `lowrank4b.sh` (wiki.test and tech-eval). *(All deleted 2026-09-27, along with the
  prototype and the captured activations they were built from. Running this test now means
  rebuilding them.)*
- Go ahead only if the held-out KL beats diag.

**Size:** `n·k` floats per distinct input. At k = 64 that's about 175 MB for Qwen3-4B and 0.55 GB
for Qwen3.8-27B. For comparison, the imatrix is about 3 MB / 14 MB, and the full covariance about
18 GB / 100 GB. Choose `k` per input from its spectrum (energy captured, off-diagonal left) rather
than a fixed number.

**What it needs:**
- **A collector.** The fork has no imatrix tool. Upstream `llama-imatrix` (checked at commit
  `53ed051`) writes only `<name>.in_sum2` and `<name>.counts` per tensor, which is `Σx²` and the
  token counts. Two options:
  - accumulate `XᵀX` per input and eigendecompose offline. That's `n²` floats, 1.2 GB for one
    27B `ffn_down` input, so it runs layer by layer over several passes.
  - a streaming sketch with `O(n·k)` memory: frequent directions [Liberty 2013], or a randomized
    range finder [Halko et al. 2011] over accumulated blocks
- **A file format:** for example, extra tensors `<name>.eig_val` (k) and `<name>.eig_vec`
  (k × n) in the imatrix GGUF, shared by the tensors of one input.
- **Calibration:** eigenvectors need more tokens than a diagonal, and the small ones are noise.
  Truncating to the top k also regularizes against the overfitting that full covariance shows.

### Other items

- **Larger models,** to confirm that the 0.6B results scale and to check the §5 cost estimate:
  - **Qwen3-4B:** `build-hq/imatrix-4b.gguf`, with reference caches for both texts. Parity target:
    the prototype's HQ4_XS 0.0383 on wiki.test.
  - **Qwen3.8-27B:** convert the safetensors in `~/Sandbox/unquantized/Qwen3.8-27B` to a BF16
    GGUF (not the abliterated Q5_K file of the earlier 27B run). Then run
    `--hadamard --imatrix imatrix_unsloth.gguf Q4_K_M` and record wall time, peak RSS, and the
    factor and encode split. Measure KL on 20 chunks against Q4_K_M + the same imatrix.
- **Mean term only:** diag + `μμᵀ`, the k = 1 case above. It needs `Σx` per channel next to
  `Σx²`, a small addition to any collector.
- **GPU factorization** when a CUDA build runs the quantizer: the prototype's cuSOLVER path,
  `gpufactor.cu` (deleted 2026-09-27; its approach is described in "The prototype").
- **Half the factor memory:** factor the fp64 matrix in packed, panel-major storage. That's
  `6·n²` bytes instead of `10·n²`, and it raises the width limit under the default cap to about
  37 000. It's worth doing only if models that wide appear.
- **Re-searching scales** with the block's in-block feedback applied, and GPTQ for the unrotated
  types. Plain IQ4_XS + full-covariance GPTQ measured 0.0341 against 0.0278 rotated, so the
  rotation still adds value.

## The prototype

> **Deleted 2026-09-27:** `build-hq/investigate/` was removed, along with everything this section
> describes: the source, the build script, the drivers, and the captured activations
> (`act4b`, `act06`). The description is kept as a record of what the prototype did.

Research code, not in the tree: `build-hq/investigate/` (git-ignored), built with
`./build.sh requant3 "" gpu`.
- **`requant3.cpp`** reads the BF16 source and an HQ file as a template, re-encodes every HQ tensor
  and writes a new GGUF with the same types and seed. Its modes:

  | mode | encoding |
  |---|---|
  | `copy` | the uniform HQ quantizer |
  | `rimat` | base quantizer with a rotated-space imatrix |
  | `gptq` | full activation covariance |
  | `gptq-diag` | `H = R·diag(E[x²])·Rᵀ` |
  | `gptq-diagmu` | adds `μμᵀ` |
  | `gptq-lr<k>` | top-k eigenpairs plus the rest of the diagonal |
  | `gptq-rimat` | GPTQ with scales from the rotated-space imatrix |

  Options:
  - `IMATRIX=<file.gguf>` takes `E[x²]` from an upstream imatrix; the activation directory is
    still read for the tensor → input mapping.
  - `OERR=1` reports in-sample output error in every mode.

  Usage:
  `IMATRIX=imatrix.gguf ./requant3 gptq-diag src-bf16.gguf template-hq.gguf act_dir out.gguf [damp=0.01]`
- **The encoder** handles Q4_K, Q5_K, IQ4_XS and IQ4_NL through a small `grid` struct. It decodes
  one super-block's levels, finds the nearest by binary search, and writes the code back. It has
  no lazy batching: each row costs `O(n²)` and streams `U`, 16 rows at a time.
- **`gpufactor.cu`:** cuSOLVER `Dpotrf` + `Xtrtri` on the reversed, damped `H` (`gpu_gptq_factor`),
  `gpu_xtx`, `gpu_quad` for output error, and `gpu_topk_eig` (`Ssyevdx`) for the low-rank mode.
- **`capture.cpp`** dumps every weight-GEMM input via `cb_eval`: `act4b` (Qwen3-4B) and `act06`
  (0.6B) on wiki.train.
- **Drivers:** `lowrank4b.sh` (the low-rank sweep) and `../q06cmp.sh` (the 0.6B table;
  `build-hq/q06cmp.sh` still exists).

## Key file index

| Item | Location |
|---|---|
| fp64 RHT (Phase 1) | `ggml/src/ggml-hadamard.c`, declared next to `ggml_rht_ref` in `ggml/include/ggml.h` |
| Factorization, LRU cache, memory cap, row-group dispatch | new `src/llama-quant-gptq.{h,cpp}` (unity build via `src/llama.cpp`) |
| The HQ driver, `quantize_hq`, `ggml_quantize_rows_gptq` | `ggml/src/ggml-quants-hq.c`; public declaration in `ggml/include/ggml.h` |
| HQ `_impl`s and the exact codebook search | `ggml-quants-hq.c`: `quantize_row_hq4_K_impl` `:191`, `hq5_K` `:247`, `hq2_xxs` `:348`, `hq2_xs` `:511`, `hq2_s` `:677`, `hq3_xxs` `:857`, `hq3_s` `:1025`, `hq_repick` `:1292`, `*_repick` `:1324–1436`, `hq4_nl_impl` `:1440`, `quantize_hq` `:1556` |
| `ggml_quantize_chunk`'s HQ case (calls `quantize_hq`, initializes grids) | `ggml/src/ggml.c:8114`, HQ case `:8164–8172` |
| Routing, the pre-pass count, the imatrix lookup, the slab loop | `src/llama-quant.cpp`: pre-pass `:1306–1337`, lookup `:1504`, rotation and quantize call `:1583–1598` |
| Worker helpers | `llama_parallel_rows` `src/llama-quant.cpp:885`, `llama_tensor_quantize_impl` `:753` |
| Tests to change / add | `tests/test-hadamard.cpp`, `tests/test-hadamard-quantize.cpp` (imatrix checks at `:291–324`), new `tests/test-hq-gptq.cpp`, `tools/quantize/tests-hq.sh` |
| Benchmark (Phase 5) | `tools/models/Qwen3-0.6B-BF16.gguf`, `build-hq/imatrix-06.gguf`; texts `tools/models/wiki.test.raw`, `tools/hessian/calib/dataset/tech-eval.txt`; caches `build-hq/q06-80.kl`, `build-hq/q06-tech.kl`; driver `build-hq/q06cmp.sh` |
| Larger-model follow-up (Future work) | 4B: `build-hq/imatrix-4b.gguf`, caches `build-hq/q4b-ref.kl`, `build-hq/tech-ref.kl`; 27B: `~/Sandbox/unquantized/Qwen3.8-27B` (safetensors, `imatrix_unsloth.gguf`) |

## Risks and sequencing

- **Order:** 1 → 2 → 3 → 4 → 5 → 6.
  - After Phase 3, the element-wise types work end to end and can be checked against the
    prototype before the codebook types (Phase 4) go in.
  - Phase 1's timing for `n = 17408` comes first, because the CPU Cholesky decides §5's estimate.
- **Riskiest:**
  - **Building `quantize_hq` on the shared driver** touches code every HQ file goes through. The
    Phase 2 hash test pins the current output for all nine types before the change.
  - **The fp64 transform matching `ggml_rht_ref`.** A mismatch makes `H` the Hessian of a
    different rotation: worse quality, no crash. Phase 1 compares them directly, and Phase 2's
    error-reduction test would catch a residual mismatch.
  - **Feedback indexing** (the packed `U` offsets, the lazy-block boundaries, the tiled trailing
    update, the group solve). The §3.3 bitwise equivalence and the error-reduction tests guard
    it.
  - **CPU Cholesky speed.** If the tile kernel misses about 100 GFLOP/s fp64, try an fp32 panel
    with fp64 accumulation, checked by Phase 1's `U·H·Uᵀ` test. After that, GPU factorization.
- **Wrong-output-without-error hazards, each with a test:**
  - `H` built with a seed other than the rows' (requantization inherits the source seed)
  - the wrong expert's imatrix slice
  - feedback written into columns that are already encoded
  - a code rule that differs between the `_impl` and the feedback step (the §3.3 test)
  - the HQ imatrix lookup left skipped, which would silently leave GPTQ off (the Phase 3
    "imatrix is used" test)

## References

Each citation above was checked against the source on 2026-09-26.

- **[GPTQ]** E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers.* ICLR 2023, arXiv:2210.17323.
  - §4 step 2: lazy batch updates, `B = 128` columns.
  - §4 step 3: the upper Cholesky factor of `H⁻¹`, and damping of 1 % of the average diagonal.
  - Algorithm 1: `E ← (W − Q)/[H⁻¹]_jj`.
- **[GPTQ repo]** `IST-DASLab/gptq` README: `--act-order`, "quantizing columns in order of
  decreasing activation size".
- **[QuIP#]** A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM
  Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024, arXiv:2402.04396.
  §4.1: g-block LDLQ, `Ŵ_k = Q(W_k + (W − Ŵ)_{<k}·A_k)` with a vector quantizer `Q`.
- **[Wilkinson 1968]** J. H. Wilkinson. *A priori error analysis of algebraic processes.* Proc.
  International Congress of Mathematicians, Moscow 1966, Mir Publishers, 1968, pp. 629–640. The
  Cholesky completion condition `20·n^{3/2}·u·κ₂(A) < 1`, as stated and cited in:
- **[Higham 1990]** N. J. Higham. *Analysis of the Cholesky Decomposition of a Semi-definite
  Matrix.* In M. G. Cox and S. J. Hammarling (eds.), *Reliable Numerical Computation*, Oxford
  University Press, 1990, pp. 161–185 (MIMS EPrint 2008.56), proof of Theorem 3.1.
- **[Sun et al. 2024]** M. Sun, X. Chen, J. Z. Kolter, Z. Liu. *Massive Activations in Large
  Language Models.* arXiv:2402.17762, 2024.
- **[Liberty 2013]** E. Liberty. *Simple and Deterministic Matrix Sketching* (frequent
  directions). KDD 2013.
- **[Halko et al. 2011]** N. Halko, P.-G. Martinsson, J. A. Tropp. *Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM
  Review 53(2):217–288, 2011.

**Checked here** (scripts in the session scratchpad; results quoted where used):
- this plan's grouped step gives the same codes as QuIP#'s g-block LDLQ
- the Woodbury form in Future work
- the loader's tensor order (`src/llama-model-loader.h:54`)
- upstream `llama-imatrix`'s output tensors (commit `53ed051`)
