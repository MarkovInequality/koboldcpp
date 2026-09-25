# Plan: GPTQ error feedback for the HQ types (rotated-space quantizer using the imatrix)

Status: not implemented. A prototype exists (see "What exists"). Follows
`plans/full_row_rht_quantization_plan.md` (the HQ types), whose Implementation record has the
measurements this plan starts from.

## Goal

Make the HQ quantizers use `--imatrix` again, through GPTQ/LDLQ-style error feedback in the rotated
space with the Hessian `H = R·diag(v)·Rᵀ`, where `v` is the ordinary (unrotated) imatrix and `R` the
row's RHT. File format, inference and the HQ types stay exactly as they are; only how the codes are
chosen changes.

## Motivation (measured, 2026-09-24)

The HQ types as specified quantize with uniform weights (nearest-neighbour rounding) and ignore the
imatrix, on the argument that after a full-row rotation every input coordinate carries the same
activation energy (full-row plan §8.1). On Qwen3-4B that loses to the ordinary imatrix quants at
every width, and for IQ4 even to plain quantization without an imatrix.

Qwen3-4B, wikitext-2 test, 80 × 512 windows (20 400 scored tokens), mean KL against BF16; single
types `--pure --token-embedding-type q6_K`; imatrix from upstream `llama-imatrix` on wiki.train:

| type | plain | imatrix | HQ (uniform) | HQ + rotated-space imatrix | **HQ + GPTQ, H = R·diag(v)·Rᵀ** |
|---|---|---|---|---|---|
| Q4_K | 0.0966 | 0.0523 | 0.0844 | – | **0.0339** |
| Q5_K | 0.0286 | 0.0170 | 0.0240 | – | **0.0111** |
| IQ4_XS | 0.0814 | 0.0611 | 0.1101 | 0.1015 | **0.0383** |
| IQ4_NL | 0.0812 | 0.0615 | 0.1036 | – | **0.0384** |
| IQ3_S | 0.6182 | 0.1663 | 0.2649 | 0.2590 | not prototyped |

GPTQ with the diagonal imatrix Hessian is **32–38 % below base + imatrix** for every type tried.
With the full activation covariance `H = E[(Rx)(Rx)ᵀ]` HQ4_XS reaches 0.0278, and plain IQ4_XS +
GPTQ reaches 0.0341, so the rotation adds value once the quantizer is covariance-aware.

**Why uniform rounding loses** (investigation, `build-hq/investigate/`):
- It isn't activation quantization: dequantizing to F32 and running without Q8 activations or a
  runtime RHT gives the same KLs (plain IQ4_XS 0.1691 → 0.1689, HQ4_XS 0.2028 → 0.2019 on 0.6B).
- A few input channels carry most of the activation energy: massive activations / attention sinks,
  e.g. in Qwen3-4B `blk.6.ffn_down`, 4 channels at up to ±5572 hold 99.8 % of the input energy. KL is
  dominated by those tensors: swapping just `blk.6` and `blk.16.ffn_down` between the plain and the
  HQ4_XS file moves HQ4_XS from 0.1101 to 0.0841.
- Plain IQ4 happens to be accurate in exactly those weight columns (their weights are near zero
  apart from a few block maxima, which IQ4's grid and x² weighting favour): 0.3–0.56× the average
  column error. The rotation spreads each such channel over every coordinate; uniform rounding gives
  that direction average accuracy (HQ error is isotropic, 1.0× in any basis).
- A rotated-space imatrix can't fix it: the direction `R·e_k` has magnitude `1/√n` in every rotated
  coordinate, so no per-coordinate weight protects it (−8 % KL only). Error feedback with a Hessian
  that contains that direction does.
- KL follows layer-output error (Spearman 0.94 across 17 Qwen3-4B variants) and sink-token output
  error (15/15 pairwise orderings), not weight MSE (7/15). GPTQ *raises* weight MSE (HQ4_XS 0.0057 →
  0.0076 relative) while cutting KL by two thirds.

## Algorithm

Per weight tensor `W` (rows `w_r` of width `n`), with the imatrix `v` for that tensor (per expert for
3D tensors) and the file's seed:

1. **Hessian.** `H = R·diag(v)·Rᵀ` (n × n, symmetric), damped: `H ← H + λI`, λ = 1 % of mean(diag H)
   (= 1 % of mean(v)). Damping commutes with R, so `H⁻¹ = R·(D + λI)⁻¹·Rᵀ` in closed form. H depends
   only on the GEMM input, so q/k/v share one and gate/up share one; compute each once per layer.
2. **Factor.** `U` = upper Cholesky factor of `H⁻¹` (`H⁻¹ = UᵀU`). The prototype gets it as the
   inverse of the lower Cholesky factor of the index-reversed H (potrf + trtri), then reverses back.
   O(n³/3) + O(n³/3) flops per distinct input; U is n² floats (1.2 GB at n = 17408).
3. **Encode each row** in the rotated space, `x = R·w_r`, super-block by super-block (256; 32 for
   IQ4_NL):
   - pick the super-block's scales (and mins) by running the existing HQ quantizer
     (`ggml_quantize_chunk(HQ type, …)` on those 256 current values) — uniform-weight scale search on
     the error-updated values — and decode the resulting per-sub-block grid levels;
   - then for each element j of the block, in order: `q_j` = nearest grid level to `x_j`; write its
     code; `err = (x_j − q_j)/U_jj`; `x_k −= err·U_jk` for every k > j (the rest of the row, not just
     the block).
4. The output bytes are the HQ type's normal layout.

The prototype does no lazy batching (GPTQ's 128-column blocks with a deferred rank-B update), so each
row costs O(n²) and streams the upper triangle of U; rows are processed 16 at a time per thread to
reuse each U row.

**IQ2/IQ3** (vector codebooks: 8-element groups for IQ2, 4-element for IQ3, with sign-parity rules)
need a group-wise variant: quantize a group jointly against the codebook, then propagate the group's
error with the g × g diagonal block of U (BlockLDLQ as in QuIP#). Not prototyped.

## What exists

Prototype (research code, not in the tree): `build-hq/investigate/` (git-ignored).
- `requant3.cpp` (`requant2.cpp` is the same without the `IMATRIX=` option): reads the BF16 source
  and an HQ file as a template, re-encodes every HQ tensor and writes a new GGUF with the same types
  and seed. Modes: `copy` (the current HQ quantizer), `rimat` (base quantizer with a rotated-space
  imatrix), `gptq` (full covariance from captured activations), `gptq-diag` (H = R·diag(E[x²])·Rᵀ),
  `gptq-diagmu` (adds the activation mean `μμᵀ`), `gptq-rimat`. `IMATRIX=<file.gguf>` takes E[x²]
  from an upstream imatrix instead of captured activations — that is the configuration that
  produced the 0.0383 above. Usage:
  `IMATRIX=imatrix.gguf ./requant3 gptq-diag src-bf16.gguf template-hq.gguf act_dir out.gguf [damp=0.01]`
  (`act_dir` is still read for the tensor → input mapping even with `IMATRIX`).
- The element-wise encoder handles Q4_K, Q5_K, IQ4_XS and IQ4_NL through a small `grid` struct that
  decodes one super-block's levels from its bytes (sorted ascending, with the code for each level),
  finds the nearest level by binary search, and writes a code back into the block (`set`).
- `gpufactor.cu`: `gpu_gptq_factor(H, n, damp, U)` (cuSOLVER `Dpotrf` + `Xtrtri` in fp64 on the
  index-reversed, damped H), `gpu_xtx` (H = XᵀX/T with cuBLAS), and `gpu_quad` (tr(D·H·Dᵀ) for
  output-error reporting). `requant2.cpp` also has an OpenMP CPU `gptq_factor` (unblocked, slow).
- `capture.cpp` dumps every weight-GEMM input via `cb_eval` (27 GB for 4B at 20k tokens);
  `oerrh.cpp` / `outerr.cpp` / `split.cpp` / `coldiag.cpp` measure output error, sink-token error and
  per-column error; `mix.cpp` swaps tensors between files; `dequant.cpp` makes the F32 comparison.
- Cost: Qwen3-4B IQ4_XS, `gptq-diag` with the GPU factor, 8 CPU threads: 436 s (ffn_down, n = 9728:
  ~7 s per tensor; attn_q, n = 2560: ~1 s). Full-covariance `gptq`: 1596 s.
- Overfitting: with the diagonal Hessian, 0.6B output error is 0.536 on the calibration activations
  and 0.549 held out (none); the full covariance overfits (0.207 vs 0.409) but still gives the best KL.

The core of the encoder, for reference (from `requant3.cpp`; `xs` holds g rows in the rotated space,
`U` is row-major upper):

```cpp
for (int64_t s = 0; s < n; s += sbs) {
    for (int64_t r = 0; r < g; ++r) {
        uint8_t * blk = q + (r0 + r)*rs + (s/sbs)*bsz;
        ggml_quantize_chunk(hq_type, xs + r*n + s, blk, 0, 1, sbs, nullptr); // scales
        gr[r].decode(blk);                                                  // levels
    }
    for (int64_t e = 0; e < sbs; ++e) {
        const int64_t j  = s + e;
        const float * uj = U + j*n;
        for (int64_t r = 0; r < g; ++r) {
            float * x = xs + r*n;
            const int k = gr[r].nearest(e, x[j]);
            gr[r].set(q + (r0 + r)*rs + (s/sbs)*bsz, e, k);
            err[r] = (x[j] - gr[r].lvl[e/32][k])/uj[j];
        }
        for (int64_t r = 0; r < g; ++r) {
            float * x = xs + r*n;
            for (int64_t k = j + 1; k < n; ++k) x[k] -= err[r]*uj[k];
        }
    }
}
```

## Implementation plan

### Phase 1 — Factorization (`ggml/src/ggml-hadamard.c` or a new `ggml/src/ggml-gptq.c`)
- `ggml_rht_gptq_factor(const float * v, int64_t n, uint64_t seed, float damp, float * U)`: builds
  `H = R·diag(v)·Rᵀ + λI` (columns `R·(√vₖ eₖ)` via `ggml_rht_ref`, then a syrk), and the upper
  Cholesky factor of its inverse. Needs a **blocked** fp64 (or fp32 with the damping) Cholesky and
  triangular inverse with threads: n = 17408 is 1.8·10¹² flops each; the unblocked prototype CPU path
  is far too slow. No LAPACK dependency in koboldcpp, so write it (blocked right-looking, 64–128
  columns, GEMM-based trailing update) or reuse ggml's matmul.
- Exploit the structure where possible: `H⁻¹ = R·(D + λI)⁻¹·Rᵀ` is available in closed form, so
  only the Cholesky itself is O(n³); worth checking whether a cheaper exact factor exists for the
  `(H_K ⊗ H_P)·diag(s)` structure (e.g. per-chunk blocks when D is near-constant).
- Memory: U is n² floats; at n = 17408 that's 1.2 GB (fp32). Keep one per distinct layer input.

### Phase 2 — Encoders (`ggml/src/ggml-quants-hq.c`)
- `size_t quantize_hq_gptq(enum ggml_type type, float * rows, void * dst, int64_t nrows, int64_t n,
  const float * U)`: rows already rotated, modified in place by the error feedback. Element-wise
  types first (HQ4_K, HQ5_K, HQ4_NL, HQ4_XS), porting the prototype's `grid` decode/nearest/set per
  type, and rows in groups to reuse U.
- Lazy batching (GPTQ's column blocks of 128 with a deferred update) to cut U traffic.
- Scale choice: start with the prototype's (HQ scale search on the error-updated super-block before
  encoding it); try re-searching with the block's own error feedback, and GPTQ's act-order, as A/Bs.
- IQ2/IQ3: group-wise BlockLDLQ with the codebook search (and sign parity) per group; the existing
  neighbour search and the stored-scale re-pick can be reused per group.

### Phase 3 — Quantizer tool (`src/llama-quant.cpp`, `tools/quantize/quantize.cpp`)
- HQ tensors use `--imatrix` when given: no more "ignored" warning; without an imatrix they keep the
  current uniform quantizer (and the IQ2 HQ types keep running without one).
- Per layer: group tensors by input (q/k/v, gate/up; per expert for 3D tensors), compute the factor
  once, then encode each tensor's rows with the existing slab/worker structure.
- Requantizing HQ → HQ with an imatrix: the source is already in the rotated space; the Hessian is
  the same `R·diag(v)·Rᵀ` for the file's seed.
- `--lora` merges keep working (the delta is added before encoding).

### Phase 4 — Tests and measurement
- Unit: factor correctness (`UᵀU·H = I` to tolerance, for several n and K, including 17408);
  with `v` constant the Hessian is `c·I`, so GPTQ must reduce to the current uniform HQ encoder
  (bitwise if the scale path is identical); determinism across thread counts; 3D experts.
- Quality: the full-row plan's Qwen3-4B table with a GPTQ column, every HQ type including the mixes
  (`--hadamard Q4_K_M`, `IQ4_XS`, `IQ3_XS`, `IQ2_XXS`), against base + imatrix. Calibration on
  wiki.train only; add a non-Wikipedia evaluation text, since calibration and test are both
  Wikipedia today.
- Quantization time for Qwen3-4B and the 27B, and peak memory.

## Open questions

- Is the diagonal-imatrix Hessian enough, or is a richer calibration worth a new format? The full
  covariance gave a further −29 % KL (0.039 → 0.028) but is n² floats per input (~18 GB fp32 for
  Qwen3-4B) and overfits in-sample; a low-rank-plus-diagonal Hessian (e.g. the top sink directions
  plus `diag(v)`) might get most of it cheaply. `gptq-diagmu` (adding `μμᵀ`) is already prototyped
  but wasn't measured on 4B.
- CPU-only factorization time for the 27B (n = 17408, 64 layers) without a GPU.
- Whether GPTQ without the rotation (plain + GPTQ + imatrix) is worth offering too (IQ4_XS 0.0341
  with the full covariance, vs 0.0278 rotated).
