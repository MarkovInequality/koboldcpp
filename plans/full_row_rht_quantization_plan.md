# Plan: Full-Row Randomized Hadamard Rotation (RHT) — the HQ quant types

Status: implemented (Phases 0–8), 2026-09-24. See the
**Implementation record** near the end for deviations, the measured items (§8.3, §8.4, Phase 7)
and the quality results — which did not come out as expected (HQ loses to imatrix quants).

## Goal

1. **Rotated quant types, "HQ".** Store each quantized weight row as `R·w`, where `R` is a
   randomized Hadamard transform over the **full input dimension**, in the QuIP#/QuaRot style.
   Rotation spreads outliers across the whole row, so every quant block sees well-behaved,
   near-Gaussian values. At inference the activation is rotated by the same `R`, so
   `W'·(R·x) = W·x`.
2. **Types:** HQ variants of Q4_K/Q5_K and of the i-quants IQ2–IQ4 (§1). The HQ types take **no
   importance matrix** and quantize with **nearest-neighbour (uniform-weight) rounding** (§8).
3. **One fused op per rotated activation,** `GGML_OP_RHT(x; seed)`. `R` is a pure function of
   `(n, seed)`, defined entirely inside ggml. Its CUDA kernels are written for this fork (§5.2).
   Decode spreads each row across SMs. Prompt processing takes one pass per row. When every
   consumer is a quantized GEMM, the kernel writes the Q8_1 that those GEMMs would otherwise
   compute themselves.
4. **Broad dimension coverage:** every row width `n` that's divisible by 4 and has an odd part of
   at most 63. That covers the hidden and FFN widths of all the common model families checked
   (§11). The non-power-of-2 Hadamard factor comes from Dao's hand-written matrices plus a table
   of verified matrices from N. J. A. Sloane's Hadamard library. No matrices are generated in code.
5. **Backends never see HQ types.** At load, each HQ tensor is created with its base type and
   recorded in a llama-side set of rotated tensors (Phase 2). Every backend then runs the base
   type's kernels unchanged. The graph guard enforces the rotation (5.3).
6. **CPU and the CUDA backend only, for now** (CUDA, plus ROCm and MUSA, which build the same
   code). A model with rotated tensors is refused at load if it would use any other device
   (§5.4). Other backends are Future work.

Out of scope: other base types (Q2_K, Q3_K, Q6_K, Q8_0, IQ1), LDLQ/GPTQ error feedback, and
collecting an imatrix in rotated space (see Future work).

## Current state and what this replaces

The fork currently has **ConvRot**: `Q4R_K`/`Q5R_K` (type indices 150/151), which apply a
block-diagonal 256-wide Hadamard rotation without random signs. Its design is in
`plans/add_hadamard_rotated_quantization_plan.md`. This plan **replaces** it, with no backwards
compatibility (§1.3).

These parts of the ConvRot implementation are kept as they are:
- **Type aliasing:** `GGML_ROTATED_TYPE_PAIRS`, `ggml_get_base_type` /
  `ggml_get_rotated_type` / `ggml_is_rotated`. The quantizer and the loader use them, and backends
  no longer do (Phase 2).
- **Graph wiring:** the injection point in `build_lora_mm` / `build_lora_mm_id`, the graph-level
  GEMM sites that call `rotate_input_if_rotated`, and the post-build guard
  `llm_graph_check_hadamard_rotation`. The guard now asks the rotated set instead of the type, and
  gains rules (5.3).
- **Rules:**
  - token embeddings are never rotated
  - the LoRA branch uses the unrotated input
  - rotated → unrotated requantization is refused
- **Build and quantizer plumbing:** the Makefile header-dependency fix, and the quantizer's slab
  accounting (`bytes_per_row`).

The rotation itself, its graph inputs and its imatrix handling are replaced.

---

## Design

### 1. Types

#### 1.1 Names and indices

| base | HQ type | `type_name` | index |
|---|---|---|---|
| Q4_K / Q5_K | `HQ4_K` / `HQ5_K` | `hq4_K` / `hq5_K` | 150 / 151 |
| IQ2_XXS / IQ2_XS / IQ2_S | `HQ2_XXS` / `HQ2_XS` / `HQ2_S` | `hq2_xxs` / `hq2_xs` / `hq2_s` | 152–154 |
| IQ3_XXS / IQ3_S | `HQ3_XXS` / `HQ3_S` | `hq3_xxs` / `hq3_s` | 155–156 |
| IQ4_NL / IQ4_XS | `HQ4_NL` / `HQ4_XS` | `hq4_nl` / `hq4_xs` | 157–158 |

- "HQ" stands for Hadamard quant. The i-quant variants drop the `I` because they don't use an
  imatrix. "The HQ i-quants" means the seven types with IQ bases.
- `GGML_TYPE_COUNT` becomes 159. `Q4R_K`/`Q5R_K` are renamed to `HQ4_K`/`HQ5_K`. They're used in
  `ggml/include/ggml.h`, `ggml/src/ggml.c`, `tools/quantize/quantize.cpp`, `tools/quantize/tests.sh`,
  `tools/quantize/README.md` and `tests/test-hadamard-ppl.cpp`. Their rows in
  `ggml/src/ggml-cpu/ggml-cpu.c` are deleted, not renamed (Phase 1: no `type_traits_cpu` rows).
- No identifier, macro or string matching `HQ<digit>` / `hq<digit>` exists in the repo. `QR2_XXS`-style
  names are avoided because `ggml/src/ggml-common.h:143–170` already defines them as quant-ratio
  macros.
- **There are no HQ ftypes.** `--hadamard` maps each tensor's chosen base type to its HQ variant
  (Phase 4). `IQ3_XS`, `IQ3_M` and `IQ2_M` are ftypes (per-tensor mixes), not tensor types, so
  there's no `HQ3_XS`. `--hadamard IQ3_XS` maps its IQ3_S / IQ3_XXS / Q4_K tensors to HQ3_S /
  HQ3_XXS / HQ4_K, and leaves types without an HQ variant as they are.

#### 1.2 Shared format and decode, separate encode

- **Decode, shared with the base type:** an HQ type has exactly its base type's block layout.
  Uniform-weight rounding only changes *which* valid codes are chosen, not how they decode. So at
  load, an HQ tensor becomes a tensor of its base type with the same bytes (Phase 2), and every
  backend decodes it with the base type's kernels, with no remaps.
- **HQ type ids exist only in files and in the quantizer.** At inference, llama knows which
  tensors are rotated; ggml and the backends don't.
- **Encode, separate:** HQ types have their own quantizers (Phase 3), which can differ from the
  base type's freely.
- If an HQ type's format ever diverges from its base (block layout, codebook or scale encoding),
  it needs its own decode kernels at that point.

#### 1.3 No backwards compatibility with ConvRot

`HQ4_K`/`HQ5_K` reuse indices 150/151. A ConvRot file has no `hadamard.*` metadata, so the loader's
missing-key check (§7) rejects it with: "rotated tensor types but no `hadamard.seed` —
probably a ConvRot file from an older build of this fork; requantize from the original model". A
new file loaded by an old ConvRot build would decode wrongly without an error. That's accepted,
because these files aren't distributed (the README forbids it).

### 2. The rotation

For a row width `n`, choose `n = K · P` (§4), where `P` is a power of 2 and `Ĥ_K` is a Hadamard
matrix of order `K` (`K = 1`, or a table order such as 12, 20, 28, 68, 108). Then

`R = (1/√n) · (Ĥ_K ⊗ H_P) · diag(s)`,   with `Ĥ_K`, `H_P` unnormalized ±1 matrices,

applied in **natural element order**: element `i = c·P + j`, where `c ∈ [0,K)` is the outer,
strided index and `j ∈ [0,P)` the inner, contiguous one. Per row:

1. **Signs:** `u = s ⊙ x`, with `s ∈ {±1}ⁿ` (§6).
2. **Power-of-2 transform:** apply the FWHT `H_P` to each contiguous length-`P` chunk (`K` chunks).
3. **Order-K mix:** for each inner position `j`, mix the `K` values `u[c·P + j]` by `Ĥ_K`:
   `out[c'·P + j] = Σ_c Ĥ_K[c', c] · u[c·P + j]`.
4. **Scale** by `1/√n`.

Steps 2 and 3 commute (they act on different Kronecker factors). This is the layout of QuaRot's
`matmul_hadU` (`x.view(-1, K, n/K)`, FWHT over the last axis, `had_K` over the middle one) and of
Dao's `fast_hadamard_transform_{12,20,28}N` kernels. Output layout equals input layout, so there's
no permutation or transpose.

- **Weights:** store `W' = W·Rᵀ`. Row `r` of `W'` is `R·w_r`, so the quantizer applies the same
  transform to each weight row.
- **Inference:** `y = W'·(R·x) = W·Rᵀ·R·x = W·x`.
- **Only orthogonality is needed, not symmetry.** Most of these matrices aren't symmetric, and
  `diag(s)` doesn't commute with `H`. Applying the same `R` to weight rows and activations is
  enough. Tests check `⟨Rw, Rx⟩ = ⟨w, x⟩`, not self-inverse identities.
- **Cost:** `n·(K + log₂P)` adds per row. For `n = 17408`, `K = 68` that's 1.3 M. What that
  costs depends on the regime, not the add count:
  - **Prompt processing is limited by memory bandwidth.** A row is read and written once. On the
    RTX 5090 there's room for about 150 instructions per element before the adds become the
    limit (F32 in, Q8_1 out), which covers every order up to about K = 100 when the mix is
    unrolled.
  - **Decode is limited by latency.** One row that goes to one thread block runs every add on
    one SM, while the GEMV it feeds uses all of them. Measured on the RTX 5090 at
    `n = 17408`, `K = 68`: a K-generic one-block mix alone takes 18.5 µs and an unrolled one
    takes 5.8 µs. The full transform, spread over SMs in two passes (§5.2), takes 2.6 µs. The
    `ffn_down` GEMV it feeds is about 33 µs at Q4_K. So the thread layout, not the add count,
    decides the overhead.

### 3. Layering: ggml defines `R`, llama decides where to apply it

Everything that defines `R` lives in **ggml**
(`ggml/src/ggml-hadamard.c`, `ggml/src/ggml-hadamard-tables.h`):
- the decomposition rule and `K_MAX`
- the sign PRNG
- the matrix table
- the reference transform

The kernels must live in ggml, and ggml can't call llama, so this placement lets every
implementation compute `R` itself from `(n, seed)`. Nothing is passed down as data: there are no
sign or matrix tensors in the graph.

**llama** owns policy:
- which tensors are rotated: chosen at quantize time, and at load recorded in the rotated set
  while the tensors are created with base types
- which devices a rotated model may use (§5.4)
- choosing and storing the seed
- GGUF metadata and loader checks
- requantize rules
- graph wiring (the rotate-once memo, the LoRA rule, the guard)

### 4. The decomposition rule: `ggml_rht_plan(n, &K, &P)`

Write `n = m · 2^t` with `m` odd. A Hadamard matrix has order 1, 2 or a multiple of 4, so the
smallest usable `K` for `m > 1` is `4m`. The table (§5) has a matrix for every such order up to
256, so the rule is closed-form:

```
K_MAX = 256
m = odd part of n;  t = number of trailing zero bits of n
if m == 1:                 return (1, n)       # pure power of 2 (Sylvester, no table)
if t < 2 or 4*m > K_MAX:   return NONE          # unsupported (§10)
K = 4*m;  P = n / K                             # Ĥ_K = the table entry for K
return (K, P)
```

- **Coverage:** every `n` divisible by 4 whose odd part is at most 63.
- **The smallest `K` is always chosen**, because the order-K mix costs `n·K`. For example, 24·2ʲ
  resolves to 12·2ʲ⁺¹. Powers of 2 are handled entirely by `H_P`.
- Examples:
  - n = 5120 = 5·2¹⁰ → (20, 256)
  - n = 14336 = 7·2¹¹ → (28, 512)
  - n = 17408 = 17·2¹⁰ → (68, 256)
  - n = 11008 = 43·2⁸ → (172, 64)
  - n = 64·65 → NONE (4·65 = 260 > K_MAX)
  - n = 6 → NONE (not divisible by 4)
- **Kernel limits aren't part of the rule.** The CUDA kernels run every width the rule accepts
  (§5.2), and rotated models only load on the CPU and CUDA-backend devices (§5.4). No width falls
  back to the CPU node by node.

### 5. Hadamard matrices: one hand-maintained header

Every order is known when the code is built, so every matrix is hardcoded in one file,
**`ggml/src/ggml-hadamard-tables.h`**. It's the only copy of matrix data in the repo.

- **Format:** each matrix is stored as its `+`/`-` row strings, copied verbatim from its source,
  one string literal per row. There's no generator and no conversion step, and each block can be
  diffed against its source file.
  - **One flat array per matrix,** `rows[r·K + c]`: the row literals sit next to each other, so
    they concatenate into one `K·K + 1` array.
  - Not a 2D array of rows: constant evaluation can't index across rows through a pointer to the
    first element. nvcc rejects it ("cannot access position 138 in array of 69 elements").
- **Comments:** one line per matrix, giving the construction and the author (for example
  `// Williamson — N. J. A. Sloane`, `// Paley II, q = 5 — Tri Dao`, or the Sloane file suffix
  when the construction isn't documented). One file-level comment credits Sloane's library, with
  its URL, and carries Dao's BSD-3 notice.
- **Index:** one array of `{K, rows}`, with each order appearing once.
- **The file compiles as C and as CUDA C++, included once per storage class.** Each matrix is
  declared as `GGML_HAD_DECL(K) = "…" "…" …;`, and an `GGML_HAD_ORDERS(X)` list at the end
  builds the index. The includer defines both macros, and the CUDA code includes the file twice:
  - **C** (`ggml-hadamard.c`): `static const char had_K[]` plus the `{K, rows}` index.
  - **CUDA, device copy:** `static __device__ const char d_had_K[]`. The unrolled mixes read it in
    ordinary `#pragma unroll` loops, and the optimizer folds the loads.
  - **CUDA, host `constexpr` copy:** used only by `constexpr` packing functions that build the
    K-generic path's `__device__ const` bit tables at compile time.
- **Checked on the RTX 5090 with the fork's nvcc flags** (orders 12, 68 and 172). Unrolled and
  K-generic results match the CPU. Each unrolled kernel's SASS has exactly `K` global loads (its
  inputs) and only FADDs, with no table loads left. The compiler even shares common prefix sums
  between rows. The whole test compiled in 3.4 s using 258 MB.
- **Don't take the signs through per-`(r, c)` templates** (variable templates or
  `index_sequence` folds). The same K = 172 mix took 39 s and 21 GB of nvcc memory that way.
- **CPU access:** `bool ggml_hadamard_matrix(int K, float * out)` returns an **unnormalized,
  row-major ±1** `K×K` matrix (row `i` = `H` row `i`; orientation matters). `ggml_rht_ref` caches
  the unpacked matrix per `K` once, thread-safely.
- **CUDA access:** unrolled paths read the device copy with constant indices, which fold away.
  Paths that load signs at run time use a bit-packed `__device__ const` copy, built from the
  same strings at compile time, in global memory. It can't go in `__constant__`, because the
  whole table is about 87 KB, over the 64 KB limit. Each block copies its one matrix (at most
  8 KB) into shared memory.
- **Size:** about 700 KB of source text. On the CPU side the strings are carried as they are; on
  the CUDA side there's the 87 KB bit table plus whatever the unrolled paths fold into
  instructions.

**Contents.** N. J. A. Sloane's Hadamard library (http://neilsloane.com/hadamard/) has at least
one matrix for every multiple of 4 up to 256. All 658 files were downloaded and verified
(`H·Hᵀ = K·I`) on 2026-09-23. Dao's five hand-written matrices (the strings in `csrc/code_gen.py`
of Dao-AILab/fast-hadamard-transform) were compared with the library: four are identical to Sloane
files, and `had_12_paley` (Paley II, q = 5) is his own.

| K | entry | source |
|---|---|---|
| 12 | `had_12_paley` | Dao's own construction |
| 20 | `had.20.will.txt` | Sloane, identical to Dao's `had_20_will` |
| 28 | `had.28.will.txt` | Sloane, identical to Dao's `had_28_will` |
| 40 | `had.40.tpal.txt` | Sloane, identical to Dao's `had_40_tpal`. **Data only:** the rule never selects it (40 = 8·5), so it has no kernel instance, and its only test is `H·Hᵀ = K·I`. |
| 36–252 (4·odd) | the next table | Sloane: for each order, the first file listed on the library page |

| K | file | K | file | K | file | K | file |
|---|---|---|---|---|---|---|---|
| 36 | `had.36.pal2.txt` | 100 | `had.100.will.txt` | 164 | `had.164.pal.txt` | 228 | `had.228.pal.txt` |
| 44 | `had.44.pal.txt` | 108 | `had.108.pal.txt` | 172 | `had.172.will.txt` | 236 | `had.236.od.txt` |
| 52 | `had.52.will.txt` | 116 | `had.116.will.txt` | 180 | `had.180.pal.txt` | 244 | `had.244.will.txt` |
| 60 | `had.60.pal.txt` | 124 | `had.124.pal2.txt` | 188 | `had.188.tur.txt` | 252 | `had.252.pal.txt` |
| 68 | `had.68.pal.txt` | 132 | `had.132.pal.txt` | 196 | `had.196.pal2.txt` | | |
| 76 | `had.76.pal2.txt` | 140 | `had.140.pal.txt` | 204 | `had.204.pal2.txt` | | |
| 84 | `had.84.pal.txt` | 148 | `had.148.pal2.txt` | 212 | `had.212.pal.txt` | | |
| 92 | `had.92.will.txt` | 156 | `had.156.will.txt` | 220 | `had.220.pal2.txt` | | |

**Writing the header:** a one-time job, by hand or with a throwaway script that isn't committed.
A one-off check confirms each block is byte-identical to its downloaded file or Dao string; the
files' SHA-256s go in the commit message. After that, the unit tests (Phase 6) check `H·Hᵀ = K·I`
for every entry.

**Why a table is safe:** any valid Hadamard matrix of order K gives an orthogonal `R`, so
correctness only needs `H·Hᵀ = K·I` (which a single wrong sign always breaks) and the quantizer and
engine using the same matrix (guaranteed, because both read the one header).
Sloane's library states no licence; the matrices are mathematical objects, and they're credited
in the header.

### 6. Random signs

`s` depends only on `(seed, n)`. Every weight that consumes the same activation must use the same
`R`, which is what lets one rotation serve QKV, or FFN up and gate. QuIP#'s incoherence bound holds
for each matrix separately under a random `s`, so sharing `s` across tensors costs nothing.

It's defined in ggml, with one `static inline` mixer shared by the CPU and CUDA code:
- `γ = 0x9E3779B97F4A7C15`, `state₀ = seed ^ (γ · n)` (all `uint64_t`, wrapping mod 2⁶⁴)
- `word(k) = splitmix64_mix(state₀ + (k+1)·γ)`, which is the k-th splitmix64 output
- `s_i = (word(i / 64) >> (i mod 64)) & 1 ? −1 : +1`

**It's counter-based:** any thread computes its own sign words directly, one mix per 64 elements,
so no kernel reads a sign array. Golden vectors on the CPU and on the device keep implementations
from drifting. The seed is chosen at quantize time: a fixed default, so builds are reproducible,
overridden by `--hadamard-seed N`, and stored in the GGUF.

### 7. File format

- **One KV entry:** `hadamard.seed` (u64), written whenever at least one tensor is rotated, by the
  quantizer (Phase 4) and by `llama_model_save_to_file` (Phase 2). ConvRot files carry no
  `hadamard.*` keys.
- **Loader:** if any tensor has an HQ type and `hadamard.seed` is missing, it throws (with the
  ConvRot hint from §1.3). Without the seed the weights can't be decoded.
- **Only the current definition of `R` is supported.** There's no version key and no check for
  files made with a different one.
- **Nothing else is stored:** a tensor records only its type id.

### 8. Quantization semantics

#### 8.1 No imatrix for HQ types

> **Measured (2026-09-24): the premise below is wrong in practice.** Real activations have a large
> mean and strongly correlated channels, so the rotated per-coordinate energy is far from constant
> (coefficient of variation median 0.48). More importantly, the few massive-activation channels are
> axis-aligned in the unrotated basis, which a per-coordinate weight in the rotated basis can't
> protect. See the Implementation record: HQ with uniform weights loses to the imatrix quants, and
> GPTQ in the rotated space with `H = R·diag(imatrix)·Rᵀ` fixes it.
>
> **Superseded (2026-09-26):** `--imatrix` now drives GPTQ error feedback for every HQ type, as
> implemented in `plans/gptq_rotated_quantizer_plan.md`. The uniform-weight quantizer below is still
> what HQ types use without an imatrix.

Under the imatrix's own diagonal assumption, the rotated imatrix is
`diag(R·diag(v)·Rᵀ)ᵢ = Σⱼ R_ij² vⱼ`. Every entry of a full-row RHT has `|R_ij|² = 1/n`, so this is
**exactly** the row mean, a constant. Quantizer weights are scale-invariant, so that constant
carries no information. Every HQ type therefore ignores `--imatrix`, with one warning per run.
Only a rotated-space imatrix (Future work) could recover importance weighting.

#### 8.2 Uniform weights (nearest-neighbour rounding)

Quantizers minimize `Σᵢ wᵢ·(xᵢ − d·qᵢ)²`. The HQ quantizers use `wᵢ = 1`.
- **Why uniform:** the real objective is output error, `E‖ΔW·x‖²`. After the rotation every
  input column carries equal energy (§8.1), and the rotation preserves length, so minimizing plain
  error on the rotated weights is the right target.
- **Why not the no-imatrix heuristics:** those upstream fallbacks weight by magnitude: `x²` for
  IQ3/IQ4, `0.25σ² + x²` for IQ2_S, `av_x + |x|` for Q4_K's `_ref`. For rotated, near-Gaussian
  data, size says nothing about importance. Magnitude weighting then chases the largest values
  and nearly ignores the many near-zero ones.
- **IQ2_XXS / IQ2_XS:** their base quantizers have no imatrix-free path (they assert
  `quant_weights`, `ggml-quants.c:3302`, `:3480`). The HQ versions don't need one.
- **What uniform weights mean for each part of the quantizer:**
  - For a given scale, each group's code is the codebook point nearest in plain L2.
  - The best scale for fixed codes is the plain least-squares `Σqx/Σq²`. The scale is still
    searched over the same candidate set as the base quantizer, because picking it by absmax is
    far worse at 2–3 bits.
  - The sign-parity fix-up (IQ2_XXS/XS, IQ3_XXS) flips the element with the smallest `|x|`. That's
    a heuristic: the true cost of flipping element i is `4·|xᵢ|·d·qᵢ`, so smallest-`|x|` is exact
    only when that element lands on grid magnitude 1, which is the usual case.
- **Not an exact minimum:** uniform weights target plain MSE, but they don't guarantee its
  minimum. The scale search tries a fixed set of values, sub-block scales are re-rounded to 4–6
  bits, and the neighbour search is approximate (§8.3).
- Pure RTN (`d = amax/qmax`, no scale sweep) could be added later as an A/B option.

#### 8.3 Grid formats: neighbour list vs exact nearest neighbour (measured)

IQ2/IQ3 find the nearest codebook point by rounding to the integer lattice. If that point isn't
in the grid, they scan a precomputed neighbour list (`iq2_find_best_neighbour`,
`iq3_find_best_neighbour`). When the rounded point is in the grid, it's the exact nearest point;
otherwise the result is exact only within the neighbourhood.

An exhaustive search (256–1024 points × 8 dimensions × 13–19 scale trials) costs roughly
10¹⁴–10¹⁵ FLOPs for 27B weights, which is minutes on a multicore CPU. So cost doesn't rule it
out. A Phase 6 test measures how often the neighbour list disagrees with an exhaustive search,
and the resulting MSE gap. An `--hq-exact-nn` option, or exhaustive search in the final assignment
pass, is added only if the gap is non-negligible.

#### 8.4 HQ4_K / HQ5_K rounding (measured)

Upstream `quantize_q4_K`/`q5_K` contain two algorithms (`ggml-quants.c:1626`, `:1851`):
- **`_ref`:** `make_qkx2_quants` with weights `av_x + |x|`.
- **`_impl`:** `make_qkx3_quants`, a wider scale/min search, with weights `qw·sqrt(σ²+x²)`, and
  a weighted super-block scale fit.

The HQ versions start from the `_impl` search with uniform weights. A Phase 6 A/B test on
Qwen3-4B, the development model (§11), compares it with `_impl` using `sqrt(σ²+x²)` weights and
with `_ref`. The winner is what stays in the HQ quantizer.

### 9. Producing HQ data

HQ data can only be produced by the HQ quantizers, through `ggml_quantize_chunk` (Phase 3).
Rotation needs the whole row and the seed, which a block encoder doesn't have. The hazard is
**unrotated** data ending up labelled as rotated: wrong output without any error.
- **No generic encoder:** HQ types have `from_float_ref = NULL` in `type_traits`. They have no
  `type_traits_cpu` entries, so their `from_float` is NULL too.
- **At inference nothing can write a rotated weight.** Backends only ever see base types (Phase
  2). The graph guard (5.3) allows a rotated weight in exactly one place: as `src0` of `MUL_MAT` /
  `MUL_MAT_ID` whose input is an RHT node. Any write into it (`CPY`/`SET_ROWS` destination,
  in-place ops) or any other use throws, on every backend.
- Checked in this fork: no inference path writes weights. The runtime `from_float` calls convert
  activations (`vec_dot_type`), llama never copies into a model tensor, and koboldcpp applies GGUF
  LoRAs at runtime. Only the legacy GGML v2/v3 formats merge LoRA into weights, and they can't hold
  HQ types.
- **Re-encoding would be harmless anyway.** The rotation lives in the values, not the format, so
  a repack or a conversion of a rotated weight's own values (for example the CPU repack of Q4_K)
  stays rotated.
- **Runtime LoRA works unchanged:** `W'·(R·x) + scale·B·(A·x)`. The adapter branch uses the
  unrotated input, and adapter tensors keep their own types.
- **Permanent LoRA merges** go through `llama-quantize --lora` (Phase 4). For an unrotated source,
  the delta is added before rotation. For an HQ source, the delta is rotated with the file's seed
  first.
- **KV cache:** its allowed types are a fixed list (`common/arg.cpp:305`) that doesn't include HQ
  types.

### 10. Unsupported dimensions

When `ggml_rht_plan(n)` returns NONE (odd part ≥ 64, or `n` not divisible by 4):
- with `--hadamard`, the tensor keeps its base type, with one warning per dimension
- with an explicit `--tensor-type …=hq*`, quantizing throws

### 11. Coverage

**Target models:**

| model | n | tensors | (K, P) | `Ĥ_K` |
|---|---|---|---|---|
| Qwen3.8-27B | 5120 | attn_q/k/v/qkv/gate, ffn_gate/up, ssm_alpha/beta, output | (20, 256) | `had.20.will` |
| | 6144 | attn_output, ssm_out | (12, 512) | `had_12_paley` |
| | 10240 | nextn.eh_proj | (20, 512) | `had.20.will` |
| | 17408 | ffn_down | (68, 256) | `had.68.pal` |
| Qwen3-4B (development model) | 2560 | hidden | (20, 128) | `had.20.will` |
| | 4096 | attn_output | (1, 4096) | — |
| | 9728 | ffn_down | (76, 128) | `had.76.pal2` |
| Qwen3-0.6B (smoke-test model) | 1024 | hidden | (1, 1024) | — |
| | 2048 | attn_output | (1, 2048) | — |
| | 3072 | ffn_down | (12, 256) | `had_12_paley` |

`token_embd` is never rotated.

Qwen3-4B is the development model for Phases 6 and 7. Its widths exercise a Dao order (20) and a
large Sloane order (76) at 4B size. Qwen3-0.6B only exercises K = 12, and its decode is dominated
by launch overhead, so it's used for quick end-to-end checks, not for tuning or quality
conclusions.

**Other common models** (no `Ĥ_K` = pure power of 2):

| model | hidden → (K, P) | FFN → (K, P) |
|---|---|---|
| Llama-3.1-8B, Mistral-7B | 4096 → (1, 4096) | 14336 → (28, 512) |
| Llama-3.1-70B | 8192 → (1, 8192) | 28672 → (28, 1024) |
| Llama-2-7B | 4096 → (1, 4096) | 11008 → (172, 64) |
| Mistral-Nemo-12B | 5120 → (20, 256) | 14336 → (28, 512) |
| Gemma-2-9B | 3584 → (28, 128) | 14336 → (28, 512) |
| Gemma-3-27B | 5376 → (84, 64) | 21504 → (84, 256) |
| Qwen3-8B | 4096 → (1, 4096) | 12288 → (12, 1024) |
| Qwen3-14B | 5120 → (20, 256) | 17408 → (68, 256) |
| Qwen3-32B | 5120 → (20, 256) | 25600 → (100, 256) |
| Qwen2.5-7B | 3584 → (28, 128) | 18944 → (148, 128) |
| Qwen2.5-14B / 32B | 5120 → (20, 256) | 13824 → (108, 128) / 27648 → (108, 256) |
| Phi-3-mini | 3072 → (12, 256) | 8192 → (1, 8192) |

The orders in these two tables (12, 20, 28, 68, 76, 84, 100, 108, 148, 172) are the initial list of
unrolled CUDA mixes (§5.2). Every other table order runs on the K-generic mix. Attention-output
widths (`n_head·head_dim`) go through the same rule; the Phase 6 unit test lists them per model.

---

## Implementation conventions (all phases)

- **Minimize comments.** Comment only what the code doesn't make obvious:
  - why a non-obvious choice was made (for example, why decode takes two passes, or why
    `rotate_input_if_rotated` asserts F32)
  - an invariant that would silently break `R` if violated (signs before the transform, row-major
    `Ĥ_K`, natural `Ĥ_K ⊗ H_P` order)
  - a hardware or toolchain workaround (CUDA-graph capture, wave64, PDL ordering)
- **Don't write** comments that restate the code, narrate steps or refer to this plan. Don't
  carry over upstream comments that no longer apply after porting (for example, in the adapted
  `test-backend-ops.cpp`).
- **Required notices are the exception:** Dao's BSD-3 notice in `ggml-hadamard-tables.h` (his
  matrix strings), the Sloane credit, and one line per matrix in the tables header. The CUDA
  kernels aren't derived from Dao's code, so they carry no notice.
- Match the surrounding code's style and comment density.
- **Every significant feature that can fail gets a test.** That includes each function, rule, op,
  type property, CLI option, file-format check and refusal path the plan adds or changes, plus
  the cases where it should fail. Tests are written **in the same phase as the feature**, and a
  phase isn't done until its tests pass. Phase 6 lists every test by area, with the phase each
  one belongs to. When the plan changes, add or update the matching test in the same change. When
  something can't be tested on this machine (a real Vulkan or Metal device, HIP), say so in the test
  list, rather than leaving it silently untested.

---

## Phase 0 — RHT library in ggml (`ggml/src/ggml-hadamard.c`, declared in `ggml/include/ggml.h`)

- `bool ggml_rht_plan(int64_t n, int * K, int64_t * P)`: the rule in §4.
- `bool ggml_hadamard_matrix(int K, float * out)`: the table accessor (§5).
- `uint64_t ggml_rht_sign_word(uint64_t seed, int64_t n, int64_t k)`: the PRNG (§6). Its mixer is
  a `static inline` function in the internal header `ggml/src/ggml-hadamard.h`, which
  `ggml-hadamard.c` and `ggml-cuda/rht.cuh` both include. The tables header is included from there
  too.
- `void ggml_rht_ref(float * x, int64_t n, uint64_t seed)`: the transform for one row, in place,
  with a thread-local scratch buffer. The quantizer and the `--lora` merge call it, and every test
  uses it as the oracle. The mix loop runs over whole contiguous chunks
  (`out[c'·P + j] ± = u[c·P + j]` for a run of `j`), so the compiler vectorizes it.
- Two internal stage functions: signs + `H_P` over a range of chunks, and mix + scale over a
  range of positions `j`. The CPU op uses them to split one row across threads, with one barrier
  between the stages (5.1). `ggml_rht_ref` runs both over the whole row.
- `ggml/src/ggml-hadamard-tables.h` (§5).

The KV cache's existing `llama_gen_hadamard_matrix` / `llama_hadamard_inplace`
(`src/llama-hadamard.h`) stay as they are (see Future work: merge).

## Phase 1 — Types (`ggml.h`, `ggml.c`, `ggml-cpu.c`)

- **Enum:** `GGML_TYPE_HQ4_K = 150` … `GGML_TYPE_HQ4_XS = 158`, and `GGML_TYPE_COUNT = 159` (§1.1).
  This replaces the ConvRot enum values, traits rows, pairs, `llama_hadamard_rotate_rows` /
  `_imatrix`, the group-`g` graph path and the group-`g` guard. The ConvRot code is deleted, not
  kept behind a flag.
- **That includes every ConvRot read-side remap in the backends,** because backends never see
  rotated types again (Phase 2):
  - CUDA: `convert.cu` (6), `mmq.cu` (3), `mmvq.cu` (1), `ggml-cuda.cu` (1)
  - Vulkan: `ggml-vulkan.cpp` (9, plus a fusion guard for the ConvRot hint)
  - Metal: `ggml-metal-device.cpp` (5), `ggml-metal-ops.cpp` (1)
  - CPU: `spacemit/ime.cpp` (4)
  - **Restore these 8 files, don't hand-edit them.** Since the plan commit `4a708cd34`, only ConvRot
    commits (`29303175c`, `f05705f1b`, `aeefdc63d`) have touched them. Their net diff is exactly
    these remaps and the guard, 79 lines added and 51 removed. So
    `git checkout 4a708cd34 -- <the 8 files>` puts back their pre-ConvRot state. Metal can't be
    compiled on this machine, so an exact restore is safer than editing it.
  - If an upstream merge touches them first, revert just the ConvRot hunks instead (from
    `git diff 4a708cd34 HEAD -- <file>`).
  - Afterwards `grep -rn 'ggml_get_base_type\|ggml_is_rotated' ggml/src/ggml-*/` finds nothing.
  - The remaining uses are `ggml.c` (the pair functions), `ggml-quants.c` (`ggml_validate_row_data`)
    and `src/`.
- **`GGML_ROTATED_TYPE_PAIRS`:** all nine pairs (`Q4_K ↔ HQ4_K`, …, `IQ4_XS ↔ HQ4_XS`). Update the
  comment at `ggml.h:786` to list them.
- **`type_traits` rows** (`ggml.c:962`): each HQ type has a complete row next to the existing
  ones, with the base type's values (`type_name` aside). For example, `HQ4_XS` has
  `blck_size = QK_K`, `type_size = sizeof(block_iq4_xs)` and `to_float = dequantize_row_iq4_xs`.
  - `to_float` is what the quantizer uses to dequantize an HQ source when requantizing.
  - **Exception:** `from_float_ref = NULL` (§9).
- **No `type_traits_cpu` rows** (`ggml-cpu.c:417`): HQ types never reach the CPU backend, so
  their entries stay zero. A stray HQ tensor fails on a NULL `vec_dot` instead of computing
  something.
- **`ggml_validate_row_data`** (`ggml-quants.c`) validates an HQ row as its base type. The
  quantizer validates its HQ output (`llama-quant.cpp:753`, `:784`), and the loader's
  `check_tensors` path validates the file's types (`llama-model-loader.cpp:1444`).
- **No runtime `from_float` changes.** Every runtime call converts activations to the
  `vec_dot_type`: the CPU matmul (`ggml-cpu.c:1292`, `:1571`), the repack matmul
  (`repack.cpp:2728`, `:2834`) and flash attention (`ops.cpp:8550`). HQ types never reach the
  ops that encode into the weight type.
- **Python tooling:** `gguf-py/gguf/constants.py` doesn't list 150–158; that's out of scope. If
  it's ever needed, add them all together.
- **Build:** confirm that every object including `ggml.h` rebuilds after the enum change. A stale
  `ggml-cpu.o` would index `type_traits_cpu` out of bounds.

## Phase 2 — Loading HQ tensors (`src/llama-model-loader.cpp`, `src/llama-model.cpp`)

An HQ tensor is loaded as a tensor of its base type, with the same bytes, and llama records that
it's rotated. No backend changes are needed: every backend already runs the base types. That
includes the CPU repack, so HQ4_K/HQ5_K/HQ4_NL get Q4_K/Q5_K/IQ4_NL's repacked layouts.

- **Where:** `llama_model_loader::create_tensor`, on its local copy `t_meta` (`:1305`, and the
  file-less path at `:1262`), before `buft_for_tensor` picks the buffer type. There,
  `t_meta.type = ggml_get_base_type(t_meta.type)`.
  - Buffer selection (the `weight_buft_supported` probe) and the created tensor then both see the
    base type.
  - `nb[]` and `ggml_nbytes` don't change (the existing assert at `:1319` still holds), so mmap,
    `load_all_data` and its validation of `cur->type` work unchanged.
- **Not in the loader's metadata:** the meta tensors in `weights_map` keep their HQ types.
  `llama-quant` reads source types through the same loader, and needs them for the requantize
  rules and the seed inheritance. The printed type summary also still shows the HQ types.
- **The rotated set:** `create_tensor` adds each demoted tensor to the loader's
  `std::unordered_set<const ggml_tensor *> rotated_tensors`, which `load_tensors` moves into
  `llama_model`. `llama_model::is_rotated(t)` follows `view_src` to the root tensor. The graph
  code (5.3) asks it instead of `ggml_is_rotated(w->type)`.
- **Device policy:** enforced here, right after the tensors are created (§5.4).
- **Saving from memory:** `llama_model_save_to_file` writes a correct HQ file.
  - In memory, rotated tensors carry base types. The saver writes `tensor->type`
    (`llama-model-saver.cpp:154`, via `gguf_add_tensor`) and a fixed list of keys, so as it
    stands it would label rotated data as the base type and drop the seed. Any build would then
    load that file as an ordinary model and never rotate the input.
  - So, in `llama_model_saver::add_tensors_from_model`, each tensor in the rotated set gets
    `gguf_set_tensor_type(ctx, name, ggml_get_rotated_type(t->type))` after it's added. The sizes
    are equal, so the offsets don't change.
  - When the set is non-empty, the saver also writes `hadamard.seed` from the model.
  - Nothing in the fork calls it today; the round-trip test (Phase 6) keeps it correct.
- **`check_tensors`:** validates the file's types before demotion, which is why
  `ggml_validate_row_data` keeps its HQ remap (Phase 1).

## Phase 3 — HQ quantizers (`ggml/src/ggml-quants-hq.c`)

The HQ quantizers are separate from the base quantizers, so HQ encoding can change freely. **The
base quantizers in `ggml-quants.c` aren't modified.**
- **Copies, not a uniform-weights flag in the base functions.** §8.3 and §8.4 may change the HQ
  encoding, and upstream's i-quant quantizers rarely change, so the copies' drift risk is low.
- **No importance-weight input.** The HQ `_impl`s have uniform weights built in and take no
  weights parameter. A future rotated-space imatrix or error feedback would add one then.

- **One `_impl` per HQ type,** each a copy of the base function with uniform weights built in
  (`weight[i] = 1`, `waux[i] = 1`). The imatrix and heuristic branches, and the
  `GGML_ASSERT(quant_weights)` of IQ2_XXS/XS, are left out.
  - `quantize_row_hq2_xxs_impl`, `hq2_xs`, `hq2_s`, `hq3_xxs`, `hq3_s`
  - `hq4_nl_impl`, shared by HQ4_NL and HQ4_XS the way `iq4_nl_impl` is upstream
  - `quantize_row_hq4_K_impl` / `hq5_K_impl` (§8.4)
  - Size: about 1.5k lines.
- **Helpers copied as `static`:** `nearest_int`, `make_qp_quants`, `make_qkx3_quants`,
  `best_index_int8`, and `hq2_`/`hq3_find_best_neighbour`. That's about 300 lines.
- **Shared format data:** the grid, map and neighbour tables stay built and owned by
  `ggml-quants.c` (`iq2_data` / `iq3_data`, from `iq2xs_init_impl` / `iq3xs_init_impl`). Add two
  small internal accessors, `ggml_iq2_entry(type)` and `ggml_iq3_entry(grid_size)`.
  `ggml_quantize_init` gets HQ cases that initialize the base grids.
- **Entry point:** one internal function,
  `size_t quantize_hq(enum ggml_type, const float * src, void * dst, int64_t nrow, int64_t n_per_row)`,
  declared in the internal `ggml-quants.h` (not `ggml.h`).
- **`ggml_quantize_chunk`:** one case group for the nine HQ types that calls `quantize_hq` and
  ignores `imatrix`. Callers pass the stored HQ type directly.
- **`ggml_quantize_requires_imatrix`:** false for every HQ type.
- **Optional exact NN:** an exact nearest-neighbour final pass (§8.3), added only if Phase 6 shows
  it matters.
- **Build:** add `ggml-quants-hq.c` to the koboldcpp `Makefile` (CPU objects are listed
  explicitly) and to the CMake lists.

## Phase 4 — Quantizer tool (`src/llama-quant.cpp`, `tools/quantize/quantize.cpp`)

- **CLI:**
  - `--hadamard` maps each tensor's chosen type to its HQ variant via `ggml_get_rotated_type`
    (§1.1). Types without an HQ variant are unchanged.
  - `--hadamard-seed N` sets the seed.
  - `--tensor-type` accepts the HQ type names.
- **Target-type pass:**
  - Call `ggml_rht_plan(ne[0], …)` for each candidate; if it returns NONE, apply §10.
  - Token embeddings are never rotated.
  - Remove the ConvRot `ggml_fwht_supports_group` warnings here and in `llama-model.cpp:1845`.
  - `tensor_requires_imatrix` is false for HQ targets, so `--hadamard IQ2_XXS` runs without an
    imatrix. Non-HQ tensors in the same ftype mix follow their normal imatrix rules.
- **imatrix:** HQ tensors ignore it, with one warning per run giving the count of affected tensors.
  Non-HQ tensors use it as usual.
- **Rotation:** each row goes through `ggml_rht_ref(row, n, seed)`, spread across the `workers`
  that `llama_tensor_quantize_impl` already uses. Rotating a 17408-wide row costs about 1.2 M MACs,
  and there are 5120 rows per tensor.
- **Metadata:** write `hadamard.seed` when any tensor is rotated.
- **Seed when requantizing:** a file holds one seed, so every rotated tensor in it must share it.
  - If the source has `hadamard.seed = S`, the output uses `S` for every tensor, including newly
    rotated ones.
  - An explicit `--hadamard-seed ≠ S` is an error.
  - Only when the source has no rotated tensors does the seed come from `--hadamard-seed` or the
    default.
- **Requantize rules:**
  - HQ → non-HQ is refused.
  - HQ → the same HQ type is a plain copy.
  - HQ → a different HQ type (for example HQ5_K → HQ4_XS): dequantize, then quantize with **no
    re-rotation**, because the data is already in the `R` space for the inherited seed.
  - A source with HQ types and no `hadamard.seed` is refused.
- **LoRA merge: `--lora FILE[:scale]` (repeatable).** Merges adapters into the weights in the same
  row pipeline. It works for any target type.
  - **Loading:** the adapter GGUF is read with the gguf API, using `llama-adapter.cpp`'s
    conventions:
    - `<tensor>.lora_a` / `.lora_b` pairs, with `adapter.type == "lora"` and
      `adapter.lora.alpha`
    - `general.architecture` must match the model's
    - A-LoRA adapters are refused
    - adapter tensors (F32/F16/BF16/Q8_0) are dequantized to F32 once
  - **Delta:** `Δ = scale_eff · B·A`, with `scale_eff = alpha ? scale·alpha/rank : scale` (the
    same as `build_lora_mm`). It's computed per slab in F32 across the `workers`. 3D expert
    tensors use the matching expert slice.
  - **Where `Δ` is added:**
    - **Unrotated source rows:** added **before** rotation.
    - **HQ source rows:** add `ggml_rht_ref(Δ_row, n, seed)` with the inherited seed.
    - Neither path uses `from_float`.
  - **Edge cases:**
    - Adapter pairs for tensors not in the model are an error.
    - `token_embd` deltas merge and stay unrotated.
    - A tensor that would otherwise be a plain copy is dequantized and requantized when it has an
      adapter.
  - **Metadata:** record `general.merged_loras` (file basenames and scales).
  - **Quality:** merging into an f16/bf16 source quantizes once. Merging into a quantized source
    quantizes twice, and the README says so.
- **Slab accounting:** `ggml_rht_ref`'s scratch is per thread and bounded by `n`, so
  `bytes_per_row` needs nothing extra.

## Phase 5 — Inference

### 5.1 The op (`ggml.h`, `ggml.c`, CPU forward in `ggml-cpu/ops.cpp`)

- `ggml_tensor * ggml_rht(ctx, x, uint64_t seed)` creates `GGML_OP_RHT`, with `x` as its only
  source.
  - **Input `x`:** F32 `[n, …]`, with contiguous rows. All dims above 0 are rows, so the 3D
    `mul_mat_id` input works unchanged. `ggml_rht` asserts F32 and that `ggml_rht_plan(n)`
    succeeds.
  - **`op_params`:** the seed only. Every implementation derives `K` and `P` from
    `ggml_rht_plan(ne[0])`.
  - **Output:** F32, same shape as `x`, equal to `R·x` per row.
- **Registration:** add it to `GGML_OP_NAME` / `GGML_OP_SYMBOL` and bump the `GGML_OP_COUNT`
  static asserts.
- **F32 only.** F16/BF16 inputs arrive with the KV-cache merge (Future work), which is their
  first user. The weight-rotation path never needs them:
  - Its input is the `src1` of a rotated weight's GEMM, which is F32 throughout. The only
    `ggml_cast` to F16 in `src/` is for attention K/V (`llama-graph.cpp:2685`, `:2689`) and the
    KQ masks, and none of them feeds a weight GEMM.
  - CUDA also requires F32 `src1` for quantized weights (`ggml-cuda.cu:1796`, `:1908`).
  - So `rotate_input_if_rotated` asserts F32. If a model ever feeds F16 there, the fix is a
    `ggml_cast` to F32 before the op.
- **CPU forward:** it runs the rotations for layers on the CPU (CPU-only runs and partial
  offload), so decode speed matters here too. Rows are split across threads. When there are
  fewer rows than threads (decode), each row is also split: the chunk stage over chunk ranges,
  one `ggml_barrier`, then the mix stage over ranges of `j` (the Phase 0 stage functions).
  Otherwise one-row decode runs single-threaded while every other op uses all threads.

### 5.2 CUDA kernels (`ggml/src/ggml-cuda/rht.cu`, `rht.cuh`)

Written for this fork, not ported. The Makefile globs `ggml/src/ggml-cuda/*.cu`. The design treats
the two regimes separately, because different things limit them (§2):
- **Decode (a few rows) is limited by latency.** The row is spread across many SMs.
- **Prompt processing (many rows) is limited by memory bandwidth.** Each row is read once and
  written once, in one pass.

On CUDA the F32 rotated activation usually isn't needed at all. Every quantized GEMM first
converts its F32 `src1` to Q8_1 (`mmvq.cu:1332`, `mmq.cu:159`), so when all of an RHT node's
consumers are such GEMMs, the RHT kernel writes that Q8_1 itself (the Q8_1 epilogue below).

The numbers in this section are from prototypes run on the RTX 5090, not from the final kernels.

**Building blocks** (device functions in `rht.cuh`, shared by every kernel):
- **Load a chunk:** a warp loads one `P`-chunk. Each lane holds `E = P/32` consecutive elements,
  with float4 loads. Chunks with `P < 128` are packed several per warp.
- **Signs:** applied right after the load. A lane's `E` consecutive elements share one splitmix64
  word (`E` divides 64), so each lane computes one word per chunk. The words don't depend on the
  input, so they're computed before `ggml_cuda_pdl_sync()`.
- **`H_P` in a warp:** `log₂E` butterfly stages in registers, then 5 `__shfl_xor_sync` stages, in
  natural order. `E ≤ E_MAX` (initially 32, so `P ≤ 1024` per warp).
- **Larger `P` is split internally.** `H_P = H_{P₁} ⊗ H_{P₂}`, both Sylvester in natural order,
  and `H_{P₁}` moves into the mix stage as an in-register butterfly. That covers `K = 1` rows up
  to 2¹⁵. The result is the same `R`; only the factoring inside the kernel differs.
- **Order-K mix, one output tile:** a lane owns one position `j` and `G` outputs `c'`. It reads
  the `K` inputs `u[c·P + j]` and accumulates `±u` with the signs of `Ĥ_K`.
  - **Unrolled:** a `#pragma unroll` loop over the tables header's device copy (§5). The loads
    fold, so the mix compiles to FADDs with sign modifiers only. Initial list: the §11 orders
    (12, 20, 28, 68, 76, 84, 100, 108, 148, 172), kept in one X-macro list.
  - **K-generic:** every other table order. It reads the order's ±1 bits (at most 8 KB) into
    shared memory and applies them as sign-bit XORs.
- **Epilogues:** a runtime, warp-uniform switch, not a template parameter, so each order costs one
  instance per kernel.
  - **F32:** store the scaled value.
  - **Q8_1 (MMVQ):** the `block_q8_1` layout of `quantize_row_q8_1_cuda`.
  - **Q8_1 (MMQ):** the `block_q8_1_mmq` layout, including its token interleaving, in the D4, DS4
    or D2S6 variant that `mmq_get_q8_1_ds_layout` picks for the consumer's `src0` type.
  - A mix warp always covers 32 consecutive `j` of one output chunk `c'`, which is exactly one
    `block_q8_1`. Absmax and sum are 5-step warp reductions, then lane 0 writes the scale. D2S6
    scales cover 64 values, so there two warps combine, or a lane owns 2 `j`.
  - The per-block packing and indexing are factored out of `quantize.cu` and called from both
    places, so the layouts can't drift. The zero blocks for the row padding (`MATRIX_ROW_PADDING`)
    are written too.

**Kernel A: decode, two passes spread over SMs** (initially used for rows ≤ 8, the MMVQ range):
1. **`rht_chunks`:** one warp per (chunk, row) does load, signs and `H_P`, then writes F32 to a
   pool scratch buffer of `n·rows` floats, which stays in L2. It triggers the next launch early
   (`ggml_cuda_pdl_lc()`).
2. **`rht_mix_split`:** one warp per (32 positions `j`, output tile of `G`, row). After
   `ggml_cuda_pdl_sync()`, each lane loads its `K` inputs from scratch (the warp's loads are
   coalesced, at stride `P`), mixes, scales by `1/√n` and runs the epilogue.
- **Prototype, full transform with F32 output, measured inside a CUDA graph:** 2.1 µs at
  (n = 5120, K = 20), 2.5 µs at (6144, 12), 2.6 µs at (17408, 68).
- For comparison, a one-CTA-per-row kernel spends 5.8 µs on the K = 68 mix alone, and a K-generic
  one 18.5 µs.
- **A cluster version was also tried** (one kernel, chunks exchanged through distributed shared
  memory). It was 1.9 µs at K = 20, but 11–14 µs at K = 68, stalled on the cluster barriers'
  memory fences and on DSMEM loads through generic addresses. It's a Phase 7 experiment, not
  the initial design.

**Kernel B: prompt processing, one pass, one CTA per row:**
- **Layout:** the CTA has `W` warps, and the `K` chunks are split across them as evenly as
  possible, `Q = ⌈K/W⌉` at most (68 = 17 × 4). `W` needn't divide `K`: 172 = 4·43 has no useful
  divisor. Stage 1 runs load, signs and `H_P` for each chunk, keeping the row in registers: at
  most `Q·E` floats per lane, 32 at n = 17408.
- **Stage 2** handles groups of `S` slabs, where a slab is 32 consecutive `j`.
  - The lanes holding those `j` write them to shared memory `[S][K][32]`, then there's a barrier.
  - `S·K/G` warps each mix one (slab, tile), reading `K` values from shared memory (the lane is
    `j`, so there are no bank conflicts), and run the epilogue. Then another barrier.
- **Shared memory is `S·K·128` bytes whatever `n` is:** 35 KB at K = 68, S = 4, and 32 KB at
  K = 252, S = 1. It stays under the 48 KB default, so no opt-in is needed and no width is too
  wide for a device. The earlier design staged the whole row (`4n` bytes), which pushed
  Qwen3-32B's 25600 and Qwen2.5-32B's 27648 past the 99 KB limit of sm_86/89/120, and 17408 past
  Turing's 64 KB.
- Prototype mix stages already ran at memory bandwidth for K ≤ 84 at 512 rows (1.5 TB/s at
  n = 17408, same as a copy).

**Choosing a kernel:** initially kernel A for rows ≤ 8 and kernel B otherwise. Phase 7 measures
the crossover per `(n, K)`; kernel B with one CTA may win for small `K` in decode because it's a
single launch.

**Q8_1 emission** (in the CUDA backend's graph evaluation):
- **When an RHT node writes Q8_1:** the backend checks its consumers in the same cgraph. One pass
  over the cgraph builds the consumer lists, and they must match `ggml_node_get_use_count`. The
  node writes Q8_1 when all of the following hold:
  - every consumer is a `MUL_MAT` with a quantized `src0`, on this device, not in a split buffer
  - every consumer will take MMVQ or MMQ (the same predicate `ggml_cuda_mul_mat` uses) and needs
    the same Q8_1 layout
  - the node isn't a graph output
  - the Q8_1 fits in the node's F32 buffer, which is about 3.5× larger except for tiny MMQ
    batches
- **What it does then:** the RHT kernel writes that layout into its own buffer and records
  `{node → layout}` in the CUDA context for this evaluation. `ggml_cuda_mul_mat_vec_q` and
  `ggml_cuda_mul_mat_q` skip their `quantize_*` call for a recorded `src1` and read it directly.
- **Otherwise the node writes F32** and every consumer converts as it does today. That covers
  mixed layouts (for example a Q6_K `attn_v` next to HQ4_K `attn_q`), `MUL_MAT_ID`, non-matmul
  consumers, split buffers, and eval callbacks. Eval callbacks split the graph, so the consumers
  aren't in the same cgraph.
- **Effect:** QKV, up/gate and every other rotated input are converted to Q8_1 once instead of
  once per GEMM. An HQ layer launches fewer kernels than its base-type layer. In prompt
  processing the RHT reads F32 once and writes Q8_1 once, the same traffic as today's
  conversion.
- The existing gate/up + GLU fusion (`ggml-cuda.cu:3774`) still applies: the two GEMMs stay
  adjacent and share `src1`.

**Launch and portability:**
- **Code size:** each unrolled order has one instance each for kernel A pass 2 and kernel B, about
  `2·K²` FADDs, so roughly 180k instructions for the initial list. Portable builds ship PTX only
  (`Makefile:250–255`), so the driver compiles them on the user's machine at first load. The
  table check (orders 12, 68 and 172, including a 29k-FADD kernel) compiled in 3.4 s, so this
  should take seconds. Phase 7 measures the JIT time and caps the list if needed.
- **CUDA-graph safe:** no host syncs, the scratch comes from the pool (as in `mmq.cu`), and any
  `cudaFuncSetAttribute` runs once per device and instance.
- **Warp-size portable,** like `fwht.cu`. The unit of work is a 32-lane group with width-32
  shuffles, so a wave64 AMD wavefront runs two groups. `ggml_cuda_kernel_launch(…)` handles
  launches, and PDL applies where available.
- **Row strides** come from `nb[1]`.

**`supports_op`:** true when `x` and `dst` are F32 with contiguous rows and `ggml_rht_plan(ne[0])`
succeeds. There's no shared-memory limit on the width.

**HIP and MUSA:** the kernels are built there too, because the CUDA code is shared and warp-size
portable, and rotated models load on those devices (§5.4). They're unverified until the
`test-backend-ops` RHT and Q8_1 cases pass on real hardware.

### 5.3 Graph (`src/llama-graph.{h,cpp}`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`)

- **Loader:** reads and requires `hadamard.seed` (§7) and stores it on
  the model. The rotated set and the device policy come from Phase 2 and §5.4.
- **`rotate_input_if_rotated(w, cur)`** returns `cur` unchanged unless `model.is_rotated(w)`. The
  weight's type no longer says it: at runtime it's the base type. The rotation is memoized on
  `cur`, so QKV share one rotation and FFN up/gate share another:
  ```
  cur_rot = ggml_rht(ctx0, cur, model.hadamard_seed)
  ```
  Non-contiguous `cur` is made contiguous first. It asserts F32 (5.1).
- **No rotation graph inputs:** the seed travels in `op_params`. Delete `llm_graph_input_hadamard`,
  its `set_input`, and the ConvRot `H_g` matrices.
- **Guard (`llm_graph_check_hadamard_rotation(gf, model)`)**, run after every graph build. It walks
  every node. A tensor counts as rotated if `model.is_rotated` holds for it, following `view_src`.
  - **A rotated tensor as a source:**
    - Views (`VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`) pass it through.
    - As `src0` of `MUL_MAT` / `MUL_MAT_ID`, it's accepted only if `src1`, after stripping
      `RESHAPE`/`VIEW`, is a `GGML_OP_RHT` node with `ne[0] == src0->ne[0]` and the model's seed.
    - Any other use throws. That includes `GET_ROWS` (so a rotated `token_embd`), `ADD`, a rotated
      tensor as `src1`, and any other op.
  - **A write into a rotated tensor:** a node that isn't a view but whose `view_src` root is
    rotated throws. That covers a `CPY` / `SET_ROWS` destination and in-place ops.
  - Model code that bypasses the wrappers fails here, on every backend. This replaces the old
    type-based refusals of ops that write weights (§9).
- **LoRA:** `build_lora_mm` / `build_lora_mm_id` compute `W'·(R·x) + scale·B·(A·x)`, with the
  adapter branch on the unrotated `cur`.
- **Kept for the KV cache:** `llama_mul_mat_hadamard`, the FWHT hint and `fwht.cu`.

### 5.4 Supported devices: CPU and the CUDA backend

A model with rotated tensors loads only if every device it would use is the CPU or a device of the
CUDA backend: CUDA, or ROCm (HIP) and MUSA, which build the same `ggml-cuda` code and so have the
same `GGML_OP_RHT` kernels. The other backends have no `GGML_OP_RHT` (Vulkan, Metal, SYCL,
OpenCL, RPC, …). There, every rotation would run on the CPU, costing a graph split and a
device→host→device round trip, four per layer, and nothing would report the slowdown.

**The check:** at the end of Phase 2's tensor creation, before any buffer is allocated or read.
It applies when the rotated set is non-empty.
- Every device in `model.devices`, and the device of every buffer type holding a model tensor,
  must either be of type `GGML_BACKEND_DEVICE_TYPE_CPU` or belong to the CUDA backend.
  - The CPU type includes the CPU's extra buffer types, such as the repack.
  - The CUDA backend registers under `GGML_CUDA_NAME`, which is `"CUDA"`, `"ROCm"` under HIP or
    `"MUSA"` under MUSA (`ggml-cuda.h:11–17`). The check accepts those three registry names.
    `llama.o` is shared across the koboldcpp backend builds, so it can't use `#ifdef`s or call
    `ggml_backend_cuda_reg()`.
- Otherwise throw: "this model has Hadamard-rotated (HQ) tensors, which are only supported on the
  CPU and the CUDA backend (CUDA, ROCm, MUSA); device <name> (<backend>) isn't. Run CPU-only
  (koboldcpp: --usecpu; llama tools: --device none) or on a CUDA/ROCm build."
- **ROCm and MUSA are unverified:** there's no AMD or Moore Threads GPU here. The README says so
  until the `test-backend-ops` RHT and Q8_1 cases have passed on one (Phase 6).
- **The whole device list is checked, not just the devices that hold weights.** The scheduler can
  offload ops to any device in the list, for example prompt-processing matmuls with `-ngl 0`. So
  on a Vulkan or Metal build, a rotated model runs only with the GPU excluded.
- Support for more backends is Future work. With the load-time demotion, a port only needs
  `GGML_OP_RHT`, since the base types' kernels already run HQ weights.

## Phase 6 — Tests and quality measurement

**Unit (`tests/test-hadamard.cpp`):**
- **Table (Phase 0):**
  - Every entry, including order 40, satisfies `H·Hᵀ = K·I` exactly, in integer arithmetic.
  - Each order appears once, and `ggml_hadamard_matrix` returns false for any order not in the
    table.
- **Rule (Phase 0):** `ggml_rht_plan` gives the expected `(K, P)` for every dimension in §11, plus:
  - n = 24·256 → (12, 512)
  - n = 128 → (1, 128)
  - n = 40·256 → (20, 512)
  - n = 11008 → (172, 64)
  - n = 64·63 → (252, 16)
  - n = 64·65 → NONE
  - n = 6 and n = 2·63 → NONE
- **Signs (Phase 0, device part in Phase 5):** golden vectors for `(seed=0, n=64)` and `(seed=42, n=5120)`, checked against
  `ggml_rht_sign_word` on the CPU and against the device mixer (a small test kernel). Computing
  word k directly must equal stepping splitmix64 k times.
- **Reference transform (Phase 0):**
  - **It's orthogonal:** `‖Rx‖ = ‖x‖` and `⟨Rw, Rx⟩ = ⟨w, x⟩` at each `n` in §11.
  - **It's the right orthogonal matrix.** Orthogonality alone doesn't catch a wrong element
    order, a transposed `Ĥ_K` or signs applied after the transform. For small shapes, (K, P) =
    (1, 64), (12, 16), (20, 8), (28, 4), (68, 4) and (172, 4), build `R` densely from its
    definition in §2 (`(1/√n)·(Ĥ_K ⊗ H_P)·diag(s)`, natural order), and check that
    `ggml_rht_ref` equals `R·x` for basis vectors and random vectors.
  - **Thread safety:** many threads calling `ggml_rht_ref` concurrently on different `K`, the
    first time each `K` is used, give the same results as a single thread (this checks the
    per-`K` matrix cache).
  - **Stage split:** running the two stage functions over any split of the chunks and of the `j`
    range (1, 3 and 8 parts) gives exactly the bytes of `ggml_rht_ref`.
- **Table accessor orientation (Phase 0):** `ggml_hadamard_matrix(K)` row `i` equals the header's
  row string `i`, not column `i`, for every `K`.
- **Traits rows (Phase 1):**
  - For every HQ type, every `type_traits` field except `type_name` and `from_float_ref` equals
    the base type's, and `from_float_ref` is NULL.
  - The `type_traits_cpu` entry is empty (`from_float` and `vec_dot` are NULL).
- **No backend remaps left (Phase 1):** `grep -rn 'ggml_get_base_type\|ggml_is_rotated'
  ggml/src/ggml-*/` finds nothing. It's a one-line check in `tools/quantize/tests.sh`.
- **Type pairing (Phase 1):**
  - For all nine pairs, `ggml_get_rotated_type(base) == hq`, `ggml_get_base_type(hq) == base`,
    and `ggml_is_rotated(hq)` holds.
  - Every non-HQ type maps to itself, and `ggml_is_rotated` is false for it.
  - `ggml_type_name(hq)` returns the §1.1 name, and parsing that name (the `--tensor-type` parser)
    returns the HQ type.
- **Row validation (Phase 1):** `ggml_validate_row_data` accepts valid HQ rows and rejects the same
  corruptions (for example NaN/Inf scales) that it rejects for the base type.
- **HQ quantizers (Phase 3):**
  - **Base quantizers are unchanged:** a hash of `ggml_quantize_chunk` output for every base type
    on a fixed input, recorded before Phase 3 and checked after. The only base edits are the grid
    accessors.
  - **Format compatibility:** HQ output dequantized with the base type's `to_float` equals
    dequantization through the HQ traits row, bitwise.
  - **No imatrix needed or used:** HQ2_XXS / HQ2_XS quantize without an imatrix (no assert), and
    passing an imatrix gives byte-identical output for every HQ type.
  - **Cold start:** quantizing an HQ i-quant as the first call, before any base-type
    `ggml_quantize_init`, works. That checks that the HQ cases initialize the base grids.
  - **Determinism:** the same input quantized twice, single-threaded and multi-threaded, gives
    identical bytes.
  - `ggml_quantize_requires_imatrix` is false for every HQ type.
  - For Gaussian and Laplacian rows (at least 4096 × 4096), `ggml_quantize_chunk(HQ type)` has
    aggregate plain MSE ≤ the baseline × (1 + ε), with ε ≈ 0.01. A per-row "≤" isn't guaranteed
    (§8.2), so it isn't asserted.
  - Baselines: `ggml_quantize_chunk(base, imatrix = NULL)` for IQ2_S, IQ3_XXS/S and IQ4_NL/XS, and
    an all-ones imatrix for IQ2_XXS/XS.
  - A failure means investigate; it isn't automatically a bug.
- **Neighbour list vs exhaustive search (Phase 3, §8.3):** for each IQ2/IQ3 grid, on rotated Gaussian
  groups, report the disagreement rate and the MSE gap. This reports numbers; it doesn't pass or
  fail.
- **Round trip (Phase 3):** rotate → `ggml_quantize_chunk(HQ)` → dequantize → `Rᵀ` comes out close to the
  original, within a per-type tolerance.

**Backend (`tests/test-backend-ops.cpp`):**
- **The harness (set up at the start of Phase 5, before the CUDA kernels):**
  - The fork has no `test-backend-ops`. Take upstream's copy at the llama.cpp commit this fork
    last merged: `git show 511f9c137:tests/test-backend-ops.cpp`, 11k lines. It compiles against
    the fork's headers unchanged, and its existing `MUL_MAT` cases pass on CUDA (checked
    2026-09-24).
  - Add the fork's cases in one separate function, `make_test_cases_fork()`, called from one line
    in `make_test_cases_eval()`, so re-syncing after an upstream merge is a copy plus that line.
  - Makefile target `test-backend-ops` (with `LLAMA_CUBLAS=1`): ggml-only objects (as
    `HADAMARD_TEST_OBJS`, with `ggml_v4_cublas.o` and `ggml-backend-reg_cublas.o`) plus
    `CUBLAS_OBJS_V4`. `$(OBJS)` can't be used because it pulls in the llama-dependent
    `common.o`.
  - Its `perf` mode is used for prompt-size shapes only. For one-row shapes it measures launch
    and sync overhead: 6.3 µs for a 4096-element `ADD`, against 0.65 µs for a launch inside a
    CUDA graph. Decode latency comes from `bench-rht` (Phase 7).
- **`RHT` op, CUDA against CPU (Phase 5):**
  - **Kernels and orders:** kernel A and kernel B for every unrolled order, each at its §11 `P`
    and at the smallest and largest `P` with `n ≤ 2¹⁵`. The K-generic mix at K = 36, 44, 116 and
    252. `K = 1` for `P` = 32 … 2¹⁵, which covers the internal `H_P` split. Chunks narrower than
    128 elements: (172, 64) and (252, 16).
  - **Rows:** 1, 7, 8, 9 (either side of the A/B switch) and 512. The 3D `mul_mat_id` shape. A
    row stride larger than `n` (contiguous rows, padded stride).
  - **Reference:** the CPU op, which is checked against `ggml_rht_ref` directly.
- **Q8_1 emission (Phase 5):** small graphs, CUDA against CPU. Each case runs with and without
  `GGML_CUDA_DISABLE_FUSION=1`, which also turns emission off:
  - `RHT → 3 × MUL_MAT` with Q4_K `src0` (the QKV shape, and what an HQ4_K weight is at
    runtime), at 1, 8, 64 and 512 rows (MMVQ and MMQ)
  - the same with one consumer whose MMQ layout differs (Q4_K next to Q6_K), which falls back to
    F32
  - `RHT → up, gate → GLU`, the fused gate/up path
  - an RHT node that also feeds an `ADD`, and one marked as a graph output: both fall back
  - `n = 5376`, whose rows need the `MATRIX_ROW_PADDING` zero blocks
- **CPU op (Phase 5):** one row computed with 1, 4 and 16 threads gives the same bytes as
  `ggml_rht_ref`, which checks the intra-row split.
- **Coverage (Phase 5):** CUDA's `supports_op` accepts every width the rule accepts.
- **CUDA graphs (Phase 5):** decoding with CUDA graphs enabled and disabled gives identical tokens
  and logits within tolerance on an HQ model. That catches a runtime API call or allocation inside
  the capture.

**Loading (Phase 2, `tests/test-hadamard.cpp`):**
- **Test file, without the quantizer:** Phase 4 doesn't exist yet, so the test makes its own HQ
  file with the gguf API. It copies a small Q4_K / IQ4_NL GGUF, relabels the eligible tensors to
  their HQ types (`gguf_set_tensor_type`) and adds `hadamard.seed`. The data isn't really
  rotated, which doesn't matter for these tests.
- **Demotion:**
  - After loading, every tensor stored as an HQ type in the file has its base type, the same
    `nb[]` and the same bytes.
  - `model.is_rotated` is true for exactly those tensors, and for views of them.
  - The loader's meta tensors (what `llama-quant` reads) keep their HQ types.
- **Buffer selection:** on the CPU, HQ4_K / HQ5_K / HQ4_NL weights get the repack buffer type,
  like Q4_K / Q5_K / IQ4_NL. The logits match a load with the repack disabled.
- **Save round trip:** save a loaded HQ model with `llama_model_save_to_file` and load the saved
  file. It must have:
  - the source's HQ types in the file (the loader's meta tensors)
  - the same `hadamard.seed`
  - byte-identical tensor data
  - identical logits
  - a non-HQ model saves without a `hadamard.seed` key
- **Device policy (§5.4):**
  - A CPU-only load and a CUDA load succeed.
  - A load with an RPC device in the device list is refused with the §5.4 message. Use a CPU-only
    `rpc-server` on localhost: the Makefile's `rpcserver` target links Vulkan, so add a CPU
    variant.
  - Vulkan and Metal take the same code path, but can't run here.

**Graph (Phase 5.3, `tests/test-hadamard.cpp` with a small model):**
- **Rotate-once memo:** the built graph has exactly one `GGML_OP_RHT` per distinct rotated
  activation: one for QKV, one for FFN up/gate, and one each for the other rotated inputs. Count
  them per layer.
- **Guard: rotated input.** A graph that feeds a rotated weight from an unrotated input, or from an
  `RHT` node with the wrong seed or width, makes `llm_graph_check_hadamard_rotation` throw.
- **Guard: other uses.** A rotated weight used by `GET_ROWS` or `ADD`, used as `src1`, or written
  (`CPY` / `SET_ROWS` destination, an in-place `ADD`) throws. A view of a rotated weight is
  checked the same way.
- **LoRA branch:** with an adapter loaded, the adapter's `mul_mat` takes the unrotated `cur`, not
  the `RHT` output.
- **No rotation graph inputs:** a graph of an HQ model contains no Hadamard input tensors.

**Quantizer tool (Phase 4):**
- **Seed:** two runs with the default seed give byte-identical files. A different
  `--hadamard-seed` changes the HQ tensors and `hadamard.seed`, and nothing else.
- **Metadata:** `hadamard.seed` is written only when a tensor is rotated, and no other
  `hadamard.*` key is written. `general.merged_loras` is written only with `--lora`.
- **Parallel rotation:** a tensor rotated with 1 thread and with N threads gives identical bytes.
- **Unsupported dimension (§10):** with a small synthetic GGUF containing a tensor of width
  64·65, `--hadamard` keeps its base type and warns once per width, and
  `--tensor-type …=hq4_K` for that tensor throws.
- **Missing seed:** a file with HQ types and no `hadamard.seed` is refused at load and when
  requantizing, with the ConvRot hint (§1.3).

**Integration (Phases 4–5, `tools/quantize/tests.sh`, Qwen3-0.6B):**
- **Quantize and generate:** `--hadamard` with Q4_K_M, Q5_K_M, IQ4_XS, IQ3_S, IQ3_XS (an ftype mix)
  and IQ2_XXS (with no imatrix), plus generation from each.
- **imatrix:** `--imatrix` with `--hadamard` gives the one-per-run warning, and the HQ tensors are
  byte-identical to a run without it.
- **Refused:**
  - a rotated `token_embd`, by the guard (a `GET_ROWS` on a rotated weight)
  - a file with `hadamard.seed` removed
  - requantizing with a `--hadamard-seed` that differs from the file's seed
- **Requantize:**
  - HQ5_K → HQ4_XS gives perplexity close to a direct HQ4_XS quantization.
  - A partially rotated file, requantized with `--hadamard`, keeps the source seed.
- **Runtime LoRA:** a small adapter on HQ4_K and HQ4_XS models. The KL of `HQ + LoRA` against
  `f16 + LoRA` must be in line with `HQ` against `f16`. Include an adapter targeting `output` and,
  if available, one targeting MoE experts.
- **`--lora` merge:**
  - An f16 source + `--lora` + `--hadamard Q4_K_M` matches `f16 + runtime LoRA` about as closely
    as `HQ + runtime LoRA` does.
  - An HQ source + `--lora` matches the f16-source merge within quantization noise.
  - A non-HQ target merges correctly.
  - Refused: a different architecture, an A-LoRA adapter, and an adapter tensor with no model
    tensor.

**Not testable on this machine (listed so it isn't forgotten):**
- **Vulkan and Metal:** the §5.4 refusal on a real device. The code path is the one the RPC test
  covers.
- **HIP/ROCm and MUSA:** the RHT kernels and the Q8_1 emission. Both load (§5.4) but are marked
  unverified.

Each is marked "unverified" in the README until it's run on real hardware.

**Quality (`tests/test-hadamard-ppl.cpp`, wikitext-2, CPU and CUDA):** perplexity, KL divergence
against f16 and top-token agreement. Differences can be around 0.05 ppl, which is why KL is
included. The conclusions come from Qwen3-4B, whose widths use orders 20 and 76. Qwen3-0.6B runs
first as a quick check, but its only non-power-of-2 order is 12.

| base | unrotated | ConvRot g256 (existing measurements) | full-row RHT |
|---|---|---|---|
| Q4_K / Q5_K | ✓ | ✓ | ✓ |
| IQ4_XS / IQ4_NL | without and with imatrix | — | ✓ |
| IQ3_S / IQ3_XXS | without and with imatrix | — | ✓ |
| IQ2_S / IQ2_XS / IQ2_XXS | with imatrix (required) | — | ✓ |

The fork has no imatrix tool. The unrotated runs' imatrix comes from upstream llama.cpp's
`llama-imatrix` on the f16 model, with wikitext-2's training split so it doesn't overlap the
evaluation text. The fork's `common/imatrix-loader.cpp` reads it (legacy `.dat` and GGUF).

This measures four things:
- full-row vs 256-wide rotation
- rotation with nearest-neighbour rounding vs no rotation with an imatrix
- the HQ4_K/HQ5_K rounding A/B test (§8.4)
- **the quality you get at a given file size.** Every run above, HQ and unrotated, is also
  reported as KL divergence and perplexity against bits per weight (file size ÷ parameter
  count), as one table sorted by bits per weight plus a plot.
  - The main question is HQ4_XS (4.25 bpw) against HQ4_K (4.5 bpw). After rotation the weights
    are symmetric around zero, so Q4_K's per-block offset (about 0.25 bpw) buys little, while
    IQ4_XS's non-uniform grid suits bell-shaped values. HQ4_XS may match HQ4_K at about 6% less
    size.
  - The single-type rows need no extra runs; they're the same measurements, viewed by size. The
    view also adds a few ftype mixes, because they're what users actually run: `--hadamard Q4_K_M`,
    `IQ4_XS`, `IQ3_XS` and `IQ2_XXS`, next to their unrotated versions. That's 4 × 2 more
    quantize-and-measure runs on Qwen3-4B.
  - The result decides which `--hadamard` recipes the README recommends (Phase 8).

**Expected:** at 4 bits, HQ is competitive. At 2–3 bits, HQ with nearest-neighbour rounding will
probably lose to the unrotated IQ types with an imatrix. QuIP# and QuaRot reach low bit widths with
LDLQ/GPTQ error feedback, which this plan doesn't include. That outcome points to the rotated-space
imatrix (Future work), not to a bug.

**Throughput:** a 9275-token prompt plus 512 decoded tokens, median of 5, comparing HQ with
unrotated for decode and prompt processing separately, on Qwen3-4B and the 27B.
- **Per-kernel times:** `ncu` over a few decode steps and one prompt batch, filtered to the RHT
  and `quantize_*` kernels. Use the CUDA 12.9 build,
  `/usr/local/cuda-12.9/nsight-compute-2025.2.1/ncu`, with
  `--cache-control none --clock-control none`; `/usr/bin/ncu` 2022.4 doesn't support sm_120.
  `nsys` records no kernel data under WSL2.
- **Kernel count:** the same run confirms the Q8_1 emission. An HQ decode step launches no
  `quantize_q8_1` for rotated inputs.

## Phase 7 — Tune the RHT kernels

Start once the Phase 6 CUDA tests pass. `bench-rht` checks every variant against `ggml_rht_ref` on
every run. The `test-backend-ops` RHT and Q8_1 cases are re-run before a change is kept.

**7.0 Tooling** (built in Phase 5, alongside the kernels):
- **`tests/bench-rht.cu`**, Makefile target `bench-rht`. It builds with plain
  `nvcc -O3 -std=c++17 -arch=native` in a few seconds, includes `rht.cuh` and links
  `ggml-hadamard.c`. For each kernel variant and shape it:
  - checks the output against `ggml_rht_ref` (max error relative to max |y|)
  - times decode as 64 dependent launches captured in one CUDA graph, in µs per rotation, which
    is how decode runs
  - times prompt processing as 512 rows cycling through buffer sets larger than L2 (96 MB on the
    5090), in µs and GB/s against a copy kernel that moves the same bytes
  - prints CSV, and with `--once` launches each variant once, for `ncu`
- **Variants are template parameters** (`E_MAX`, `W`, `G`, `S`, rows per CTA, kernel A or B), so
  one build sweeps them. Only the winners are instantiated in `rht.cu`.
- **Profiler:** the CUDA 12.9 `ncu` (see Throughput in Phase 6). The useful metrics are the
  `smsp__average_warps_issue_stalled_*` stall reasons (they diagnosed the cluster prototype),
  `smsp__inst_executed` and registers.
- **Static checks:** `-Xptxas -v` for registers and spills. `cuobjdump -sass` instruction counts
  show whether the signs folded: an unrolled tile should be `K·G` FADDs and no sign loads.
- **Makefile dependencies (fix in Phase 5):** the per-file CUDA rule (`Makefile:270`) depends
  only on `ggml.h`, `ggml-common.h` and `common.cuh`. Editing `rht.cuh`, the tables header or
  the sign-mixer header wouldn't rebuild `rht.o`, so stale code would get timed. Add those headers
  to the rule, or generate dependencies with `-MMD -MP`.
- **End to end:** decode and prompt tokens/s for HQ against the base type, on Qwen3-4B for every
  step and on the 27B once per milestone.

**7.1 Baseline.**
- For every `(K, P)` in §11 and Qwen3-4B, at rows = 1, 8 and 512, record µs, GB/s, registers,
  spills and the top stall reasons.
- Per model, record rotation µs per layer against the layer's GEMV time (weight bytes ÷ measured
  bandwidth).

**7.2 Experiments**, roughly in order of expected gain:
1. **Kernel A/B crossover** by rows and `(n, K)`. Kernel B with one CTA may beat kernel A for
   small `K` in decode, because it's a single launch.
2. **Kernel A:** `G` (smaller tiles mean more warps and less serial work per warp), warps per CTA
   in both passes, and chunks per warp in pass 1.
3. **Kernel B:** `W`, `S` (slabs in flight), `G`, and registers kept at 64 or fewer so two CTAs
   fit per SM.
4. **Cluster kernel A (sm_90+):** one kernel, with chunks exchanged through distributed shared
   memory. The prototype was 1.9 µs at K = 20 but 11–14 µs at K = 68. It stalled on the cluster
   barriers' fences and on DSMEM loads through generic addresses, which go through the load/store
   queue (`lg_throttle`). Retry with `ld.shared::cluster` on `mapa` addresses and a single
   arrive/wait.
5. **Faster K-generic mix,** then shrink the unrolled list if the generic mix gets close. Two
   routes, for K ≥ 100 where the adds rather than memory are the limit:
   - ±1 values held in uniform registers (the tile index is warp-uniform)
   - a tensor-core mix: `Ĥ_K` is exact in BF16, and the F32 input needs a hi/lo split
6. **`RMS_NORM → MUL → RHT`** through `ggml_cuda_can_fuse`. In decode the norm already holds the
   row in one CTA, so kernel B's stages can run in the same kernel. That removes one launch per
   hidden-size rotation.
7. **Q8_1 emission for `MUL_MAT_ID`** (MoE), on the MMVQ id path.
8. **Unrolled list vs PTX JIT time:** measure first-load time on a portable build with
   `CUDA_CACHE_DISABLE=1`, and the binary size.

**Targets and stopping.** Budget: about a week. Stop when the targets are met, or when a step
gains < 5%.
- **Decode** (rows = 1, in a CUDA graph): ≤ 2.5 µs per rotation for n ≤ 6144, and ≤ 3 µs at
  n = 17408.
- **Prompt processing** (512 rows): ≥ 80% of copy bandwidth for K ≤ 100.
- **End to end:** HQ within 3% of the base type for decode and 2% for prompt processing, on
  Qwen3-4B and the 27B.

**Guards:** no shape gets slower than the Phase 5 baseline, and every instantiated variant has its
own `test-backend-ops` case.

**7.3 Record** in this plan:
- per shape: the chosen kernel and parameters, with µs before and after
- the unrolled list
- PTX JIT time and binary size
- what was rejected, and why

## Phase 8 — Docs

Rewrite the rotation section of `tools/quantize/README.md`:
- the RHT description and the HQ type table (base type and rotation for each)
- nearest-neighbour semantics, and that the imatrix is ignored
- `--hadamard-seed`
- `--lora` merging, and that merging into a quantized source quantizes twice
- that ConvRot files no longer load (requantize them)
- that HQ models run on the CPU and the CUDA backend only (CUDA, and ROCm/MUSA, which are
  unverified), and are refused at load with any other device in use (§5.4). On a Vulkan or Metal
  build, run them CPU-only.
- the `hadamard.*` metadata
- the measured tables, and the recommended `--hadamard` recipes from the equal-size comparison
  (Phase 6)

The "files are fork-only" warning stays.

---

## Implementation record

### Deviations from the plan
- **Tests are split by what they link:** `tests/test-hadamard.cpp` (ggml only: table, rule, signs,
  reference, stage split, types, row validation, the CPU op), `tests/test-hadamard-quants.cpp`
  (the HQ quantizers), `tests/test-hadamard-quantize.cpp` (the quantizer tool, through
  `llama_model_quantize`), `tests/test-hadamard-llama.cpp` (+ `-cuda`: loading, saving, device
  policy, graph, guard), `tests/test-hadamard-ppl.cpp` (+ `-cuda`: PPL, KL against a cached FP16
  reference, top-1, bpw), and `tools/quantize/tests-hq.sh` (integration; `CUDA=1` for the GPU run).
- **Device-policy test:** the RPC server runs in-process (`ggml_backend_rpc_start_server` on a
  thread) instead of a CPU `rpc-server` binary.
- **Graph access to the model:** `llm_graph_params` / `llm_graph_context` carry `hq_model`, a pointer
  to the model, for `is_rotated` and the seed.
- **§8.3:** no exact-NN option. Instead every grid type (HQ2_*, HQ3_*) gets a final pass that
  re-picks each group's code exhaustively over the decode grid at the *stored* scale, only when it
  lowers that group's error (see §8.3 results below).
- **§5 / §5.2 CUDA:** the tables header is included once, as the host `constexpr` copy; both the
  unrolled and the K-generic mixes read one bit-packed `__device__ const` table (87 KB). Unrolled
  mixes use constant indices, which fold away (K = 172 compiles to 29 308 FADDs, no table loads).
  A lane holds elements `32e + lane` (scalar coalesced loads, no row-stride alignment needed).
  Unrolled tiles are one template instance per output tile. The K-generic mix reads its sign words
  from global memory (L1), not shared memory. Rows the in-warp split can't take (K = 1 above 2¹⁵,
  K > 1 with P > 1024) use extra strided passes between kernel A's passes.
- **Q8_1 emission** is from kernel A only (≤ 8 rows, plus widths kernel B can't take). Kernel B
  (prompt processing) writes F32: its Q8_1 epilogue spilled past the 64-register budget and made
  emission slower than F32 plus a separate quantize. `GGML_CUDA_RHT_F32=1` turns emission off alone.
  The layout decision reuses the path predicate factored out of `ggml_cuda_mul_mat`
  (`ggml_cuda_mul_mat_path`); the Q8_1 block packing is shared with `quantize.cu` (`quantize.cuh`).
- **Makefile:** the per-file CUDA rule now generates dependencies (`-MMD -MP`) instead of listing
  headers.
- **koboldcpp.py:** `dump_gguf_metadata` names the HQ types.
- **`tensor_type_fallback`** handles an explicit HQ target of an incompatible width by falling back
  like its base type (to the fallback's HQ variant when there is one).
- **Hadamard sources** (verified 2026-09-24; every block byte-identical to its file):
  `code_gen.py` (Dao-AILab/fast-hadamard-transform master)
  10f349a305808b3bd935b64ed4fab3aad13cc1c3b9096e36032a27e5d5ac1206; Sloane files:
  - `had.20.will.txt` 34822da355777eb508e7bd38cf855a0ec964428f6572cd160421820185c10138
  - `had.28.will.txt` 70f6ddeee69365097964236477424ac52713f60d0ad7be4402b5d86a6feb650f
  - `had.36.pal2.txt` 31180207ac6916ae73fb94e9d1b59c2151dd3e68226b87d609c13466e1fc405d
  - `had.40.tpal.txt` d53d1b6efa05eaf2d9fa0f589274eb4cd23d747d2d3d8aae9b1677e7ce7061da
  - `had.44.pal.txt` 537abf889a1d9416cec210a6c1a315462a02a35f03b8df676b3bd6a4b42b32a2
  - `had.52.will.txt` b09945ddadc1d9ccf7a871bc350808ed6551e2de13523a15099457b432c89512
  - `had.60.pal.txt` eb568e43184ab1933e335a769ed01c130ef99eac3a08f420e256e508758a873c
  - `had.68.pal.txt` 7b250339aff3113a58f6cae8c89b47b365be16e51776795373dd50b5d4632700
  - `had.76.pal2.txt` 5c26648d5f36e166e94470d2e18a0e57eada5cc4dbd5a5bf0143f794ce56ddb7
  - `had.84.pal.txt` 15305511924de3f2907c65178830ab5aec04944b028104ee247324d846b5680f
  - `had.92.will.txt` 0ae509b1d7ad87c71d9fccb434c85d24c7bef601568a32300639babc97f51611
  - `had.100.will.txt` 2b0c1b3e0315f54a079188d4da86dff6c249eb63efc3e12448d0e57ddcd73801
  - `had.108.pal.txt` bcc8faf141eba6d59f26a0159b27c80f3d8e0f0bf87816f93d90977135fbde2a
  - `had.116.will.txt` c819980c357390b3d0705baaaa36850e1e5c2016c2fbc637743ebde77775d2d9
  - `had.124.pal2.txt` 210abd4db59a18f52cfd710f66819fe56f6f4141886c861e8e30700cb08f0733
  - `had.132.pal.txt` 9cadd54a150e8f123dc1474143c1c2c8754f845dd91f5f44b55168c78159db1a
  - `had.140.pal.txt` 4d87e06ea4df3e910ca688d47cca8d799fc4bb196ac021ee6e15b22a06baefe1
  - `had.148.pal2.txt` 76c7374d95facd930e6c92edaadf954d471d13d4c412cffc40260a000125c95d
  - `had.156.will.txt` 4d2f4179cd753e6438a96c6d1c51bae8b9abca7f02ab662a7ab8b66a6e4b5543
  - `had.164.pal.txt` 83f4fe5e6181a4ff3f8f019532db0737b031f3a04be0bd4547c0e5e1aa12f7ce
  - `had.172.will.txt` f235a78cccb5a6ce1acc4208275dda1ef0040e05788b05c97a43feaa3555e862
  - `had.180.pal.txt` 8afc37534c4fb0b9d7015457c1a95040aa57119b19cf5b82efa5bee817036232
  - `had.188.tur.txt` cfa9e9b8f4a61d8275e7bfa8d25dbbfd21cebdb218b5c715c49fc40270cdd107
  - `had.196.pal2.txt` dbd2c4041f7d81530d8f4de3b946c31fbc5b18a2a4c2eeffacff79237fc9c531
  - `had.204.pal2.txt` 6d81eff9a2e273192657210c6e5394502fddf7e164bee145cc784bb73350d67c
  - `had.212.pal.txt` 096361b90f631a1a0828847735d342f16f70b9d8a714f02d2853e4309611135d
  - `had.220.pal2.txt` a81ea6e46d1a3c4b62598f0f60ee7bf308e29a57de83af3c4c1f778f3277f87a
  - `had.228.pal.txt` 445e678e0e7152f4fa9b4111b7da5e4cf0b14be0364428d90891ae8d5c3a8ea2
  - `had.236.od.txt` b28c3992a4d78dc68d1017c331b580972811b885d92c11a814f671205cf16ad8
  - `had.244.will.txt` 38335c91929e3845c32eaeea6aeb45dcd34ac58fbb6c988bb80ddf66eee43946
  - `had.252.pal.txt` 5aaf5c3169df4db9007b697e6cb03892aab705c4317160515baaa87018172e42
  - `code_gen.py` 10f349a305808b3bd935b64ed4fab3aad13cc1c3b9096e36032a27e5d5ac1206

### §8.3 (measured): neighbour list vs exhaustive search
256 × 5120 rotated Gaussian rows. The quantizers' own lattice search disagrees with an exhaustive
search in at most 0.22 % of groups (HQ2_S), with at most 0.1 % MSE gap, so no `--hq-exact-nn`. But
re-picking the codes at the stored (re-rounded, fudged) scale over the actual decode grid gained
0.3–3.5 % MSE (HQ3_S 3.5 %, HQ2_XXS 1.4 %, HQ3_XXS 0.9 %), for 6–18 % more quantization time; it
is in. HQ4_NL/XS and HQ4_K/HQ5_K have no such mismatch.

### §8.4 (measured): HQ4_K / HQ5_K rounding
Qwen3-4B, `--pure`, KL against BF16: HQ4_K uniform 0.0844, `sqrt(σ²+x²)` 0.0845, `_ref` 0.0946;
HQ5_K 0.0240 / 0.0236 / 0.0233 (within noise). Uniform weights stay.

### Phase 6 (measured): quality
Qwen3-4B, wikitext-2 test, 80 × 512 windows (20 400 scored tokens), KL against BF16; the imatrix
is upstream `llama-imatrix` on wiki.train (100 chunks). Single types: `--pure --token-embedding-type
q6_K`.

| type | plain | imatrix | HQ (uniform) |
|---|---|---|---|
| Q5_K | 0.0286 | 0.0170 | 0.0240 |
| Q4_K | 0.0966 | 0.0523 | 0.0844 |
| IQ4_NL | 0.0812 | 0.0615 | 0.1036 |
| IQ4_XS | 0.0814 | 0.0611 | 0.1101 |
| IQ3_S | 0.6182 | 0.1663 | 0.2649 |
| IQ3_XXS | — | 0.2689 | 0.5145 |
| IQ2_S | — | 0.7201 | 1.1914 |
| IQ2_XS | — | 0.9122 | 3.0025 |
| IQ2_XXS | — | 1.4076 | 3.3513 |

Mixes (plain / imatrix / `--hadamard`): Q4_K_M 0.0690 / 0.0451 / 0.0561; IQ4_XS 0.0739 / 0.0578 /
0.0940; IQ3_XS — / 0.2092 / 0.3168; IQ2_XXS — / 1.1819 / 2.5960.

- **HQ loses to the imatrix quants at every width**, as expected at 2–3 bits but not at 4–5.
- **HQ4_XS / HQ4_NL even lose to plain IQ4** (no imatrix), while HQ4_K / HQ5_K / HQ3_S beat their
  plain types. HQ4_XS ≈ HQ4_K did not happen; once HQ errors are isotropic, KL follows weight MSE and
  HQ4_K has more bits.
- **Cause (investigated, `build-hq/investigate/`):** not activation quantization (dequantized F32
  runs give the same KL). A few massive-activation channels (attention sinks: e.g. 4 channels at
  ±1000–5572 carrying 99 % of an `ffn_down` input's energy) dominate KL. Plain IQ4 happens to be
  2× more accurate than HQ in exactly those weight columns (near-zero weights plus block maxima,
  which IQ4's grid and x² weighting favour); the rotation spreads those directions over every
  coordinate, where uniform rounding gives them average accuracy. Swapping two `ffn_down` tensors
  moves most of the IQ4_XS gap. KL tracks layer-output error (Spearman 0.94) and sink-token error,
  not weight MSE.
- **Rotated-space imatrix barely helps** (IQ4_XS 0.1101 → 0.1015): an axis direction `R·e_k` has
  magnitude `1/√n` in every rotated coordinate, so no per-coordinate weight can protect it.
- **GPTQ in the rotated space with `H = R·diag(v)·Rᵀ`** (`v` = the ordinary imatrix; `H⁻¹ =
  R(D + λI)⁻¹Rᵀ` in closed form, λ = 1 % of mean(v)), prototyped for the element-wise grids: HQ4_XS
  0.0383, HQ4_NL 0.0384, HQ4_K 0.0339, HQ5_K 0.0111 — **32–38 % below base + imatrix**. With the
  full activation covariance HQ4_XS reaches 0.0278 (plain + GPTQ: 0.0341), so the rotation adds value
  once the quantizer is covariance-aware. The file format and inference don't change.

### Phase 7 (measured): kernel tuning
RTX 5090. Decode = µs per rotation, 64 dependent launches in one CUDA graph; 512 rows = % of a copy
kernel moving the same bytes. Kernel A for ≤ 8 rows; at 512 rows kernel B, except K = 1 rows up to
16384 wide, which stay on kernel A (`rht_cfg::k1_a_max`).

| n (K, P) | 1 row: stage 2 → final | 8 rows | 512 rows |
|---|---|---|---|
| 2560 (20, 128) | 1.67 → 1.69 | 1.86 → 1.87 | 69 % → 72 % |
| 4096 (1, 4096) | 2.28 → 2.31 | 2.41 → 2.40 | 62 % → 74 % (kernel A) |
| 9728 (76, 128) | 3.27 → 2.41 | 3.94 → 3.97 | 75 % → 74 % |
| 5120 (20, 256) | 1.79 → 1.79 | 1.95 → 1.96 | 90 % |
| 6144 (12, 512) | 1.79 → 1.80 | 2.02 → 2.02 | 101 % |
| 17408 (68, 256) | 3.04 → 2.59 | 5.34 → 4.84 | 63 % → 79 % |
| 11008 (172, 64) | 5.81 → 3.68 | 8.51 → 7.78 | 33 % → 35 % |
| 3072 (12, 256) | 1.61 → 1.62 | 1.76 → 1.76 | 79 % |

Also at 1 row: 18944 (K = 148) 5.99 → 3.64, 13824 (108) 4.69 → 3.05, 25600 (100) 4.28 → 3.18,
21504 (84) 4.39 → 2.93 µs. At 512 rows ≥ 80 % of copy for 5120, 6144, 8192, 10240, 12288, 14336;
short for 2560, 4096, 9728, 21504 (72–74 %), 5376 (58 %), 25600 (47 %).

- **Kept:** kernel A pass 2 splits each tile's K-sum over 4 warps for K ≥ 68 (compile-time split
  points, partial sums added in a fixed order), jobs ordered tile-slowest so warps share the
  unrolled code, an unsplit K = 76 instance above 2 rows; kernel B picks 8/16/32 warps by shape and
  compiles only the chunk widths an order can reach.
- **Q8_1 emission policy** (`ggml_cuda_rht_write_q8_1`): ≥ 2 consumers always; 1 consumer only from
  kernel A and only when K < 64 (or MMVQ and K < 100); kernel B emits through its F32 scratch plus
  one quantize pass. The planner bounds MMQ's `J_max` by `min(ne11, 512)` (2.66 ms → 68 µs per graph
  evaluation).
- **Rejected:** persistent kernel-B CTAs (row spilled to local memory, 4× slower), a global 32-warp
  kernel B (hurts 9728, 5376), a runtime K-split (defeats specialisation), a 2-way split (no gain),
  the row-slowest job order (11008 at 8 rows 8.5 → 14 µs).
- **Unrolled list** stays {12, 20, 28, 68, 76, 84, 100, 108, 148, 172}: the generic mix is 1.2–2×
  slower on them. `rht.o` (sm_120) is 2.76 MB / 624k SASS instructions; PTX 17 MB; a PTX-only build's
  first load takes 18–21 s with `CUDA_CACHE_DISABLE=1` (0.28 s native). Dropping 148 and 172 would
  remove ~35 % of the SASS at the cost of the Llama-2 and Qwen2.5-7B FFN widths.
- **Not done:** `RMS_NORM → MUL → RHT` fusion and `MUL_MAT_ID` emission. Kernel B at 512 rows is
  instruction-fetch bound (48 % `no_instruction` stalls) for small and medium widths.

**End to end** (idle machine, 9285-token prompt + 512 generated, median of 5, t/s):

| model | build | prompt: base → HQ | decode: base → HQ |
|---|---|---|---|
| Qwen3-4B pure Q4_K | Makefile (no CUDA graphs) | 16032 → 14680 (−8.4 %) | 142.8 → 133.0 (−6.8 %) |
| Qwen3-4B Q4_K_M (imatrix vs `--hadamard`) | Makefile | 15380 → 14449 (−6.1 %) | 148.1 → 126.6 (−14.5 %) |
| Qwen3.8-27B Q4_K_M | Makefile | 3321 → 3171 (−4.5 %) | 47.7 → 45.2 (−5.2 %) |
| Qwen3-4B pure Q4_K | + `GGML_CUDA_USE_GRAPHS` | 15392 → 14661 (−4.7 %) | 235.6 → 209.7 (−11.0 %) |
| Qwen3-4B Q4_K_M | + `GGML_CUDA_USE_GRAPHS` | 15351 → 14463 (−5.8 %) | 223.7 → 208.6 (−6.7 %) |
| Qwen3.8-27B Q4_K_M | + `GGML_CUDA_USE_GRAPHS` | 3262 → 3167 (−2.9 %) | 69.0 → 65.8 (−4.6 %) |

CPU (8 threads, `-march=native`, Qwen3-4B pure Q4_K, 500-token prompt + 64 generated): prompt 252 →
239 t/s, generation 18.8 → 17.8 t/s (about −5 % each, noisy). Getting there took three changes to the
CPU op: `ggml-hadamard.c` built per arch like `ggml-quants.c` (it had no SIMD flags), decode rows
split out of place through the op's work buffer over (tile × output group) instead of in place over
tiles only, and K = 1 rows split as `H_P1 ⊗ H_P2` (`ggml_rht_stage_fwht_outer`); the mix keeps its
32-wide accumulators in vector registers (GCC/Clang vector extensions). One decode row at n = 4096
went from 38.5 to 2 µs, n = 9728 from 39 to 11 µs; all splits stay bitwise equal to `ggml_rht_ref`.

The 3 % decode / 2 % prompt targets are not met. Decode pays ~3.7 µs per rotation (two kernels plus
graph gaps) × 144 rotations per token on Qwen3-4B; the norm fusion is the next lever. The Makefile
didn't enable CUDA graphs at the time of these measurements (the CMake build did for CUDA 12), which
cost both models far more than the rotation does; it now defines `GGML_CUDA_USE_GRAPHS` for CUDA 12+
(`LLAMA_CUDA_NO_GRAPHS=1` turns it off), and CUDA objects rebuild when the CUDA flags change.

## Future work

- **GPTQ/LDLQ error feedback in the rotated space (done 2026-09-26, see
  `plans/gptq_rotated_quantizer_plan.md`):** HQ quantizers take
  `--imatrix` again, as a diagonal Hessian `H = R·diag(v)·Rᵀ` (Phase 6 results above). Needs
  `U = chol(H⁻¹)` once per distinct GEMM input per layer (O(n³); n = 17408 needs a blocked
  CPU/GPU Cholesky), a per-row sequential encoder with the existing scale search at each
  super-block, and a group-wise (vector) variant for the IQ2/IQ3 grids. Evaluate changes by KL or
  sink-token output error, not weight MSE.
- **Rotated-space imatrix:** measured at only −8 % KL for HQ4_XS; superseded by the item above.
- **Merge the KV-cache FWHT into `GGML_OP_RHT`:** the KV cache's hinted `mul_mat` + `fwht.cu` is
  the `K = 1`, no-signs case. It needs a "no signs" flag in `op_params`, and F16/BF16 inputs
  (converted to F32 on load, F32 compute). It leaves one Hadamard subsystem instead of two,
  removing `llama_gen_hadamard_matrix` / `llama_hadamard_inplace`, the FWHT hint and `fwht.cu`.
- **More backends,** each lifting the §5.4 refusal for itself. Because HQ tensors load as base
  types, a backend only needs `GGML_OP_RHT` (and passing its `test-backend-ops` cases); no type
  remaps.
  - **Vulkan:** start from the existing KV-cache FWHT shader (`ggml_vk_fwht_pipeline_idx`,
    64–512). It needs a cross-subgroup exchange for larger `P`, the in-register signs, and the
    order-K mix from the same tables header.
  - **Metal**, and the others (SYCL, OpenCL), the same way.
- **Rotate the residual stream (QuaRot-style) where the architecture allows it.** Fold each
  norm's scale into the following weights and rotate `token_embd` and the residual stream by `R`.
  Then the QKV, up/gate and output inputs are already rotated and need no runtime RHT; only
  `attn_output` and `ffn_down` do, which halves the rotations per layer.
  - It doesn't work with post-norms (Gemma's sandwich norms).
  - It needs every consumer of the residual rotated, including F32 routers, which have no HQ
    type. The file would list its rotated tensors in metadata; the runtime rotated set (Phase 2)
    already works for any type. It also conflicts with "`token_embd` is never rotated".
  - The guard would need to accept an input that's already in the rotated space.
- **`K_MAX` beyond 256,** for odd parts of 64 or more. Sloane's library stops at 256 (apart from
  428), so other sources are needed. The mix cost grows as `n·K`, so at K = 508 prompt processing
  would be limited by the adds rather than memory (§2).
- **ROCm and MUSA verification on real hardware:** run the RHT and Q8_1 `test-backend-ops` cases
  and a perplexity check, then drop "unverified" from the README.
- **More base types** (Q2_K/Q3_K/Q6_K, IQ1).
- **Perplexity/KL and throughput on Qwen3.8-27B** (HQ4_K + HQ4_XS).

## Key file index

| Item | Location |
|---|---|
| Hadamard matrices (the only copy) | `ggml/src/ggml-hadamard-tables.h` |
| RHT definition (rule, PRNG, table accessor, reference) | `ggml/src/ggml-hadamard.c`, declared in `ggml/include/ggml.h`; sign mixer and tables include in the internal `ggml/src/ggml-hadamard.h` |
| RHT op: CUDA kernels / CPU forward | `ggml/src/ggml-cuda/rht.cu`, `rht.cuh` / `ggml/src/ggml-cpu/ops.cpp` |
| Q8_1 emission hooks | `ggml/src/ggml-cuda/ggml-cuda.cu` (graph evaluation), `mmvq.cu:1327–1332`, `mmq.cu:139–159`, packing shared with `quantize.cu` |
| Kernel bench / backend tests | `tests/bench-rht.cu` / `tests/test-backend-ops.cpp` (from upstream `511f9c137`) |
| Demotion to base types, rotated set | `src/llama-model-loader.cpp` (`create_tensor`, `:1076`; `t_meta` at `:1305`, `:1262`) |
| Device policy (CPU + CUDA backend), rotated set on the model | `src/llama-model.cpp` (`load_tensors`) |
| Saving HQ types and `hadamard.seed` from memory | `src/llama-model-saver.cpp` (`add_tensors_from_model`, `add_tensor` at `:143`) |
| HQ quantizers | `ggml/src/ggml-quants-hq.c`, internal declarations in `ggml/src/ggml-quants.h` |
| Base quantizers (copied from, not modified) | `ggml/src/ggml-quants.c:3270–4360`, `iq4_nl_impl` at `:4966`, `q4_K` at `:1626`, `q5_K` at `:1851` |
| Type enum / pairs X-macro | `ggml/include/ggml.h:449`, `ggml/src/ggml.c:1400` |
| `type_traits` (HQ rows) / `type_traits_cpu` (no HQ rows) | `ggml/src/ggml.c:962`, `ggml/src/ggml-cpu/ggml-cpu.c:417` |
| ConvRot backend remaps to delete | `ggml/src/ggml-cuda/{convert,mmq,mmvq,ggml-cuda}.cu`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `ggml/src/ggml-metal/ggml-metal-{device,ops}.cpp`, `ggml/src/ggml-cpu/spacemit/ime.cpp` |
| `ggml_quantize_requires_imatrix` / `ggml_quantize_chunk` | `ggml/src/ggml.c:8022`, `:8030` |
| imatrix-required check / row rotation | `src/llama-quant.cpp:1165` / `:1388` |
| `rotate_input_if_rotated` / guard | `src/llama-graph.cpp:1543`, `:1569` |
| KV-cache FWHT (kept) | `src/llama-impl.h:57` (`llama_mul_mat_hadamard`), `ggml/src/ggml-cuda/fwht.cu`, Vulkan `ggml_vk_fwht_pipeline_idx` (`ggml-vulkan.cpp:10052`) |
| Portability idioms | `ggml/src/ggml-cuda/fwht.cu` |
| Dao upstream (matrix strings only) | `Dao-AILab/fast-hadamard-transform`: `csrc/code_gen.py` |
| Sloane Hadamard library | http://neilsloane.com/hadamard/ |

## Risks and sequencing

- **Order:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each step is checkable on its own:
  - Phase 0 is unit-tested on its own.
  - Types load after Phase 2.
  - The HQ quantizers are tested after Phase 3.
  - An HQ file exists after Phase 4.
  - Phase 5 builds the op's CPU forward first (a correct end-to-end path). Next come the
    `test-backend-ops` port and `bench-rht`, then the CUDA kernels (A, B, F32 epilogue) gated by
    the RHT cases, and last the Q8_1 emission gated by its graph cases.
  - Phase 6 establishes quality before Phase 7 tunes speed.
- **Riskiest:**
  - **The CUDA kernels must match `ggml_rht_ref` exactly:** element order, `Ĥ_K` orientation,
    signs before the transform, and the internal `H_P` split. A mismatch doesn't crash; it gives
    perplexity in the hundreds. The RHT backend tests must pass before any perplexity run.
  - **The Q8_1 emission:** a consumer that reads the node as F32 after it wrote Q8_1 gets
    garbage. The all-consumers rule and the fallback graph cases (Phase 6) guard it, and
    `GGML_CUDA_DISABLE_FUSION=1` turns it off for bisecting.
  - **Where the demotion happens.** It has to come before buffer selection (otherwise CPU repack
    and the buffer probes see HQ types), and never in the loader's metadata (otherwise
    `llama-quant` loses the HQ types). The Phase 2 loading tests and the Phase 4 requantize tests
    each catch one side.
- **Wrong-output-without-error hazards, each with a test:**
  - CPU and CUDA sign mixers diverging
  - `Ĥ_K` orientation (row-major vs transposed), or a transcription error in the table
  - signs applied after the transform instead of before
  - a requantize run mixing two seeds in one file
  - any path that writes into a rotated weight at runtime (the guard), or encodes HQ data without
    rotating (`from_float_ref = NULL`, no CPU traits)
  - a model saved from memory with rotated tensors labelled as base types, or without its seed
    (the saver restores both; the save round-trip test checks it)
- **Verification limits:** no Vulkan driver on this machine (WSL2), no Metal hardware, no AMD GPU.
