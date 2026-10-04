# Plan: inference speedups

## Summary

A first pass over the inference path on the RTX 5090 (WSL2), with Qwen3.8-27B (`qwen35`, hybrid:
48 gated-delta-net + 16 attention layers) and Qwen3-0.6B, each quantized from BF16 as plain
`Q4_K_M` and `--hadamard Q4_K_M` (no imatrix; speed only), found three groups of costs:

1. **koboldcpp's request loop.** On the hybrid model its SmartCache takes up to three ~390 MB state
   snapshots per request. Each spends 220–294 ms allocating the host buffer (zero-fill plus first-touch page
   faults) and 23–26 ms copying the state.
2. **HQ prompt processing** is 5.7 % slower than the base type on the 27B (pp512 3713 → 3501 t/s). About half
   of that is the producer kernels in front of the rotations: RMS norm and SWIGLU each write an F32 tensor
   that the RHT kernel reads straight back.
3. **HQ decode** is 3.5 % slower on the 27B (tg128 75.6 → 72.9 t/s) and 24 % slower on the 0.6B
   (670 → 512 t/s). There, writing Q8_1 from the rotation is itself the regression: with `GGML_CUDA_RHT_F32=1`
   the 0.6B does 593 t/s.

The plan fixes the snapshot allocation (K1), the one-warp Q8_1 epilogue and the K-split epilogue (D1, D2),
and fuses RMS norm and GLU into the RHT kernels (F1, F2). A draft had a fourth item, a new prompt kernel that
writes Q8_1. The audit and a prototype rejected it (see "Audit").

Two follow-ups came from questions after the implementation:
- **G1:** the GLU of a rotated gate is built in place, so upstream's gate/up/GLU decode fusion is no longer refused
  in 24 of the 27B's 64 layers.
- **M1:** each activation is rotated once. MoE models with a shared expert rotated the same input twice per layer,
  and the experts' GEMMs now read the rotation's Q8_1.

A third came with the 2026-10-04 upstream merge:
- **G2:** upstream's allocation dependencies (`add_alloc_dep` in `graph_optimize`) keep each fusion's inputs
  allocated. This replaces G1's in-place GLU and the F1/F2 aliasing rule, and also covers MoE experts and every
  fusable GLU op.

"Future work" has what's left, starting with rotated MoE routers, so the FFN norm can fuse in MoE layers too.

## Measurements (initial pass)

Tools:
- `llama-bench`, built from upstream's `tools/llama-bench/llama-bench.cpp` at `511f9c137` against the fork's
  CUDA objects. The fork ships only its `main.cpp` stub.
- `bench-rht` and `ncu`.
- Timing probes patched into `gpttype_adapter.cpp`, then reverted.

All of these live in the session scratch directory (see "Harness").

| model | pp512 plain → HQ | tg128 plain → HQ |
|---|---|---|
| 27B Q4_K_M | 3713 → 3501 (−5.7 %) | 75.6 → 72.9 (−3.5 %) |
| 0.6B Q4_K_M | ~40–46k (noisy) | 670 → 512 (−24 %); HQ with `GGML_CUDA_RHT_F32=1`: 593 |

Run-to-run noise is about ±3 % on pp512, so decisions use interleaved A/B runs and medians.

**koboldcpp, 27B plain.** Compared with `--benchmark` at ctx 4096 (3840-token prompt, 256 generated), llama-bench
at the same depth gets 3595 t/s prompt and 75.9 t/s generation. `--benchmark` gets 2531–3007 and 67.7–68.8.

Per generated token (the same for HQ):

| step | time |
|---|---|
| `llama_decode` returns (graph build + launch) | 0.57 ms |
| logits fetch waits for the GPU | 12.7 ms plain, 13.1 ms HQ |
| sampling | 0.39 ms |
| rest | 0.13 ms |

Per request there are three snapshots:

| snapshot | size | allocation + `llama_state_get_data` |
|---|---|---|
| lifeboat at 3072 tokens | 341 MB | 219 + 23 ms |
| before the last 32 prompt tokens | 387 MB | 268 + 26 ms |
| at generation start (same slot) | 389 MB | 294 + 26 ms |

The last one reallocates because the state grew by 2 MB. In a standalone test, one 390 MB buffer costs:

| how | time |
|---|---|
| `std::vector<uint8_t>(n)` (zero-fill + faults) | 120–280 ms |
| `new uint8_t[n]`, then a copy into it (faults) | 117 ms |
| a copy into an already-touched buffer | 19 ms |

`MADV_HUGEPAGE` doesn't help under WSL2.

**RHT kernels** (`bench-rht`, µs; 1 row = in a CUDA graph, 512 rows = prompt):

| n (K, P) | 1 row F32 | 1 row F32 + quantize | 1 row emit Q8_1 | 512 rows B | B + quantize (MMQ) |
|---|---|---|---|---|---|
| 1024 (1, 1024) | 1.83 | 2.50 | **5.77** | (A1) 4.21 | 5.68 |
| 2048 (1, 2048) | 2.14 | 2.85 | 2.40 | | |
| 3072 (12, 256) | 1.60 | 2.33 | 2.13 | 9.96 | 13.07 |
| 5120 (20, 256) | 2.06 | 2.88 | 2.62 | 14.78 | 18.89 |
| 6144 (12, 512) | 2.02 | 2.90 | 2.47 | 15.85 | 20.62 |
| 17408 (68, 256) | 2.97 | 3.85 | 4.08 | 59.13 | 70.25 |

- **1024-wide row** (K = 1, one pass): one warp holds 32 values per lane. Writing Q8_1 runs 32 block reductions
  of 10 shuffles each, one after another, in that warp (`rht_chunks`, `M == 1` branch).
- **Kernel A pass 2 with the K-split** (K ≥ 68): the first warp of each group of 4 runs the epilogue for all 4
  output chunks. So at 17408, writing Q8_1 (4.08 µs) is slower than F32 plus the consumer's quantize (3.85 µs).
- **Where the rotation inputs come from** (qwen35, per layer):
  - `attn_norm` (RMS_NORM, MUL) → RHT(5120), 4 consumers
  - `ssm_norm·silu(z)` or `attn·sigmoid(gate)` → RHT(6144), 1 consumer
  - `post_attention_norm` → RHT(5120), 2 consumers
  - SWIGLU(gate, up) → RHT(17408), 1 consumer
  - In decode the SWIGLU is already fused into the gate/up MMVQ. In prompt processing it is a separate kernel.
- **Graph order** (`GGML_SCHED_DEBUG=2`, 27B):
  - In the 16 attention layers and at the 64 post-attention norms, RMS_NORM, MUL, RHT are consecutive nodes.
  - In the 48 gated-delta-net layers they are not. `qwen35.cpp` expands `attn_norm` right after building it,
    and the RHT only enters the graph when a consumer does, after `build_rs`'s SCALE/GET_ROWS/CPY nodes
    (nodes 1–2, then 5–11, then 13).

**Prototypes** (session scratch, 512 rows; the build limited the unrolled orders to 12, 20, 68):

| n | norm + B | B with norm prologue | SWIGLU + B | B with SWIGLU prologue |
|---|---|---|---|---|
| 3072 | 23.1 | 15.1 | 21.2 | 15.7 |
| 5120 | 33.2 | 19.2 | 31.6 | 24.6 |
| 6144 | 38.5 | 29.2 | 36.3 | 32.9 |
| 17408 | 121.9 | 73.2 | 125.3 | 90.6 |

- The SWIGLU prologue is bitwise-identical to SWIGLU then B. The norm prologue differs by ≤ 5e-7 relative:
  the scale `1/rms` moves after the transform.
- The unfused "norm" and "SWIGLU" there are stand-in kernels. Phase 3 re-measures against the real ones.
- **D1 prototype:** the 1024-wide Q8_1 emission goes from 5.77 to 1.63 µs, faster than F32 output (1.76). All
  264 of bench-rht's Q8_1 cases stay byte-identical to `quantize.cu`.
- **Staged-row kernel ("C")**, the whole row in shared memory, then mix and epilogue:
  - F32 output is bitwise-equal to B and 0–15 % faster at most widths.
  - Q8_1 output is slower than B plus one quantize pass at 5120 (22.6 vs 22.6), 6144 and 17408 (86–92 vs 74).
    The per-block shuffle reductions of the epilogue dominate.

## Audit (of draft 1) and what changed

A separate agent audited draft 1 against the code, and the prototypes above checked its two performance
assumptions.

| finding | change |
|---|---|
| **Blocker:** a fused RHT reads producers' inputs while writing its output, and the allocator may place that output in an input's freed block. gate/up are freed as soon as the SWIGLU is allocated, so the 17408 RHT often sits on one of them. Unfused, that is harmless. | Every fusion checks memory ranges, allowing only exact per-row aliasing that the chosen kernel handles (F1/F2, "Aliasing"). |
| F1 cannot fire at the 48 gated-delta-net norms: the nodes aren't adjacent. | `rotate_input_if_rotated` expands the RHT into the graph when it creates it (F0). |
| P1 (a new kernel C) duplicates kernel B; its gain was overstated. | Dropped. The prototype's Q8_1 epilogue lost to B + quantize. Future work: a reduce-scatter epilogue. |
| The kernel choice must agree between the Q8_1 planner and the op, and would depend on the device's shared memory. | Moot without a new B mode. Prologues work in every shape of A and B, so fusion never depends on the kernel choice. |
| `ggml_cuda_can_fuse`'s generic RMS_NORM/MUL branch would accept a size-3 pattern with no RHT checks. The Q8_1 counter only counts the unfused path. | The matcher lives in `rht.cu` with its own checks. `ggml_cuda_try_fuse` gets a one-line hook ahead of the RMS_NORM fusions. Counters move into the op. |
| D1: static shared memory in `rht_chunks` would cost every instance occupancy. A serial per-lane sum breaks byte identity. | Dynamic shared memory, only for `M == 1` rows that write Q8_1. The lane sums in the exact xor-tree order of the quantize kernels (prototyped: byte-identical). |
| The tests would pass even if the fusion never fired. | Fused-launch counter, and test cases that must fuse and cases that must not. |
| K1: `reserve(2×)` commits memory on Windows. The old buffer is held during reallocation. A failed save leaves a non-empty buffer with size 0, and load then wipes the context. | Grow by 1.25× rounded up; free first; a slot is valid iff its size > 0. |
| Minor: D2 must keep the summation order. | Warp p sums `part_acc` for q = 0..3, in the same order as now. |

Not taken from the audit:
- The `{SIGMOID, MUL, RHT}` pattern: 16 sites, one launch each.
- A single-launch kernel A for decode. The cluster variant was measured at 11–14 µs at K = 68, and B at 1 row
  takes 5–14 µs.

## Items

| id | what | files | expected (27B unless noted) |
|---|---|---|---|
| K1 | Snapshot buffers without zero-fill, with growth headroom, freed before reallocation | `gpttype_adapter.cpp`, `otherarch/otherarch.h` | −190 to −270 ms per snapshot into a warm slot, −100 to −160 ms into a cold one |
| D1 | Lane-per-block Q8_1 epilogue for one-warp rows | `rht-impl.cuh` | 0.6B decode: ~+15 % |
| D2 | K-split epilogue spread over the split's warps; re-measure the one-consumer emission limits | `rht-impl.cuh`, `rht.cu` | ~1 µs per 17408 decode rotation |
| F0 | Expand each RHT into the graph where it is created | `src/llama-graph.cpp` | lets F1 fire at all 128 norms |
| F1 | RMS_NORM → MUL → RHT: the norm moves into the RHT kernels | `rht-impl.cuh`, `rht.cu`, `ggml-cuda.cu` (hook) | pp ~1.8 ms per ubatch; decode ~0.15 ms/token |
| F2 | GLU (SWIGLU, GEGLU) → RHT | same | pp ~2 ms per ubatch where the aliasing check passes (always since G2) |
| G1 | Follow-up: GLU of a rotated gate built in place over the gate's output, so upstream's gate/up/GLU MMVQ fusion isn't refused. Superseded by G2 | `ggml.h`, `ggml.c`, `src/llama-graph.{h,cpp}`, `rht.cu` | measured: decode fusion 40 → 64/64 layers, +0.3 % decode |
| G2 | Follow-up (2026-10-04): the fusions' inputs stay allocated through upstream's `graph_optimize` allocation dependencies; G1 and the F1/F2 aliasing rule removed | `ggml-cuda.cu`, `rht.{cu,cuh}`; reverts `ggml.{h,c}`, `src/llama-graph.{h,cpp}` | measured: G1's fusion counts, decode unchanged |
| M1 | Follow-up: one RHT per activation. The rotation cache looks through row-keeping reshapes; the Q8_1 planner and MMVQ let expert `MUL_MAT_ID`s read the RHT's Q8_1 | `src/llama-graph.cpp`, `ggml-cuda.cu`, `mmvq.cu` | measured on Qwen1.5-MoE: decode 943 → 847 kernels, ~+1.7 % decode and prompt |

Why these gains aren't simply additive:
- F1 and F2 are measured against the 8.4 ms HQ prompt overhead per ubatch. Together they may bring HQ prompt
  processing close to the base type.
- Decode keeps kernel A's two launches per rotation. That is the remaining gap.

Not pursued, and why:
- **Sampler speed:** 0.39 ms/token on a 248k vocabulary. It is upstream koboldcpp code, already bucket-sorted,
  and ~1–2 % on the 27B.
- **Fusing the gated MULs in front of the 6144 RHT:** at most one launch per layer.
- **Rotating the residual stream QuaRot-style:** it changes the file format and needs requantization.
- **The base engine's decode, at 65 % of weight bandwidth:** that is upstream's `qwen35` kernels and graph.

## Design

### K1. Snapshot buffers

- **Buffer type.** `savestate_data::current_savestate_buffer` and `current_draft_savestate_buffer` become
  `std::vector<uint8_t, kcpp_noinit_allocator<uint8_t>>`. The allocator's `construct(U *)` default-initializes,
  so `resize` doesn't write the bytes. The saved and restored bytes are unchanged.
- **Growing.** `gpttype_save_state_kv` works on the buffer's existing capacity. When that is too small it:
  - calls `clear()` + `shrink_to_fit()`, which frees the old memory first;
  - reserves `needed × 1.25`, rounded up to 16 MiB;
  - then resizes to `needed + 512`.

  So a slot re-saved after its state grew reuses its touched pages.
- **Validity.** A slot is valid iff `current_savestate_size > 0` (and `current_draft_savestate_size` for the
  draft):
  - `gpttype_load_state_kv` refuses a slot whose size is 0 *before* clearing the context;
  - `gpttype_clear_state_kv` keeps its `.empty()` test, which still means "never allocated".
- **No policy change.** Which snapshots are taken, and when, stays as it is. The file is upstream koboldcpp, so
  keep the diff small and upstreamable.

### D1. One-warp rows (`rht_chunks`, `M == 1`, Q8_1 output)

- **Transpose.** The warp writes its E values, scaled, into dynamic shared memory `[32][33]`. Lane `b < E` then
  quantizes block `b` alone.
- **Arithmetic.** Lane `b` uses each layout's own arithmetic and summation tree:
  - Q8_1: `d = amax/127`, `q = round(x/d)`, and the sum over the 32-lane xor tree;
  - MMQ D4/DS4: `d_inv = 127/amax` and the sum of 4-value groups over the 8-lane xor tree.

  So the bytes equal `quantize.cu`'s.
- **Launch.** `rht_launch` passes `a_warps·32·33·4` bytes only for that case; every other launch keeps 0.
  `rht_store_pad` is kept.

### D2. Kernel A K-split epilogue

- After the partial sums, warp `p` of a split group finishes output chunk `p` of the tile. It adds `part_acc`
  for `q = 0..KS−1` in order, so the result is bitwise-identical to today's, then runs its epilogue.
- `static_assert(GT == KS)` for the split instances.
- Then re-measure the one-consumer limits in `ggml_cuda_rht_write_q8_1` with bench-rht, and keep them only
  where writing Q8_1 wins.

### F0. RHT placement (`rotate_input_if_rotated`)

- **Change.** When it creates an RHT node, it calls `ggml_build_forward_expand(gf, cur_rot)`. The RHT then
  follows its input immediately:
  - if the input was expanded just before (qwen35's `attn_norm`), it lands right after the MUL;
  - otherwise the DFS adds the norm, the MUL and the RHT together.
- **Semantics.** Only node order changes. The RHT reads its input at the point where model code asked for the
  GEMM, not later.
- **Tests.** test-hadamard-archs checks every architecture; the 27B graph dump shows the new order.

### F1/F2. Prologue fusion

**Kernel side** (`rht-impl.cuh`):
- `rht_args` gets `pro`, a runtime switch: `NONE`, `NORM` (`x·w`, per-row scale `1/rms`), or `GLU` (`act(g)·u`
  with act = SiLU or GELU, via `ggml_cuda_op_silu_single` / `ggml_cuda_op_gelu_single` from `unary.cuh`).
  - It also gets a second source (pointer and row strides) and `eps`.
  - A template parameter would triple the instances; a warp-uniform switch in the load doesn't.
- **Kernel A:**
  - Pass 1 loads through the prologue.
  - For `NORM` it also writes each unit's sum of squares to `tmp[n·rows + row·units + unit]`. That area comes
    after the rows, so the strided passes don't touch it.
  - Pass 2 sums them in a fixed order: lane-strided, then `warp_reduce_sum`, identical in every warp.
  - The `M == 1` single pass sums inside its warp.
  - The RMS scale multiplies the store's scale (F32 and Q8_1 paths).
- **Kernel B:**
  - Stage 1 loads through the prologue.
  - For `NORM`, the sum of squares goes through a CTA reduction: `W` floats of shared memory, counted inside the
    48 KB budget so the dynamic part shrinks. The prototype hit "invalid argument" when static and dynamic
    shared memory together passed 48 KB.
  - The per-row scale is applied at the store.
  - With `via_f32`, B writes the pool temporary as today, then quantizes once.
- The GLU prologue produces exactly the GLU kernel's F32 values, so the rotation's F32 output is bitwise-identical.
  The norm prologue is within ~5e-7 relative.

**Matching** (`rht.cu`: `int ggml_cuda_rht_try_fuse(ctx, cgraph, i)`, which returns the number of nodes to skip).
`ggml_cuda_try_fuse` gets one hook, ahead of its RMS_NORM, GLU and UNARY fusions.
- **Norm pattern:** `{RMS_NORM, MUL, RHT}` at `i`, via `ggml_can_fuse` (consecutive, single use, not an output).
  - The MUL's other operand is contiguous F32 `[n,1,1,1]`.
  - `eps` comes from the RMS_NORM's op params.
  - The RMS_NORM input is F32 with contiguous rows.
  - The RHT's src is the MUL itself (not a CONT or view).
- **GLU pattern:** `{GLU, RHT}` at `i`, with op SWIGLU or GEGLU (not SWIGLU_OAI, GEGLU_ERF/QUICK or REGLU).
  - It handles both forms: split (`src1` given, own row strides) and single-tensor (halves, `swapped`).
  - Inputs are F32 with `nb[0] = 4` and uniform row strides.
- **Aliasing.** The launch reads the fused external inputs (RMS_NORM's `x`; GLU's `src0`/`src1`) and writes the
  RHT output. The fusion is allowed when either:
  - no input overlaps the output (`ggml_cuda_check_fusion_memory_ranges` with the RHT as output node; leaves are
    skipped as there); or
  - the output is written as F32 directly, and the only overlapping input has the output's data pointer and row
    strides. Every kernel reads a row completely before writing it:
    - B stages the row before its first barrier;
    - two-pass A reads everything in pass 1, and pass 2 writes after the grid dependency;
    - the `M == 1` warp loads its whole row first.

  Q8_1 written by kernel A directly has a different layout, so it needs no overlap. `via_f32` never writes the
  output while reading inputs. The check is made in `rht.cu`, which knows how the launch will write.

  *Superseded by G2:* the inputs are kept allocated until the RHT node, so the output never overlaps them.
- **Counters.** A fused-launch counter and the Q8_1 counter live in the op, both exported through
  `get_proc_address` like `ggml_backend_cuda_rht_q8_1_count`. `GGML_CUDA_DISABLE_FUSION=1` disables the fusions.

### G1. In-place GLU for rotated gates (follow-up)

*Superseded by G2 (2026-10-04); kept as the record of the problem.*

- **Problem.** The gate and up GEMMs read an RHT that the graph allocator frees once both are allocated, and the
  GLU output can land on it (24 of the 27B's 64 layers). Upstream's fused gate/up/GLU kernel reads that RHT's Q8_1
  in every block while other blocks write the GLU. `ggml_cuda_check_fusion_memory_ranges` therefore correctly
  refuses. A kernel-side fix would need a grid-wide barrier or a copy of the input, which costs about what the
  fusion saves.
- **Change.**
  - `ggml_glu_split_inplace` (new in ggml) gives the GLU's output as a view of its gate input, as the
    `ggml_*_inplace` unary ops do.
  - `build_ffn` builds the GLU through `build_glu_split`, which uses the in-place variant when the gate weight is
    rotated (`rotation_owner`, factored out of `rotate_input_if_rotated`).
  - This is safe: the gate output has no other consumer, it is allocated while the RHT is live (so it never
    overlaps it), and the fused kernel doesn't write it.
- **Scope.**
  - Rotated models only. They run only on the CPU and CUDA backends, whose GLU kernels are elementwise without
    restricted pointers; upstream's allocator doesn't treat GLU as in-place.
  - The split form only, with a contiguous gate.
  - `build_moe_ffn` is unchanged: no expert-path collisions were seen (M1's findings).
- **Knock-on.** `ggml_can_fuse` refuses a view as a fused intermediate, which would switch off {GLU, RHT} (F2).
  `rht_glu_can_fuse` accepts the case where the view's source is the GLU's own input and has no other consumer.

### M1. One rotation per activation (follow-up)

- **Problem.** The rotation cache is keyed by the input tensor. `build_moe_ffn` gives the experts a 3D reshape of
  the input the shared expert reads, so MoE models with a shared expert rotated that input twice per layer.
- **Graph.** `rotate_input_if_rotated` looks through `RESHAPE`s that keep the row width, which keep the rows and
  their order. It creates the RHT on the source, and each caller gets it, or a reshape of it, cached under the
  caller's own input: R·reshape(x) = reshape(R·x).
- **Q8_1 planner.** Without more, the shared RHT would have to write F32, since expert GEMMs never took Q8_1, and
  every consumer would quantize for itself. So `ggml_cuda_rht_plan_q8_1` changes:
  - An RHT's family is the RHT plus those reshapes, and consumers may read any member. Every use of the family
    must be a consumer or one of the family's own reshapes. The chosen format is registered for every member.
  - `ggml_cuda_rht_consumer_fmt` gives `MUL_MAT_ID` Q8_1 when `ggml_cuda_mul_mat_id` takes its MMVQ path, fusions
    included: `ne2 <= min(MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch)`.
  - Through a reshape only MMVQ's row-major layout lines up. MMQ layouts depend on the shape, so they stay limited
    to direct consumers.
- **MMVQ.** `ggml_cuda_mul_mat_vec_q` takes RHT-written Q8_1 with `ids` when src1 is contiguous, because the RHT
  wrote its rows in src1's (i11, i12, i13) order. It used to assert `!ids`.
- **Test.** test-hadamard-archs asserts that no two RHT nodes read the same rows. It checks through `cb_eval`
  queries, which leave the graph unsplit.

### G2. Allocation dependencies for the fusions (follow-up after the 2026-10-04 merge)

- **Upstream mechanism** (#27301, in since the merge of llama.cpp `53ed051ce`):
  - A backend's `graph_optimize` gets `ggml_backend_graph_optimize_params`. Its `add_alloc_dep(tensor, until)`
    keeps `tensor` allocated until node `until` has been computed.
  - The scheduler adds a `GGML_OP_NONE` node after `until`, with the kept tensors as sources. That node goes only
    into the copy of the graph that is allocated, never into the graph the backend computes, so the Q8_1 planner
    doesn't see it as a consumer.
  - Upstream uses this for its MoE weighted-reduction and top-k MoE fusions. It is the general form of what G1
    did from the graph side.
- **gate/up/GLU** (`ggml_cuda_glu_add_alloc_deps`, `ggml-cuda.cu`):
  - For every split GLU, both inputs are traced back through the scale MUL and bias ADD/ADD_ID that the fusions
    accept, to two `MUL_MAT(_ID)`s.
  - If those share src1 and the vector fusion can apply (`ggml_cuda_should_fuse_mul_mat_vec_q/f`), src1 stays
    allocated until the GLU, and so do the `ids` of `MUL_MAT_ID`.
  - It isn't limited to rotated models. It covers routed experts and all four fusable GLU ops (G1 had only
    SWIGLU/GEGLU in `build_ffn`). Upstream's `ggml_cuda_check_fusion_memory_ranges` stays as the guard.
- **Prologues** (`ggml_cuda_rht_add_alloc_deps`, `rht.cu`):
  - The RMS norm's input, and the GLU's `g`/`u`, stay allocated until the RHT node.
  - The matcher is shared with `ggml_cuda_rht_try_fuse` (`rht_norm_fusable`, `rht_glu_fusable`), so dependencies
    and fusions can't disagree.
  - `rht_fusion_safe`, with its per-kernel exceptions, is gone. `ggml_cuda_rht_try_fuse` asserts that the output
    doesn't overlap the inputs, so a missing dependency fails loudly instead of racing.
- **Order:**
  - The dependencies are added before upstream's optional stream reordering (`GGML_CUDA_GRAPH_OPT=1`), on the node
    order the fusions match.
  - Within concurrent regions, upstream restores that order for fusion.
- **Removed:**
  - `ggml_glu_split_inplace`: the GLU constructor is back to upstream's.
  - `build_glu_split`: `build_ffn` calls `ggml_swiglu_split`/`ggml_geglu_split` again.
  - The in-place view case of `rht_glu_can_fuse`, and the `RHT_FUSED` in-place test cases.
- **Kept:** kernel B still writes Q8_1 for a single consumer. That is the same work as the consumer's own
  quantize; the aliasing argument for it no longer applies.
- **Cost:** the kept tensors live a few nodes longer. In decode that is a small src1. In prompt processing it is
  the norm input (the residual, which stays alive anyway) and the GLU inputs up to the next node.

## Implementation conventions

- **Comments:** keep them minimal and match the surrounding style.
- **Upstream code:** `ggml-cuda.cu` and `gpttype_adapter.cpp` get small hooks; the logic goes in fork files.
- **Tests:** every step has its test and passes it before the next step starts.
- **Bitwise claims** are checked bitwise:
  - D1, D2: bench-rht byte identity and the RHT test-backend-ops cases;
  - F2 (F32): bench-rht;
  - K1: saved state bytes.

  F1 is checked by tolerance, and by KL on the 0.6B.
- **Performance changes** are kept only when interleaved A/B medians show a gain.

## Harness

Session scratch, not in the repo:
- `llama-bench-cuda`: upstream `llama-bench.cpp` plus a make fragment that links it with `LLAMA_TOOL_OBJS_CUDA`.
- `ab.sh`: interleaved llama-bench rounds, medians.
- `kcpp-prof.patch`: the adapter probes.
- `protoB/`, `protoC/`: the prototypes.
- Models: `m/q06-{q4km,hq4km}.gguf`, `m/q27-{q4km,hq4km}.gguf`, and for G1/M1:
  - `m/granite-{f16,q4km,hq4km}.gguf` (granite-3.1-1b-a400m);
  - `m/qwen15moe-{q4km,hq4km}.gguf` (Qwen1.5-MoE-A2.7B; its HQ copy is requantized, which is enough for placement
    and speed, not quality).
- The placement log: a temporary `GLU_ALLOC_LOG` block in `ggml_cuda_try_fuse` that prints each gate/up/GLU
  pattern's compute-buffer offsets and fusion verdict. Removed afterwards.

## Phases

0. **Baselines.**
   - The tables above, plus bench-rht CSVs saved to scratch.
   - Oracles that pass before any change:
     - `test-backend-ops -o RHT` (and the full CUDA run of the RHT and GLU cases);
     - `test-hadamard-archs-cuda`, `test-hadamard-llama-cuda`, bench-rht;
     - `test-hadamard-ppl-cuda` KL of the HQ 0.6B against its BF16 reference.
1. **K1.**
   - *Tests:* the saved state bytes and the generated text are identical before and after, over three
     koboldcpp requests (same prompt twice, then an extended prompt) at temperature 0.
   - Snapshot timings come from the probe patch.
2. **D1, D2.**
   - *Tests:* bench-rht byte identity, with the Q8_1 checks extended to `M == 1` widths in all three layouts and
     to the split orders; test-backend-ops RHT cases.
   - *Measure:* bench-rht, then 0.6B and 27B decode A/B.
3. **F0, F1, F2.**
   - F0 first, with test-hadamard-archs(-cuda) and the 27B node order.
   - Then prologues in A and B, with bench-rht checks: GLU bitwise, norm by tolerance, all shapes (K = 1,
     unrolled, generic, narrow, strided/multipass), rows 1/7/9/512.
   - Then the matcher. New test-backend-ops cases check the result against the CPU and the fused-launch counter:
     - must fuse: norm and GLU (split, single-tensor, swapped) at rows 1, 8, 9 and 512; widths of every class;
       with and without Q8_1 consumers;
     - must not fuse: a MUL with two uses, an output-flagged MUL, a GLU op outside the set;
     - an aliased output: the graph allocator places the RHT over a freed input, and the result must still be right.
   - Count the fusions on the 27B prompt and decode graphs.
   - KL on the 0.6B must not change beyond noise.
4. **End to end.** llama-bench A/B (0.6B and 27B, plain vs HQ, before and after) and koboldcpp `--benchmark`,
   recorded below.

## Implementation record

All phases implemented 2026-09-29, follow-ups G1 and M1 on 2026-09-29/30. Uncommitted. Files:
- `gpttype_adapter.cpp`, `otherarch/otherarch.h` (K1)
- `src/llama-graph.{h,cpp}` (F0, G1, M1)
- `ggml/include/ggml.h`, `ggml/src/ggml.c` (G1: `ggml_glu_split_inplace`, removed again by G2)
- `ggml/src/ggml-cuda/{rht-impl.cuh, rht.cu, rht.cuh}` (D1, D2, F1, F2, G1)
- `ggml/src/ggml-cuda/ggml-cuda.cu`: the fusion hook, the counters, and the Q8_1 planner changes (M1)
- `ggml/src/ggml-cuda/mmvq.cu` (M1)
- `tests/bench-rht.cu`, `tests/test-backend-ops.cpp`, `tests/test-hadamard-archs.cpp`
- the `bench-rht` Makefile rule (`unary.cuh` dependency)

### K1 (snapshot buffers)

- **Buffer type and helper.** `kcpp_state_buffer` (a vector with a default-initializing allocator) and
  `kcpp_state_buffer_fit`: 1.25× headroom rounded to 16 MiB; frees before reallocating. A slot is valid iff
  its size > 0.
- **Deviation: page faults.** Without the zero-fill, a fresh buffer's page faults happened inside
  `llama_state_get_data`, and a CUDA device-to-host copy into unfaulted pageable memory is about twice as slow
  (standalone, 300 MB: 214 ms, against 98 ms with the pages touched first and 17 ms into a warm buffer). The
  first build was therefore *slower* on cold slots (206 and 439 ms).
  - `kcpp_state_buffer_fit` now touches one byte per 4 KiB page of the part not written before.
  - The "JIT free" keeps the buffers, with the sizes marking them invalid, so a re-save knows which part is
    warm.
- **Result**, koboldcpp server on the 27B, three greedy requests (a 2.3k-token prompt, the same again, then
  extended): the outputs are identical before and after.

  | snapshot | before (allocation + copy) | after |
  |---|---|---|
  | cold slot | 96–140 + ~20 ms | 100–134 + ~21 ms (page faults either way) |
  | re-save into a warm slot | 128 + 21, 140 + 21 ms | 32 and 55 ms |

  In a chat session the slots are reused, so most snapshots become warm.

### D1, D2 and the emission policy

**D1 and D2**, as designed. bench-rht's Q8_1 identity check now also covers the one-warp widths 32–512: 360
cases, all byte-identical.

| rotation (1 row) | before | after |
|---|---|---|
| 1024, writing Q8_1 | 5.77 µs | 1.60 µs |
| 17408, writing Q8_1 | 4.08 µs | 2.68 µs |
| 17408, F32 | 2.97 µs | 2.48 µs |

After D1/D2, writing Q8_1 wins at 1 row for every order. With one consumer at 2–8 rows it wins only for
K < 64 (K = 68–172 lose by up to 2.7 µs).

**Deviation, kernel B:** kernel B now writes Q8_1 for a single quantized consumer too, through its F32 temporary
and one quantize pass. That is the same work as the consumer's own quantize, and B then never writes the RHT
output while reading its inputs. Without this, the allocator put the 17408 RHT output over part of the freed
gate/up in 32 of the 27B's 64 layers, and the aliasing rule refused the GLU fusion there.

New `ggml_cuda_rht_write_q8_1`:
- 2 or more consumers, or kernel B: write Q8_1;
- kernel A with one consumer: 1 row, or K < 64.

### F0

The RHT is expanded where it is created. In the 27B graph, RMS_NORM, MUL and RHT are now nodes 1–3 of the
gated-delta-net layers (they were 1, 2 and 13). test-hadamard-archs(-cuda) still gives 254/254.

### F1/F2 (prologues)

- **Deviation, loads.** `rht_load_pro` loads x and the second operand first, then applies the prologue in a loop
  per case. A `switch` inside the unrolled load loop made each iteration's loads control-dependent, so they
  weren't batched: +4.6 µs on one-warp rows (E = 32).
- **Deviation, kernel B's sum of squares** goes through the start of its dynamic shared memory, not static shared
  memory. The prototype's static array pushed some 48 KB launches over the limit ("invalid argument").
- **Deviation, cutoff (`ggml_cuda_rht_fusion_pays`).** Kernel B isn't fused for orders above 148. At 512 rows its
  mix is compute-bound there: order 172 was 0.5–7.5 % slower fused, while 148 was 20–27 % faster. At 9 rows
  every order wins.
- **Costs** (bench-rht, against a producer kernel + rotation; the producers are stand-ins for the backend's RMS
  norm and GLU kernels):

  | shape | producer + RHT | fused | plain RHT |
  |---|---|---|---|
  | 1 row, 5120 norm | 6.88 µs | 2.50 µs | 2.02 µs |
  | 1 row, 17408 SWIGLU | 13.88 µs | 3.00 µs | 2.48 µs |
  | 512 rows, 5120 norm | 25.2 µs | 16.9 µs | |
  | 512 rows, 5120 SWIGLU | 30.0 µs | 20.2 µs | |
  | 512 rows, 17408 norm | 102.5 µs | 71.2 µs | |
  | 512 rows, 17408 SWIGLU | 122.0 µs | 79.9 µs | |

  The unfused paths are unchanged within ±3 %.
- **Fusions on the 27B graphs** (temporary log, removed):
  - prompt (512): all 128 norm sites and all 64 GLU sites;
  - decode: 128 norm sites (+ the output norm).
  - In decode, upstream's gate/up/GLU MMVQ fusion fires in only 40 of the 64 HQ layers, against all 64 on the
    plain model. In the other 24 the GLU output sits in the rotation's freed block, and
    `ggml_cuda_check_fusion_memory_ranges` rightly refuses. Fixed afterwards: see "Follow-up: gate/up/GLU fusion
    in HQ decode".

### Tests

- **bench-rht:** sign mixer; 796 correctness cases; 360 Q8_1 cases byte-identical; 312 prologue cases. Those
  cover norm, SWIGLU and GEGLU; every width class (one-warp, mixed K = 1, strided, unrolled split and unsplit,
  generic, narrow); rows 1/7/9 and 4D strides; kernels A and B. GLU F32 is bitwise-equal to producer +
  rotation, and Q8_1 byte-identical to quantize.cu.
- **test-backend-ops:**
  - `RHT` 426/426.
  - New `RHT_FUSED` 384/384, with the fused-launch counter asserted, with fusion enabled, with
    `GGML_CUDA_DISABLE_FUSION=1` and with `GGML_CUDA_RHT_F32=1`. It covers every kernel path, Q8_1 consumers,
    3D rows, inputs computed in the graph (aliasing), a MUL with two uses and SWIGLU_OAI (must not fuse), and
    order 172 at 512 rows (not paying).
  - The full CUDA suite: 14790/14790.
- **test-hadamard-archs-cuda:** 254/254.
- **KL** (0.6B, 40×512 wikitext test, vs BF16):

  | model | mean KL | p99 | top-1 |
  |---|---|---|---|
  | HQ before | 0.103783 | 0.718 | 83.65 % |
  | HQ after | 0.104165 | 0.709 | 83.74 % |
  | plain | bit-identical | | |

  The norm prologue changes the rotated activations by ~3e-7 relative, which re-rolls Q8_1 rounding.

### End to end

llama-bench, interleaved rounds (27B: 8, 0.6B: 5), median (max). "old" is the build before this plan.

| model | pp512 | tg128 |
|---|---|---|
| 27B plain | 3528 (3670) | 74.59 (75.89) |
| 27B HQ old | 2337 (2444)¹ | 70.73 (73.97) |
| 27B HQ new | 3563 (3623) | 72.19 |
| 0.6B plain | 45714 (46177) | 650.1 (655.6) |
| 0.6B HQ old | 41235 (42732) | 505.0 (508.1) |
| 0.6B HQ new | 45453 (46012) | 590.3 (597.3) |

¹ The old build was in the slow mode (see "Open") in all 8 rounds. In fast mode it ran 3465–3515 elsewhere.

**Against plain:**

| model | prompt | decode |
|---|---|---|
| 27B HQ | now level with plain (fast mode, was about −3 to −5 %) | −5.2 % → −3.2 % |
| 0.6B HQ | −9.8 % → −0.6 % | −22 % → −9 % |

**koboldcpp `--benchmark`** (27B, ctx 4096, 3840-token prompt, 256 generated):

| model | generation | prompt |
|---|---|---|
| plain | 67.7–68.8 → 71.3–71.5 t/s | 2531–3007 → 2807–2929 t/s |
| HQ | 67.4 → 69.1–69.3 t/s | 2876 → 2720–2909 t/s |

- The generation gain is the snapshot at generation start, which re-saves into the slot written 32 tokens earlier:
  ~320 ms → ~50 ms.
- The prompt still pays two cold-slot snapshots in a single-request benchmark. In a chat session those slots
  are warm.

### Follow-up G1: gate/up/GLU fusion in HQ decode

**Cause, confirmed** with a temporary placement log (offsets in the compute buffer, 27B decode):

| layers | norm | RHT | gate | up | SwiGLU | fused |
|---|---|---|---|---|---|---|
| 40 HQ layers | [21.1,41.1)K | [41.1,61.1)K | [61.1,129.1)K | [129.1,197.1)K | [197.1,265.1)K | yes |
| 24 HQ layers | [0,20)K | [21.1,41.1)K | [109.1,177.1)K | [177.1,245.1)K | **[21.1,89.1)K** | no |
| plain, all 64 | [0 or 21.1)K, 20 KB | — | [41.1,109.1)K | [109.1,177.1)K | [177.1,245.1)K | yes |

- The SwiGLU output starts at the freed RHT block and runs into a 68 KB free gap after it. The norm output is not
  involved.
- Upstream's fused kernel reads the whole RHT (the Q8_1 it wrote) in every block while other blocks write the
  SwiGLU output. The overlap would therefore be a real race, and the memory-range check correctly refuses.

**Fix:** G1 (see Design).

**Result:**
- Fusion counts:
  - decode: gate/up/GLU fused in 64/64 layers (was 40);
  - prompt: {GLU, RHT} still 64/64 (in place) and norm 128/128.
- Tests:
  - `test-hadamard-archs` CPU and CUDA 254/254 (the rotated fixtures now build the in-place GLU);
  - `RHT_FUSED` 460/460, with 76 in-place cases asserting the fusion;
  - `RHT` 426/426.
- Values are unchanged: in place writes the same numbers, and upstream's fused kernel computes the same GLU.

**Speed** (27B decode, llama-bench tg128, 10 interleaved rounds):
- The GPU showed several speed modes that session (~60, ~70, ~79 and ~103–109 t/s) while another client kept it at
  7–10 % utilization, so only runs in the dominant ~70 mode were compared:
  - previous build: 69.68 t/s median (69.59–69.84, n = 5);
  - in-place GLU: 69.88 (69.69–70.16, n = 8), i.e. **+0.3 %**;
  - plain: 71.52 (n = 4), so the HQ gap is now about −2.3 %.
- That matches one launch saved per layer in 24 layers. An earlier noisy A/B suggested ~40 µs per layer, which was
  the modes switching, not the fusion.
- KL on the 0.6B is unchanged (0.104165). The full CUDA test-backend-ops suite passes: 14866/14866.

### Follow-up M1: one rotation per activation (MoE)

**Findings** (placement log on granite-3.1-1b-a400m and Qwen1.5-MoE-A2.7B, plain and HQ, decode):
- The expert path has no gate/up/GLU collisions: routed experts fused 24/24 in both models, the shared expert
  24/24.
- An overlap wouldn't be a race there anyway: the experts' GEMMs (`MUL_MAT_ID`) got F32 from the RHT and quantized
  a private copy first.
- **Models with a shared expert rotated the same input twice per layer.** The rotation cache is keyed by the input
  tensor, and `build_moe_ffn` gives the experts a 3D reshape of the input the shared expert reads. The
  architecture fixtures show that this hit about every MoE architecture: qwen2moe, qwen3next, qwen35moe, qwen4exp,
  deepseek/2/3.2/4, glm4moe, glm-dsa, cohere2moe, exaone-moe, granite(moe), grok.

**Fix:** M1 (see Design): the graph change, plus the Q8_1 planner and MMVQ changes. Without the latter two, the
shared RHT would have had to write F32. With them, the single rotation also saves the experts' quantize launches,
in MoE models without a shared expert too.

**Tests:**
- **`test-hadamard-archs`** now checks through `cb_eval` queries (which leave the graph unsplit) that no two RHT
  nodes read the same rows. With the look-through disabled, 30+ MoE configurations fail ("ffn_norm-0 rotated by two
  RHT nodes"); with it, 254/254 pass on CPU and CUDA.
- **test-backend-ops `RHT_Q8_1_MOE`** (a shared-expert `MUL_MAT` on the RHT, two expert `MUL_MAT_ID`s on its 3D
  reshape): Q8_1 is written up to 8 rows (Q4_K up to its MMVQ-expert limit) and F32 beyond, as predicted, and the
  results match the CPU.
- RHT 438/438, RHT_FUSED 460/460, full CUDA suite 14878/14878.
- **Qwen1.5-MoE HQ:** greedy generation is byte-identical between writing Q8_1 and `GGML_CUDA_RHT_F32=1`, and so is
  KL (0.041628 against the plain Q4_K_M).

**Result** (Qwen1.5-MoE HQ, one decode step, ncu):

| | kernels | RHTs | quantize launches |
|---|---|---|---|
| before | 943 | 145 | 48 |
| after | 847 | 121 | 0 |

That's −10 % kernels.

**Timing** (llama-bench, 8 interleaved rounds, paired new vs previous build, on a GPU another client kept busy):

| | median gain | rounds faster |
|---|---|---|
| decode | +1.7 % | 6/7 (one round's reading was invalid) |
| prompt | +1.8 % | 8/8 |

Against plain (Qwen1.5-MoE Q4_K_M, medians):
- decode: 316.6 t/s plain vs 270.6 t/s HQ, −15 %;
- prompt: 14295 vs 13063 t/s, −9 %.

MoE HQ is further behind plain than the dense models are. Its layers are small, so fixed per-rotation costs weigh
more, and the norm→RHT fusion can't fire: the router, which isn't rotated, also reads the norm output (see
"Future work").

### Follow-up G2: allocation dependencies (2026-10-04)

Done right after the merge of upstream `concedo` (llama.cpp `53ed051ce`), on branch `merge-concedo`. The design is
under G2.

**Fusion counts**: a temporary `FUSELOG` counter in the CUDA evaluate loop, removed afterwards, recorded during
koboldcpp `--benchmark` runs.

| model | graph | gate/up/GLU (MMVQ) | {RMS_NORM, MUL, RHT} | {GLU, RHT} |
|---|---|---|---|---|
| 27B HQ4_K_M | decode | 64/64 | 128/128 | — (the GLU is in the MMVQ fusion) |
| 27B HQ4_K_M | prompt (512) | — | 128/128 | 64/64 |
| 0.6B HQ4_K_M | decode | 28/28 | 56/56 | — |
| 0.6B HQ4_K_M | prompt (512) | 1 (the last layer, which runs on the output row only) | 56/56 | 27/28 |
| 27B UD-Q4_K_M (plain) | decode | 36/64 | — | — |

- The 27B HQ counts are G1's.
- On the plain UD-Q4_K_M, exactly the 36 layers whose `ffn_gate` and `ffn_up` have the same type fuse. Unsloth's
  mix uses different types in the other 28 layers, and upstream's fusion needs equal types.

**Tests:**
- `RHT_FUSED`: 384/384, both with fusion enabled and with `GGML_CUDA_DISABLE_FUSION=1`.
  - The 76 in-place cases are gone.
  - The computed-input cases must now fuse too. test-backend-ops allocates every tensor separately and never calls
    `graph_optimize`, so its inputs never alias.
- `RHT` 438/438, `MUL_MAT_VEC_FUSION` 1265/1265, `MUL_MAT_ID_FUSION` 13/13, `SWIGLU` 24/24.
- `test-hadamard-archs`: 264/264 on CPU and on CUDA. The CUDA sweep runs through the scheduler, so a missing
  dependency would have hit the new assert.
- `golden.sh check` against master: byte-identical.

**Speed and memory** (27B HQ4_K_M, koboldcpp `--benchmark`, ctx 4096, 4 interleaved rounds):
- Decode is unchanged: G1 gave 65.92–66.80 t/s, G2 66.05–67.02 t/s.
- One G2 round read 151 t/s. koboldcpp times generation with `std::chrono::high_resolution_clock`, which is the
  system clock on libstdc++, and WSL clock corrections make single readings invalid (an earlier run read a
  negative time).
- Prompt speed couldn't be compared: `--benchmark` swung between 2.0 and 4.5k t/s across rounds on this box. The
  prompt fusion counts are the same.
- The compute buffers are identical: 519.27 MiB CUDA0, 24.27 MiB host.

### Open: HQ prompt processing has a temporal slow mode

- **What:** on this box, 27B HQ pp512 sometimes runs at ~2300–2600 t/s instead of ~3500.
  - It affects the old build and the new one alike.
  - It lasts from a single repetition to several minutes of fresh processes.
  - The plain model never showed it: 3150–3700 over ~40 runs.
- **Not the cause:** it showed up with fusion on; with `GGML_CUDA_RHT_F32=1` or `GGML_CUDA_DISABLE_FUSION=1` it
  didn't in 4 tries each, which is too few to rule them out.
- **Suspicion:** interference from other GPU clients on the Windows side (~2.9 GB and 8–28 % utilization while
  "idle"), to which kernel B's 1024-thread, one-CTA-per-SM launches are more exposed.
- **Next step:** ncu on a slow and a fast process.
- **Measurement rule adopted:** interleaved rounds, reporting the median and the max.

## Future work

### Rotated MoE routers, so the FFN norm fuses in MoE layers

**Why.**
- In a MoE layer, the FFN norm's output feeds the RHT and also two GEMMs that aren't rotated:
  - the router, `ffn_gate_inp` (n_embd × n_expert);
  - with a shared expert, its gate, `ffn_gate_inp_shexp` (n_embd × 1), e.g. in `qwen35moe.cpp`.
- Both are float tensors (F32 in the files checked), for different reasons: the quantizer skips `ffn_gate_inp` by
  name (`llama-quant.cpp`), and the one-row shared gate is a vector. So `--hadamard` leaves them unrotated, and they
  read the norm output directly.
- The norm's MUL then has several uses, and {RMS_NORM, MUL, RHT} (F1) can't fuse. The norm output is written and
  read back in every MoE layer: a launch in decode, a full pass in prompt processing.
- It is one of the remaining costs behind MoE HQ's gap to plain: −15 % decode and −9 % prompt on Qwen1.5-MoE,
  against about −2 % for the dense 27B.

**Idea.**
- Store those gates rotated like the other weights: row i becomes R·w_i. In exact arithmetic
  (R·w)ᵀ(R·x) = wᵀx, so the logits are unchanged.
- The models already compute them through `build_lora_mm`. Once they're in the rotated set, they read the RHT
  their layer's experts read, the same node memoized by M1.
- With no unrotated reader left, the norm fuses into that RHT.

**Design questions.**
- **Where the rotation happens.**
  - *At quantize time:* `--hadamard` rotates the gates (kept float), and the file lists its rotated float tensors
    in metadata, which the loader adds to the rotated set. Explicit and checkable. Existing HQ files need a pass
    that rotates the gates; the other tensors are copied, not requantized.
  - *At load time:* rotate the gates after loading when the layer's experts are rotated. It works on existing
    files, but mmap'd CPU weights are read-only, so that path needs a copy. It also makes the loader infer
    rotation instead of reading it from the file.
  - Recommended: quantize time. The loader should trust only what the file states, as it does for the HQ types.
- **The gates' input format.** A float gate reads F32. The Q8_1 planner requires every consumer of an RHT family to
  take the same format, so the RHT would write F32, and in decode the experts' quantize launches would come back
  (M1 removed two per layer). Options, to decide by measurement:
  - *Accept F32.* In prompt processing the expert GEMMs (MMQ) already take F32, so the fusion is a clear win there.
    In decode it trades one norm launch for up to two quantize launches.
  - *An RHT epilogue that writes both F32 and Q8_1.* It needs room for both (4n + 1.125n bytes against the F32
    node's 4n), so the RHT output would have to be allocated larger.
  - *An 8-bit gate (HQ8_0)* that takes Q8_1 like the experts. That changes routing numerics, which needs its own
    KL check.
- **Numerics.** F32 rounding changes the logits by ~1e-7 relative, enough to flip a top-k choice on near-ties.
  Evaluate by KL on real MoE models (Qwen1.5-MoE and a qwen35moe), not by weight error.
- **Scope.**
  - Affected: architectures whose router reads the FFN norm output, which is most MoE ones (qwen2moe, qwen3moe,
    qwen35moe, qwen3next, granitemoe, the deepseek family, glm4moe…).
  - Not affected: models that weight the input before the experts (`weight_before_ffn`) or route from another
    tensor.

**Tests.**
- test-hadamard-archs: rotated gates give the same logits (NMSE), and the census still sees one RHT per activation.
- A fusion count showing {RMS_NORM, MUL, RHT} firing in MoE layers.
- Real models: the ncu kernel census and interleaved A/B on Qwen1.5-MoE, plus KL.

### Other next steps

- **The HQ prompt slow mode:** see "Open"; next is ncu on a slow and a fast process.
- ~~In-place GLU for routed experts~~: G2's allocation dependencies cover the expert path.
- **Kernel B's Q8_1 epilogue** through a reduce-scatter over the tile's blocks. The draft's P1 was rejected because
  its per-block shuffle epilogue was too slow. This would be worth ~1.5 % of prompt time on the 27B.
- **Kernel A's two launches per decode rotation.** That is the largest remaining decode cost of dense HQ (−2.3 %
  on the 27B).
