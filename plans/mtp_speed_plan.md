# Plan: MTP generation, prompt processing and long-context attention on CUDA

## Summary

The target is the setup the user actually runs, `~/AI/qwen3/Qwen3.8-27B-HQ4_K_M.kcpps`:
- Qwen3.8-27B HQ4_K_M (`qwen35`: 48 gated-delta-net layers and 16 attention layers);
- MTP speculative decoding with 4 draft tokens;
- q5_1 K/V, flash attention, context 262144, batch 1024, RTX 5090 under WSL2.

Each MTP cycle runs, in order:
1. 4 one-token draft decodes on the MTP head;
2. one 5-token verify on the trunk, with `n_rs_seq = 4` rollback snapshots;
3. one catch-up decode on the MTP head;
4. CPU sampling of the accepted rows.

Speed means that cycle, prompt processing, and attention at long depth. The HQ rotation overhead is covered by
`inference_speedup_plan.md` and is not revisited here.

**Every change ships with its tests.** No change lands without both:
- a **regression test** that runs from the repo (a test-backend-ops case, a test program mode, or an end-to-end
  koboldcpp check), added in the same commit or before it. Where the change fixes a bug or claims exactness, the test
  fails without the change or compares bitwise.
- a **performance check**: a named benchmark with a baseline recorded before the change. The change stays only if the
  interleaved A/B median beats the noise.

The checks of earlier changes keep running at every phase, so a later change can't silently undo an earlier gain.
The "Test policy" section has the rules, and each item lists its own tests.

| # | change | effect | regression test | performance check |
|---|---|---|---|---|
| 1 | CUDA graphs for multi-token batches: delete koboldcpp's extra check, as upstream did | measured: MTP generation +33 %; 5-token batch 26.5 → 17.0 ms | test-hadamard-archs `-g`: logits memcmp with graphs on and off, launch/capture counters | pp2/pp5/pp8 ms; kcpp-e2e t/s |
| 2 | No all-token logits on prompt batches when MTP is on | measured: prompt +9 %; −1 GB pinned host buffer | kcpp-e2e text hashes; guidance + MTP first token | long-prompt t/s, cold and warm |
| 3 | Attention: MMA kernel reading quantized K/V tiles (3a); no GQA padding in prefill (3b) | est.: decode −2 ms/token at 32k (−6 at 96k); verify −2.9 ms at 32k (−7.7 at 96k); prefill FA up to −25 % | new FLASH_ATTN_EXT cases; bitwise against the conversion path; racecheck; KL | FA µs and GB/s per layer; pp1/pp5/pp1024 at 32k and 96k |
| 4 | MTP loop host work: batched input uploads (4a); reduced-vocabulary draft head (4b); gated follow-ups (4c, 4d) | measured: 4a +6–9 % generation; est.: 4b −1.9 ms/cycle before acceptance loss | `-g` matrix in both upload modes; draft logit rows bitwise; kcpp-e2e hashes | per-decode upload time; draft step time; acceptance and t/s |
| 5 | CPU sampler: one scratch row, chunked bucket select, lazy lowest logit | est.: −0.8 ms/cycle | test-kcpp-sampler: recorded rows, bitwise against today's code | per-row µs; time between cycles |
| 6 | Kernels: gated-delta-net fusions (6a); MMVQ table for sm_120 (6b); GDN prefill (6c) | est. per verify: 6a −0.8 to −1.0 ms, 6b −0.2 to −0.5 ms; 6c up to −7 % of prefill | test-backend-ops case per pattern with fusion counters; rollback suite; MUL_MAT tolerance; KL | pp5 kernel census at `n_rs_seq = 4`; bench-mmvq GB/s; GDN µs per ubatch |
| 7 | Correctness: drafting with guidance is wrong today; drafting is disabled under grammars | fix, and investigate | kcpp-e2e: guidance with MTP equals guidance without MTP (fails today) | — |

**Expected (estimates):**
- **Short context:** one MTP cycle (~4.4 tokens) is 35.3 ms today, 26.6 ms with item 1 and ~24.7 with 4a added.
  Items 4b, 5 and 6 should bring it to about 21 ms.
- **Long context:** item 3a dominates, at about −10 ms per cycle at 96k.

**Prerequisites already on branch `mtp-speed`**, cherry-picked from concedo_experimental (upstream llama.cpp):
- 7cc0fdffd (1ab7e5ad2): CUDA RMS_NORM + SCALE fusion, used by `build_gdn_l2_norm`.
- bbd645262 (08618ff8e): K/V and recurrent state cleanup after failed restores, with upstream's
  `tests/test-save-load-state.cpp`. The fork adds `test-save-load-state(-cuda)` make targets.
- 22be82de0 (42d958167): routes sm_70 to the Turing MMVQ table. It has no effect on sm_120, and it touches the lines
  6b edits.
- 869034b4b was skipped: it fixes `llama_memory_recurrent::is_empty()`, which the fork doesn't have yet (upstream
  added it with c061df198).

<!-- prerequisite test results are recorded in "Implementation record" -->

## Test policy

**Two gates for every change.**
- **Regression gate.** An automated test, runnable from a make target or a repo script, in the change's commit or an
  earlier one.
  - **Exact changes** (graph capture, upload batching, logit rows, sampler restructuring, prologue fusions that reuse
    the same arithmetic) are tested bitwise: memcmp of logits, token rows or outputs.
  - **Other changes** use test-backend-ops NMSE against the CPU, plus KL against a recorded reference on the eval set
    (`hessian-tokenize`/`tensor-kl` tooling).
  - **Bug fixes** come with a test that fails before the fix.
  - **Fusions** come with a counter that proves the fused path ran, and a negative case where it must not fuse.
- **Performance gate.**
  - A named benchmark from the harness below, with the baseline recorded in "Implementation record" before the
    change.
  - Interleaved A/B, at least 3 rounds; report median and max.
  - Record the GPU contention level (`stall` probe) with every session.
  - The change stays only if the median gain beats the run-to-run spread.

**The standing suite** runs before and after every change, and fully at every phase end:
- test-backend-ops: FLASH_ATTN_EXT, MUL_MAT(_ID), the fusion cases, RHT, RHT_FUSED, GATED_DELTA_NET*, RMS_NORM_SCALE;
- test-hadamard-archs-cuda, plus the `-g` mode once it exists;
- test-save-load-state-cuda, plus the rollback cases;
- test-kcpp-sampler, once it exists;
- kcpp-e2e (text hashes, draft counts, t/s);
- the perf suite (llama-bench rows, kcpp-e2e t/s, FA and MMVQ microbenchmarks).

**Determinism caveats** that tests must allow for, not hide:
- Different verify batch shapes take different kernel paths, so greedy text can differ between draft lengths, or when
  a draft head changes which tokens are verified together. Such tests accept divergence only where the top-2 logit gap
  is below ~1e-3.
- argsort/top-k and mean choose algorithms by CUDA graph capture state.
- koboldcpp's first request after a load sometimes reports a nonsense eval time. The perf scripts discard it.

## Test and benchmark infrastructure (Phase 0, in the repo)

Fork-owned, so a later change can rerun any earlier check:
- **`tools/perf/llama-bench`**:
  - a make target `llama-bench-cuda` that builds upstream's `llama-bench.cpp` (pinned in-tree, since the fork ships
    only a stub) against `LLAMA_TOOL_OBJS_CUDA`;
  - an `-nrs` option for `n_rs_seq`, so pp5 at `n_rs_seq = 4` reproduces the MTP verify.
- **`tools/perf/kcpp-e2e.py`:**
  - starts koboldcpp from a given build directory, runs fixed greedy prompts (`sampler_seed` 42; 3 short prompts and
    one 9.6k-token prompt), and records text hashes, draft accepted/rejected counts, t/s, prompt t/s and first-request
    time;
  - configs: MTP on/off, guidance, grammar;
  - modes: `record` (golden file) and `check` (hashes must match; t/s compared against the baseline);
  - it also supports a scratch library next to a symlinked tree, so a change can be A/B tested before it is built into
    the repo.
- **`tools/perf/stall.cu`:** the globaltimer contention probe.
- **`tools/perf/bench-mmvq.cu`:** item 6b's microbenchmark.
  - It cycles ≥ 384 MB of weight copies (L2 is 96 MB) and launches from a captured graph.
  - It reports GB/s against 1792 and an LDG.128 roofline.
- **FA perf cases** in test-backend-ops (`-o FLASH_ATTN_EXT perf`): hs 256, 4 KV heads, GQA 6, kv {8k, 32k, 96k},
  nb {1, 2, 5, 8}, f16 vs q5_1.
- **test-hadamard-archs `-g` mode** (item 1).
  - Use `n_layer` 4, so qwen35 has an attention layer.
  - Prefill 16 tokens, then 4 batches each of T ∈ {2, 4, 5, 8, 9, 33, 64}, asserting n_kv is stable.
  - Dump all logits, repeat in a child process with `GGML_CUDA_DISABLE_GRAPHS=1`, and memcmp.
  - Proc-address counters for CUDA graph launches and captures, which the test asserts.
- **Rollback and multi-sequence cases**, added to test-save-load-state or ported from upstream's
  `tests/test-recurrent-state-rollback.cpp`:
  - `n_rs_seq` 0 and 4;
  - a 2-sequence cell swap (seq_cp/seq_rm, then a batch ordered [B, A]);
  - two fresh sequences in one ubatch;
  - a save/load round trip;
  - each run on CUDA fused, with `GGML_CUDA_DISABLE_FUSION=1`, and on CPU.
- **`tests/test-kcpp-sampler.cpp`** (item 5).
  - It replays recorded verify rows with their sampler configs through a frozen copy of today's `SampleLogits` and
    through the new code, and compares bitwise.
  - Configs: DRY, bans, ±bias, mirostat, xtc, adaptive_p, dynatemp, nsigma; several seeds.
- **Timers**, for diagnosis only and not committed. Scratch patches print per-phase times every N cycles:
  - adapter cycle phases;
  - the draft loop (launch / GPU wait / sample);
  - `llama_context::decode` (an RAII guard: build, set_inputs, sched);
  - `ggml_backend_sched_compute_splits` (uploads, compute calls).
- **ncu:** CUDA 12.9 with `--cache-control none --clock-control none`.
  - Keep only CSV lines that start with `"`.
  - Use `-k regex:` to profile one kernel family.
  - Split decode steps at the lm_head grid (248320).
  - Take per-kind medians against preemption outliers.

## Measurements (2026-10-04, RTX 5090, WSL2)

**Environment.**
- The display runs on the 5090, and a Windows-side client holds ~7 % utilization and ~3.5 GB.
- The `stall` probe lost 12–13 % of wall time to preemptions of 150–570 µs, some mid-kernel.
- The same build therefore varies: plain decode 57–73 t/s, MTP generation 113–162 t/s. Comparisons are only valid
  within one interleaved session.

**Decode, one token, `n_rs_seq` 0** (ncu, depth 0; stalls removed by per-kind medians).

| kernel class | count | time | share |
|---|---|---|---|
| MMVQ (weights) | 461 | 10.81 ms | 75 % |
| RHT | 514 | 1.44 ms | 10 % |
| small elementwise, norm, copy | 802 | 1.44 ms | 10 % |
| flash attention (VEC, 22.8 µs/layer) | 16 | 0.36 ms | 2.5 % |
| GDN and conv | 96 | 0.23 ms | 1.6 % |

- MMVQ moves 15.4 GB at 1422 GB/s, 79 % of 1792. By shape:
  - lm_head: 90 %;
  - 17408-wide gate/up and down: 81–85 %;
  - 5120 × 6144: 73–76 %;
  - 1024 × 5120: 45–55 %;
  - the 48-row alpha/beta Q8_0 matmuls: 2.3 µs each.
- This is not the MTP workload. With MTP:
  - the 48 GDN layers run only inside the 5-token verify, with 5 snapshots;
  - MMVQ fusions fire only at 1 column;
  - the conv path writes 5 snapshot copies per layer.
  
  Phase 0 re-profiles pp5 at `n_rs_seq = 4`, and item 6 uses that.

**Small batches** (llama-bench, ms per batch, depth 0).

| tokens | today (graphs off for > 1 token) | graphs allowed |
|---|---|---|
| 1 | 15.0–15.5 | same |
| 2 | 25.2–25.5 | 15.8 |
| 5 | 25.6–26.6 | 17.0–17.3 |
| 8 | 29.5–30.1 | 19.0–19.3 |
| 9–64 (slower GPU mode) | 37.8–48.9 | 28.5–41.1 |

**MTP cycle in koboldcpp.**
- Setup: greedy, three prompts of 368–400 tokens.
- Draft acceptance: 299/42, 285/73 and 276/39 (accepted/rejected).
- Outputs are byte-identical across builds at the same draft length.

| per cycle (ms) | today | graphs | graphs + batched uploads (slow mode) |
|---|---|---|---|
| 4 draft steps | 6.56 | 6.62 | 7.63 → 6.07 |
| verify `llama_decode` (CPU launch) | 14.44 | 1.32 | 1.55 → 1.19 |
| wait for verify GPU, then catch-up | 12.15 | 16.55 | ~20 |
| between cycles (CPU sampling, bookkeeping) | 2.11 | 2.09 | 2.04 |
| **total** | **35.3** | **26.6** | |

- **Generation:**
  - graphs: 110.9 / 100.0 / 113.8 → 147.3 / 133.2 / 151.8 t/s;
  - batched uploads on top, slow mode: 123.3 / 113.3 / 129.6 → 130.8 / 123.1 / 140.4 t/s.
- **One draft step** (a 96-node graph, 2 splits):
  - 0.71 ms of `llama_decode` before the GPU starts, of which:
    - scheduler 0.56 ms, including 0.52 ms for 9 input uploads, each with two syncs;
    - build/reuse 0.04 ms (a rebuild is ~0.15 ms, twice per cycle);
    - set_inputs 0.01 ms;
    - CUDA graph launch 0.04 ms;
  - then 1.1–1.25 ms of GPU, against ~0.87 ms of weight traffic;
  - 0.01 ms of sampling (the top-k runs on the GPU).
- **Draft length** (greedy, graphs on): 3, 4 and 6 are equal where their texts match; 8 is slower (a 9-token verify
  leaves MMVQ).
- **Argmax** instead of the draft's top-k(10): 5.44 vs 5.37 ms of drafts per cycle, no gain.

**Prompt processing.**
- **koboldcpp, 9.6k tokens, MTP on, warm:** 3030 t/s with all-token logits, 3300 without. Output identical.
- **First request:** 4.7 s with all-token logits, 3.1 s without.
- **Per 1024-token ubatch (ncu):**

| depth | MMQ | GDN | other | RHT | FA |
|---|---|---|---|---|---|
| ~1k (303 ms) | 69 % | 14 % | 8 % | 6.6 % | 2.3 % |
| 32k (426 ms) | 47 % | 8.7 % | — | — | 31 % |

- **FA at 32k:** 131 ms for 16 layers, 103–127 TFLOPS useful.
  - It runs (8,8) tiles, so 2 of 8 GQA slots are padding.
  - The q5_1 → F16 conversion takes 63–69 µs per layer and tensor (~1.45 TB/s).
- **GDN prefill:** a serial scan, 0.8 ms per layer per ubatch (~780 ns per token).

**Attention at depth (q5_1 K/V, per layer).**

| case | 32k | 96k |
|---|---|---|
| decode, VEC, grid (1,14,24) | 165 µs (~300 GB/s) | ~8 ms/token total |
| decode, f16 K/V (MMA (1,8), GQA-packed) | 88 + 4 µs (~1.5 TB/s) | |
| 5-token verify, q5_1 vs f16 K/V | +2.9 ms per verify | +7.7 ms per verify |

- 4× the VEC parallel blocks gives 165 → 151 µs.
- VEC takes 22.8 µs per layer even at depth 0.
- The MTP head's own layer uses the same q5_1 VEC path: 4 decodes and one converted catch-up per cycle.

**No effect:** `GGML_CUDA_GRAPH_OPT=1` (multi-stream).

## Items

### 1. CUDA graphs for multi-token batches

**Why.**
- `ggml_cuda_graph_check_compability` (ggml-cuda.cu ~2778) has a koboldcpp-only check that turns graphs off whenever
  an ADD has `src[1]->ne[1] > 1`.
- History: upstream removed it in #19645 (ad8207af7). Concedo reverted that on 02-20, re-applied it on 02-22
  (edc04f3f7), reverted again 14 h later (71d42fae8), and kept the check in 06c0ffaea.
- Every verify and catch-up therefore launches ~2,000 kernels one by one, which is 14 ms of CPU launch work per
  verify.
- `src[1]->ne[1]` is no batch measure: it is n_layer (gemma3n/4), n_expert_used or head counts in other graphs, and a
  spatial size in sd.cpp, whisper, TTS and CLIP graphs, all of which share ggml-cuda.

**Change.**
- Delete the check. Since February, upstream has added:
  - a warmup of two identical calls;
  - src ne/nb in the graph equality (a29e4c0b7, the likely cause of the February failure);
  - a capture-aware argsort and a static cuBLAS workspace.
- Disable graphs while the device uses the legacy (non-VMM) pool. A captured graph keeps pool addresses, which only
  the VMM pool guarantees.
- Ask Concedo/Henk what broke in February.
- MTP verify and catch-up shapes repeat:
  - the MTP context keeps catch-up and drafts in separate graph arenas (`gf_res_prev[n_outputs > 0]`), so their CUDA
    keys differ;
  - re-capture happens once per 256 tokens, when n_kv grows;
  - large prompt ubatches never finish warmup, because n_kv changes each time.

**Regression tests.**
- **test-hadamard-archs `-g`** (bitwise, counters asserted):
  - all architectures;
  - an HQ-quantized fixture (MMVQ, RHT, MMID ≤ 4);
  - a MoE with F16 experts (which needs sync, so zero launches);
  - a build with `GGML_CUDA_NO_VMM` (zero captures).
- **Configs:** two virtual devices (`GGML_CUDA_DEVICES`) with split-mode layer; partial `-ngl`; pipeline parallel.
- **kcpp-e2e:** hashes and draft counts.
- **Smoke tests** of the non-LLM users: SD at 512 and 1024 px, whisper, qwen3tts/kokoro, a CLIP/mmproj image, and TTS
  alongside MTP generation.
- **Fallback:** if a failure can't be fixed, guard on the largest `src1` row count over weight `MUL_MAT`/
  `MUL_MAT_ID` nodes, never an ADD.

**Performance check.**
- llama-bench pp2/pp5/pp8 at depth 0. Baseline: 25.2 / 25.6–26.6 / 29.5 ms. Target: ≤ 16 / 17.5 / 19.5.
- kcpp-e2e generation t/s. Baseline: 110.9 / 100.0 / 113.8 in a fast session.

### 2. No all-token logits on prompt batches

**Why.**
- With MTP on, `kcpp_embd_batch(..., draft_is_mtp)` requests logits for every prompt token (gpttype_adapter.cpp 6695,
  6724, 6750). That is a 1024-row lm_head plus a 1 GB device-to-host copy per ubatch.
- None of the draft types need them:
  - MTP reads unmasked nextn, and DFlash/DSpark read `layer_inp`, from every row regardless of outputs;
  - nothing in speculative.cpp reads target logits;
  - readers use `llama_get_logits_ith(-1)`.
- Upstream's server already outputs prompt rows only for embeddings.

**Change.**
- Pass `false` at the three prompt-batch sites. 6724 is dead under MTP; change it for consistency.
- The verify (1155) keeps `true`. The checkpoint replay (7272) can pass `false`, since nothing reads its logits.

**Regression tests.**
- kcpp-e2e long-prompt hashes.
- `latest_logits` (savestate) within a tolerance. The last row now goes through a 1-row MMVQ instead of MMQ.
- A DFlash or DSpark fixture and a second MTP architecture produce the same text.
- Guidance + MTP: the first generated token now gets the guidance mix on the right row.

**Performance check.** kcpp-e2e long-prompt t/s and first-request time. Baseline: 3030 t/s warm, 4.7 s cold. Target:
≥ 3250 t/s and ~3.1 s.

### 3. Attention

**3a. MMA flash attention reading quantized K/V tiles** (n_tokens ≤ 8, decode included).
- **Why.** For quantized K/V on Ada and newer, `fattn.cu` (~707-722) picks:
  - VEC for `ne[1] ≤ 2`: one block per Q head and no GQA packing. It sits at 255 registers and 2 blocks per SM, so it
    is latency-bound.
  - Otherwise, MMA_F16 after converting the whole K/V window to F16 (`launch_fattn`), per layer, on every verify.
  
  The f16 MMA (1,8) kernel is already the GQA-packed decode kernel; it only needs to read quantized tiles.
- **Kernel.**
  - Add a `ggml_type type_KV` template parameter (default F16) to `flash_attn_ext_f16` and its `_process_tile`,
    `_iter` and `_case` (fattn-mma-f16.cuh). Quantized code sits behind `if constexpr`.
  - The ldmatrix, mma, softmax, sinks, softcap, stream-k and mask code are reused unchanged.
- **New fork header `fattn-mma-q.cuh`:**
  - **Raw loads.** `cp.async` of raw rows into shared memory: 16 B, or an 8 B `.ca` helper for rows that are only
    8-byte aligned (q8_0/q4_0/q5_0 at D = 128).
  - **Dequant.** One warp per row into the existing swizzled F16 tile, with dequantize.cuh's arithmetic (fp32
    `q·d + m`, then `__float2half_rn`), so the tile bytes equal the conversion path's. The half2 FMA helpers
    (`dequantize_V_*<half>`) round differently and are not used.
- **Budget:**
  - ≤ 48 KB of shared memory, so (1,8) runs 2 blocks per SM. Stage F16 in halves of D with the existing
    `nbatch_K2`/`nbatch_V2` loop.
  - The alternative, 1 block per SM with double-buffered raw tiles, is picked by benchmark.
  - Separate config entries (`nbatch_fa`, staging width, raw stages).
- **Dispatch.** One predicate inside `ggml_cuda_get_best_fattn_kernel`, shared by `get_alloc_size` and dispatch,
  holds when:
  - an instance exists (gated by `GGML_CUDA_FA_<K>_<V>`);
  - `K->type == V->type`;
  - `gqa_opt_applies`;
  - `ne[1] ≤ 8`;
  - alignment of `nb[]` and `view_offs` (never the data pointer).
  
  HIP, Volta, MLA (D 576, V-is-K-view) and sparse top-k keep conversion. Decode leaves VEC only where the benchmark
  shows the new path faster, at depths {0, 4k, 32k, 96k}.
- **Instances:** files named `fattn-mma-q*.cu` (the Makefile wildcard `fattn-mma*.cu` picks them up) for q5_1 first,
  then q8_0 and q4_0, × D ∈ {256, 128} × (1,8), (2,8), (4,8), (8,8), plus (·,4) for GQA 3–4.
- **The F16 scratch stays:** prefill still converts, and gallocr sizes for the worst graph (1.07 GB at 262k).
- **Regression tests:**
  - **New FLASH_ATTN_EXT cases, NMSE ≤ 5e-4 against the CPU:**
    - {q5_1, q8_0, q4_0} × hs {128, 256} × GQA {2, 4, 6, 8} × nb {1, 2, 3, 5, 8, 16} × kv {512, 8192};
    - both permutations;
    - kv 1025 (falls back);
    - sinks, softcap at hs 256, nr3 = 2, and hs 128 q8_0.
  - **Bitwise against the conversion path** (an env toggle forces it) where the config and grid match (kv 512).
    Elsewhere the stream-k split points differ.
  - **Per-step tooling checks:** compute-sanitizer racecheck, and cuobjdump showing no new spills.
  - **KL** of the q5_1 cache against the recorded reference, unchanged.
  - **kcpp-e2e** after 32k and 96k prompts.
- **Performance check:**
  - FA perf cases, µs and GB/s per layer. Baseline at 32k: VEC 165 µs, f16 MMA 92 µs. Target: decode ≤ 45 µs (the
    floor for 50 MB is ~34 µs); verify(5) ≤ 50 µs against ~130 µs of conversion plus MMA today.
  - llama-bench pp1/pp5 at 32k and 96k. Baseline: 20.3 / 23.2 ms and 24.3 / 31.3 ms.
  - kcpp-e2e t/s after long prompts. The MTP context adds an expected −0.7 ms/cycle at 32k and −2.5 at 96k.

**3b. No GQA padding in prefill.**
- **Change:** in `switch_ncols2` (fattn.cu ~324), for `Q->ne[1] ≥ 32` use the largest power of two dividing the GQA
  ratio (6 → 2, so (32,2), which spills less than (8,8)). Keep 8 for small batches.
- **Regression tests:** FLASH_ATTN_EXT prefill cases at GQA 3, 5, 6, 7 and 8, nb 32–512, f16 and q5_1; KL.
- **Performance check:**
  - ncu FA time per 32k-deep ubatch (baseline 131 ms for 16 layers);
  - llama-bench pp1024 at 32k and 96k (baselines recorded in Phase 0);
  - target −20 % FA time.
  - Before tuning further, record `dram__bytes_read`, L2 hit rate and tensor-pipe use.

**3c. Low priority: small attention-layer fusions.**
- Per layer and token today: 4 `fwht_cuda` (the Q/K pair share a matrix), 2 single-block `k_set_rows_quant`, and a 4 µs
  stream-k fixup. About 0.2 ms per token.
- Fuse the Q/K rotation pair and the K/V set_rows pair as CUDA patterns.
- Tests: test-backend-ops cases in the exact node order, with a fusion counter and a negative case. Perf: kernel count
  and tg1.

**3d. Measure first: the host KQ mask.**
- `set_input_kq_mask` scans n_kv on every decode, six times per cycle.
- Time it at 32k, 96k and 262k (timer at llama-kv-cache.cpp ~1770).
- If it costs more than ~0.5 ms per cycle, add a fast path for causal, no-SWA, single-sequence masks. Its regression
  test compares masks against the existing builder over random cell layouts, with multi-sequence and SWA falling back.

### 4. MTP loop host work

**4a. Batched input uploads in the scheduler.**
- **Why:** without events (`n_copies == 1`, the single-GPU case), `ggml_backend_sched_compute_splits` (ggml-backend.cpp
  ~1684) uploads each `FLAG_INPUT` with a backend synchronize plus a blocking copy. That is 0.52 ms of a draft step's
  0.74 ms of host time, and upstream (4ebdf2c74) does the same.
- **Change:** when the source buffer is host memory and the split backend has `set_tensor_async`:
  - issue `ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, nbytes)` on the split's stream, for
    user inputs and for host-resident intermediate inputs (a CPU split's output; WEIGHTS buffers excluded);
  - then one `ggml_backend_synchronize(split_backend)` after the loop.
- **Why it's correct:**
  - Stream order keeps device copies after earlier readers of `input_cpy`.
  - The one sync keeps the "caller may overwrite inputs after compute returns" contract. It also stops a later CPU
    split from reusing a host region before its copy lands.
  - The events path (pipeline parallel) is unchanged.
- **Also:** cache `graph_host_work_is_light` per graph build (llama-context.cpp ~2605).
- **Upstream:** a candidate for an upstream PR.
- **Regression tests:**
  - the `-g` matrix in both upload modes, bitwise; inputs change every batch, which exercises the overwrite contract;
  - partial `-ngl` (a CPU split feeding the GPU);
  - two virtual devices;
  - pipeline parallel;
  - kcpp-e2e hashes and draft counts.
- **Performance check:**
  - scheduler upload time per draft step (timer patch). Baseline: 0.52 ms. Prototype: 0.12 ms.
  - kcpp-e2e t/s. Prototype: +6–9 %.
  - llama-bench tg1 for plain decode.

**4b. Reduced-vocabulary draft head.**
- **Why:** each draft step multiplies by the full `model.output`, 1.04 GB of the 1.39 GB it reads. Only the top
  candidate matters: `p_min` is 0, drafts are top-k(10), and acceptance compares ids only (gpttype_adapter.cpp ~7060).
- **Change.** Reuse EAGLE3/DFlash's full-width reduced head pattern (`fill(-inf)` + `set_rows`, eagle3.cpp ~307):
  - Logits stay `n_vocab` wide: `concat(mul(build_lora_mm(view(head_w, rows 0..N)), head_s), fill(-inf, T−N),
    mul(build_lora_mm(view(head_w, rows T..n_vocab)), head_s))`.
  - T is the smallest control/EOG id ≥ N (here 248058).
  - Samplers, the raw-logit copy, `build_sampling` and the CPU fallback stay untouched.
- **Why views:**
  - row views keep whole quant blocks and pass `llm_graph_keeps_rows`;
  - `is_rotated` follows `view_src`, and both views share one RHT through the rotation cache;
  - LoRA is skipped on the draft head (views keep their "(view)" name), which only affects acceptance.
- **Hook:**
  - one place in `build_lora_mm`/`build_mm`: when `gtype == DECODER_MTP`, a draft vocabulary is set, and `w->ne[1] ==
    n_vocab`;
  - it covers all 14 MTP graphs (deepseek4 uses `build_mm`) and future ones.
- **Setting N:**
  - koboldcpp `--mtpvocab N` (0 = off), passed through `load_model_inputs` to a llama-ext setter before the MTP
    context's first reserve;
  - off for SPLIT_MODE_TENSOR.
- **If a prefix isn't enough:** reuse `d2t` + `set_rows` with a gathered head registered in `rotated_tensors`.
- **Regression tests:**
  - **Draft logit rows:** a test-hadamard-archs MTP-fixture mode, or a new test on the 27B, checks that rows < N and
    ≥ T are bitwise equal to the full head and the rest are −inf. It includes an HQ-rotated `output.weight`.
  - **kcpp-e2e:** text equal, or diverging only at near-ties.
  - **Smoke tests:** qwen35moe and step35.
- **Performance check:**
  - draft-step GPU time (timer). Baseline: 1.1–1.25 ms per step. Estimate: −0.47 ms.
  - kcpp-e2e acceptance and t/s for N ∈ {48k, 64k, off}, on the opencode-style replay prompts. Keep the smallest N that
    wins.

**4c. Gated: fewer decodes per cycle.**
- **Gate:** run only if Phase 0's GPU-busy measurement shows ≥ 0.3 ms of host gap per draft step after 4a.
- **Options:**
  - **Deferred catch-up:** decode the accepted prefix together with draft step 1. Saves one MTP decode, mask build and
    rebuild per cycle (~−0.5 ms). Touches `common/speculative.cpp`, which upstream is reworking (f1ea20621,
    60e9cf7a7, 1fb7ef3e3).
  - **An unrolled 4-step draft graph** with on-device argmax and embedding lookup. Ceiling ~−1.5 ms/cycle. It needs
    `tok_embd` rows on the GPU and per-step KV slots and masks inside one graph.
- **Tests:** kcpp-e2e hashes and draft counts identical; the rollback suite. **Perf:** cycle timer, kcpp-e2e t/s.

**4d. Retune draft length after 4a, 4b, 5 and 6a.**
- Sweep `--draftamount` 4–7 × `p_min` ∈ {0, 0.3, 0.5}. `p_min` is hard-coded to 0 (gpttype_adapter.cpp 901) and
  becomes an option.
- Cap at 7, so the verify stays ≤ 8 tokens and within MMVQ. Each extra draft adds a ~151 MB GDN snapshot plane.
- **Test:** kcpp-e2e with the option at its default gives unchanged hashes.
- **Perf:** kcpp-e2e t/s per setting. Change the default only on a clear win.

### 5. CPU sampler (`SampleLogits` and the verify loop, gpttype_adapter.cpp)

**Why.** About 0.46 ms per sampled row, ~4.4 rows per cycle. Three full-vocabulary passes run per row:

| pass | location (gpttype_adapter.cpp) | cost |
|---|---|---|
| `LowestLogit` | 6972 | 102 µs |
| array-of-structs candidate fill | 2343-2347 | 80 µs |
| top-3000 bucket sort | 2378, fn 1210-1290 | ~300 µs; 1.2 ms when more than 3000 logits fall in the top bucket |

**Change** (upstream koboldcpp code; keep the diff small; the adapter is built without AVX2, so scalar code):
- **Scratch row.**
  - memcpy the logits row into a thread-local scratch (~30 µs).
  - Apply the logit biases in their existing order.
  - `sample_dry` takes the float row (single caller).
  - Bans keep writing in place.
  - llama's logits buffer is never mutated (savestate copies it).
- **Chunked bucket select** from the scratch row:
  - the same 128 fixed buckets ([-10, 10]), 4 sub-histograms;
  - the bucket index computed exactly like today's `int()` plus clamp (+inf, NaN and overflow land as now);
  - collect in id order into `candidates`, then run the existing sort/partial_sort code.
- **Fallback:** the reasoning-budget and grammar paths build the array-of-structs from the scratch row and run today's
  code.
- **Lazy `LowestLogit`:** compute it only when a ban applies this token, before `sample_guidance` edits the row.
- **Stop sequences:** search only the tail of `concat_output` (it is append-only).

**Regression tests.**
- `test-kcpp-sampler`: recorded verify rows (the record step runs first, from kcpp-e2e prompts) through the frozen
  reference and the new code, bitwise.
- Stop-sequence cases: sequences spanning appends, overlapping matches, multi-byte pieces.
- kcpp-e2e hashes at a fixed seed with temperature, DRY and logit bias.

**Performance check.**
- `test-kcpp-sampler --bench` per-row µs at the adapter's flags. Baseline: fill + top-k 359–444 µs. Prototype: ~255 µs.
- Cycle timer "between cycles": baseline 2.1 ms, target ~1.3 ms.
- Record how often the top bucket overflows (the 1.2 ms case).

### 6. Kernels

All per-verify estimates and checks use pp5 at `n_rs_seq = 4` (`llama-bench -nrs 4`), the graph MTP actually runs.

**6a. Gated-delta-net layer fusions.**

Each step is a CUDA-side pattern in `ggml_cuda_try_fuse`, with a counter and graph_optimize alloc deps. There are no
op semantics changes and no other backends to update. In order of verify value:
1. **Conv update:** fuse {CONCAT, CPY × K, SSM_CONV, [ADD], SILU}, reading the already-gathered conv state.
   - −6 kernels per layer at K = 5; multi-sequence safe.
   - Also matches mamba2, nemotron-h and granite.
   - Model edit: expand `conv_output_silu` right after `build_conv_state`.
2. **State in place:** fuse {GET_ROWS, RESHAPE, GDN, CPY} so GDN reads the state through s_copy.
   - Only when `n_seqs == 1`. With several sequences, cell swaps or a shared zero row would let blocks read rows that
     others are writing.
   - Saves the 3 MB-per-layer gather.
3. **Gating prologue:** softplus(α + dt)·A and sigmoid(β) computed inside GDN, bitwise through the shared unary
   helpers.
   - Needs alloc deps, and β's matmul expanded before α's elementwise ops.
   - Move the gating and `build_norm_gated` into `delta-net-base` (qwen35, qwen35moe, qwen3next, qwen4exp share them).
4. **MMVQ fusion for 2–8 columns.** The kernel already sizes per-column state; the `has_fusion` instantiation and
   guards block it. In the verify this enables:
   - gate/up/GLU in 36 layers;
   - residual adds on down, ssm_out and wo;
   - alpha + dt.
   
   About 250 kernels per verify.
5. **α/β in one launch:** a second dst on the gate plumbing (both are HQ8_0 [5120, 48] and share the attn_norm RHT's
   Q8_1).
6. **Reshape-tolerant matchers:** the mm+ADD matcher and `rht_glu_fusable` look through one row-keeping RESHAPE. No
   model edits.
7. **Merged q/k L2 norm:** one norm over a [128, 2·H_k, T, S] view, now that RMS_NORM+SCALE is fused (7cc0fdffd).
   - Only on the fused GDN path: the unfused paths call `ggml_scale(q)`, which needs padded rows, so they use
     `ggml_cont`.

- **Also:** conv snapshot planes min(T, K) as mamba does. That saves 192 kernels per non-MTP decode graph, but the saved
  state bytes of unused planes change, so the state test is updated deliberately.
- **Regression tests:**
  - one test-backend-ops case per pattern, in qwen35's exact node order, CUDA vs CPU, with the fusion counter;
  - negative cases: an extra consumer, the output flag, `n_seqs > 1`;
  - the rollback and multi-sequence suite on CUDA fused vs `GGML_CUDA_DISABLE_FUSION=1` vs CPU, with `n_rs_seq` ∈ {0, 4}
    and fused GDN on and off;
  - kcpp-e2e hashes.
- **Performance check:**
  - pp5 (`-nrs 4`) kernel census and ms per verify, against the Phase 0 baseline;
  - a per-fusion env-toggle A/B in graph mode (ncu's serialized times overstate tiny kernels);
  - expected −0.8 to −1.0 ms per verify, mostly from steps 1–3.

**6b. MMVQ table for consumer Blackwell.**
- **Table:** add `MMVQ_PARAMETERS_BLACKWELL`. The device and host sides agree in every build variant via
  `__CUDA_ARCH_LIST__`.
- **Sweep it** (don't copy GB10's rules):
  - ncols 1–8;
  - nwarps including 5 (two full trips at K = 5120 Q4_K/Q5_K) and 8;
  - N ∈ {48, 96, 1024, 6144, 10240, 17408};
  - K ∈ {5120, 6144, 17408};
  - types Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0, Q3_K.
  
  Prefer rules aware of SM count.
- **Generalize** `c_promoted` and `should_halve_iters`, which are hard-wired to GB10.
- **Optional:** issue the first trip's weight loads before `pdl_sync`, only for WEIGHTS buffers (template flag).
- **Regression tests:**
  - test-backend-ops MUL_MAT for all types and ncols 1–8 (tolerance; the reduction order changes);
  - KL on the 27B eval set;
  - a LLAMA_PORTABLE CU13 build smoke test.
- **Performance check:**
  - bench-mmvq GB/s per shape. Baseline: 79 % overall, 73–76 % at 5120 × 6144.
  - llama-bench tg1 and pp5 (`-nrs 4`).
  - Expected −0.4 to −0.65 ms per decode token, −0.2 to −0.5 ms per verify.

**6c. GDN prefill.**
- **Step 1:** in `gated_delta_net_cuda`, prefetch the next token's q/k/v/g/β into registers (the loop prefetches
  nothing today) and add `__restrict__`.
  - Test: bitwise, via test-backend-ops GATED_DELTA_NET and kcpp-e2e hashes.
  - Perf: GDN µs per layer per ubatch (baseline 0.8 ms) and pp1024.
- **Step 2:** measure upstream's chunked graph path (`build_delta_net_chunking`: cumsum/tri/solve_tri/mul_mat, all on
  CUDA) at `n_rs_seq` 0. If it's competitive, use a graph split: a chunked prefix plus the fused K-tail that produces
  the snapshot states. That needs no new kernel.
  - Tests: tolerance against the recurrent path, snapshot states within tolerance, the rollback suite, KL.
- **Step 3**, only if step 2 falls short: a chunked tensor-core kernel with a recurrent K-tail.
- **Ceiling after step 1:** ~7 % of a short-depth ubatch.

**Not kernel work:** 28 of 64 layers don't fuse gate/up in 1-token decode because this quant mixes their types. Fix it
in the HQ quant recipe (same type for gate and up), checked by KL.

### 7. Correctness

**Guidance with drafting is wrong today.**
- `sample_guidance` mixes `llama_get_logits(ctx)` (row 0) into every sampled row (gpttype_adapter.cpp ~2259), but the
  verify samples rows 0..4.
- `guidance_n_past` advances by 1 per cycle, while the main context advances by accepted + 1.
- **Fix:** add the guidance condition to the draft gate (~6692).
- **Regression test:** kcpp-e2e with a negative prompt at temperature 0 gives the same text with and without MTP.
  It fails before the fix.

**Investigate: drafting under a grammar.**
- `grammar != nullptr` disables drafting (6692; f75bbb945, no reason recorded).
- Verification samples each row with the grammar state advanced only through accepted tokens (~7076), so it looks
  exact.
- It matters if OpenCode's tool calls run under a grammar.
- **Test:** kcpp-e2e grammar prompts at temperature 0, with drafting allowed vs disabled: identical text. **Perf:** t/s.

## Not pursued, and why

- **A shape-aware CUDA graph key:** catch-up and drafts already get distinct keys from the two graph arenas; grammar
  disables drafting; the draft shrinks only in a request's last cycles. Item 1's capture counter shows whether a
  chained-head architecture (step35) ever needs it.
- **A second scheduler for the MTP context's two graphs:**
  - rebuilds cost ~0.15 ms, twice per cycle;
  - it would touch ~35 `sched.get()` sites and could double the compute buffer;
  - 4a and 4c address the real host costs.
- **A GPU top-K prefilter for the CPU sampler:**
  - a backend sampler on the target disables the raw-logit copy that bans, guidance and savestate need;
  - CUDA top-k sorts all 248k values (CCCL 2.8);
  - exactness is fragile.
- **Argmax instead of the draft's top-k(10):** measured, no gain.
- **Optimistic overlap of the next draft with CPU sampling:** it only hits when every draft and the bonus match (~55 %
  of cycles under greedy), worth ~−0.45 ms. It needs `draft()` split while upstream reworks it.
- **A faster q5_1 → F16 conversion kernel:** already ~1.45 TB/s, and 3a removes it from the verify path.
- **VEC tuning** (occupancy cap, more parallel blocks, more columns):
  - the kernel is register-saturated;
  - measured 8 % at 4× blocks;
  - 3a's MMA (1,8) is the GQA-packed decode kernel.
- **Caching the F16 conversion across prefill ubatches:** 17 GB at 262k.
- **q8_0 K/V instead of q5_1:** 9.1 vs 6.4 GB at 262k, and 1.42× the bytes read.
- **`GGML_CUDA_GRAPH_OPT=1`:** measured, no effect.
- **Merging all of concedo_experimental now:** only the four commits above are needed; the rest waits for concedo's
  next stable merge.
- **HQ rotation cost:** see `inference_speedup_plan.md`, "Future work".
- **Windows-side GPU contention:** environmental. Measure around it.

## Implementation conventions

- **Tests first or alongside:** every change follows the test policy above. The regression test goes in the same
  commit as the change, or an earlier one, and the baseline is recorded before the change.
- **Comments:** minimal, in the surrounding style; no narration and no references to this plan.
- **Upstream files:** `ggml-cuda.cu`, `ggml-backend.cpp`, `fattn*`, `mmvq.cu`, `llama-graph.cpp`, the model files and
  `gpttype_adapter.cpp` get small hooks.
  - Logic goes in fork files (`fattn-mma-q.cuh`, `tools/perf/*`, `tests/test-kcpp-sampler.cpp`) or existing helpers
    (`delta-net-base`).
  - Prefer existing mechanisms: upstream's graph rules, EAGLE3's full-width reduced logits, the gate/up MMVQ
    plumbing, `ggml_cuda_try_fuse` with alloc deps, `rotated_tensors`.
- **Toggles:** env toggles used only for A/B measurement come out before commit. A toggle stays only if a regression
  test uses it, as with the conversion-path switch in item 3a.

## Phases

Each phase ends with the full standing suite and an A/B of the phase against the previous phase end, recorded in
"Implementation record".

0. **Infrastructure and baselines.**
   - **Prerequisites:** the cherry-picks and their tests (see "Implementation record").
   - **Build the in-repo infrastructure:**
     - `tools/perf/` (llama-bench with `-nrs`, kcpp-e2e, stall, bench-mmvq);
     - the `-g` mode and graph counters;
     - the rollback and multi-sequence cases;
     - the FA perf cases;
     - the sampler row recorder.
   - Rebuild `koboldcpp_cublas.so`.
   - **Record baselines** at depth 0, 32k and 96k:
     - llama-bench tg1 / pp1 / pp5 (`-nrs` 0 and 4) / pp1024;
     - kcpp-e2e (hashes, draft counts, t/s, prompt t/s);
     - the timers;
     - GPU-busy fraction per draft step;
     - KQ mask time;
     - a pp5 `-nrs 4` ncu census.
1. **Items 1, 2, 4a, 7 (guidance fix).** Each with its regression test committed first. Ask Concedo/Henk about
   February in parallel.
2. **Item 5.** Record rows, then the reference test, then the change.
3. **Item 4b**, then 4d.
4. **Item 3a**: tests and perf cases, then q5_1 (1,8)/(8,8), then the other types. Then 3b.
5. **Item 6a**, one pattern per commit, each with its test-backend-ops case and counter, then a pp5 census.
6. **Item 6b**: sweep, table, A/B.
7. **Gated items:** 6c, 4c, 3c, 3d and the grammar investigation, each on the measurement its section names.

## Implementation record

### Prerequisites (branch `mtp-speed`)

- **Cherry-picks** (with `-x`), on top of master 922ef81fb:
  - 7cc0fdffd: RMS_NORM + SCALE fusion. Clean.
  - bbd645262: state restore cleanup. Clean except `tests/test-save-load-state.cpp`, which concedo deletes; upstream's
    file was taken.
  - 22be82de0: sm_70 MMVQ table. One conflict, resolved by keeping the fork's `defined(__CUDA_ARCH__)` guard with the
    new Volta lower bound.
  - 869034b4b: skipped. Its target `llama_memory_recurrent::is_empty()` doesn't exist in the fork.
- **Fork commit 4bbe23800:** `test-save-load-state` and `test-save-load-state-cuda` make targets, plus a .gitignore entry.
- **Regression tests:**
  - test-backend-ops (CUDA0) all pass:

| case | passed |
|---|---|
| RMS_NORM_SCALE | 10/10 |
| NORM_SCALE | 10/10 |
| RMS_NORM | 51/51 |
| RMS_NORM_MUL_ADD | 36/36 |
| GATED_DELTA_NET | 36/36 |
| RHT | 438/438 |
| RHT_FUSED | 384/384 |

  - test-hadamard-archs-cuda: 264 passed, 0 failed (4 skipped as before).
  - test-save-load-state-cuda, all 9 tests: pass on Qwen3-0.6B Q4_K_M and on Qwen3.8-27B HQ4_K_M (hybrid, `-c 4096`).
  - The same test linked against the pre-cherry-pick `llama.o` fails test 9 on both models ("logits changed after
    failed restore", every logit NaN). So it guards the fix.
- **Performance** (27B HQ4_K_M, q5_1 K/V, slow GPU mode, 3 interleaved rounds):
  - kernels per decode token: 1953 → 1857; all 96 `scale_f32` are gone, so the fusion fires;
  - tg128 median: 2288 → 2272 ms (−0.7 %, matches ~0.13 ms of removed launches per token);
  - pp5 with graphs still off: 33.7 → 30.9 ms (noisy);
  - one bogus llama-bench row (tg128 405 ms) was discarded.
- **sm_70 table:** no effect on sm_120; covered by the build and the cases above.

### Phase 0: infrastructure (d48556e4b, 2f5e7d6ea; branch `mtp-speed-impl`)

- **In the repo:**
  - `tools/perf/llama-bench.cpp` + `make llama-bench-cuda`, with `-nrs`. With `-nrs > 0` the context is at least 256
    tokens, since n_ubatch (capped by n_ctx) must exceed n_rs_seq.
  - `tools/perf/kcpp-e2e.py`: configs mtp, nomtp, guidance (MTP vs no MTP must match), grammar (same), media (vision,
    TTS, whisper, TTS during MTP generation), and the sampler paths (sampled with DRY/bias, mirostat, xtc,
    dynatemp+nsigma, adaptive-p, bans). The long prompt is pinned to `53ed051ce:src/llama-graph.cpp`. A cross-config
    difference is accepted only at a top-2 logprob gap below 1e-3, probed at the first differing token. Golden:
    `tools/perf/golden/kcpp-e2e-27b.json`.
  - `tools/perf/stall.cu` (`make tools/perf/stall`), `tools/perf/standing-suite.sh` (the correctness half of the
    standing suite).
  - `test-hadamard-archs -g`, with `ggml_backend_cuda_graph_launch_count`/`_capture_count`, and
    `test-hadamard-archs-cuda-novmm` (ggml-cuda.cu with `GGML_CUDA_NO_VMM`).
  - `tests/test-recurrent-state-rollback.cpp` (upstream's at 53ed051ce plus the fork's cases) and
    `tests/test-recurrent-state-rollback.sh` (CUDA, CUDA without fusion, CPU with `-ub 16`).
  - FLASH_ATTN_EXT perf cases at the 27B's shapes (hs 256, 4 KV heads, GQA 6, kv 8k/32k/96k, nb 1/2/5/8/1024).
- **Deviations from the plan:**
  - `-g` keeps the fixture's 2 layers: `full_attention_interval` is 2 there, so qwen35 already has an attention layer.
    It runs each batch size on a fresh 16-token prefill with `n_ctx` 256, which caps n_kv at 256 (4 batches, 3 for
    T = 64).
  - F16-expert MoE models launch graphs too (MMF needs no sync), so `-g` asserts graph launches only for dense
    models, plus no re-capture churn for all.
  - The HQ4_K_M fixture is skipped where the model saver lacks the arch (gemma3n, bitnet, apertus, step35) and for
    qwen4exp, whose Q4_K_M quantizes a tensor its graph adds (the CPU add aborts) — a quantizer bug, not pursued.
  - No SD model on this box, so no SD smoke test; no DFlash/DSpark or second MTP architecture either.
  - bench-mmvq and the sampler-row recorder are built with the items that need them (6b, 5).
- **Bug found and fixed (6712d8ab0):** a recurrent rollback deeper than the last ubatch's snapshots was accepted and
  restored a stale gated-delta-net state. A ubatch of T tokens writes snapshots for its last min(T, n_rs_seq + 1)
  states, so rolling back T tokens (the state before the ubatch) read a plane from an older ubatch. Upstream's own
  multi-seq split replay does exactly that (a 3-token tail rolled back 3), and failed here at nmse 1.9e-3 on both 27B
  quants. koboldcpp could hit it when a reused prompt trims a context whose last ubatch was short (single-token
  decodes with MTP loaded). The memory now tracks per cell how many snapshots the last ubatch wrote and refuses a
  deeper rollback; the fork's guard case fails without the fix (nmse 3.3e-3) and passes with it. Upstream's
  multi-seq case now rolls back 3 of a 4-token tail (nmse 4.4e-5 against a reference that took a 1-token step).
- **Rollback suite:** CUDA, CUDA without fusion and the CPU pass every case. The CPU leg runs with `-ub 16`: at the
  default ubatch each context's buffer-discovery decode of 512 tokens made a 27B HQ run take over 30 minutes.
- **Environment during Phase 1:** `stall` lost 13–16 % of wall time to preemptions in every session (a slower mode
  than the plan's measurements: MTP generation was 88–92 t/s at baseline).

### Phase 1 (303f03c52, 0af97dedc, 6942c1bd2, 8ff930639)

All A/Bs interleaved, 3 rounds, 27B HQ4_K_M, q5_1 K/V, FA, contention 13–16 %.

| item | regression test | result |
|---|---|---|
| 1 graphs | `-g`: 249 configs bitwise graphs on vs off, every T in 2..64 launched as a graph (before: "no graph at T=2"); NO_VMM: 0 launches; `-ngl 1`, 2 virtual devices, pipeline parallel pass; kcpp-e2e and media hashes equal | pp2/pp5/pp8 29.7/32.3/37.7 → 21.7/22.0/25.0 ms; pp5 at nrs 4 33.1 → 24.0 ms; tg32 unchanged; MTP generation 88.8/80.6/92.0 → 120.6/106.8/124.5 t/s (+33–36 %) |
| 2 prompt logits | kcpp-e2e: long-prompt texts equal, draft counts equal | 9.6k prompt cold 1725 → 2402 t/s (wall 6.60 → 5.10 s), warm 2456 → 2614 t/s |
| 7 guidance | kcpp-e2e guidance: MTP and no-MTP texts differed at token 15 (top-2 gap 1.05) with item 2 alone; identical after | drafting off under guidance |
| 4a uploads | `-g` with the reference child on sync uploads, in both parent upload modes, `-ngl 1`, 2 devices, pp | draft-step uploads 0.53–0.64 → 0.13–0.18 ms; MTP generation 122.1/109.4/124.3 → 129.8/118.0/134.3 t/s (+6–8 %); tg32 588 → 572–581 ms |

- Item 1 keeps the legacy-pool gate inside ggml-cuda.cu (`ggml_cuda_graph_set_enabled` returns false without the
  VMM pool) rather than a field in common.cuh, which would rebuild every CUDA object.
- Item 2: the 1 GB host output buffer grows lazily, so its size isn't in the startup log; not re-measured.
- Phase 1 end: MTP generation 88.8/80.6/92.0 → 129.8/118.0/134.3 t/s (+42–46 %), cold long prompt 1725 → ~2500 t/s.
- Standing suite at Phase 1 end: every step passes. The first run failed only the RHT counter assertions because
  `test-backend-ops -j 4` ran other cases on parallel threads against the global counters; the suite now runs the
  counter-asserting cases on one thread.

### Phase 2: item 5 (17c6aa479)

- **Change as planned**, with two refinements found by profiling the new selection on recorded rows (copy 26, bucket
  index 61, histogram 41, scatter 82, sort 75 µs per row):
  - the bucket indices are computed with SSE2 (`cvttps2dq` returns INT_MIN for NaN and out-of-range values exactly like
    the scalar `int()`, and mul then add rounds the same; a scalar loop elsewhere);
  - the scatter skips 16-entry chunks without a candidate (~1 % of the vocabulary reaches the kept buckets).
  - The sorts are unchanged, so the selection is bitwise the old one; it lives in `otherarch/kcpp_sampler_util.h`,
    shared with the test, together with the stop-sequence tail scanner.
- **Regression:** `test-kcpp-sampler` on 974 rows recorded from kcpp-e2e (KCPP_SAMPLER_RECORD) plus edge rows (ties,
  everything in the top bucket, a tied boundary group, ±inf, NaN, 3e38, vocabularies of 100 and 2000, everything
  below the lowest bucket) × biases/bans/DRY penalties × k ∈ {3000, 40, 1}: 29460 cases bitwise; stop scan equal to a
  full find on 2000 random append sequences (pieces spanning appends, overlaps, multi-byte). kcpp-e2e: every text
  and draft count unchanged across greedy, DRY/bias, mirostat, xtc, dynatemp+nsigma, adaptive-p, bans, grammar and
  guidance.
- **Performance:** selection 404 → 286 µs/row (`test-kcpp-sampler --bench`); the boundary bucket held > 6000 entries
  in 0 of 974 real rows. Scratch timer around the sampling loop of an MTP cycle: 1.99–2.15 → 1.16–1.29 ms
  (550–578 → 320–356 µs/row), the plan's target. kcpp-e2e MTP generation +2–6 % (median of 3 interleaved rounds;
  within the run-to-run spread for some prompts, which the cycle timer resolves).
- koboldcpp sometimes reports a bogus eval time for a request that is not the first (e.g. 400 tokens "at 264 t/s"):
  `e2esum` medians over 3 rounds absorb single glitches.

### Phase 3: items 4b (50cccab69) and 4d

- **4b as planned.** T for Qwen3.8 is 248044 (the first control/EOG id at or after N), so N = 65536 keeps rows
  [0, 65536) ∪ [248044, 248320). `is_rotated` follows `view_src`, so the HQ-rotated head works through the views,
  and both views share one RHT through the rotation cache.
  - Regression: `test-mtp-draft-vocab` (27B HQ4_K_M, HQ-rotated output.weight): kept rows bitwise equal to the full
    head's, the rest -inf, for a 5-row batch and three single-token steps.
  - Not run: qwen35moe/step35 smoke tests (no such models here).
  - Performance (interleaved, 3 rounds): MTP generation off/48k/64k short0 145.7/162.0/154.0, short2
    141.8/151.2/152.3, sampled 124.6/140.9/135.4 t/s; accepted/rejected drafts at 64k 275/40 vs 276/39 (48k 272/44).
    The option stays off by default; 65536 keeps acceptance closer to the full head at the same speed.
- **4d.** `--draftpmin` makes `p_min` an option (default 0).
  - Bug found: with `p_min > 0` a draft can stop before its first token, and koboldcpp aborted the generation ("Draft
    model produced no draft tokens", empty text). It now verifies the single token. With `--draftpmin 0.3` 9 of 11
    kcpp-e2e texts equal the golden; short1 diverges at token 370 of 400 and the temperature-0.7 run diverges (the
    verify batch shapes now vary between 1 and 5 tokens).
  - Draft length at p_min 0 (mtpvocab 64k, contention 0.3 %, geomean over 9 prompts): 4 → 155.9, 5 → 155.2,
    6 → 149.3, 7 → 150.1 t/s. The default stays 4.
  - p_min (with the fix, mtpvocab 64k, geomean): d4 165.4; d4/0.3 139.1 (−16 %), d5/0.3 136.9, d6/0.3 131.1,
    d6/0.5 113.1, d7/0.5 111.2 (−33 %). Shorter, varying verify batches lose both accepted tokens per cycle and
    CUDA graph reuse. The default stays 0.

### Phase 4: items 3a (348af46d5) and 3b (54d3ce936)

- **3a, simpler than planned.** The quantized loader dequantizes each 8-value piece of a K/V row with the conversion
  path's functions (`dequantize_q5_1` etc., then `__float2half`) and writes it straight into the swizzled F16 tile,
  synchronously: no raw `cp.async` staging, the kernel runs with `nstages = 0`. The shared memory is the 1-stage F16
  budget (about 33 KB at (1,8), D 256), so 2 blocks per SM without separate config entries. Instances: q4_0, q5_1,
  q8_0 × D {128, 256} × (1,8), (2,8), (4,8), (8,8), (2,4), (4,4), (8,4) (21 files); GQA 2 keeps the old path.
  - `GGML_CUDA_FA_Q_CONVERT=1` selects the same MMA kernel on the F16 copy (not VEC), so the test compares exactly
    the dequantization step. The plan's `get_alloc_size` sharing: `BEST_FATTN_KERNEL_MMA_Q` needs no F16 scratch.
  - Decode (1–2 tokens) leaves VEC everywhere the predicate holds: at every measured depth the new kernel is faster
    (8k: 42.5 → 31.2 µs/layer; llama-bench decode at depth 0: 22.79 → 22.05 ms, 4k 25.67 → 23.98 ms).
  - Regression: `test-fattn-mma-q`, 180 cases (3 types × hs × GQA 4/6/8 × kv 512/8192 × 1–8 tokens): bitwise at
    kv 512; at kv 8192, 1–2 tokens differ by ≤ 1e-7 nmse, because the quantized kernel's smaller shared memory fits
    more blocks and stream-k splits the KV range elsewhere. test-backend-ops: the plan's new cases plus every other
    FLASH_ATTN_EXT case pass.
  - Spills (ptxas): q5_1 (1,8) D 256 197 registers, none (F16: 254, none); (8,8) D 256 80 B (F16: 330 B);
    (8,8) D 128 32 B (F16: none) is the only new spill.
  - Not run: compute-sanitizer racecheck, which cannot attach under WSL2 ("Failed to initialize WDDM debugger
    interface", needs EnableDebuggerInterface.bat as administrator).
  - Performance, per layer at the 27B's shapes (q5_1): 8k decode/verify(5) 42.5/74.9 → 31.2/46.1 µs; 32k
    147.5/313.7 → 88.2/121.9 µs; 96k 570/951 → 348/449 µs. llama-bench pp5 at nrs 4: 32k 37.2 → 29.4 ms, 96k
    41.2 → 35.6 ms; decode at 32k 22.95 → 21.79 ms (VEC → MMA-Q; the MMA kernel on the F16 copy: 24.13 ms).
    kcpp-e2e (3 interleaved rounds): MTP generation after a 27.5k-token prompt 59.9 → 62.8 t/s, after 88.6k
    34.3 → 41.0 t/s (+20 %); short prompts +1–6 %.
- **3b, generalized:** for 32+ tokens the tile's head count is the largest power of two dividing the GQA ratio
  (GQA 3/5/7 → 1, 6 → 2, 12 → 4), not just 6 → 2.
  - Per layer at the 27B's shapes: 1024-token prefill at 32k 7.0 → 5.3 ms (q5_1 and f16), 96k 27.0 → 20.8 ms;
    f16 nb 512 at 4k/16k/128k −26/−26/−19 %. llama-bench pp1024 at depth 0/4k/8k unchanged (341.7/344.7/358.9 →
    336.1/344.5/357.0 ms); kcpp-e2e prompt rates at 9.6k tokens are within the run spread, 88.6k +5 %.
  - KL: the deltas are smaller than the metric's own movement under equivalent rounding changes (q5_1 opencode
    0.00610 → 0.00702 at --parallel 1 but 0.00704 → 0.00614 at --parallel 2; wiki/heldout ±1 %; q8_0 opencode +1.4 %
    with top-1 up).
- The kcpp-e2e golden was re-recorded after 3a: decode with MTP off now runs the MMA kernel, so those greedy texts
  moved; the guidance/grammar cross-checks, media and deep configs pass. `deep` is a new e2e config (32k and 96k
  prompts at a 131072 context).
- Standing suite at Phase 4 end: PASS (all steps, including the CPU rollback leg and the e2e check with deep).

### Phase 5: item 6a (996092a1f, 24a3cd980, 1eb9fde73, 1265c94af, 0c8ee129b, 456805cd7, 756c559de, 3e8ae2c1f, dc62c870e)

The pp5 verify graph at `n_rs_seq = 4` went from 2389 kernels (Phase 4 census) to 1553; ncu's serialized time at base
clocks from 21.76 to 19.52 ms. Each step is one commit with its test and an interleaved llama-bench A/B (8 rounds
unless noted; "paired" is the per-round difference).

- **Step 1, conv update** (996092a1f): {CONCAT, CPY × K, SSM_CONV, [ADD], SILU} as one kernel, with alloc deps for
  its inputs. pp5 19.99 → 19.22 ms, tg32 471 → 458 ms. SSM_CONV_UPDATE in test-backend-ops (counter; concat as output
  and more than 32 tokens not fused).
- **Step 2, state in place** (24a3cd980): with one sequence the GDN reads its state row through `s_copy`, the gather
  skipped. pp5 19.69 → 18.72 ms. GATED_DELTA_NET_STATE_IN_PLACE (rows the snapshots overwrite; two sequences not).
- **Step 4 before 3, MMVQ fusions for 2–8 columns** (1eb9fde73): gate/up/GLU and the residual adds of down and wo.
  - First attempt slower (pp5 19.29 → 20.35 ms): the fused kernel checks for a gate at run time, which at 5 columns
    cost enough registers (134–137 vs 96–118) that fused GLU ran ~40 % slower than the two matmuls. With more than one
    column the gate is now a template flag. bench-mmvq at 5 columns: GLU equal or faster, residual add ~1 µs faster,
    except IQ4_XS at K 17408 (~3 µs slower; loading the bias in the epilogue instead was slower for every type).
  - Fused only where the node alone runs MMVQ (`ggml_cuda_mul_mat_runs_mmvq`), as an RHT may have written src1 in
    that kernel's Q8_1 layout; biases must have the output's strides.
  - pp5 19.05 → 18.62 ms (6 rounds), tg32 neutral. MMVQ_FUSION (counter, broadcast bias, matmul as output).
- **Step 3, gating in the GDN** (1265c94af): sigmoid(β) and softplus(α + dt)·A computed in the kernel through the
  unary ops' helpers (now in unary.cuh); add, softplus, mul and sigmoid skipped, planned once per evaluation with the
  in-place gathers. Bitwise the same (`test-gdn-gating`). pp5 20.59 → 20.34 ms in 10 rounds under 13 % GPU contention,
  8 of 10 paired rounds faster. GATED_DELTA_NET_GATING (second consumer of β, g as output).
- **Step 5, α and β in one launch** (0c8ee129b): two adjacent MUL_MATs of the same input with weights of one type and
  shape run as one MMVQ launch, the second through the gate plumbing written to its own output. The models expand the
  β and α projections next to each other. At 5120 → 48 Q8_0 (24 blocks) a pair takes 7.5 µs instead of 13.5 at 5
  columns, 3.4 instead of 5.5 at one. pp5 18.28 → 18.12 ms, tg32 462.5 → 455.6 ms.
- **Step 6, reshape-tolerant matchers:**
  - {MUL_MAT, RESHAPE, ADD} with a row-keeping reshape (ssm_out and the residual; 456805cd7). It fused 1 of 48 at
    first: the allocator puts the sum in place of the residual, which the generic memory-range check refuses. The
    kernel allows that alias (each block reads its rows' bias before writing them); only src1 must not overlap
    (3e8ae2c1f). `test-cuda-fusion-alloc` allocates with ggml_gallocr as the scheduler does, which test-backend-ops
    can't (6 of 6 failed with the old check). pp5 18.20 → 18.09 ms after the fix.
  - Not a GLU but the same prologue: {UNARY silu, MUL, [RESHAPE], RHT}, the gated norm before ssm_out's rotation,
    computed in the rotation's SWIGLU prologue (756c559de). unary_gated kernels 64 → 16; A/B within noise.
- **Step 7, one L2 norm over q and k** (dc62c870e): on the fused GDN path the models norm a [d, 2·H_k, T, S] view and
  pass two strided views; every backend's GDN reads q/k through strides. rms_norm 177 → 129. pp5 18.10 → 17.84 ms.
- **Not done:** α + dt through MMVQ (superseded by step 3); conv snapshot planes min(T, K) (after step 1 the copies
  are inside the fused kernel, so only the unused planes' bytes would go, not kernels).
- Every step: kcpp-e2e hashes unchanged.
- **Phase 4 end → Phase 5 end** (interleaved, 0 % contention): llama-bench pp5 `-nrs 4` 20.08 → 17.87 ms (8 of 8
  rounds), tg32 483 → 451 ms; at depth 88k pp5 37.8 → 32.6 ms, tg16 359 → 345 ms. kcpp-e2e (3 rounds): MTP generation
  +5–11 %, MTP off +3–4 %, short prompts +5–14 %, long prompts unchanged.
- **Timing on this box:** the WSL2 guest TSC ran 9.8 % fast against the Windows clock and time sync stepped the wall
  clock back 2.7 s every 32 s. Durations from the wall clock (koboldcpp's timer, llama-bench) came out negative or
  seconds off when they spanned a step; the deep96k generation rates of single kcpp-e2e runs were unusable for that
  reason (one Phase 5 run at 41.8 t/s, two at 33.3). Both timers now use the steady clock (9d3152d27); absolute rates
  read ~9 % low until the VM's clock is recalibrated (`wsl --shutdown`), ratios are unaffected.
- Standing suite at Phase 5 end: PASS, with the new steps test-gdn-gating and test-cuda-fusion-alloc and the counted
  cases MMVQ_FUSION, SSM_CONV_UPDATE, GATED_DELTA_NET_STATE_IN_PLACE (the last two had been missing from the list).

### Phase 6: item 6b (88504aed4, golden 3689ffed3)

- **Sweep:** bench-mmvq built 14 times with the GENERIC table overridden (nwarps 1/2/4/5/8 × rows per block 1/2/4;
  8 × 4 exceeds static shared memory for the fused kernels), the 27B's shapes (K 5120 → N 48/1024/6144/10240/12288/17408,
  6144 → 5120, 17408 → 5120), Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0, 1/2/5/8 columns, 2 rotated rounds (bench-mmvq now warms up
  for 300 ms: the GPU idles at ~200 MHz).
  - The clear result: launches with few warps in flight (5120 → 48 and → 1024) are 15–45 % faster with 5 warps.
  - Re-tuned tables gained up to 2.7 % of MMVQ time in isolation (weighted by the 27B's shape counts) but lost 5–10 %
    on some shapes; a K-aware one (1 warp over 4 rows for K ≥ 6144 from 5 columns, 2 over 2 below) gained 2.7 % in
    bench-mmvq and nothing in the model's graph (pp5 17.74 → 17.79 ms), where these shapes run as fused kernels.
- **Kept:** `MMVQ_PARAMETERS_BLACKWELL` = the generic table, but a launch with fewer than 6 warps per SM in flight
  takes 5 warps per block (the `halve_iters` variant, now also from 2 columns and at 1 column outside GB10). Host and
  device pick the table as for GB10, so a portable build agrees: a CU13 portable mmvq (PTX for compute_75..120,
  `KCPP_LIMIT_CUDA_MAX_ARCH=1200`) passes the matmul tests through JIT and launches the same block shapes (ncu).
- **Bug found on the way:** a 5-row block over 512 MUL_MAT_ID rows wrote past its rows: the kernel's last-block bound
  is `stride_col_dst`, which for MUL_MAT_ID spans the experts. Latent upstream (rows per block were powers of two);
  `small_k` launches keep 4 warps. Caught by the existing MUL_MAT_ID sentinels.
- **Results:** bench-mmvq (3 rounds, 27B-weighted) 1/2/5/8 columns −0.5/−1.4/−1.0/−0.3 %, no shape slower beyond
  noise; llama-bench pp5 `-nrs 4` 17.68 → 17.64 ms (7 of 8 paired rounds), tg32 neutral. Under the plan's estimate
  (−0.2 to −0.5 ms).
- **Quality:** KL against BF16 in 5-token ubatches (new `test-hadamard-ppl -ub N`, the verify regime), mean KL
  wiki/heldout/opencode 0.00932/0.00814/0.00594 → 0.00935/0.00806/0.00588. Greedy texts moved at near-ties: golden
  re-recorded (cross-checks pass).

### Phase 7: gated items (ab916ca46 3d, 7086d7f44 grammar, 0432296e9 GDN perf cases)

Gate measurements with a scratch build timing every graph (CUDA events), each draft call and each KQ mask build,
under kcpp-e2e. On this VM the monotonic clock ran 9.8 % fast (see Phase 5), which first looked like a 1.4 ms host
gap per verify: spin-polling the events showed the GPU starts within 0.1 ms and the gap is exactly the clock's 9.8 %.

- **3d, KQ mask (done):** at 88k context a 1-token mask took ~100 µs (the first row scans every cell with per-cell
  checks; later rows copy), ~6 per MTP cycle: 0.4–0.6 ms per cycle at 96k, linear in the context (the user runs 262k).
  When every used cell carries the sequence (`llama_kv_cells::seq_has_all_used`, O(1)) the row needs only the
  positions array: causal masks without SWA or ALiBi, 2D (M-RoPE, qwen35) checked on the cells at the token's
  position. 1-token mask 94–105 → 32 µs, 5-token 123–131 → 59 µs at 88k. Masks identical: kcpp-e2e hashes (incl.
  deep 32k/96k) and the rollback suite unchanged. `test-kv-mask`: 16000 random layouts (empty cells, cells used
  without a sequence, shared cells, positions out of order/repeated/past, 1D and 2D) against the per-cell checks.
- **Grammar (done):** drafting was disabled under any grammar (f75bbb945, no reason). The verify samples each row
  with the grammar advanced through accepted tokens only, so it is exact. kcpp-e2e's new `grammar_long` prompt
  (212 tokens of JSON) gives the no-MTP text with MTP, 58.1 → 105.9 t/s (142 of 190 drafts accepted); the check
  requires drafting under it (failed before).
- **4c (gate not met):** per cycle the 5 MTP graphs (catch-up 0.375 ms GPU, 4 draft steps 1.005 ms each) run in
  5.14 ms of real time against 4.40 ms GPU: ~0.18 ms host gap per draft step, under the 0.3 ms gate. The catch-up
  decode itself is 0.49 ms of wall per cycle; deferring it into draft step 1 stays a candidate (~2 %).
- **6c (no gain):**
  - Step 1: prefetching the next token's q/k/v/g/β in `gated_delta_net_cuda` (bitwise) made it 5–10 % slower at
    1/5/1024 tokens (the kernel already hides load latency across warps; the extra registers cost occupancy);
    `__restrict__` alone is neutral. Not committed.
  - Step 2: upstream's chunked graph path (`build_delta_net_chunking`, `fused_gdn_ch` off) is 17 % slower on pp1024
    (2875–2922 vs 3468–3488 t/s).
  - Step 3 (a chunked tensor-core kernel with a recurrent K-tail) not attempted: the GDN is 1.02 ms per layer per
    1024-token ubatch, ~17 % of prefill, so this stays the largest prefill opportunity, as its own project.
  - GDN perf cases at the 27B's shapes added to test-backend-ops.
- **3c (not pursued):** the Q/K rotation pair and the K/V set_rows pair are adjacent and fusable, but together ~32
  launches (~0.05 ms) per graph, ~0.3 % of a verify.
- **Phase 5 end → Phase 7 end** (llama-bench, interleaved, steady clock): depth 0 pp5 17.64 → 17.60 ms, tg32 neutral;
  at 88k pp5 28.03 → 27.74 ms and tg16 288.7 → 286.3 ms (3 of 3 rounds, the KQ mask); MTP under a grammar +82 %.
- **Standing suite at Phase 7 end:** PASS after one test refinement: test-mtp-draft-vocab compared the reduced draft
  head's tail bitwise in a 5-row batch, which the Blackwell table's widened launch for that small matmul rounds
  differently (all within 1e-4); the head rows and the single-token draft steps stay bitwise (7f84a2224).

### After the plan (2026-10-05): 6932598d0, 80960a5dd, 043f43968, 5d1b23b35, d4c86c5c3

After a WSL restart the VM's clock is right again: over 90 s the guest's monotonic clock and Windows agree to 0.16 %
(9.8 % before). Absolute rates in the records above read ~9 % low; plain decode is now 415 ms per 32 tokens in
llama-bench where Phase 7 recorded 448.

- **Deep context (finding):** MTP generation after an 88.6k prompt was 41–49 t/s while a cycle inside the loop takes
  26.5 ms there (24.3 ms at short context: drafts 5.5, verify launch 0.2–1.0, wait 17, sampling 1.0). The rest was
  SmartCache: hybrid models snapshot their whole state (the 88.6k context's KV is 2.2 GB) at the start of every
  generation, and koboldcpp turns SmartCache on for them by itself. The time went into the host buffer, not the
  copy: a buffer too small was freed and a new one faulted in with 4 KB pages, 1.3 s per 2.2 GB against 0.13 s for
  the device-to-host copy. On WSL2 freed memory goes back to the host within seconds (free page reporting) and
  costs ~0.4 s/GB to fault in again (0.036 s/GB right after the free).
- **SmartCache buffers (6932598d0):** on Linux one anonymous mapping with transparent huge pages that grows with
  mremap, keeping its pages, the kernel faulting in only the tail (MADV_POPULATE_WRITE). Snapshots that grow a slot
  754–797 → 358–362 ms (1.3 GB); the 88.6k gen-start snapshot 1.9–3.4 → 1.3–1.6 s; in an agentic session the
  per-turn snapshot is 89–115 ms with no reallocation spikes (the old code paid 461 ms whenever a slot outgrew its
  headroom). test-kcpp-state-buffer (growth keeps the bytes written, huge pages; both fail with the old vector).
- **Tools (80960a5dd):** kcpp-e2e `agentic`: a 27.5k-token conversation extended by ~3k tokens of new input per turn,
  150 tokens generated per turn, `--smartcache 2`; turns at 30–42k tokens take 1.7–2.5 s (~1 s prompt processing,
  ~1 s generation, ~0.1 s snapshot). `tools/perf/lb-ab.py` replaces the scratch A/B scripts lost with /tmp.
- **Deferred MTP catch-up (4c's first option): dropped.** With the right clock the catch-up decode costs 0.29 ms
  of host time per cycle (not 0.49), and its GPU time runs while the CPU samples the verify rows; merged into draft
  step 1 it would save under 1 % of a cycle.
- **`--mtpvocab 65536`** re-measured (3 interleaved rounds): short-context MTP generation +5–9 %, generation in the
  agentic turns +3 %, turn wall −1.9 %. Still off by default.
- **GDN, two state columns per warp (043f43968):** bitwise; for prompt batches when the one-column grid fills 3/4 of
  the resident block slots (it lost 34 % at 4.5 blocks per SM and gained 30 % at 12). pp1024 275.1 → 268.9 ms.
  Register prefetch of the next token was slower again (580 → 860 µs); one fused reduction per token
  (attn = g S.q + delta k.q) gave only −7 % and is not reproducible across launch shapes without explicit FMAs.
- **GDN chunked kernels (5d1b23b35, 6c step 3, in FP32 rather than on tensor cores):** prep (T by forward
  substitution with shuffles, P, decays; k.k and q.k once per q/k head) and a scan over blocks of 16 state columns,
  the recurrent kernel for the snapshot tail. The first version was slower (813 vs 588 µs); staging k/q in shared
  memory, fitting 3 blocks per SM, register-tiled products, a bank-conflict-free lane order and the per-q/k-head
  prep brought the 27B layer at 1024 tokens to 306–316 µs (recurrent 583–587). llama-bench pp1024 268.1 → 248.1
  ms (−7.5 %), pp512 −6.4 %, verify and decode unchanged.
  - Accuracy: both kernels at NMSE ≤ 1e-12 against an FP64 recurrence (also with correlated keys and slow decays);
    on the 27B's real activations each call matches the recurrent kernel to NMSE 1e-13 to 3e-12.
  - **The model amplifies any rounding change.** Against the recurrent run the chunked one has a self-KL of 0.0014
    (top-1 99.3 %), but so does the recurrent run itself with 256-token ubatches (0.0016) or a q8_0 K/V cache
    (0.0013): about a fifth of the model's KL against BF16. KL against BF16 moved within that noise
    (wiki/heldout/opencode 0.009498/0.008381/0.007022 → 0.009566/0.008600/0.006219), perplexity unchanged.
    Greedy texts diverged at top-2 gaps of 0.13–0.15 after 9.6k-token prompts; KL checks of future changes should be
    read against this floor.
- Prefill at 1024-token ubatches after this batch: 275.1 → 248.1 ms (two GDN commits), with pp5 and tg unchanged.
- Golden re-recorded after the chunked kernels (d4c86c5c3); the standing suite (now with test-kcpp-state-buffer and
  the agentic e2e config) passes in full, including the rollback suite on CUDA, without fusion and on the CPU.

### Open after this plan

- Taking the SmartCache gen-start snapshot off the generation path: the attention KV below the prompt end doesn't
  change while generating, so it could be copied on a second stream after the recurrent state; ~0.1 s per agentic
  turn now, more at 100k+ tokens. Needs a deferred-copy hook in llama's state writer.
- The prefill KQ mask at long context: ~3.9 ms per 1024-token ubatch at 48k on the host, plus its upload (100 MB);
  only the decode/verify masks have the fast path.
- The MMVQ block shape by K from 5 columns (1 warp over 4 rows for K ≥ 6144): 2.7 % of MMVQ time in isolation,
  nothing measurable in the graph.
- `--mtpvocab 65536` (4b) stays off by default; +5–9 % short-context MTP generation, +3 % in agentic turns.
- The latent MMVQ bound in MUL_MAT_ID (rows per block not a power of two write past their rows; Phase 6 keeps 4 warps
  for those launches): an upstream report.
- Not pursued further: the deferred MTP catch-up (see above); a tensor-core (TF32/BF16) GDN scan, whose rounding would
  exceed the FP32 kernels' and gains less on consumer Blackwell, where TF32 runs at the FP32 rate.
