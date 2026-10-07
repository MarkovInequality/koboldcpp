# Plan: Qwen Image 2.1 from upstream stable-diffusion.cpp

## Summary

Qwen Image 2.1 (text-to-image, editing, transparent RGBA output) is supported in stable-diffusion.cpp and in
concedo_experimental, but not in our tree. Master's sd.cpp is the master-866 sync (0bd509c59). concedo_experimental
(6ef0d37bd) gets 2.1 through its master-885 (a915109d8) and master-917 (40fbf48a1) syncs. Those syncs, and the
master-872 sync under them, also carry ~280 commits of unrelated work: a tokenizer rewrite on oniguruma, new models,
and performance changes. This plan brings in the 2.1 support alone.

| # | change | effect |
|---|---|---|
| 1 | **Replay the eight upstream 2.1 commits and one prerequisite** onto pure sd.cpp master-866 in a scratch clone, then three-way merge the result into `otherarch/sdcpp` | 2.1 models load, generate and edit; nothing else from the 917 sync comes along |
| 2 | **Adapt to the 866 base:** the tokenizer's `encode`, no SageAttention, `ggml_ext_conv_3d` without the direct path, RoPE without the circular refactor | builds on our tree with upstream's numerics |
| 3 | **kcpp side, as in concedo_experimental:** no TAESD for 2.1; `get_model_info` treats 2.1 as an editing Qwen model; a hint when an edit fails without the vision mmproj | koboldcpp's adapter passes reference images through and avoids the TAESD crash |
| 4 | **Tests without downloaded weights:** synthetic models, `test-sd-qwen21`, `test-sd-wan-vae`, `sd-e2e.py` through koboldcpp | regression coverage in the standing suite |
| 5 | **One-time fidelity check with real weights** against a concedo_experimental build, then delete the downloads | the port matches upstream kcpp bitwise |

**Test policy** (as in `mtp_speed_plan.md`):
- Every change ships with a regression test runnable from the repo, in the same commit or an earlier one. Fixes come
  with a test that fails before the fix; exact changes are compared bitwise.
- Every change gets a performance check against a baseline recorded before it: interleaved A/B, at least 3 rounds,
  median.
- Earlier checks rerun at each step's end.
- Tests must not depend on downloaded model files (the user's requirement); real weights are for the one-time
  fidelity check only.

## Background

### The model (upstream commits and `docs/qwen_image_2.1.md`)

- **Detection:** `model.diffusion_model.txt_in.text_norm.weight`.
- **DiT:**
  - 32 layers, hidden 4096, 128-wide heads; the published GGUF and Comfy exports fuse the MLP into one `gate_up`
    weight, gate first.
  - Text features enter through a zero-centered RMS norm (`txt_in`).
  - Every block output passes through `tanh(modulation)` gates.
  - A reference image takes its latent grid in the token layout through "image slots": vision tokens in the
    prompt, one per 2×2 latent patch.
- **VAE:**
  - Its own Wan-derived VAE: 64 latent channels, 16× spatial, 4 image channels (RGBA).
  - The BF16 export stores its Conv3d weights with a temporal size of 1.
  - Its activations overflow FP16 convolutions, so convolution inputs are scaled by 1/128 and the outputs scaled back.
- **Text encoder:**
  - Qwen3-VL-8B (arch `QWEN3_VL`): last layer, 4096 wide.
  - Editing needs its vision tower, as an mmproj or in the same file, and reference sizes in multiples of 32.
- **Schedule:** flow, with an exponential shift interpolated between (256, 0.5) and (8192, 0.9), and the last model
  step stretched onto `shift_terminal` 0.02.
- **Prefix KV cache:** text and reference K/V don't change between steps; they are cached per condition (types auto,
  f32, f16 or quantized).
- **LoRAs** are trained on the unfused `gate_layer` and `proj`. They must land on the two halves of `gate_up`.
- **Weights for the fidelity check** (not committed):
  - leejet/Qwen-Image-2.1-GGUF Q4_K (4.2 GB);
  - the Comfy-Org 2.1 VAE (0.68 GB);
  - Qwen3VL-8B-Instruct Q4_K_M (5.0 GB) with its F16 mmproj (1.2 GB).

### Upstream commits (stable-diffusion.cpp, master-866..master-917)

| commit | what | notes |
|---|---|---|
| 137f740 | add Qwen Image 2.1 support (#1994) | version, DiT, VAE config, conditioner, Flux scheduler for 2.1, 4 image channels, name conversion |
| e112ab5 | alpha channel input (#2021) | in `src/` only `video.cpp`; the rest is the CLI and server |
| 241518b | latent2rgba preview (#2032) | a 64-channel latent preview with alpha |
| 97d932b | VAE tiling retries only on allocation failures (#2019) | prerequisite: adds `last_compute_status()`, which the prefix cache's out-of-memory retry reads |
| 2dc7f54 | prefix KV cache (#2035) | runner cache, per-condition prefix ids in `sample()` |
| 740c7ae | configurable cache types, early cache scheduling (#2045) | `used_flash_attn` out-parameter; `get_context(graph)` in control, sensenova, ltx_vae, wan_vae |
| b167b94 | flow schedule defaults (#2048) | `FluxScheduler` takes the model version |
| 0a9340c | scale the VAE convolutions (#2054) | a `scale` parameter on `ggml_ext_conv_3d` |
| 510bccf | map LoRAs onto the fused MLP (#2057) | |

Left out, as not 2.1 or speed only:
- e6281b6 configurable conditioning cache;
- 88411ef circular RoPE refactor;
- 70c1dbc one-frame 2D Wan convolutions; 3e037a8 direct 3D VAE convolutions;
- 187b256 SageAttention;
- 275ab58 and 17860c0, CPU overhead; adcac69, flash-attention head padding;
- the tokenizer rewrite (4964abd, f9ddc0f, 59c23bc and the oniguruma import);
- every model added since 866.

### kcpp side in concedo_experimental

- **40fbf48a1** (squashed sync), "fix Qwen Image 2.1 edit mode detection":
  - `get_model_info` sets `is_qwenimg` through `sd_version_is_qwen_image`;
  - `supports_ref_image` also covers `is_qwenimg`.

  Upstream removes 2.1 from `sd_version_supports_ref_latent_img_cfg` because 2.1 routes references through the VLM.
  Without this, koboldcpp's adapter would drop the images.
- **40fbf48a1**, "disable TAE for Qwen Image 2.1": the TAESD substitution in the kcpp block skips 2.1. The Wan 2.1 tiny
  VAE it would pick can't take 64-channel latents and crashes.
- **7e6bb3845:** prints a hint when a generation with reference images fails.
- **No Python changes.**

### Our tree (master 7442881ca)

- **Layout:** `otherarch/sdcpp` mirrors sd.cpp's `src/`, `include/`, `examples/common/` and `examples/cli/`.
  - 177 of its 198 sd.cpp files are byte-identical to master-866.
  - 21 carry kcpp edits, among them the kcpp block in `diffusion_engine.cpp`, the `kcpp_sd` namespace in
    `stable-diffusion.cpp`, `model_loader.cpp`, `media_io` and `ggml_extend.cpp`.
- **Code points:**
  - TAESD substitution: `src/pipeline/diffusion_engine.cpp:876`.
  - `get_model_info`: `src/stable-diffusion.cpp:815`, `:828`.
  - Failure return in `sdtype_generate`: `sdtype_adapter.cpp:1745`.
  - `SDVersion`: `src/model.h:41`; version detection: `src/model_loader.cpp:570`.
  - Header dependency list: `SDCPP_COMMON_BASENAMES`, `Makefile:731`.
  - GUI tooltip for Clip-1: `koboldcpp.py:10113`.
- **Tokenizer:** `encode` returns the token vector: `src/tokenizers/tokenizer.h:38`.

## Findings that shape the design

### Experimental's sync commits can't be taken as a unit

- **Size:**
  - a915109d8 touches 44 files;
  - 40fbf48a1 touches 69 (+4580/−987);
  - both apply cleanly only on top of 15e978865, the master-872 sync: 81 files, +97k/−448k, the oniguruma tokenizers.

  Taking them means taking most of the 917 sync.
- **Upstream is granular:** in sd.cpp each feature and fix is its own commit, so replaying those commits is precise.

### A three-way merge carries upstream deltas over kcpp edits

Our files are sd.cpp 866 plus kcpp edits. For every file the replay changes, `git merge-file` merges three versions:
- base: sd.cpp 866;
- ours: our file;
- theirs: the replayed file.

The merge applies upstream's change and keeps the kcpp edits. Files only the replay adds are copied.

### Dependencies visible in the diffs (expected adaptations)

- **`last_compute_status_`:** 2dc7f54 reads it, and 97d932b adds it. Bring in 97d932b.
- **Tokenizer:**
  - 137f740's conditioner calls the bool-returning `tokenizer->encode(text, out, cb)` from the master-872 rewrite.
  - Use 866's vector form, as the neighbouring branches do (`conditioner.hpp:2254`).
  - 137f740's `model_builders.cpp` also passes `tokenizers` to `LLMEmbedder` (4964abd); drop the argument.
- **SageAttention (187b256):** 740c7ae and 2dc7f54 reference it.
  - Leave out the `sage_attn` parameter.
  - `!sage_attn_enabled` is always true here.
- **Convolution scale:** 0a9340c adds `scale` after `direct` (3e037a8) in `ggml_ext_conv_3d`.
  - Add `scale` after `force_prec_f32` and keep our 3D path.
  - Skip the one-frame 2D path (70c1dbc).
- **RoPE:** `qwen_image_2_1.hpp` at 917 uses the circular-RoPE helpers (88411ef). Keep the earlier `Rope::embed_nd`;
  without circular padding the embeddings are the same.
- **Formatting:** b56c686 reformatted the 2.1 file, which gives whitespace-only conflicts.
- **Enum shift:** inserting `VERSION_QWEN_IMAGE_2_1` into `SDVersion` shifts every later value.
  - `model_version_to_str` gets the matching entry.
  - Nothing outside the library may depend on the numbers; check `koboldcpp.py` (it doesn't).

### Shared code the replay changes

- **`wan_vae.hpp`:**
  - singleton temporal kernels imported in `CausalConv3d::init_params`;
  - `is_2D` detected by an exact name;
  - `Resample` for 2D VAEs;
  - the feature cache persisted through `persist_cache_tensor`;
  - the convolution scale (1 for every other model).
- **`ggml_runner`:** `cache(name, tensor, graph)` schedules cache outputs where they are registered;
  `get_context(graph)` in four runners.
- **`ggml_extend`:** `used_flash_attn` (null everywhere else) and the convolution scale.
- **`FluxScheduler`:** its anchors become members. Flux's arithmetic is unchanged (3840.f either way) and must stay
  bitwise.
- **VAE decode retries:** only on `GGML_STATUS_ALLOC_FAILED` (97d932b), for every model.

So every model on the Wan VAE (Wan, Qwen Image, LingBot, Krea2, Anima) needs a bitwise regression check against master.

### Testing without downloaded weights

- **Tiny models are valid models.** The 2.1 DiT, the Qwen3-VL encoder and the LingBot DiT read their sizes from the
  weights. The encoder's heads stay fixed at 32×128.
- **VAEs must be full size.** Their dimensions are fixed:
  - 2.1: dim 96, dec_dim 144, z 64, ≈340M parameters;
  - Wan 2.1: ≈127M.

  Generate them at test time from the published files' tensor layouts (names, types, shapes, ~24 KB) with
  deterministic values. That also exercises the name conversion and the singleton-kernel import.
- **Through the public API.** Runners refuse to execute without a residency manager, so the tests go through
  `new_sd_ctx` with `generate_image` or `generate_video`, which also covers loading and detection.
- **No small Wan.**
  - Wan's DiT width follows its layer count (30 layers → the 1.3B dimensions, 40 → 14B).
  - Its text encoder is a 4096-wide UMT5; the 256k vocabulary alone is ≈1B parameters.
  - LingBot uses the Wan 2.1 VAE and a Qwen3-VL encoder and reads its sizes from the weights, so it can decode
    multi-frame clips through that VAE.
  - koboldcpp doesn't treat LingBot as a video model, so that test goes through the C++ API.
- **Synthetic models must respond to conditioning.** At random-init scale the `tanh(modulation)` gates sit near 0,
  and conditioning would barely reach the pixels. Give the synthetic modulation weights a gain, so edits and LoRAs move
  pixels well above rounding.

## Design

### Replay and merge

1. **Clone and branch:** clone leejet/stable-diffusion.cpp without blobs into the scratchpad and branch at
   `master-866-42d6c0a`.
2. **Cherry-pick:** 137f740, e112ab5, 241518b, 2dc7f54, 740c7ae, b167b94, 0a9340c, 510bccf, then 97d932b.
   - Resolve each conflict against upstream's own history, following the rules above.
   - Take ours for `examples/server/`, which isn't vendored.
3. **Merge:** for every file the replay changes under `src/`, `include/`, `examples/common/` and
   `examples/cli/main.cpp`, run `git merge-file` (base 866, ours, theirs) and copy new files. Docs and assets stay
   out.
4. **Build:** fix the adaptations that only show at compile time (tokenizer, sage).
5. **kcpp side:**
   - In `diffusion_engine.cpp`'s kcpp block, the TAESD substitution skips `VERSION_QWEN_IMAGE_2_1`.
   - In `get_model_info`, `is_qwenimg = sd_version_is_qwen_image(...)` and `supports_ref_image ... || is_qwenimg`.
   - In `sdtype_generate`, when a generation with reference images fails on a Qwen model, print the hint about the
     Qwen3-VL mmproj.
6. **Makefile:** add `src/model/diffusion/qwen_image_2_1.hpp` to `SDCPP_COMMON_BASENAMES`.
7. **Diff against 917:** compare every 2.1-only file with sd.cpp 917. Only the circular-RoPE refactor and formatting
   may differ.

### Branch and commits

- Branch `qwen-image-2.1` off master; master stays untouched.
- **First commit:** test tooling.
- **Second commit:** the replay with the kcpp side. Its message lists the upstream commits and the adaptations.

### Using it

- `--sdmodel`: a 2.1 GGUF or safetensors.
- `--sdvae`: the 2.1 VAE. The Qwen Image and Wan VAEs don't fit.
- `--sdllm`: Qwen3-VL-8B or a finetune or quant of it. Nothing else works: the architecture is fixed, and the DiT takes
  4096-wide last-layer features.
- `--sdclip1`: its mmproj, for editing.
- `--sdflashattention`: makes the prefix cache F16.
- `--sdclipdevice main`: koboldcpp runs the image text encoder on the CPU by default.

## Phase 0: baselines (before any change)

1. **Master library:**
   - Make a git worktree at master and tar-copy this tree's `*.o`, `*.d` and `*.so` files into it.
   - `touch` the objects one minute ahead, then `touch` two minutes ahead every worktree file that fails `cmp`
     against this tree.
   - `make koboldcpp_cublas` rebuilds only what differs. Use the result with `sd-e2e.py --lib`.
2. **Experimental reference:** the same, in a worktree at `upstream/concedo_experimental`. Compare files by content
   path by path: experimental moved the root sources into `kcpp_src/`, so a diff list misses them.
3. **Real weights:** download the four files listed above (~11 GB) and Wan 2.1 T2V 1.3B (fp16 diffusion, UMT5
   Q4_K_M, Wan 2.1 VAE, ~6.7 GB) into scratch folders. Confirm master can't load 2.1 ("Could not load image model").
4. **`test-sd-wan-vae` golden:** record it with the master build. The test uses only headers that exist on master.
5. **Timings:**
   - Wan 1.3B on master (text encoder on the GPU);
   - Qwen 2.1 on the experimental build at 512×512, 12 steps.

## Steps

Each step is a commit that builds and passes its own tests and all earlier ones.

1. **Test tooling:**
   - `tests/sd-synthetic.h` writes synthetic safetensors from a block's parameters or a recorded layout.
   - `tests/data/*.tensors` hold the two VAE layouts.
   - `test-sd-wan-vae`, with its golden from master.
   - Makefile rules (`SD_TEST_OBJS`), `.gitignore`, and the standing suite.

   Passes on master.
2. **The replay and the kcpp side:**
   - with `test-sd-qwen21`;
   - with `sd-e2e.py` and its golden.

   `test-sd-qwen21` can't build on master (fails before). Check each fix by reverting it (see Tests).
3. **Fidelity and A/B with real weights,** then delete the downloads.

## Tests

### Regression

1. **`tests/test-sd-qwen21.cpp`** (make target `test-sd-qwen21`; CPU, ~10 s, no downloads):
   - **Flow schedule (b167b94):**
     - 2.1 sigmas within 1e-5 of the official formula, computed independently in double precision;
     - the last model step at 0.02;
     - Flux sigmas bitwise those of the fixed anchors.

     Reverting the 2.1 defaults fails it.
   - **Convolution scale (0a9340c):**
     - small inputs match a direct convolution;
     - unscaled inputs past 65504 overflow an F16 im2col;
     - scaled by 1/128 they stay finite and accurate.
   - **Token layout:**
     - text then the target image;
     - a reference's vision slots take its latent grid (positions and segments as specified);
     - mismatched slots are rejected.
   - **Detection:** 2.1 from its text norm; Qwen Image and Qwen Image Layered still detected.
   - **Generation** with synthetic models: tiny DiT and Qwen3-VL with a vision tower, the 2.1 VAE layout, 64×64,
     3 steps.
     - Loads as "Qwen Image 2.1".
     - `get_model_info` reports an editing Qwen model with a 16× VAE. Reverting the kcpp fix fails it.
     - RGBA output, not flat, a bitwise repeat.
     - The prefix cache stored in F32: from the library log, since there's no flash attention on CPU.
     - Cache off agrees within one pixel level.
     - A q8_0 cache stays close, and the log shows q8_0.
     - Latent previews arrive RGBA. Reverting 241518b fails it.
     - An edit through the vision encoder changes the image.
     - A LoRA on `gate_layer` and `proj` matches the same delta baked into `gate_up`, against a control with the halves
       swapped. Reverting 510bccf fails it.
     - An edit without vision weights fails instead of ignoring the image.
     - Loading with koboldcpp's `embd_res/taesd.embd` still decodes with the 2.1 VAE. Without the exclusion the
       process aborts.
   - **`--write-models DIR`:** writes the synthetic diffusion model, VAE and combined encoder for `sd-e2e.py`.
2. **`tests/test-sd-wan-vae.cpp`** (make target `test-sd-wan-vae`; CPU):
   - A tiny LingBot video model decodes a 9-frame 64×64 clip through the Wan 2.1 VAE layout.
   - The clip is bitwise equal on a repeat and not flat.
   - Its hash must equal the golden recorded with master. CPU results depend on the instruction set, so the golden is
     per machine.
3. **`tools/perf/sd-e2e.py`** (koboldcpp, GPU, flash attention, synthetic models from `test-sd-qwen21 --write-models`;
   golden `tools/perf/golden/sd-e2e.json`):
   - t2i at 128×128 twice (bitwise, RGBA);
   - an edit through the vision tower in the combined encoder file.
4. **Fidelity, once, with real weights:**
   - **Qwen 2.1:** t2i, a repeat, an RGBA prompt and an edit at 512×512 must be bitwise equal between this build and the
     experimental build. Look at the images: the sign text, the edit applied, the transparent background.
   - **Wan 1.3B:** a frame and a 9-frame clip must be bitwise equal between master and this build.
   - Afterwards delete the downloads.
5. **Build:** `sdmain` compiles, covering `examples/common` and `examples/cli` against the new `media_io` signature.
6. **Standing suite:** `tools/perf/standing-suite.sh` builds and runs `test-sd-qwen21` and `test-sd-wan-vae`, and runs
   `sd-e2e.py check`.

### Performance (interleaved, ≥ 3 rounds, median)

- **Wan** (the real 1.3B and the synthetic LingBot clip), this build against master: expected equal. The shared changes
  are cache scheduling and a scale of 1.
- **Qwen 2.1,** this build against experimental at 512×512, 12 steps: expected a small gap from the excluded speed
  commits. There's no master baseline to gate on, so record it.
- **Text encoder on the GPU** for timings (`--sdclipdevice main`); otherwise the CPU encoder dominates.

## Risks

- **A hidden dependency** on a skipped upstream commit that still compiles. The bitwise fidelity check against
  experimental catches it.
- **Tokenizer differences** between 866 and 917 (59c23bc's pre-tokenization) could misalign image slots. The edit
  fidelity check and the slot layout test cover it.
- **The next concedo merge** conflicts in every replayed file. Resolve by taking concedo's side; the tests stay.
- **The VAE retry change** (97d932b) affects every model: no more tiled retries after execution errors.

## Not in this plan

- **The excluded upstream commits:**
  - circular RoPE (seamless tiling) for 2.1;
  - the generic conditioning cache;
  - the Wan VAE speedups;
  - SageAttention;
  - the CPU overhead and flash-attention padding changes.
- **Qwen Image 2.1 ControlNet** (the "fun controlnet union" in the Comfy repo): not in these commits.
- **The Qwen3.5-9B prompt enhancers:** they rewrite prompts, which is a chat model's job, not a text encoder's.
- **The Clip-1 tooltip** (`koboldcpp.py:10113`) still says Qwen2.5VL; experimental hasn't changed it either.
- **LingBot in koboldcpp:**
  - koboldcpp treating LingBot as a video model;
  - LingBot's head-count detection dividing the default width when `blocks.*` sorts before `patch_embedder`.
- **Upstream 2.1 fixes after master-917.**
