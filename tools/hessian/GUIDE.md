# From a calibration dataset to a Hessian

This guide walks through the whole pipeline with this repo's tools: build a calibration dataset, run it through a
model to collect the full input Hessians, check the file, and quantize with it. The commands are the ones that
produced `qwen38-calib-v1` and `qwen38-27b-hessian-v1.gguf` for Qwen3.8-27B. The reference docs for each part:

- the dataset scripts: `calib/README.md`
- the collector and its file format: `README.md`
- the evaluation harness: `eval/README.md`
- the quantizer's `--hessian`: `../quantize/README.md`, "With a hessian file"
- the design and every measurement: `plans/hessian_collection_plan.md`

```
calib/ scripts ──> calib/dataset/calib.jsonl ──> hessian-collect ──> model-hessian.gguf (+ imatrix)
                                                                          │
                                         quantize_gguf --hadamard --hessian ──> quantized model ──> eval/
```

## What it takes

| | Qwen3.8-27B |
|---|---|
| GPU | 32 GB (an RTX 5090 here) |
| RAM | 30 GB (collection peaks at 25 GB) |
| disk | ~3 GB while building the dataset; the Hessian is 101 GB (`1.57 GB` of Grams per layer) |
| time | dataset: ~4.5 h of generation plus ~6.5 h of OpenCode sessions; Hessian: ~3.3 h |

Build everything from the repo root:

```
make LLAMA_CUBLAS=1 -j8 koboldcpp_cublas hessian-tokenize hessian-collect-cuda test-hessian tools test-hadamard-ppl-cuda
```

- `koboldcpp_cublas`: the server the dataset scripts generate replies with.
- `hessian-tokenize`: the tokenizer every dataset script uses, identical to the collector's.
- `hessian-collect-cuda`: the collector.
- `test-hessian`: checks the Hessian file.
- `tools`: builds `quantize_gguf`, among others.
- `test-hadamard-ppl-cuda`: scores quantized models.

Inputs:

- **The generator model** (`GEN_MODEL`): the model whose replies fill the dataset. Use the model you're calibrating
  for, so the replies look like its real output. Here that's `Qwen3.8-27B-HQ8_0.gguf`, the 8-bit HQ copy, which fits
  the GPU and is within KL 0.0004 of BF16.
- **The model to collect on:** normally the same HQ8_0 file. The collector captures each GEMM's input before the
  Hadamard rotation, so an HQ model gives natural-space Grams, the same as BF16 up to 8-bit noise: 2–8 % relative
  Frobenius on Qwen3-0.6B, against 140 % for a wrongly rotated capture. The Hessian then serves every quantization
  of the model, rotated or not, whatever the seed.
- **For the OpenCode part of the dataset:** OpenCode (v1.18.32 here) on `PATH`, your own OpenCode sessions (read
  from `$XDG_DATA_HOME/opencode`), and your repos that `calib/opencode/tasks.jsonl` names (`USER_REPOS`).
- **Python 3** with `pyarrow requests numpy jinja2 pyyaml` for the dataset scripts; `analyze.py` also needs `scipy`.
- **wikitext-2** (`wiki.test.raw`, which `decontam.py` and `eval/` use, and `wiki.train.raw` for the collector's
  tests): from `wikitext-2-raw-v1.zip` in the `ggml-org/ci` dataset on Hugging Face. Keep them in `tools/models/`
  (git-ignored):

  ```
  curl -L -o tools/models/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  unzip -j tools/models/wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw wikitext-2-raw/wiki.train.raw -d tools/models
  rm tools/models/wikitext-2-raw-v1.zip
  ```

## 1. Build the calibration dataset

Run from `tools/hessian/calib/`. Temporary files go to `calib/work/` (git-ignored, about 3 GB); the result goes to
`calib/dataset/` (git-ignored: it holds third-party text and your own sessions).

```
cd tools/hessian/calib
export PY=python3                                  # with the packages above
export GEN_MODEL=/path/to/Qwen3.8-27B-HQ8_0.gguf
export USER_REPOS=/path/to/your/repos

# the sources and the prompts
$PY fetch.py                      # 1. slices of the sources -> work/sources/; pins their revisions in sources.json
$PY prompts.py                    # 2. the prompt list -> work/gen/prompts.jsonl

# the model's own replies
./serve.sh gen                    # 3. koboldcpp for generation (port 5002)
$PY generate.py                   # 4. replies -> work/gen/docs.jsonl (about 4.5 h; resumes where it stopped)
$PY raw.py                        # 5. untemplated text (FineWeb-Edu, Wikipedia, PG-19, code) -> work/gen/raw_docs.jsonl

# full OpenCode sessions, driven by the model on clones of real repos
./serve.sh opencode               # 6. koboldcpp with the settings OpenCode uses (port 5003)
opencode/setup.sh                 # 7. an isolated OpenCode config; a snapshot of what must not change
opencode/run.sh --user-config < /dev/null     # 8a. the prompt check (below): one request with your own config,
opencode/run.sh e01-ripgrep-loc < /dev/null   #     one short calibration task,
$PY opencode/promptdiff.py work/opencode/captures/_user-config.jsonl work/opencode/captures/e01-ripgrep-loc.jsonl
opencode/run.sh < /dev/null       # 8. the other tasks -> work/opencode/docs.jsonl (about 6.5 h; resumes)
./serve.sh stop

# the evaluation set, decontamination, the split
$PY opencode/evalset.py           # 9. your recorded sessions -> dataset/eval-opencode.jsonl (evaluation only)
$PY techeval.py                   # 10a. the out-of-domain evaluation text (below) -> dataset/tech-eval.txt
$PY decontam.py --eval ../../models/wiki.test.raw --eval dataset/tech-eval.txt   # 10. drop overlaps with eval texts
$PY split.py                      # 11. calib/held-out split -> dataset/, with manifest.json
$PY stats.py                      # 12. the checks
./cleanup.sh                      # 13. removes work/
python3 stats.py --after-cleanup  #     the kept files still match the manifest
cd ../../..
```

- **Every step can be rerun.**
- **Steps 3–4 and 6–8 need the GPU for hours:** run them with nothing else on it. Under WSL the CUDA runtime doesn't
  see other processes' VRAM.
- **Always run `opencode/run.sh < /dev/null`:** `opencode run` reads a piped stdin and waits for its end.
- **Each OpenCode run is sandboxed:** `$HOME` is overlaid with a throwaway layer in `$TMPDIR` (keep it outside
  `$HOME`). `stats.py` checks afterwards that your OpenCode data and repos are unchanged.
- **Without OpenCode,** steps 6–9 can't run.
  - `split.py` still builds a dataset from what exists, but it lacks the `opencode` category and the OpenCode
    episodes in `agentic`, together about 15 % of the target.
  - `stats.py`'s token-target checks then fail.
  - `evalset.py` needs step 8's captures too, so `eval/` loses its opencode set.
  - Lower `TARGETS` in `calib/common.py` to match.
- **The prompt check (step 8a).**
  - *Why:* OpenCode builds each session's first ~11k tokens (the system prompt and the tool schemas) from its config.
    Every calibration session starts with them, and they're about a fifth of the OpenCode tokens that count. So
    the isolated config must make OpenCode send the same prefix as your real sessions, or the Hessian is calibrated
    on a prompt you never send.
  - *What it catches:* the prompt names the model ("You are powered by the model named …"), and OpenCode picks its
    prompt by model id. A tool that the config denies disappears from the tool list.
  - *What it did here:* the first check caught both. The calibration config used a different provider and model
    name, and it denied `webfetch`. `setup.sh` now copies your provider and model names and leaves webfetch
    allowed.
  - *How:* `run.sh --user-config` sends one request, "hi", through OpenCode with a copy of your own config. The copy
    changes only the server URL, to the logging proxy, and runs in the same sandbox as the tasks. `promptdiff.py`
    compares that request with a calibration task's first request: the tool list, each tool's schema, and the
    system prompt as a diff.
  - *Expected result:* identical tools, and a system prompt that differs only in its `Working directory:` line.
    `max_tokens` may differ, since the calibration config caps replies at 8192, but it isn't part of the prompt.
  - *When:* before the long runs, and again whenever you change `setup.sh`'s config or upgrade OpenCode. It needs
    `serve.sh opencode` running and the public repos from step 1.
- **The evaluation texts for `decontam.py` (step 10).**
  - *What to pass:* every text you'll score quantizations on, one `--eval` each. `decontam.py` drops any document
    that shares a 13-word sequence with them, or with the GSM8K and MATH test sets, which it always checks.
    Otherwise a quantization calibrated on eval text would score better there than in real use.
  - *For `qwen38-calib-v1`:* wikitext-2's `wiki.test.raw` (downloaded above) and `dataset/tech-eval.txt`. The
    check dropped 5 documents, all math problems overlapping GSM8K/MATH.
  - *`eval/`'s chat sets need no entry:* heldout is split off by source unit, and your own sessions are never in the
    calibration set.
  - *`tech-eval.txt`* is an out-of-domain text for measuring quantization drift away from the calibration domain:
    about 210 KB of this repo's own docs and C++. The calibration set takes no text from this repo, so any
    similar slice serves the same purpose. `techeval.py` builds one from git: the top-level docs (`README.md`,
    `docs/*.md`, `tools/*/README*.md`) cut at 115,000 bytes, then the first 95,000 bytes of `src/llama-model.cpp`.
    - Its default revision, `7957c3ad6`, reproduces the original byte for byte, which the earlier HQ measurements
      used. `--rev HEAD` gives a current one.
    - If your dataset does include this repo's text, pick another out-of-domain source instead.
- **To change the mix:** the per-category token targets and the held-out share are `TARGETS` and `HELDOUT` in
  `calib/common.py`. The sources are in `calib/sources.json`, the OpenCode tasks in `calib/opencode/tasks.jsonl`.
  `calib/README.md` describes every script and check.

The result:

| file | what |
|---|---|
| `dataset/calib.jsonl` | the calibration documents: `{"id", "category", "unit", "templated", "text", "count"?}` per line. `count` lists the `[start, end)` token spans whose rows enter the Hessian; without it the whole document counts. |
| `dataset/heldout.jsonl`, `calib-heldout.txt` | the held-out 10 %, split by source unit |
| `dataset/eval-opencode.jsonl` | your own sessions, for evaluation only |
| `dataset/manifest.json` | sources, token counts, drops, sha256 of every file, and `dataset_tag` (`qwen38-calib-v1:<sha256(calib.jsonl)[0:12]>`) |

## 2. Collect the Hessian

`hessian-collect-cuda` runs every document through the model and accumulates, for each distinct input of a weight
GEMM, the Gram matrix `G = Σ x·xᵀ` over the counted rows. That's GPTQ's Hessian up to scale. It writes one GGUF that
is also a valid imatrix. From the repo root:

```
M=/path/to/Qwen3.8-27B-gguf
TAG=$(python3 -c "import json; print(json.load(open('tools/hessian/calib/dataset/manifest.json'))['dataset_tag'])")

./hessian-collect-cuda -m $M/Qwen3.8-27B-HQ8_0.gguf --docs tools/hessian/calib/dataset/calib.jsonl \
    -o $M/qwen38-27b-hessian-v1.gguf --imatrix-out $M/qwen38-27b-imatrix-v1.gguf --dataset "$TAG" \
    -ngl 99 -ot 'blk\.[0-9]+\.ffn_(gate|up|down)\.weight=CPU' --no-mmap -ctk q8_0 -ctv q8_0
```

- **Placement** (`-ngl`, `-ot`, `--no-mmap`, `-ctk/-ctv`): the Grams need VRAM, 1.57 GB per layer here. The FFN
  weights stay in RAM and stream to the GPU once per ubatch, while attention, DeltaNet, the KV cache and the LM head
  stay on the GPU. `--no-mmap` keeps the page cache from evicting the weights.
- **Passes:** the documents run once per pass, and each pass accumulates a group of layers.
  - `--group-layers auto` (the default) sizes the group from the VRAM free after loading, with 1 GB of margin: 7 of
    64 layers here, so 10 passes of about 14 minutes.
  - The result is the same for any grouping.
- **Context:** `--ctx` defaults to the longest document (up to 128k tokens here).
- **The LM head** only sees positions that request logits: every 8th counted position (`--output-stride 8`).
  Plain Qwen3 models select their output rows inside the last layer, so they need `--output-stride 1` for that
  layer to see every row. Qwen3.8 doesn't.
- **Interrupted?** Rerun the same command with `--resume`. Finished layers are recorded in the file and skipped.
- **Try it first:** `--dry-run` lists the inputs and weights it will collect without running anything, and
  `--layers 0,out` collects just a subset.

Outputs:

- **`qwen38-27b-hessian-v1.gguf`** (101 GB) holds three kinds of tensor:
  - `<owner>.in_gram` F32 `[n, n]`: one per distinct input.
  - `<w>.in_sum2`: the diagonal, one per weight.
  - `<w>.counts`: the number of rows summed.
  - Weights that share an input (q/k/v, gate/up, the DeltaNet inputs) name its owner in `hessian.alias.*`. The
    format is in `README.md`.
- **`qwen38-27b-imatrix-v1.gguf`** (14 MB): the diagonals alone, a standard imatrix for any quantizer.

## 3. Check it

```
./test-hessian check $M/qwen38-27b-hessian-v1.gguf        # symmetric, finite, in_sum2 = diag(G) bitwise (2.5 min)
python3 tools/hessian/analyze.py sanity $M/qwen38-27b-hessian-v1.gguf --layers 0,31,63 --expect-count 1001545
```

`analyze.py sanity` checks the counts and finiteness, and runs an fp64 Cholesky of `G/mean(diag G) + 0.01·I` for
every input of the given layers and the LM head. `--unsloth IMATRIX` also correlates the diagonals with another
imatrix; for Qwen3.8-27B that's at least 0.99 on the normalized inputs, and a median of 0.80 on `ffn_down`, whose
inputs depend more on the text.

## 3b. Diagnostics: is the dataset big enough, and does position matter? (optional)

Two questions the Hessian itself can't answer: how well a dataset of this size determines each Gram, and whether
long-context positions look different enough to deserve more share. Both are answered by collecting a few layers
from subsets of the dataset and comparing the Grams:

```
D=tools/hessian/calib/work/diag             # git-ignored; cleanup.sh removes it
python3 tools/hessian/analyze.py subsets tools/hessian/calib/dataset/calib.jsonl $D
for s in half-a half-b pos-0-8k pos-8-32k pos-32-64k pos-64-128k; do
    ./hessian-collect-cuda -m $M/Qwen3.8-27B-HQ8_0.gguf --docs $D/$s.jsonl -o $D/$s.gguf --dataset $s --layers 0,31,63 \
        --trim-docs -ngl 99 -ot 'blk\.[0-9]+\.ffn_(gate|up|down)\.weight=CPU' --no-mmap -ctk q8_0 -ctv q8_0
done
python3 tools/hessian/analyze.py epsilon $D/half-a.gguf $D/half-b.gguf
for pair in "pos-8-32k pos-0-8k" "pos-32-64k pos-8-32k" "pos-64-128k pos-32-64k" "pos-64-128k pos-8-32k"; do
    python3 tools/hessian/analyze.py epsilon $D/${pair% *}.gguf $D/${pair#* }.gguf
done
```

- **The subsets** (`analyze.py subsets`):
  - `half-a` and `half-b` split the documents by source unit, so no conversation, book or repo is on both sides.
  - `pos-*` are the OpenCode documents with their count spans limited to token positions 0–8k, 8–32k, 32–64k and
    64–128k. `--trim-docs` drops each document after its last counted token, which saves most of the time for the
    early buckets.
- **The measure** (`analyze.py epsilon A B`), per input, with both Grams divided by their counts:
  - `ε = ‖(G_B + λI)^-1/2 (G_A − G_B) (G_B + λI)^-1/2‖₂`, with `λ = 0.01·mean(diag G_B)`: the largest relative
    change of `xᵀGx` over all directions `x`, at 1 % damping.
  - The relative Frobenius difference.
- **What `qwen38-calib-v1` gave:**
  - **Split-half ε** was 8–46, and 363 for layer 0's `ffn_down`, where rare massive-activation tokens land in one
    half. Independent rows would give about 0.2. Rows within a document are correlated, so the effective sample is
    closer to the ~200 documents per half.
  - **So the low-variance directions are poorly determined.** GPTQ with the raw Gram would overfit, which is why
    the quantizer shrinks toward the diagonal (`--hessian-alpha`, tuned to 0.1 on held-out KL).
  - **Position:** beyond 8k the buckets differ by 1–4× the split-half ε, from far fewer documents (5–31). Position
    doesn't clearly move the Grams more than document sampling does, so there was no reason to shift the mix
    toward long sessions.
  - The 0–8k bucket is almost all OpenCode's shared system prompt, so comparisons against it measure content, not
    position.
- **Cost:** six collections of 3 layers, about 4.7 GB of Grams each. The full numbers are in
  `plans/hessian_collection_plan.md`, Phase 5.

## 4. Quantize with it

```
./quantize_gguf --hadamard --tensor-type-file $M/Qwen3.8-27B-UD-Q4_K_M.tensor-types.txt \
    --hessian $M/qwen38-27b-hessian-v1.gguf $M/Qwen3.8-27B-bf16.gguf $M/Qwen3.8-27B-HQ-Q4_K_M.gguf Q4_K_M
```

- **What it does:** every HQ tensor gets GPTQ with its input's full Gram. The defaults, α = 0.1 and damping 0.01,
  were the best of the tuning.
- **The imatrix:** `--hessian` is also the importance matrix, so no `--imatrix` is needed.
- **Cost:** about 27 minutes on 8 cores, 8 GB of RAM at peak.
- **Result** (KL to BF16 on wikitext / held-out chat / your OpenCode sessions):
  - this quant: 0.0091 / 0.0079 / 0.0058
  - Unsloth's UD-Q4_K_M, the same size: 0.0090 / 0.0121 / 0.0084

To score a quantization, or to repeat the tuning and the rotated-vs-unrotated comparison, use `eval/`:

```
export REF=$M/Qwen3.8-27B-bf16.gguf WIKI=tools/models/wiki.test.raw
REF_NGL=33 tools/hessian/eval/ref.sh              # once: the reference's log-probs (~6 min)
tools/hessian/eval/eval.sh $M/Qwen3.8-27B-HQ-Q4_K_M.gguf
```

## Tests

`tools/hessian/tests.sh <Qwen3-0.6B-BF16.gguf> <wiki.train.raw>` runs the collector's test suite on Qwen3-0.6B.
It covers the graph hook, the file format and loaders, count spans, resume, and the CUDA accumulation against
fp64. `test-hq-gptq` covers the quantizer's full-Hessian factor.
