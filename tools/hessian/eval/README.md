# eval: scoring quantizations against their BF16 model

Scripts that score quantized models by KL divergence and top-1 agreement against the BF16 model they were made from,
on three evaluation sets, and that run the full-Hessian GPTQ tuning and the rotated-vs-unrotated comparison recorded in
`plans/hessian_collection_plan.md` ("Follow-up: full-Hessian GPTQ"). They call `test-hadamard-ppl-cuda` and
`quantize_gguf` from the repo root (`make LLAMA_CUBLAS=1 tools test-hadamard-ppl-cuda`).

## Settings

Everything comes from the environment; nothing points at a particular machine.

| variable | needed by | meaning |
|---|---|---|
| `REF` | all | the BF16 reference model; the quantizations are also made from it |
| `WIKI` | all | wikitext-2's `wiki.test.raw` (e.g. `tools/models/wiki.test.raw`) |
| `TYPES` | `iter.sh`, `compare.sh` | a tensor-type file for `--tensor-type-file`, e.g. Unsloth's UD-Q4_K_M mix |
| `HESSIAN` | `iter.sh`, `compare.sh` | the `hessian-collect` file |
| `IMATRIX` | `compare.sh` | an imatrix from the same calibration data, for the unrotated quant (`hessian-collect --imatrix-out`) |
| `LOG` | `iter.sh`, `compare.sh` | the markdown log (default `work/tuning.md`) |
| `REF_NGL` | `ref.sh` | GPU layers of the reference while its log-probs are computed (default 99) |
| `FTYPE` | `iter.sh`, `compare.sh` | the ftype for tensors `TYPES` doesn't list (default `Q4_K_M`) |
| `BASELINE` | `compare.sh` | optional: a model scored alongside, e.g. the original UD-Q4_K_M |
| `QUANT_DIR` | `iter.sh`, `compare.sh` | where temporary quantizations go (default `work/`) |
| `WORK`, `TEXTS`, `PPL`, `QUANTIZE` | all | the work and text directories, and the two binaries (defaults below and in the repo root) |

## Files

- **`texts/`** (git-ignored, kept): the two chat eval texts. `texts.py` builds them from
  `../calib/dataset/` when they're missing, and the build depends only on the dataset.
  - `heldout.txt`: the first 9000 characters of each of the calibration set's held-out documents.
  - `opencode.txt`: a 9000-character slice of each of the user's OpenCode sessions, from the first turn start after a
    seeded random point. It's the user's own text, which is why the directory is git-ignored.
- **`work/`** (git-ignored, disposable): the reference caches (`<set>.kl-cache`, 42 GB for Qwen3.8-27B's 248k
  vocabulary), `ref.sh`'s logs, `results.tsv` (every scored row, tab-separated), `queue.txt`, the per-run logs, and
  the temporary quantizations.

## The evaluation sets (`common.sh`)

| set | text | windows | scored |
|---|---|---|---|
| wiki | `WIKI` | 80 × 512 | every position in the second half of each window |
| heldout | `texts/heldout.txt` | 64 × 1024 | with `--chat-mask` |
| opencode | `texts/opencode.txt` | 36 × 2048 | with `--chat-mask` |

**Why the chat mask:** chat models are trained only on their own turns. In system, user and tool turns both the
BF16 and the quantized model mostly predict `<|im_end|>`, and their disagreement there is noise: unmasked,
UD-Q4_K_M scored KL 0.137 on heldout and 0.858 on opencode (p99 20 nats), against 0.009 on wikitext.
`test-hadamard-ppl --chat-mask` scores only the tokens the model generates: assistant turns, and text outside any turn.

## Scripts

| script | does |
|---|---|
| `ref.sh` | computes the reference's log-probs for each set into `work/<set>.kl-cache`; about 2 minutes per set for the 27B (`REF_NGL=33` on a 32 GB GPU). A valid cache is reused. |
| `eval.sh MODEL` | scores a model on every set: one line per set, `set bpw NLL PPL meanKL p99KL top1`. About 2 minutes for a 16 GB model. |
| `iter.sh NAME ALPHA DAMP [NOTE]` | one tuning run: quantizes `REF` with `TYPES` as HQ types and `--hessian HESSIAN --hessian-alpha ALPHA --gptq-damp DAMP`, scores it, appends a row to `LOG`, and deletes the quantization |
| `iter.sh --baseline MODEL NAME` | scores and logs an existing model; nothing is deleted |
| `queue.sh` | runs the lines of `work/queue.txt` (`NAME ALPHA DAMP [NOTE]`) through `iter.sh` one at a time; lines can be appended while it runs |
| `compare.sh ROTATED.gguf` | quantizes `REF` rotated (HQ, `--hessian` at the defaults, kept as `ROTATED.gguf`) and unrotated (the plain types with `IMATRIX`, deleted), scores both, and appends the comparison to `LOG` |
| `texts.py OUTDIR` | builds the chat eval texts |

## The Qwen3.8-27B runs

```
M=~/Sandbox/unquantized/Qwen3.8-27B-gguf
export REF=$M/Qwen3.8-27B-bf16.gguf WIKI=tools/models/wiki.test.raw \
       TYPES=$M/Qwen3.8-27B-UD-Q4_K_M.tensor-types.txt HESSIAN=$M/qwen38-27b-hessian-v1.gguf \
       IMATRIX=$M/qwen38-27b-imatrix-v1.gguf LOG=$M/hessian-gptq-tuning.md
REF_NGL=33 tools/hessian/eval/ref.sh
tools/hessian/eval/iter.sh --baseline $M/Qwen3.8-27B-UD-Q4_K_M.gguf UD-Q4_K_M
printf '%s\n' 'a1-d01 1 0.01' 'a0-d01 0 0.01' 'a01-d01 0.1 0.01' > tools/hessian/eval/work/queue.txt
tools/hessian/eval/queue.sh            # about 30 minutes per run
BASELINE=$M/Qwen3.8-27B-UD-Q4_K_M.gguf tools/hessian/eval/compare.sh $M/Qwen3.8-27B-HQ-Q4_K_M-fullH.gguf
```

The results are in `$LOG` and in the plan.
