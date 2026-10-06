# Plan: partial-state checkpoints for SmartCache on hybrid and recurrent models

## Summary

A hybrid or recurrent model can't rewind its recurrent state, so SmartCache reuses work after a prompt change by
restoring a snapshot taken at an earlier position. Every such snapshot today is a full copy (attention KV plus
recurrent state, ~2.5 GB at 100k tokens on Qwen3.8-27B). Each request writes up to three:
- the regen slot at the end of the prompt;
- the lifeboat partway through long prompts;
- a snapshot before the last 32 tokens, which is skipped whenever a draft model or MTP is loaded.

On top of that, every request whose prompt doesn't fully contain the live context saves it to a slot, another full
copy. On the 27B that includes every OpenCode step whose re-rendered reply differs.

Only the recurrent state actually needs saving: the attention KV can be cut back with `seq_rm`. llama.cpp saves just
that part with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, ~155 MB on the 27B at any context length. Upstream llama-server
builds its context checkpoints on it, and koboldcpp uses it for one MTP rollback checkpoint.

| # | change | effect |
|---|---|---|
| 1 | **Partial checkpoints per context** (the live context and each SmartCache slot): up to 8; checkpoints in the first 40 % of the context are evicted first, then by min-gap; the 3 newest and the system-prompt checkpoint are never evicted | cheap restore points, concentrated in the later part of a conversation |
| 2 | **Per request: "latest"** (end of the prompt) **and "latest − 32"**; once per conversation, a checkpoint at the end of the system prompt | regenerate, re-rendered replies, changed prompt tails and new conversations that share a system prompt restore in ~10 ms |
| 3 | **Remove the regen slot, the lifeboat and the before-the-last-32 full snapshot** on these models | per-request snapshot cost falls from up to three full copies to two ~155 MB copies |
| 4 | **Save the live context to a slot only when the new prompt keeps less than 70 % of it;** matching across the live context and all slots, with checkpoints traveling with the snapshots | no full-copy save on re-rendered steps or edits near the end; conversations brought back from a slot can still rewind |
| 5 | **MTP draft state saved with every checkpoint and snapshot** | the first draft and the MTP head's KV are right after a restore; fixes today's lowered acceptance after every SmartCache load |
| 6 | **The cut before the last 32 tokens also runs with a draft model or MTP** | latest − 32 exists on the 27B config, which uses MTP |
| 7 | **2 slots, not 5, when SmartCache turns itself on** for these models | bounded memory for users who never chose a slot count |

Transformer models keep today's behavior, including their 70 % similarity rule, except item 4's choice of which slot
a save overwrites. Their KV already rewinds, and their partial state is empty.

**Test policy** (as in `mtp_speed_plan.md`):
- Every change ships with a regression test runnable from the repo, in the same commit or an earlier one. Fixes come
  with a test that fails before the fix; exact changes are compared bitwise.
- Every change gets a performance check against a baseline recorded before it: interleaved A/B, at least 3 rounds,
  median and max.
- Earlier checks rerun at each step's end.

## Background

### SmartCache today (`gpttype_adapter.cpp` unless noted)

- **On by default for these models.** Fast forward and context shift are on by default. Python passes 5 slots even
  without `--smartcache` (`savestate_limit_default`, `koboldcpp.py:60`, `:2197`). For hybrid and recurrent models C++
  then turns SmartCache on by itself (`:3642-3647`) and adds:
  - the regen slot (`rnn_reusable_slot_idx`, index N);
  - when N ≥ 4, a hard-reserved lifeboat slot (N + 1).
- **Matching** (`:6263-6355`): the longest slot whose tokens are a full prefix of the prompt (`FullyContainedPrefix`).
  Whenever the live context isn't fully contained in the new prompt, it's saved into `get_oldest_slot(match)` (unless
  an identical slot exists), however much the two share. Transformer models instead save when the prompt is under
  70 % similar and context shift can't help (`:6362`).
- **Full snapshots during a request,** via `gpttype_save_state_kv` (`:7614`):
  - before the last 32 tokens (`:6738-6772`), only when `draft_ctx == nullptr`;
  - the lifeboat (`:6619-6620`, `:6903-6910`) at "65 %" of a prompt with ≥ 2048 new tokens. Its target is computed
    after fast forward but compared with `n_past`, so on continued conversations it fires after the first batch;
  - the regen snapshot at generation start (`:6953-6963`).
- **Saved logits:** each snapshot stores the last logits. A restore with nothing left to decode samples from them
  (`:7027`).
- **Prompt batches** are filled in the loop at `:7349-7511` and capped at `n_batch`. Media spans are decoded there
  with plain `llama_decode`, so the draft or MTP context doesn't see them.
- **MTP rollback:** with MTP, `n_rs_seq = draftamount` (`:3324`). The recurrent memory keeps rollback copies of its
  state on the GPU. Models without bounded rollback use one `PARTIAL_ONLY` checkpoint, `mtp_spec_ckpt` (`:1144`,
  `:7297`).

### What llama.cpp provides

- **What a partial save holds.** `llama_state_seq_get_data_ext(ctx, …, seq, PARTIAL_ONLY)` on a hybrid memory writes
  only the recurrent part (`src/llama-memory-hybrid.cpp:191`): one row per layer, the current state, read from a
  rollback copy if one is pending (`src/llama-memory-recurrent.cpp:901`). Loading one resets the rollback index
  (`:877`) and replaces the sequence's recurrent cell (`state_read_meta`).
- **Order matters on restore.** The hybrid `seq_rm` tries the recurrent part first (`llama-memory-hybrid.cpp:143`).
  While the recurrent cell is past the cut (and outside the MTP rollback window), that fails without removing
  anything. So a restore loads the partial state first and cuts afterwards. With the cell back at the checkpoint, the
  recurrent part is a no-op and the attention cut succeeds.
- **`common_prompt_checkpoint`** (`common/common.h:1166`) holds a target partial state, a draft partial state and a
  speculative state (`data_spec`). Upstream and `mtp_spec_ckpt` both use it.
- **Pinned memory.** `ggml_backend_cuda_register_host_buffer` (`ggml/include/ggml-cuda.h:41`) can pin host memory.

### Upstream reference (llama-server)

- **Checkpoints per slot:** up to 32, at user-message starts and at end − 4 − n_ubatch and end − 4. Spacing of 8192 is
  enforced only when the list is full and never against the current request's checkpoints; after that, the oldest is
  dropped.
- **A RAM prompt cache** holds full states of swapped-out prompts with their checkpoints under a MiB limit, and drops
  entries that are prefixes of a newly saved prompt.
- **Neither stores MTP draft state.** The MTP driver doesn't implement `get_state`.

## Findings that shape the design

### The skipped cut with a draft model is a leftover

- **When it was added.** The cut and its `draft_ctx==nullptr` condition came in 42134db6b (2026-03-02). Back then the
  loop decoded the main context, and afterwards decoded the whole batch into a separate draft model. A snapshot
  between the cut's two parts would have held the two contexts at different positions.
- **What changed.** 104b41091 (2026-06-12) routed every decode through `kcpp_decode_main_and_spec` (`:1083`), which
  advances the draft side on the same batch through `common_speculative_process`. The cut decodes each part through
  it, so both contexts stay in step. MTP support (10e4b6d5e) landed a day earlier, and the condition was never
  revisited.

### MTP's `pending_h` isn't restored

- **What it is.** The MTP driver (`common/speculative.cpp:1340`) keeps `pending_h`, the target's hidden row for the
  last processed token.
  - `process()` pairs it with the next batch's first token (`:1544`).
  - `draft()` uses it as the first draft input (`:1623`).
  - `accept()` refreshes it after each verify (`:1751`).
- **Nothing saves it.** Neither llama's state API nor the driver's `get_state` stores it.
- **Effect.** After a restore it's stale. The first draft gets the wrong input, and the first batch writes one wrong
  entry into the MTP head's KV, lowering acceptance for the rest of that generation. Text is unaffected, since the
  target verifies drafts. It happens today on every SmartCache load with MTP.

### How OpenCode builds consecutive prompts (1.18.34; bundle and the user's session database)

- **Reasoning is sent back.** The bundled `@ai-sdk/openai-compatible` conversion sends reasoning parts back as
  `reasoning_content`. koboldcpp passes messages to the template unchanged, and the Qwen3.8 template renders
  `<think>\n{reasoning}\n</think>\n\n{content}` for every assistant turn.
  - So a step's prompt is the previous prompt plus the re-rendered reply plus the tool results.
  - If the re-rendered reply is token-identical to what was generated, the live state continues with no restore.
    Otherwise the restore point is **latest**, and the reply is reprocessed.
  - Where re-rendering differs, if at all, is unknown and is measured in Phase 0.
- **Pruning is off.** Tool-output pruning runs only with `compaction.prune` set, which the user's config doesn't set.
  None of 11,821 stored parts has been pruned.
- **Plan-mode reminders are transient.** Default plan mode pushes its reminder onto the last user message on every
  request without saving it, and after plan mode, build mode does the same with a plan-to-build reminder. A new user
  message removes them from the previous message, so the prompt changes before the whole previous turn. That affects
  17 of 289 user turns, in 2 of 97 sessions.
- **Session shape:** 2,259 requests; a median of 477 new tokens per request (70 % at most 1024); a p90 of 19 steps per
  user turn.

### Why 32 tokens and not a batch

latest − k serves prompts whose last tokens change: a re-rendered generation prompt, a client that drops reasoning
(`<think>\n\n</think>`, where `\n\n` is one token, 271), or a changed tail. A restore there reprocesses k tokens.
- In a replay of the 2,259 requests, assuming every step diverged at the generation prompt, k = 32 reprocessed 70k
  tokens in total and k = 1024 reprocessed 2.2M (about 0.24 s more per step on the 27B).
- For OpenCode as it behaves, neither is used on a step.
- k = 32 is the cheap choice. A batch-deep checkpoint would cover changes 32–1024 tokens from the end (Lite's author's
  note, world info, editing the last message). It's left as a follow-up.

### Eviction, simulated

The setup: an agentic session with a system prompt ending at 2k, a 27.5k first prompt, then turns of ~150 reply tokens
plus 3k of input, two checkpoints per request. The metric is the extra tokens reprocessed for an edit at a position,
beyond everything after the edit, as mean / worst by where in the conversation the edit falls:

| 8 checkpoints | 10 turns (56k) | 40 turns (150k) | 100 turns (339k) |
|---|---|---|---|
| this plan's rule, edits at 0–40 % | 9.3k / 20k | 28.1k / 58k | 65.9k / 134k |
| this plan's rule, edits at 40–70 % | 9.0k / 25k | 39.9k / 82k | 74.0k / 177k |
| this plan's rule, edits at 70–100 % | 3.0k / 6k | 11.9k / 25k | 25.8k / 60k |
| min-gap only, edits at 70–100 % | 3.0k / 6k | 18.6k / 38k | 47.1k / 98k |

- **The rule concentrates checkpoints in the last 60 %.** Edits that keep at least 70 % of the conversation, the ones
  that don't trigger a save, are covered best.
- **The band just above the 40 % line is thin.** Checkpoints are evicted as the growing context pushes them below the
  line, so the first surviving one can sit well above it: at 40 turns the positions are 2k, 84k, 97k, 119k, 144k,
  147k and 150k twice, with the line at 60k.
- Keeping the checkpoint nearest the line would trade some of the back-part gain for the 40–70 % band (16.1k mean at
  40 turns). That isn't part of this plan.

## Design

### Terms

- The **live context** is the conversation in the GPU's KV cache: its tokens, their attention KV, and the recurrent
  state after the last token.
- A **cut** to position p shortens it: tokens from p onward are removed from the attention KV, and the recurrent state
  is replaced by a checkpoint's state at p. Afterwards it holds tokens 0 … p − 1, and everything after p has to be
  reprocessed or restored from a slot.
- The **restore point** is the p chosen for a request.
- **Keep fraction:** the common prefix of the live context and the new prompt, divided by the live context's length.

```
live context:  [ system 0–12k | A's turns 12k–40k | A's last prompt 40k–52k | A's last reply ]
                              ^ system checkpoint                           ^ latest
OpenCode step on A:  restore point 52k (latest) → only A's last reply is discarded; keep ≈ 99 %
new conversation B with the same system prompt: restore point 12k → keep = 12k / 52k = 23 %
```

### Checkpoint contents and storage

- **A checkpoint is a `common_prompt_checkpoint`** plus koboldcpp fields:

  | part | 27B size | source |
  |---|---|---|
  | target partial state (`data_tgt`) | ~155 MB | `llama_state_seq_get_data_ext(…, PARTIAL_ONLY)` |
  | draft partial state (`data_dft`) | ~0 for MTP | the same, on `draft_ctx` |
  | speculative state (`data_spec`) | 20 KB (`pending_h`) | `common_speculative_get_state` |
  | position, creation serial, kind (latest, latest − 32, system) | — | position = `seq_pos_max + 1` at creation |
  | last logits, "latest" only | ~1 MB | as `gpttype_save_state_kv` takes them |

- **Writes and loads:**
  - Writes fill the vectors like `update_tgt`/`update_dft` do.
  - **Loads call `llama_state_seq_set_data_ext` directly and check its result.** On failure the request falls back to
    reprocessing from the start, as today's failed `seq_rm` does (`:6541-6551`). `common_prompt_checkpoint::load_*`
    would abort the process on a size mismatch instead.
- **Recycling.** Every checkpoint of a model has the same size. A checkpoint dropped by every holder returns its object
  to a free list, so the vectors keep their memory: no zero-fill and no page faults after warm-up.
- **Sharing.** A checkpoint never changes after it's written, so it's held by `std::shared_ptr` and can be shared
  between the live context and slots.
- **Optional, CUDA builds:** pin each recycled checkpoint's buffers once with `ggml_backend_cuda_register_host_buffer`.
  Kept only if the A/B shows faster checkpoint copies.

### Where checkpoints are made

Let L be the prompt length after this request, r the restore point, n_new = L − r the tokens processed, and S the
system-prompt position, if known.

- **Tiny contexts:** no checkpoints for contexts of 32 tokens or fewer.
- **latest:** at the switch to sampling (where the regen snapshot is taken today), if n_new ≥ 1. If n_new = 0 (a
  restore exactly to an existing latest), touch it instead.
- **latest − 32:** if n_new > 32 and L − 32 > S.
- **System:** at S, if r < S < L. One checkpoint, with no −32 companion.
- **Batch cuts.** The fill loop also stops a batch when it reaches a cut position (L − 32, S): `n_past +
  embd.size() == cut`. The checkpoint is taken at the top of the decode step, when `n_past == cut` and the batch is
  about to be decoded.
  - This replaces the split in two at `:6738-6772` and its "≤ 48 tokens" case, which snapshotted at the batch start.
  - `draft_ctx==nullptr` goes.
  - The retry-in-128-token-chunks path is unchanged.
- **Media:** a cut position must not split a media span (`kcpp_media_span_boundary_ok`, `:2771`). Otherwise that
  checkpoint is skipped.
- **Duplicates:** a checkpoint at an existing position replaces the old one.
- **Not made:** at user-turn starts, at regular spacing, or during generation.

### Eviction and invalidation

- **Capacity:** 8 per context. Only the live context's list gains checkpoints, so eviction only runs there.
- **Never evicted:** the 3 newest by creation serial, and the system-prompt checkpoint (or the oldest checkpoint,
  when the context has none).
- **Eviction order** when a 9th is added:
  1. evictable checkpoints positioned before 40 % of the context's current length, earliest first;
  2. if there are none, the evictable checkpoint whose neighbors are closest together. The first one's left neighbor
     is position 0, and the newest's right neighbor is the context's end. Ties go to the older one.
- **Invalidation (not eviction):**
  - Cutting the live context back to p drops its checkpoints past p.
  - Clearing the live KV (a full reprocess, a model reload, or continuous batching touching sequence 0 in
    `batch_invalidate_legacy_context_locked`) drops all of them.
  - Overwriting a slot releases its list.
- **Total:** at most (N + 1) × 8 checkpoints, often fewer because the live context shares checkpoint objects with the
  slot it came from.

### Restoring (hybrid and recurrent models)

This replaces `:6263-6355` and the recurrent fast-forward block at `:6445-6497`.

1. **Candidates:** the live context and every slot with the same media signature. For each, find the common prefix
   d with the prompt (`ComputeSharedPrefixLength`).
   - The restore point is the context's length if the prompt fully contains it.
   - Otherwise it's its latest checkpoint at or before d. For the live context, also no later than `seq_pos_max + 1`.
2. **Choice:** the furthest restore point. Ties go to the live context, then the most recently used slot.
3. **Saving the live context first.** A cut, or loading a slot in its place, discards part or all of the live
   context.
   - If the keep fraction is below 0.7, save the live context to a slot first (see "Slots"). That covers another
     conversation, or an edit far back.
   - Otherwise discard the tail. That covers re-rendered replies, regenerate, and edits near the end.
   - The denominator is the live context's length. `ComputePrefixMatchPercent` divides by the shorter of the two,
     which would make a short new conversation sharing only the system block look more than 90 % similar.
4. **Loading a slot.** If the winner is a slot, load its full snapshot and its speculative state, and take shared
   references to its checkpoints.
5. **Cutting back to a checkpoint** at p, earlier than the context's end:
   1. Load the checkpoint's target and draft partial states and its speculative state. **Loading comes first.**
   2. `seq_rm(p, -1)` on the main and draft contexts.
   3. Drop live checkpoints past p.
   4. Set `n_past`, `current_context_tokens` and `last_n_tokens` to p, the way fast forward sets them.
   5. If nothing is left to decode, set `loaded_latest_logits` from the checkpoint.
6. **When the live context is fully contained,** plain continuation is unchanged, including the existing correction
   for the last undecoded token (`:6471`). Checkpoint positions are exact, so the restore path needs no correction.
7. **If nothing is usable:** save the live context if rule 3 says so, clear, and process from the start.

### Slots (full snapshots)

- **Count:** N slots (`--smartcache N`), numbered `0..N-1` for the admin endpoints. The regen and lifeboat slots go
  away, so `savestate_limit` = N for every model.
  - When `--smartcache` isn't given and SmartCache turns itself on for a hybrid or recurrent model, N = 2. C++ knows
    this case: `inputs.smartcache` is false.
  - `--smartcache` with or without a number is unchanged.
  - The admin endpoints use the effective count, reported by C++, instead of Python's `savestate_limit`.
- **A slot holds:** a full snapshot (main and draft), tokens, last logits, media signature, speculative state, a
  checkpoint list and a last-used counter.
- **Where a save goes** (when restoring rule 3 calls for one, or when a transformer model's 70 % rule does), in this
  order:
  1. the live context is 32 tokens or fewer → don't save;
  2. an identical slot → touch it;
  3. a slot whose tokens are a prefix of the live tokens (an older snapshot of the same conversation) → replace it.
     This applies to transformer models too;
  4. an empty slot;
  5. the least recently used slot, never the one about to be loaded.

  The live context's checkpoint list moves into the slot. It already holds references to the checkpoints of any slot
  it was loaded from.
- **Admin endpoints** (`save_state`, `load_state`, `check_state`) keep their slot numbers and carry the checkpoint list
  and speculative state. `check_state` adds each context's checkpoint positions and kinds. It's a koboldcpp admin
  endpoint, not part of the OpenAI API.
- **Memory:** N full snapshots plus at most (N + 1) × 8 checkpoints. With N = 2 on the 27B at 100k tokens, the worst
  case is 5 GB plus 3.7 GB. Today it's 3 full snapshots, 7.5 GB.

### MTP draft-state fix

- **`common/speculative.cpp`:** `common_speculative_impl_draft_mtp` gets `get_state` and `set_state`, writing and
  reading `pending_h[seq_id]` with its width for a size check. That's the only change to the upstream file.
- **Used for:** every checkpoint (`data_spec`), every full snapshot (`savestate_data`), and `mtp_spec_ckpt`. The last
  stashes it at `update` and restores it before the replay at `:7297`.
- **Not covered:** rewinds without a checkpoint on transformer MTP models. No row exists to restore.

### System-prompt position

S is computed inside koboldcpp from the request's own content and passed to C++ in a new `generation_inputs` field.
HTTP requests and responses don't change.

- **Chat completions, jinja:**
  - Render the system message(s) and tools alone, with the same template kwargs and no generation prompt. If the
    rendered prompt starts with that text, pass its character length.
  - C++ tokenizes that prefix of the prompt. S is the `memory` token count plus that length, used only if those
    tokens match the start of the prompt's tokens.
  - That check also fails, safely, when koboldcpp trims the start of an over-long prompt.
- **Chat completions, adapters:** the same, with Python recording the offset after the leading system messages while
  it builds the prompt. Tools are in `memory`.
- **`/api/v1/generate`:** S = the `memory` token count, which koboldcpp already tokenizes separately.
- **No S:** no system checkpoint. The oldest checkpoint is protected in its place.

### Removed

`rnn_reusable_slot_idx`, `rnn_lifeboat_slot_idx`, `rnn_lifeboat_hard_reserved`, the `smartcache_rnn_lifeboat_*`
constants, `smartcache_quick_snapshot`, `get_nearby_compatible_smartcache_slot`, the split before the last 32 tokens,
and the full snapshots before the last 32 tokens and at generation start.

### Code structure

- **`otherarch/kcpp_smartcache.h`** (new, no llama dependency): checkpoint-list policy over (position, serial, kind)
  metadata, with the payload as a template parameter; restore-point lookup; the keep-fraction rule; the save order. A
  unit test drives it without a model.
- **`savestate_data`** (`otherarch/otherarch.h`) gains a checkpoint list and the speculative state. The live context's
  list sits beside `current_context_tokens`.
- **`gpttype_adapter.cpp`:** checkpoint writing and loading, the batch-cut condition, matching, the keep-fraction
  rule, the slot save and load paths, and the default of 2 slots.
- **`koboldcpp.py`:** S for chat completions, the new input field, `check_state`'s report, and the admin range from
  the effective slot count.
- Minimal comments, in the surrounding style.

## Phase 0: measurements and baselines (before any change)

1. **Baseline build:** the library built from 4f96f3249 (saved 2026-10-05 before any change), run from a git worktree
   at 4f96f3249 via `kcpp-e2e.py --build`.
2. **Write the `checkpoints` e2e config** (see Tests). Record its baseline numbers on the old build: processed tokens
   and wall time per request. Assertions that need `check_state` run only on the new build.
3. **Capture one real OpenCode session.**
   - Serve with the user's `~/AI/qwen3/Qwen3.8-27B-HQ4_K_M.kcpps`. `serve.sh opencode` serves the HQ8_0 generator with
     FFN offload instead.
   - Run one task through `tools/hessian/calib/opencode/run.sh` with `UPSTREAM` pointing at that server. It runs
     OpenCode in its sandbox, through the logging proxy.
   - For each step record: whether the re-rendered history equals the generated tokens; if not, where it diverges
     (reasoning, the reasoning/content boundary, content, the tool call, the generation prompt); and how many tokens
     the baseline reprocesses.
4. **Record baselines:**
   - `kcpp-e2e.py check tools/perf/golden/kcpp-e2e-27b.json`;
   - `agentic` and `checkpoints`: per-request wall time, its snapshot part, processed tokens;
   - peak RSS;
   - the GPU contention level.

## Steps

Each step is a commit that builds and passes its own tests and all earlier ones. Its A/B goes into "Implementation
record".

1. **MTP draft-state fix,** with `test-mtp-spec-state`. On its own it fixes acceptance after today's SmartCache loads.
2. **`kcpp_smartcache.h` and `test-kcpp-smartcache`.**
3. **Live-context checkpoints:** creation, batch cuts, dropping `draft_ctx==nullptr`, restore within the live context,
   and removal of the three full snapshots and the lifeboat.
4. **Slots carry checkpoints:** matching across slots, the keep-fraction rule, the save order, the default of
   2 slots, the admin endpoints, and `check_state`.
5. **The system-prompt position:** Python and `memory`.
6. **The full e2e and performance A/Bs.** The golden file changes only if a text hash changes for a documented
   reason.

## Tests

### Regression

1. **`tests/test-kcpp-smartcache.cpp`** (make target `test-kcpp-smartcache`; CPU, no model):
   - **Eviction:** capacity 8; checkpoints before 40 % of the context length go first, earliest first, then min-gap
     (neighbor rules and ties as specified); the 3 newest and the system checkpoint (or the oldest) never evicted; a
     duplicate position replaces.
   - **Lookup and invalidation:** the latest checkpoint at or before a position; truncation drops later ones.
   - **Sharing and recycling:** a shared checkpoint survives one holder dropping it; the last drop returns the object
     to the free list; the next checkpoint reuses its buffers (same data pointer).
   - **Matching:** the furthest restore point; ties to the live context, then the most recently used slot; the media
     signature respected.
   - **Keep fraction:** below 0.7 saves, at or above it doesn't. The denominator is the live length: a 13k prompt
     sharing 12k with a 52k live context saves.
   - **Save order:** as specified.
   - **Placement:** the simulated agentic session above gives the recorded positions at 10, 40 and 100 turns.
2. **`tests/test-mtp-spec-state.cpp`** (`test-mtp-spec-state-cuda MODEL`, a model with MTP layers):
   - Process a prompt, save the partial states and the speculative state, draft (tokens and draft logits), process
     another batch, restore, draft again. The drafts must match bitwise.
   - Without restoring the speculative state, they must differ. **This fails before the fix.**
3. **`tools/perf/kcpp-e2e.py` config `checkpoints`** (27B, MTP, `--smartcache 2`, `--admin` with a temporary
   admindir; checks processed tokens and `check_state`):
   - **OpenCode-style steps:** chat completions with tools, replies sent back with `reasoning_content` and structured
     `tool_calls`, tool results appended. Each step restores no further back than latest, processes at most the
     re-rendered reply plus the new input plus 32 tokens, and saves or touches no slot. Phase 0's capture sets the
     exact expectation.
   - **Regenerate:** restores latest with nothing to decode. Text identical (bitwise) to the first generation, and
     equal draft acceptance. **Acceptance fails before step 1.**
   - **Restore to a checkpoint, bitwise:** after a request, restore its latest − 32 and decode the same last 32
     tokens. The logits at L must equal those the original request produced. Same state, same batch.
   - **Changed tail** (`enable_thinking` false on the next request): restores latest − 32 and processes ~32 tokens plus
     the new input.
   - **The cut with MTP:** text identical to the baseline build, or diverging only at a near-tie (top-2 gap below
     ~1e-3). Draft acceptance within noise. The cut changes batch shapes, and the 27B's self-KL floor is ~0.0015.
   - **Edit the last message** (keep ≥ 70 %): restores latest − 32 or earlier; no save; slots unchanged.
   - **Edit far back** in an 8-turn conversation (keep < 70 %): the old version is saved to a slot; processed tokens
     are at most the distance to the nearest checkpoint plus the rest; positions follow the eviction rule.
   - **Two interleaved conversations:** returning to main restores its slot and checkpoint. A save replaces main's
     older snapshot, not the other conversation's.
   - **New conversation with the same system block:** the previous conversation is saved, then the system checkpoint
     is restored (processed ≈ total − S).
   - **Media at the end of the prompt:** no checkpoint inside the span, and the text matches a run without caching.
   - **Default slots:** without `--smartcache`, `check_state` reports 2 slots on the 27B.
   - **Admin:** save and load by slot number, with checkpoints carried.
4. **Standing checks:** test-kcpp-state-buffer; test-save-load-state-cuda; `kcpp-e2e.py check` against the golden
   file. The mtp, nomtp and agentic hashes must not change: the agentic config appends raw text, so its live state
   continues without a restore.

### Performance (interleaved A/B against Phase 0, ≥ 3 rounds, median and max)

- **`agentic`:** per-turn wall time and its snapshot part. Expected: the snapshot part falls from two full copies
  (~0.1 s at 30–40k tokens, ~0.3 s at 100k) to two partial copies (~20–40 ms). The cut adds one 32-token decode
  (~10–30 ms). Prompt processing and generation unchanged.
- **`checkpoints` OpenCode-style steps:** per-step wall time, also at ~90k tokens. A restore is a ~155 MB load instead
  of a full one (0.1–0.2 s at 100k tokens), and no full-copy save of the live context.
- **Pinned checkpoint buffers:** checkpoint save and load time with and without pinning. Pinning is kept only if it
  wins.
- **Peak RSS:** within N full snapshots plus (N + 1) × 8 checkpoints.

## Not in this plan

- **A batch-deep checkpoint** (upstream's end − 4 − n_ubatch): covers prompt changes 32–1024 tokens from the end in
  chat frontends.
- **A checkpoint at the end of reasoning** (`</think>` accepted during generation). If Phase 0 shows OpenCode's
  re-rendered replies diverge after the reasoning, it limits a step's reprocessing to content and the tool call.
- **Round-trip fidelity:** making koboldcpp's reply re-render through the template into the generated tokens, so that
  steps never restore. Also decided by Phase 0.
- **Keeping the checkpoint nearest the 40 % line:** see "Eviction, simulated".
- **Incremental full snapshots** that copy only new KV on a switch.
- **Asynchronous checkpoint copies:** llama's state API is synchronous.
- **`ON_DEVICE` states:** only one per sequence, so they can't hold a checkpoint list.
- **SWA models:** partial checkpoints of the sliding-window cache could replace koboldcpp's rewind workaround
  (`ff_swa_retain_amount`).
- **`LongestCommonSubseq`'s full (m+1)×(n+1) table** (`model_adapter.cpp:435`), used only by transformer context
  shift.

## Implementation record

(empty)
