# Offline batch inference — implementation plan

Scope: add graph-level batching (process B utterances per dispatch) to the
offline `transcribe_run` path. Streaming is explicitly out of scope.

## Status

- **Phase 1 — API + dispatcher + serial fallback: DONE.** `transcribe_run_batch`
  + `transcribe_batch_*` accessors shipped; optional `Arch::run_batch` hook with
  a generic serial fallback so all 13 families accept the batch API today.
  `transcribe_run` is now the n==1 case. Verified: batched output is
  byte-identical to single-shot.
- **Verification infrastructure: DONE** (this is the current focus; see
  "Verification strategy" below). Parity harness, frozen golden, and a
  batch-sweep bench binary are committed.
- **Phase 2 — parakeet batched-encoder fast path: SAME-LENGTH DONE.**
  The conformer helpers thread a batch axis (`B` at the activation's `ne[2]`,
  shuffled to `ne[3]` inside attention/conv), `build_encoder_graph` takes an
  `n_batch`, and parakeet's `run_batch` fuses B same-length utterances through
  one encoder dispatch, then host-decodes each. `B==1` is bit-identical to the
  pre-batch graph (cohere unaffected). Verified:
    - Text: B=4 jfk == single-shot, all utterances.
    - Tensor (`scripts/batch_tensor_parity.py`): batched per-utterance
      `dec.enc_out` == single-shot **bit-exact** (max_abs 0.0) on CPU, with
      `TRANSCRIBE_NO_FLASH=1` so both sides use manual F32 attention. (Flash
      casts the rel-pos mask to F16; the batched path always runs manual,
      which is why the default-vs-batched diff is ~1e-3, not a bug.)
    - Internal: two identical utterances in one batch are bit-equal.
    - Throughput (M4 Max / Metal, jfk 11s): 65 → 44 ms/utt at B=4 (~1.5×);
      conservative vs CUDA because the batched path forces manual attention
      and the host decode is per-utterance. Mixed-length batches currently
      fall back to per-utterance (still correct).
- **Phase 2 remainder — variable-length batching: DONE.** `run_batch` pads
  every mel to the batch's T_max and builds two per-utterance masks so a padded
  tail cannot corrupt real frames: an attention key-padding mask
  `[T_enc, 1, 1, B]` added onto the score matrix in `rel_pos_mhsa`, and a conv
  valid-frame mask `[T_enc, 1, B, 1]` applied in `conv_module` before the
  depthwise conv. Each utterance is decoded at its own `T_enc` (the
  3×stride-2 subsample of its real mel length). Verified: varied-length
  batches (4.5s/11s/35s/56s) batched 2–4 == single-shot **text-exact**; the
  most-padded utterance's encoder output matches single-shot to ~7e-3 at the
  tail boundary frames (inherent to padded batching — the greedy decode is
  unaffected, text is identical). Flash batches correctly here too.
  *Efficiency note:* padding to T_max wastes compute when lengths vary widely,
  so callers (and the WER harness manifest) should length-sort / bucket; the
  same-length path stays mask-free and bit-exact.
- **Phase 3 — sensevoice (SAN-M + CTC): DONE.** First non-conformer family.
  `run_batch` fuses B utterances through one SAN-M encoder dispatch, then
  host-decodes each utterance's CTC log-probs slice (greedy argmax — no
  autoregressive decoder, so decode is genuinely free and never batched).
  The batch axis rides the activation's `ne[2]`; the shared `src/sanm/`
  helpers derive `B` from `x->ne[2]` so `funasr_nano` (the other SAN-M
  consumer) stays bit-identical at `B == 1`. Same-length batches run the
  flash path mask-free; variable-length batches pad each LFR feature to
  `T_max` and add two per-utterance masks — an attention key-padding mask
  `[T, 1, 1, B]` (forces the manual SDPA path) and an FSMN valid-frame mask
  `[1, T, B]` applied to V before the depthwise conv. The FSMN depthwise
  conv switches to `ggml_conv_2d_dw_direct` for `B > 1` (the `conv_1d_dw_f32`
  im2col path collapses the batch axis); `B == 1` keeps the validated im2col
  path. Verified:
    - Text: B=4 same-length == single-shot; varied-length (jfk 11s /
      product-names 56s / dots 35s / whole-earth 84s) batched 2,4 ==
      single-shot **text-exact**, CPU and Metal, gated vs a frozen golden
      (`tests/golden/batch/sensevoice-small.cpu.json`).
    - Tensor (`batch_tensor_parity.py --dump-name ctc.log_probs`): same-length
      per-utterance `ctc.log_probs` vs single-shot — **bit-exact (max_abs 0.0)
      on Metal**; CPU max_abs 5e-5 (im2col-vs-direct FSMN conv reduction
      order, not a mask bug — a wrong mask perturbs real frames by orders
      more).
    - Single-shot ref numerics unchanged: `validate.py all --family
      sensevoice` 18/18 tensors within tolerance, transcript exact.
  *Throughput:* flat on this box — M4 Max / Metal jfk 43.2→48.5 ms/utt
  (b1→b4), CPU dead-flat at ~237 ms/utt. SenseVoice's SAN-M encoder already
  saturates the local GPU/CPU on a single 11s clip, so there is no launch
  headroom to amortize; the win is a discrete-GPU (L40S/CUDA) phenomenon, as
  documented for parakeet below. Because there is **zero** host decode (unlike
  parakeet's flat ~6.5s TDT decode), sensevoice has no Amdahl decode ceiling —
  end-to-end should track the full encoder batching win wherever the encoder
  is launch-bound.
- **Shared parallel mel front-end: DONE.** The per-utterance feature
  extraction (mel / kaldi-fbank / LFR) every `run_batch` runs is pure host
  code with no cross-utterance state, and is frequently the *dominant* wall
  cost once the encoder is on a fast accelerator. `transcribe::parallel_for_all`
  (`src/transcribe-batch-util.{h,cpp}`) dispatches the B extractions across CPU
  workers; sensevoice and parakeet `run_batch` both use it (any new family gets
  it for free by calling the helper instead of a serial mel loop). Verified
  batched-vs-serial text parity unchanged for both families.
  - **Measured (L40S, 128 LibriSpeech utts, F16, length-sorted):** for
    **sensevoice** the threaded mel is what unlocks the end-to-end win — the
    SAN-M encoder is so fast on an L40S that mel dominates. b4 end-to-end went
    **1.04× → 1.88×** once mel was threaded (mel 2.49s→0.65s, ~3.8×; encoder
    1.77s→0.69s, ~2.6×). (b2 is overhead-noisy at ~0.87×; b4 is the real
    signal.)
  - **Measured (M4 Max / Metal, F16, jfk):** the SAN-M encoder doesn't batch on
    Metal (already saturates), but mel is ~11ms of a ~43ms wall, so threading
    it flips batching from a *loss* to a *win*: per-utt b8 **48.2 → 30.9 ms
    (~1.5×)**; the trend reverses from rising (43→48) to falling (48→31).
  - GPU mel (DFT-as-matmul front-end) is a documented follow-up — moderate
    effort with its own F32-vs-host-F64 parity gate; the threaded host mel
    captures most of the available headroom first.
- **granite_nar (Conformer + NAR LM): INVESTIGATED → NOT SHIPPED (serial
  fallback).** A device-batched encoder was built (Shaw block-local attention
  folds `(n_blocks, B)` into one axis; batched depthwise conv; per-utterance
  key-padding folded into the block mask; var-len pad-to-`T_max`) and verified
  **text-exact** (single-shot, same-length, and variable-length, CPU + Metal,
  incl. a different-content leakage test). But it is a **throughput regression**
  on L40S, so `granite_nar` keeps `run_batch = nullptr` (the generic serial
  fallback — correct, never slower than a caller's own loop). Why it doesn't pay:
  - The encoder is **dominated by the 1024→100,352 BPE-CTC head** (a giant
    matmul that's compute-bound and gains nothing from batching), not the
    conformer blocks — so batched `encode` time stays flat (~11s) across batch
    sizes, unlike parakeet's conformer (2.2×). Decode is only ~11% on L40S.
  - A **pool-then-project** rework (`W·pool(x)` instead of `pool(W·x)`, ~4× less
    BPE matmul + readback) is exact single-shot and would help, but the pooled
    argmax is numerically less robust than per-frame CTC and drifts a few tokens
    on the *batched* path (the batched direct-conv carries ~1e-3 vs single-shot
    im2col). Shelved.
  - **Two measurement traps recorded here as lessons:** (1) the WER sweep prints
    "WER identical across batch sizes" as an *assumption*, not a check — it does
    not catch degraded b>0 transcripts; verify per-utterance parity separately.
    (2) `ggml_backend_tensor_get` with a **non-zero byte offset is unreliable**
    (returned wrong data for b>0 on **both CPU and Metal**) — read tensors in
    full (offset 0) and slice on the host. A per-utt offset readback produced a
    *fake* 1.38× by silently corrupting later utterances. sensevoice/parakeet
    `run_batch` were never affected (they full-read + host-slice).
  - **Kept (reusable, zero-impact at B==1):** the shared `conv_1d_dw_f32`
    batch-aware upgrade and the `shaw_block_attn` batch fold — both bit-identical
    at B==1, available for a future conformer family that *is* encoder-bound.
- **gigaam (Conformer + rotary attn + RNN-T / CTC): DONE.** `run_batch` fuses B
  utterances through one Conformer encoder dispatch, then host-decodes each
  utterance's RNN-T / CTC slice (small vocab, no BPE-head bottleneck — so this
  batches like parakeet, the real win, *not* like granite_nar). The batch axis
  rides the activation's `ne[2]`. gigaam owns two custom pieces the shared
  conformer helpers don't cover, both threaded for B here:
    - **2-conv1d pre-encode** (`build_pre_encode`): batch carried at the conv
      `N` axis; for variable lengths, masked subsampling zeros the padded time
      region after each conv-relu (mask_s1/mask_s2) so a stride-2 conv can't
      leak padding into a real boundary frame. *This also required fixing the
      shared `conv_1d_f32`*, whose final `[OW*N, OC] -> [OW, OC, N]` reshape was
      only valid at N==1 (the `OW*N` flatten interleaves batch with width); the
      N>1 path now mul_mats the 3-D im2col directly and permutes — bit-identical
      per-utterance at N==1.
    - **rotary attention** (`build_rotary_attn`): RoPE + QKV + SDPA threaded over
      B; variable-length batches add a `[T_k,1,1,B]` key-padding mask and force
      the manual SDPA path (same recipe as sensevoice). Same-length keeps flash.
  The shared `conv_module` (depthwise + conv_pad_mask) was already batch-aware
  from parakeet. Full-read encoder output + host-slice per utterance (NOT
  offset reads). Verified (RNN-T + CTC variants):
    - Same-length: **bit-exact** per-utterance `dec.enc_out` (max_abs 0.0, CPU,
      `--no-flash`) at B=4; text parity OK.
    - Variable-length: **F32 text-exact** (b2/b4, jfk/death/love-loss/dots) —
      proves the masking is correct. Per-utterance `dec.enc_out` drift on the
      most-padded utterance scales purely with weight precision (signal
      rms≈0.49): **F32 max_abs 1.2e-6, F16 6.0e-3, Q8_0 6.3e-2**. At Q8_0 (and
      to a lesser degree F16) that ~1% re-quantization noise flips a few tokens
      on heavily-padded utterances — **the identical phenomenon parakeet-Q8_0
      shows** (parakeet was only "text-exact" at F32). It is *not* a masking
      bug; same-length is bit-exact at every precision, and length-bucketing
      (already the documented caller recommendation) shrinks the padding → the
      drift → ~0. Callers wanting bit-stable var-len batching should bucket or
      run at F16/F32.
  Single-shot numerics unchanged; `transcribe_run_dispatch` ctest passes.
- **Phase 3+ — remaining families / autoregressive: NOT STARTED.** The encoder
  batching scaffolding (mask building, per-utterance slicing) plus the shared
  parallel-mel helper are reusable for the remaining conformer/CTC families.

## Measured results (parakeet, this work)

All head kinds validated (bit-exact same-length tensor gate, max_abs 0.0, +
varied-length text parity, CPU):

| variant | head | text parity | tensor gate |
|---|---|---|---|
| parakeet-ctc-0.6b  | CTC   | OK | bit-exact |
| parakeet-rnnt-0.6b | RNN-T | OK | bit-exact |
| parakeet-tdt-0.6b-v3 | TDT | OK | bit-exact |
| parakeet-tdt-1.1b  | TDT   | OK | bit-exact |

**Where the win comes from — and its ceiling.** L40S, 128 LibriSpeech utts,
length-sorted, with the encode/decode split:

| n_batch | wall | encode | decode | end-to-end |
|---|---|---|---|---|
| 1 | 41.4s | 32.4s | 6.3s | 1.00× |
| 4 | 26.5s | 18.8s | 5.0s | 1.56× |
| 8 | 24.9s | **15.9s** | 6.6s | 1.66× |

The encoder itself batches ~2× (32.4 → 15.9s = 2.04×) — that **is** the
original 2× experiment, which measured an encoder-dominated single clip. But
the host-side decode is **flat** (~6.5s, un-batched), so end-to-end is capped
by Amdahl at ~1.66× on TDT. Decode-light heads recover more of the encoder
win: Metal jfk, parakeet-**ctc** 57→33.5 ms/utt (1.70×) vs **tdt** 64.6→45.4
(1.42×). So encoder batching pays off most for CTC / big-encoder / NAR-decoder
families; autoregressive/host-decode families dilute it.

**Two efficiency observations for follow-up.** (1) The encoder is
*launch-bound* on a discrete GPU — ~253 ms to encode ~9s of audio on an L40S
(~35× RTF for encode alone), i.e. ~2200 conformer ops ≈ 2200 kernel launches;
batching amortizes the launches (a big part of the win). **CUDA Graphs**
(capture + replay the static graph) would cut single-shot encode and stack
with batching. (2) Length-sort/bucket the WER manifest: naive (unsorted) L40S
batching at b16 was 42.7s vs 31.4s sorted (36% slower) — random-order batches
pad short clips up to the group's longest. (3) `run_batch` rebuilds the encoder
graph + reallocs scheduler buffers per call; caching across same-shape calls
would remove that overhead.

## Next-family plan + per-family checklist

Order by (batching payoff × ease), reusing the batched conformer + the
`run_batch` recipe:

1. ~~**sensevoice**~~ **DONE** (SAN-M + CTC, *not* the shared conformer as
   originally assumed) — encoder-bound, zero host decode. See the Phase 3
   entry above. The conformer scaffolding didn't transfer (SAN-M is its own
   encoder in `src/sanm/`), but the `run_batch` recipe, mask design, and
   parity-gate strategy did.
2. ~~**gigaam**~~ **DONE** (Conformer + rotary attn + RNN-T / CTC) — reused
   parakeet's pattern; needed custom pre-encode + rotary batching and a
   `conv_1d_f32` N>1 fix. See the Phase 3 entry above.
3. **parakeet 1.1b / ctc / rnnt** — already work via shared `run_batch`;
   validated above. Bench-only.
4. **granite_nar** (Conformer + NAR LM) — shared conformer **and** a single-
   forward decoder → both encode AND decode batch (biggest end-to-end win in
   the easy tier).
5. **funasr_nano** (SAN-M + NAR LM) — both paths, but SAN-M is a custom encoder
   (not shared) → must batch its ops too.
6. autoregressive (cohere, canary, qwen3_asr, whisper, moonshine) — per-utt KV
   cache, variable finish: hard, and dilutes the encoder win. Big conformer
   encoders (canary/cohere) still benefit from encoder-only batching.

Per-family checklist (conformer + CTC/RNN-T families are largely mechanical):
1. Add `n_batch` + `batch_var_len` to the family's encoder graph builder
   (mel_in 4-D `[T_mel, n_mels, 1, B]`, output `[d, T, B]`).
2. Copy-adapt parakeet's `run_batch` (mel pack/pad → build the two masks →
   batched encode → per-utterance decode loop → `capture_result`).
3. Wire `Arch.run_batch`.
4. Run the gates: `batch_tensor_parity.py` (same-length bit-exact) +
   `batch_parity.py` (varied-length text + golden) + ctest.
5. Bench same-length on the target GPU.

**Known gap:** only the *direct* conv paths are batched (parakeet uses those on
every backend). The im2col fallbacks (`conv_1d_dw_f32`, im2col GLU) are NOT
batched — relevant for **cohere**, which uses im2col on Metal/CPU; that family
needs those paths batched too.

## Verification strategy (build this BEFORE touching the encoder)

The throughput win is only measurable on a big GPU, but *correctness* must be
provable on CPU/Metal. Three layers, cheapest first:

1. **Text parity (same build), `scripts/batch_parity.py`.** Runs a varied-length
   utterance set through the CLI serially and via `transcribe_run_batch`
   (`--batch-size N`) and asserts per-utterance hypothesis text is identical.
   Because serial is the established source of truth, this catches any
   dispatch/plumbing regression a batched path introduces. Exit-code driven.
2. **Frozen golden (cross-build).** `batch_parity.py --golden-out` captures
   today's serial output to `tests/golden/batch/<variant>.<backend>.json`
   BEFORE any encoder change. A change to the encoder itself moves serial AND
   batched together, hiding it from layer 1 — so `--golden-in` additionally
   gates both against the frozen text. Captured now at
   `tests/golden/batch/parakeet-tdt-0.6b-v2.cpu.json` (7 EN utterances, CPU).
3. **Tensor self-consistency vs reference (Phase 2 gate).** We do NOT need to
   re-capture NeMo reference dumps for varied-length batches. The existing
   `validate.py all` already gates single-shot encoder output against the NeMo
   reference within `tests/tolerances/parakeet.json`. So the batch gate is just:
   *the batched encoder's per-utterance output equals the single-shot output for
   the same wav, tensor-for-tensor.* Single-shot-vs-reference (existing) +
   batched-equals-single-shot (new) ⇒ batched-equals-reference, end-to-end.
   Mechanism: dump `enc.out` per utterance in batch mode into per-utterance
   subdirs and diff against the single-shot dump via `compare_tensors.py` with a
   tight tolerance (exact when the pad/mask is correct; small tol otherwise).
   The padding mask is the thing this gate is designed to catch — a wrong mask
   perturbs real frames and shows up as drift here, not in WER.

Bench: `examples/bench/batch_bench.cpp` (`transcribe-batch-bench`) loads the
model once and times only the `transcribe_run_batch` calls across a batch-size
sweep, emitting the historical JSON schema to
`reports/perf/<machine>/<variant>_batch_<backend>.json`. The pre-Phase-2
baseline (serial fallback) shows flat per-utterance latency by construction —
that is the number the fused encoder must beat.

## Why

- The existing CLI `--batch <list.txt>` is a serial for-loop over
  `transcribe_run` (`examples/cli/main.cpp:511-587`). No device-level batching
  exists anywhere in the library.
- A throwaway 2026-05-28 experiment on an L40S
  (`reports/perf/l40s/parakeet-batch/*.json`) measured ~2× parakeet throughput
  at batch 4–8 (313 → 158 ms/utt), regressing past 16 from memory pressure. The
  harness was not committed; the win is real and is dominated by the **encoder**
  saturating the GPU.

## Confirmed design decisions

1. **API shape**: one new entry point, `transcribe_run_batch`. Single-shot
   `transcribe_run` stays the default path and becomes the B==1 case.
2. **Result access**: index-parameterized batch accessors
   (`transcribe_batch_*(session, i, ...)`).
3. **Coverage**: *every* family must support the batch API on day one via a
   generic serial fallback (correctness); families opt into a fast batched
   `run_batch` over time. Parakeet proves the fast path first. New ports are
   expected to wire `run_batch` or consciously accept the fallback.
4. **Variable lengths**: caller hands in raw utterances; the library pads the
   batch to max length and applies a per-utterance length mask. Bucketing by
   length (caller-side) minimizes wasted compute and is documented, not
   enforced.

## API surface (Phase 0)

```c
/* One shared run_params for the whole batch in v1 (per-utterance params is a
 * later extension). pcm/n_samples are arrays of n entries. */
transcribe_status transcribe_run_batch(
    struct transcribe_session*           session,
    const float* const*                  pcm,
    const int*                           n_samples,
    int                                  n,
    const struct transcribe_run_params*  params);

int                       transcribe_batch_n_results(const struct transcribe_session*);
transcribe_status         transcribe_batch_status(const struct transcribe_session*, int i);
const char*               transcribe_batch_full_text(const struct transcribe_session*, int i);
transcribe_timestamp_kind transcribe_batch_returned_timestamp_kind(const struct transcribe_session*, int i);
const char*               transcribe_batch_detected_language(const struct transcribe_session*, int i);
int                       transcribe_batch_n_segments(const struct transcribe_session*, int i);
int                       transcribe_batch_n_words(const struct transcribe_session*, int i);
int                       transcribe_batch_n_tokens(const struct transcribe_session*, int i);
transcribe_status         transcribe_batch_get_segment(const struct transcribe_session*, int i, int j, struct transcribe_segment*);
transcribe_status         transcribe_batch_get_word(const struct transcribe_session*, int i, int j, struct transcribe_word*);
transcribe_status         transcribe_batch_get_token(const struct transcribe_session*, int i, int j, struct transcribe_token*);
```

Semantics:
- Reuses the existing `transcribe_segment/_word/_token` copy-out structs and
  their `_init` functions — no new row structs, minimal ABI growth.
- **Storage unification**: `transcribe_run` == `transcribe_run_batch` with n==1.
  Legacy single accessors (`transcribe_full_text(s)`, …) alias batch index 0.
  Internally one `std::vector<ResultSet> results;` (see Phase 1).
- **Error model**: top-level return is OK when the dispatch ran; per-utterance
  failures (e.g. unsupported language, abort) are read via
  `transcribe_batch_status(s,i)`, with completed utterances retained. Top-level
  non-OK only for whole-batch faults (bad args, struct-size, OOM,
  NOT_IMPLEMENTED, TRANSLATE unsupported).
- **Abort**: polled between utterances and decode steps; completed utterances
  retained, same partial-retain contract as single-shot.
- Text-pointer lifetime: same rule as single-shot — valid until the next
  mutating call on the session.

## Phase 1 — dispatcher, Arch hook, generic fallback, result refactor

- **Arch trait** (`src/transcribe-arch.h`): add optional
  `transcribe_status (*run_batch)(session, const float* const* pcm,
  const int* n_samples, int n, const run_params*)`. NULL = use fallback.
- **Result storage refactor** (`src/transcribe-session.h`): extract the current
  per-result fields (`tokens/words/segments/full_text/detected_language/
  result_kind/has_result` + per-utterance status/was_aborted) into a reusable
  `ResultSet`, and hold `std::vector<ResultSet> results;`. Keep the existing
  single fields as the "scratch slot" a family `run()` writes, so per-family
  `run()` needs **no change** for the fallback path. `clear_result()` resets to
  empty.
- **`transcribe_run_batch` dispatcher** (`src/transcribe.cpp`): mirror
  `transcribe_run` validation (struct sizes, ext shape/kind, common run-param
  checks, TRANSLATE support, `run_validate`), validate the arrays
  (`n>0`, every `pcm[i]!=NULL`, every `n_samples[i]>0`), clear, then:
  - `arch->run_batch != NULL` → call it (writes directly into `results`).
  - else → **generic serial fallback**: for each i, call `arch->run` into the
    scratch slot and snapshot it into `results[i]`, polling abort between.
- Rework `transcribe_run` to be `transcribe_run_batch` with n==1 (or a thin
  shared inner) so the two never diverge.

This phase alone ships a correct batch API for all 13 families.

## Phase 2 — parakeet fast path (`run_batch`) + reusable scaffolding

Build the batched-encoder pattern that later families reuse:

1. **Batched mel pack**: compute mel per utterance, pad each to `T_max` frames,
   pack to the conv-friendly layout, record valid lengths per utterance.
2. **Batched encoder graph**: raise `ne[3]` from 1→B across the pre-encode
   `conv2d` stem, conformer blocks, and output (`encoder.cpp:330,396,574`).
   Conformer blocks already broadcast over `ne[3]` (`conformer.cpp:118`).
   - **Main risk**: the conformer conv module uses `ne[2]` as batch in its
     depthwise-conv im2col (`conformer.cpp:164`) while the *utterance* batch is
     `ne[3]`. Confirm these don't collide and add reshapes if they do — read
     this carefully before coding.
   - **Length mask**: build a per-utterance additive attention/padding mask so
     padded frames never perturb real frames (extends the existing
     `chunked_mask` path with a B dimension). This is the core new code and the
     numerical-parity gate.
3. **Readback + slice**: encoder out `[d_enc, T_enc, B]`; slice per utterance to
   its valid length.
4. **Host TDT decode**: loop b in 0..B, run the existing `decode_tdt_greedy` on
   slice b into `results[b]`. No lockstep batching of the host decoder needed
   (utterances finish at different frames/durations and it is cheap relative to
   the encoder).
5. **Buffers**: size graph/scheduler for `B*T_max`, reuse across calls when
   shape is stable; document a recommended max batch and the memory-pressure
   cliff (the batch-16 regression).

Acceptance: per-utterance batched output must match single-shot — **text
identical**, tensors within `tests/tolerances/parakeet.json`.

## Phase 3 — generalize

- Extract shared helpers (mel pack+pad, length-mask builder, batched-output
  slicer) so encoder-heavy / CTC / RNN-T families (sensevoice, gigaam, …) reuse
  them quickly.
- Autoregressive families (whisper, canary, canary_qwen, cohere, qwen3_asr,
  moonshine) need a separate design: per-utterance KV-cache slabs, masked
  early-stop, per-utterance temperature fallback / prompt / lang-detect. Larger
  effort; they keep the correct serial fallback until then.
- **Porting convention**: update `src/transcribe-arch.h` docs + the
  `porting-4-cpp` / `porting-6-bench` skills so each new family either wires
  `run_batch` or consciously accepts the fallback, plus a batch bench cell.

## Phase 4 — harness, validation, bench

- **CLI**: add `--batch-size N` so `--batch <list.txt>` groups N files into
  `transcribe_run_batch` calls; serial remains the default for parity.
- **Validation**: add a B==1 vs B==N parity check (text identical, tensors
  within tol) to `scripts/validate.py` or a dedicated test.
- **Bench**: commit a proper batch-sweep bench under `scripts/bench/` producing
  `reports/perf/<machine>/<variant>_batch_<backend>.json` (per-utt ms +
  throughput across batch sizes). One process doing internal batching is the
  intended use; do not run multiple bench processes concurrently.

## Open questions / deferred

- Per-utterance `run_params` (different language/prompt per utterance) — v1
  shares one params across the batch; revisit if a real consumer needs it.
- Per-utterance timings accessor (`transcribe_batch_get_timings`) — v1 may
  expose aggregate only.
- Library-side auto-bucketing by length — deferred; caller-side bucketing is
  documented instead.

## Rough effort

- Phase 0+1 (API + dispatcher + fallback + result refactor): ~2–3 days; unlocks
  the correct batch API for all families.
- Phase 2 (parakeet fast path incl. masking + parity): ~1 week.
- Phases 3–4: incremental, per family + harness.
