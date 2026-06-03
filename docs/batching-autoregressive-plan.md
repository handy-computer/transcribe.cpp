# Batched autoregressive decode — bulletproof plan

Scope: add device-level batching to the **autoregressive decoders** (the
families whose decode is a token-at-a-time KV-cache loop). The non-autoregressive
families (parakeet / sensevoice / gigaam) are already batched — see
`batching-plan.md`. Goal here is **maximum robustness and numerical fidelity**,
not minimum effort: batched decode must be provably equivalent to the
single-shot path that `validate.py` already gates against the reference.

## Why this is the high-value tier (and not a dead end)

Autoregressive = the model emits the transcript one token at a time, feeding each
token back to predict the next; token N+1 cannot be computed before token N. So
decode is a sequential loop of full decoder forward passes (≈100–256 per
utterance). For these families the **decode dominates** runtime, and a single
sequence underutilizes a GPU — which is exactly why batching pays off (it is what
LLM servers do). The CTC/RNN-T families were batched first because they were easy
*and* their decode was already free; this tier is harder to build but is where the
real speedup lives.

## The landscape (9 families)

| Family | Decoder | Shared decode lib | Notes |
|---|---|---|---|
| funasr_nano | autoregressive Qwen3 LM | `qwen3_lm` | SAN-M enc (already batched) + adaptor + greedy LM |
| qwen3_asr | autoregressive Qwen3 LM | `qwen3_lm` | pure-LM style; audio scattered in prefill |
| canary_qwen | autoregressive Qwen3 LM | `qwen3_lm` | conformer enc + Qwen3 LM |
| granite (AR) | autoregressive Qwen3 LM | `qwen3_lm` | conformer enc + Qwen3 LM |
| cohere | autoregressive enc-dec transformer | bespoke | self **+ cross** KV; greedy |
| canary | autoregressive enc-dec transformer | bespoke | self + cross KV; greedy; mirrors cohere |
| moonshine | autoregressive enc-dec transformer | bespoke | self + cross KV; **partial RoPE + head-dim pad** |
| whisper | autoregressive enc-dec transformer | bespoke | **temperature fallback, 30s windowing, lang-detect, no-speech, prompt, segment slicing** |
| moonshine_streaming | autoregressive (streaming) | bespoke | out of scope (streaming, not offline batch) |

Two facts shape the whole plan:

1. **4 families share `qwen3_lm`.** Batch the decode there once → funasr_nano +
   qwen3_asr + canary_qwen + granite all benefit. Highest leverage; do it first.
2. **Every greedy decoder already has the same shape**: a static step graph with
   per-step scalar inputs `token[1]`, `pos[1]`, `kv_idx[1]`, `mask[max_n_kv,1]`
   (+ `cross_mask` for enc-dec), KV written via `ggml_set_rows(kv, kv_idx)` and
   read as a full masked window. **Batching = give those inputs and the KV cache
   a batch axis B and step all rows in lockstep**, moving argmax/EOS to per-row
   host control. So the core machinery is shared across *all* AR families; only
   the per-block math and prefill differ.

## Core decomposition: serial prefill + batched step

The single most important robustness decision:

- **Prefill serially, one utterance at a time, into its own batch slot of a
  batched KV cache.** Prefill is one forward pass over the prompt (cheap relative
  to the 100–256-step decode). Doing it per-utterance makes each prefill
  *byte-identical to single-shot* — zero ragged-length/position-offset risk. This
  sidesteps the most error-prone part of LLM batching (ragged prefill alignment).
- **Then batch the step loop**, which is the expensive part and where the win is.

This keeps the hard numerical surface area tiny: prefill is inherited-correct;
only the batched step needs a parity gate. (Batched ragged prefill is a *later*
throughput optimization, gated separately, never a correctness prerequisite.)

### Batched step loop (the new core)

- KV cache gains a batch axis: per-K/V `[head_dim · n_kv_heads · n_ctx · n_layer · B]`.
  `B == 1` must be **byte-identical** to today (regression guard for every
  single-shot family).
- Step graph inputs become per-row: `token[B]`, `pos[B]`, `kv_idx[B]`,
  `mask[max_n_kv, B]` (causal + key-padding + finished-row mask), and for
  enc-dec families a batched cross-KV cache + `cross_mask[T_enc_pad, B]`.
- One `ggml_set_rows` writes B new KV rows at B indices; attention reads each
  row's full window under its own mask. Logits come back `[vocab, B]`.
- **Per-row host control**: each utterance tracks its own `n_past`, `kv_idx`,
  EOS flag, and generated-token list; argmax is B-way.
- **Static-batch finish**: step *all* rows until every row has hit EOS or
  `max_new`. Finished rows keep stepping but their output is ignored (their KV is
  independent, so they can't corrupt live rows). No mid-loop batch compaction —
  that ("continuous batching") is a later optimization, not needed for
  correctness. Each utterance's tokens are recorded up to its first EOS.
- **Readback**: full-read `[vocab, B]` then host-slice per row. Never use
  non-zero-offset `ggml_backend_tensor_get` (the granite lesson — it returned
  wrong data for b>0 on CPU and Metal).

## Shared infrastructure to build once

1. **Batched KV cache** (extend `qwen3_lm::KvCache`, then generalize): batch axis
   + batched `set_rows` write + per-row masked read. The enc-dec families use the
   same set_rows/masked-window pattern, so the cache abstraction is reusable
   (cross-KV cache is the same shape with `T_enc_pad` instead of `n_ctx`).
2. **Generic batched AR driver**: per-row `n_past`/EOS/token-list tracking, B-way
   argmax, per-step mask updates, static-batch termination, result capture into
   `session->batch_results`. The family supplies four hooks: (a) per-utterance
   prefill, (b) batched step-graph builder, (c) token embedding / EOS id, (d)
   result populate. This driver + batched cache is the leverage; each family is
   then a thin wiring layer.

## Numerical-accuracy strategy (the heart of "bulletproof")

Mirror the encoder strategy that already works, extended to decode. Build the
gates **before** touching the decoder.

**Proof chain.** Single-shot decode is already validated tensor-for-tensor
against the reference (`validate.py` + `tests/tolerances/<family>.json`). So the
batch gate only needs: *batched row b's decode == single-shot b's decode*.
Single-shot-vs-reference (existing) + batched-equals-single-shot (new) ⇒
batched-equals-reference.

**Three verification layers (cheapest first):**

1. **Text parity** — `scripts/batch_parity.py` already runs serial vs
   `transcribe_run_batch` and diffs per-utterance transcript text. Decoder-
   agnostic; works as-is for AR families. Catches any dispatch/plumbing
   regression.
2. **Frozen golden** (cross-build) — `batch_parity.py --golden-out/--golden-in`
   freezes single-shot text so a decoder change that moves serial *and* batched
   together is still caught. Capture per target family before any change.
3. **Per-step decoder-logits parity** (NEW, the decode analog of the encoder's
   bit-exact `dec.enc_out` gate). Extend `batch_tensor_parity.py`: feed B copies
   of one clip, dump per-utterance per-step logits (`dec.logits_raw.b{i}` at a
   chosen step), and assert batched row == single-shot. **Same-length / same-
   content ⇒ bit-exact (max_abs 0.0).** This is the strongest gate and the one
   that proves the batched step math.

**Precision discipline (from the gigaam finding):** quantized matmuls are
T-sensitive, so variable-length batches drift ~1% at Q8_0 and flip a few tokens —
inherent, not a bug, and identical to parakeet-Q8_0. Therefore: run the **strict
gates at F32** (convert F32 GGUFs for the gated families), expect **F32 var-len
text-exact** and **same-length bit-exact at every precision**, and document that
Q8_0 var-len callers should length-bucket. Bit-exact same-length is the headline
correctness guarantee.

**Per-op batch verification.** Each ggml op on the new batch axis is validated in
isolation before trusting it: `set_rows` with B indices, `soft_max_ext` /
`flash_attn_ext` with a `[..,B]` mask, RoPE over `[..,T,B]`. (gigaam taught us an
op can silently behave differently at B>1 — e.g. `conv_1d_f32` N>1.)

## Phased rollout

**Phase 0 — verification scaffolding (no decoder change).**
Extend `batch_tensor_parity.py` for per-step decoder-logits parity; capture
frozen single-shot goldens for the Phase-1 families; convert F32 GGUFs for the
gated families. Deliverable: a red/green decode parity gate ready before any
batched-decode code exists.

**Phase 1 — shared batched `qwen3_lm` decode (highest leverage).**
Add the batch axis to `qwen3_lm::KvCache` + `block_prefill`/`block_step`; build
the generic batched AR driver; land it on **one** family first (funasr_nano, the
user's target; qwen3_asr is the simplest if we want an even cleaner first cut).
Gate: same-length bit-exact per-step logits (F32) + F32 var-len text-exact +
B==1 byte-identical regression + dispatch ctest. Then wire the other three
`qwen3_lm` families (qwen3_asr, canary_qwen, granite) — each is just its prefill +
audio-injection + result populate on top of the shared driver.

**Phase 2 — enc-dec greedy families (cohere, canary, moonshine).**
Generalize the batched cache to carry the **cross-attention** KV cache too
(batch axis on the encoder-derived cross-K/V, which we already batch on the
encoder side) and a per-row `cross_mask`. cohere and canary are near-identical, so
do cohere first then canary cheaply; moonshine adds partial-RoPE + head-dim
padding inside its block math (the driver is unchanged). Same gate suite.

**Phase 3 — whisper (highest risk, last).**
Whisper's per-utterance control flow diverges hard: temperature-fallback tiers,
30s windowing with data-dependent seek, language detection, no-speech skip,
prompt conditioning, segment slicing — two utterances take different numbers of
windows and tiers. Bulletproof options, in increasing ambition:
- **(a) Encoder-only batching** (batch the conformer encoder + cross-KV across
  utterances; decode per-utterance). Low risk, partial win; a safe floor.
- **(b) Cohort step-batching**: batch the inner greedy step *only across
  utterances currently in the same (window, tier) state*; fall back to
  per-utterance when they diverge. The decode win without reimplementing the
  fallback orchestration in batched form.
- **(c) Full batched fallback/windowing** — large, and likely not worth the
  complexity/risk vs (b). Decide after Phase 2 with real profiling.
Whisper gets its own mini-design doc before any code.

## Risk register / bulletproofing checklist

- [ ] `B == 1` byte-identical to current single-shot for every touched family
      (run existing `validate.py` per family — must stay green unchanged).
- [ ] Same-length batched per-step logits **bit-exact** (max_abs 0.0, F32, CPU).
- [ ] Variable-length batched **text-exact at F32**; document Q8_0 drift.
- [ ] Every new batch-axis op validated in isolation (set_rows, masked
      soft_max/flash, RoPE) before integration.
- [ ] Full-read + host-slice readback only (no non-zero-offset tensor_get).
- [ ] Per-row EOS/finish handled by static-batch stepping; finished rows proven
      not to perturb live rows (independent KV slots).
- [ ] Memory: batched KV cache is B×; size for `B · n_ctx · n_layer` and
      document the memory cliff + a recommended max batch.
- [ ] Per-utterance params (language/prompt) — v1 shares one `run_params` across
      the batch (matches the current `transcribe_run_batch` API); per-utterance
      params is a documented later extension.
- [ ] Abort polled between steps; completed utterances retained (partial-retain
      contract, same as single-shot).

- **Phase-1 first family**: Start with Qwen3 ASR

Questions:
- **Whisper depth**: (a) encoder-only vs (b) cohort step-batching — decide after
  Phase 2.
- **Batched ragged prefill**: deferred throughput optimization; serial prefill
  ships first.
