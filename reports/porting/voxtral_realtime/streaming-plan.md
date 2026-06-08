# Voxtral Realtime — incremental streaming implementation plan (Step 2b)

## SHIPPED (2026-06-05): true incremental streaming (reference StaticCache)

The streaming path implements the **reference generate() mechanism**, not a
reprocess. The five `stream_*` arch hooks + `accepts_ext_kind` are wired
(`src/arch/voxtral_realtime/model.cpp`), the public stream extension lives at
`include/transcribe/voxtral_realtime.h`
(`TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM` = `VRST` / `0x54535256`), and
`--stream-chunk-ms` / `--stream-voxtral-delay` drive it from the CLI.

**Mechanism (matches `modeling_voxtral_realtime.py`).** Per the reference
`prepare_inputs_for_generation` + `get_audio_features`, the encoder transformer
runs INCREMENTALLY: each new group of `downsample_factor=4` embedder frames is
pushed through the 32 encoder layers against an **encoder KV cache** (the
reference `StaticCache`, sliding_window=750) under a windowed-causal mask — each
frame processed exactly once — then grouped → projector → one audio embed. The
decoder prefills the 39-token prompt once, then steps one token per audio frame
against its **persistent sliding KV(8192)**. New ggml graphs:
`build_embedder_graph` (conv stem) and `build_encoder_chunk_graph`
(`mha_cached` writes the chunk's K/V at cache rows `[n_kv_before, n_kv_total)`,
reads `[0, n_kv_total)`); the decoder reuses the validated `build_prefill_graph`
/ `build_step_graph` with the KV persisted across feeds.

The cheap mel + 2-layer conv stem are recomputed over the accumulated buffer and
sliced (bit-exact: causal conv == padding cache; frame-independent global mel ==
per-chunk mel). Making those incremental too (conv padding cache + streaming mel)
is a numerically-identical constant-cost micro-opt, deferred. The EXPENSIVE
parts — the 32-layer encoder transformer and 26-layer decoder — process each
frame/token exactly once via real KV caches.

**Unbounded length (shipped).** The encoder cache is a 750-frame ROLL matching
the reference `StaticSlidingWindowLayer`: a `k_enc_ring_ctx`=1536-slot ring with
contiguous writes, periodic compaction (copy the last 750 frames to the front,
`compact_enc_ring`) so memory is constant for any length, and new frames pushed
in ≤`k_enc_max_batch`=512 batches so a single huge feed can't overflow. The
windowed mask uses ABSOLUTE positions (`stream_enc_abs_base` tracks slot-0's
absolute index). The decoder KV grows once to the 8192 sliding window. Validated:
jfk (748 < 750, no wrap) `proj.out` 0.0 vs offline; whole-earth (84 s, 4416
frames, multiple compactions) byte-exact transcript + `proj.out` 1.2e-2 vs the
offline whole-clip encoder; dots-full (306 s, ~15300 frames) correct at 3.6× RT.
Clips > 8192 tokens (~10.9 min) are clamped at the decoder window — a decoder
ring is the remaining follow-up.

**Verification (tested, not by construction).**
1. **Incremental encoder == offline encoder, BIT-EXACT.** Stream `proj.out`
   (the StaticCache audio embeds) vs offline `proj.out`: **max|d| = 0.000e+00**,
   both for a single finalize AND for multi-feed 1000 ms chunks (proves
   cross-feed cache accumulation). This is the decisive test of the new encoder.
2. **Streaming gen0/gen8 vs the reference `stream` oracle, within tolerance.**
   Even on Metal: gen0 max|d|=0.161 (tol 0.26), gen8 0.184 (tol 0.31), argmax
   matches both. CPU source-of-truth run tightens this further.
3. **Transcript byte-exact** vs offline and the reference, at 1000/2000/20000 ms
   chunk sizes; partials grow monotonically token-by-token.
4. Configurable delay (`--stream-voxtral-delay {2,6,10}`) transcribes jfk
   byte-exactly; the knob threads right-pad + prompt + ada scales.

---

### Original design notes (pre-implementation)

Status: design complete, implementation in progress. The OFFLINE forward is
fully validated (44/44 tensors, exact transcript, WER passing). Streaming is
**numerically equivalent to offline** — proven: reference `stream.logits_raw`
== reference `dec.logits_raw` (gen0 max|d|=0.0, gen8 within bf16 noise), and my
C++ offline logits already satisfy the `stream.*` tolerances (gen0 0.167, gen8
0.216, argmax exact). The incremental encoder is therefore a **throughput**
mechanism with identical numerics; its correctness gate is:

> the incremental (chunked + cached) encoder output must equal my already-
> validated OFFLINE encoder output on the same padded buffer, bit-for-bit on CPU.

## Why offline ≡ streaming

- The offline encoder already applies the **sliding-window(750) causal mask** —
  the model's intrinsic constraint, identical to the encoder `StaticCache(750)`.
- The causal left-pad conv (offline `pad(x,(left_pad,0))`) equals the streaming
  conv padding cache initialized to zeros on the first chunk.
- So whole-buffer compute == chunked-with-cache compute. Verified design-level
  via `VoxtralRealtimeCausalConv1d` (modeling:201-230) and the conv cache
  (modeling:56-113).

## Chunk alignment (the crux)

Feed mel in **8-frame chunks** (`audio_length_per_tok`). Per chunk:

- conv0 (k3 s1, `left_pad=2`): prepend 2 cached mel frames → 10 in → **8** conv0
  frames; cache ← last 2 mel frames. (+ bias, GELU)
- conv1 (k3 s2, `left_pad=1`): prepend 1 cached conv0 frame → 9 in → **4** enc
  frames; cache ← last 1 conv0 frame. (+ bias, GELU)
- 4 enc frames → 32 encoder blocks (RoPE at absolute positions, windowed-causal
  attention against the ≤750-frame StaticCache) → final RMSNorm.
- group 4 enc frames → **1** projector token → 1 audio embed `[3072]`.

`left_pad = (kernel-1)*dilation+1 - stride` (modeling:215-218): conv0=2, conv1=1.
Stride-2 continuity holds because 8 new conv0 frames + 1 cached = 9 → conv-s2
emits 4 and leaves exactly 1 for the next chunk.

## Session streaming state (added to `Session`)

- `stream_pcm` (accumulated PCM incl. the 32-token left-pad of zeros at front),
  `stream_audio_frames_done` (enc frames produced), `stream_mel_done`.
- Conv caches: `conv0_cache [n_mel=128, 2]`, `conv1_cache [d_model=1280, 1]`
  (host f32, carried across chunks).
- Encoder StaticCache: per-layer K,V `[head_dim=64, n_heads=32, ≤750]` (ggml
  backend buffer, ring/trim to 750), `enc_kv_n`, `enc_pos` (absolute frame).
- `stream_audio_embeds` (host `[3072, n_tok_done]`).
- Decoder: reuse `kv_cache` (persistent across feeds), `dec_pos`, `n_committed`,
  `prompt_done` flag, `generated_ids`.
- `ada_scale_all` (already computed for the active delay).

## Per-hook behavior

- **stream_validate(ctx, run, stream)**: pure; validate the family stream-ext
  (delay override range 80 ms–2.4 s → token multiple); accept the stream kind.
- **stream_begin**: init buffers; left-pad `stream_pcm` with `n_left_pad(32)*1280`
  zeros; reset conv caches + encoder StaticCache + decoder KV; compute ada scales
  for the requested delay; build the prompt `[BOS]+[STREAMING_PAD]*(32+delay)`;
  set `prompt_done=false`.
- **stream_feed(pcm,n)**: append PCM. While ≥ 8 new mel frames are available:
  compute the next 8 mel frames (from `stream_pcm`, fixed-max global mel),
  run the incremental-encoder chunk → 1 audio embed; append. Once enough audio
  embeds exist to cover the prompt (39 tokens) prefill the decoder once; then
  decode each newly-available position with the persistent decoder KV, emit
  partial text. Honor a decode throttle (`--stream-chunk-ms` / family ext).
- **stream_finalize**: process the right-pad tail (delay+1+10 tokens of zeros),
  decode remaining positions to EOS or `n_audio` clamp, mark all committed.
- **stream_reset**: clear streaming state, keep allocations.
- **accepts_ext_kind**: true for `(STREAM slot, voxtral_realtime stream kind)`.

## Validation

1. **Internal (primary)**: feed jfk through the stream path in 8-frame chunks;
   the accumulated encoder output and audio embeds must equal the offline
   `enc.out` / `proj.out` (CPU, bit-exact). The conv caches + StaticCache are
   correct iff this holds.
2. **Oracle**: stream gen0/gen8 logits vs `build/validate/.../jfk/stream/ref/
   stream.logits_raw{,.gen8}.f32` within the `stream.*` tolerances (already met
   by offline).
3. **Transcript**: streaming transcript == offline transcript (byte-exact).

## CLI

`--stream-chunk-ms <N>` drives the feed loop in `transcribe-cli` (mirror
moonshine_streaming): read PCM, feed N-ms chunks via `transcribe_stream_feed`,
print incremental text, `transcribe_stream_finalize` at EOF. Family ext header
`include/transcribe/voxtral_realtime.h` with a stream-ext struct (delay + decode
throttle), `accepts_ext_kind` kind constant.

## RECOMMENDED implementation: bounded-window re-encode (lower risk, same outcome)

A hand-rolled encoder `StaticCache(750)` + stride-2 conv padding cache is the
literal reference mechanism but is ~600 lines of high-risk new ggml. There is a
**numerically identical, constant-per-chunk-cost** alternative that REUSES the
already-validated offline `build_encoder_graph` almost unchanged:

- Because enc frame `i` attends only to `[i-749, i]` (the sliding-window mask),
  re-encoding the **last ~760 enc frames** (≈1520 mel frames, conv-aligned to an
  even offset) and keeping only the **last 4** enc frames yields those frames
  bit-exact (full attention window + full conv receptive field inside the slice).
  The first frames of the window are conv/attention-contaminated and discarded.
- `build_encoder_graph` already takes `positions_in` + `mask_in` as inputs, so a
  window just feeds the **absolute** positions `[w_enc_start … ]` and a
  windowed-causal mask — no code change to the encoder graph.
- Cost per chunk is `O(760² · 32)` regardless of total clip length ⇒ **constant
  per-chunk cost** (the perf goal), and memory is bounded.
- Mel is computed offline-style over the padded buffer (validated, exact) — no
  per-chunk center=False mel, no conv cache to get bit-right.

This achieves the user-selected outcome (constant per-chunk cost, exact numerics)
by reusing validated code; the literal KV-StaticCache is a further micro-opt with
identical numerics, deferrable. **Gate identically**: windowed enc frames ==
offline `enc.out` rows; audio embeds == offline `proj.out`; stream logits ==
oracle.

## Risk / notes

- The stride-2 conv cache alignment is the highest-risk piece; gate it against
  offline `enc.embedder.out` first (just the conv stem), then add attention.
- The encoder StaticCache windowed attention is a 4-query-frame "prefill"
  against ≤750 KV — analogous to the decoder step but multi-query.
- For jfk (748 enc frames < 750) the window never trims, so the first validation
  exercises the un-trimmed path; a >750-frame clip is needed to exercise trim.
