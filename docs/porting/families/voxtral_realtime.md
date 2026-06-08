# Voxtral Realtime

Status: shipped (Stage 8, 2026-06-06)

Release WER (full LibriSpeech test-clean, 2620 utts, offline b8, L40S): BF16
**2.08%** vs measured Oracle reference **2.08%** → ref-dtype gate PASS; all
shipped quants 2.07–2.09%. See
`reports/wer/voxtral-mini-4b-realtime-2602.librispeech-test-clean.summary.md`.

Mistral's **Voxtral Realtime 2602** — a *streaming* audio-LLM. Architecturally
distinct from the 2507 `voxtral` family (own `model_type`, own arch dir); it
shares only the projector shape, the tekken tokenizer, and the family brand.

See `reports/porting/voxtral_realtime/forward-map.md` for the full per-stage
forward map and ggml mapping.

## Identity

- Family key: `voxtral_realtime`
- Upstream architecture string: `voxtral_realtime`
  (`VoxtralRealtimeForConditionalGeneration`, transformers ≥ 5.x)
- Hugging Face repo: `mistralai/Voxtral-Mini-4B-Realtime-2602`
  (pinned `2769294da9567371363522aac9bbcfdd19447add`)
- License: Apache-2.0
- Reference framework: transformers (v5.10.2 oracle); tokenizer + streaming
  prompt template via `mistral-common` (tekken).
- Variants:
  - `voxtral-mini-4b-realtime-2602` — ~3.4B Ministral LM + ~970M causal audio
    encoder; BF16; streaming. Only variant.

## Architecture (summary)

Pattern: **audio-llm**, streaming, ADDITIVE audio fusion.

- **Frontend**: streaming log-mel, 128 mel / n_fft 400 / hop 160 / win 400
  periodic Hann / slaney 0–8000 Hz / drop-last-STFT-frame, **FIXED
  `global_log_mel_max=1.5`** (`max(log,−6.5)`, `(log+4)/4`). `center=True`
  first chunk only; conv padding cache continues across chunks.
- **Encoder** (`VoxtralRealtimeEncoder`): causal left-pad conv stem
  (128→1280 k3 s1, 1280→1280 k3 s2, GELU), 32 pre-norm RMSNorm blocks,
  **NEOX RoPE θ=1e6 hd 64**, **causal + sliding-window(750)** attention
  (q/v/o bias, k none), SwiGLU/silu MLP (down has bias), final RMSNorm.
- **Projector**: reshape 4× frame group `[1280,T]→[5120,T/4]` → Linear 5120→3072
  → GELU → Linear 3072→3072 (both no bias).
- **Decoder** (Ministral, 26 layers): token embed, **additive** audio fusion
  (`inputs_embeds += proj_out` at STREAMING_PAD positions), GQA 32q/8kv hd128,
  **NEOX RoPE θ=1e6**, sliding 8192, SwiGLU (interm 9216). Per layer, a
  **delay-conditioned ada-norm FFN scale**: `post_attention_layernorm(h) ·
  (1 + ada(t_cond))`, `ada=linear→gelu→linear`, `t_cond` a fixed sinusoid of
  `num_delay_tokens` ⇒ per-layer constant scale precomputed once per run.
  **Tied lm_head** (= `embed_tokens`).
- **Streaming**: conv padding cache + encoder StaticCache(750) + decoder
  sliding-KV(8192) re-run incrementally; downsample_factor=4 enc frames per
  decode step (12.5 Hz); output length clamped to
  `ceil(mel_frames / audio_length_per_tok=8)`; configurable `num_delay_tokens`
  (default 6 = 480 ms).

## Family-specific requirements (do not flow through convert/validate)

These need explicit C++ work and are NOT covered by the converter or the
per-tensor validate gate:

1. **Streaming scheduler** (true incremental, user-signed 2026-06-05): conv
   padding cache, encoder StaticCache(750), decoder sliding-KV(8192), windowed-
   causal masks at two widths, 12.5 Hz chunk scheduling, delay-token prompt
   layout, `--stream-chunk-ms` CLI + the 5 `stream_*` arch hooks, incremental
   token emission. Gated against the `stream` oracle for numerical equivalence
   with the offline path.
2. **Ada-norm FFN scale**: per-layer constant `[3072]` scale precomputed from
   `dec.time_embed.inv_freq` + `dec.blocks.l.ada.linear_{1,2}` at session setup
   (depends on the runtime `num_delay_tokens`). Applied via a new optional
   `view.ffn_scale` on the shared `causal_lm` block (null for all other callers).
3. **Streaming log-mel with fixed global max** (`MelFrontend normalize=global`,
   `global_log_mel_max=1.5`) — distinct from the per-utterance max used by
   whisper / voxtral-2507. Causal conv stem (left-pad only).
4. **Additive audio fusion** onto STREAMING_PAD(32) placeholder positions.

## Capability Validation

Targets below are **user-signed (2026-06-05)**. Acceptance dataset:
**LibriSpeech test-clean** (32-utterance subset; English; **auto language** —
the streaming processor is auto-detect only, `--language` is accepted but
ignored). Oracle reference WER (our own run): **1.05 %**
(`reports/wer/voxtral-mini-4b-realtime-2602-REF.librispeech-test-clean.b8`).
`Status` is filled at Stage 4.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602-BF16.gguf samples/jfk.wav` | non-empty plausible English transcript on the auto-detect path | MUST PASS | **PASS** — jfk (cpu non-flash AND metal) → byte-exact reference transcript "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." |
| Streaming | realtime (`capabilities.streaming`) | `build/bin/transcribe-cli -m .../voxtral-mini-4b-realtime-2602-BF16.gguf --stream-chunk-ms <N> samples/jfk.wav` | incremental emission; final transcript byte-equal to the offline path and to the `stream` oracle (`stream.logits_raw`/`gen8` within tolerance) | MUST PASS (forced) — true incremental streaming, user-signed 2026-06-05 | **PASS** — **true incremental** scheduler matching the reference generate() mechanism: conv stem → **encoder StaticCache** (4 new embedder frames/step against the cached K/V under the 750-frame sliding-window mask) → projector → **persistent-KV decoder** (1 token/frame). Validated (not by construction): (1) the StaticCache encoder is **bit-exact** vs the whole-clip encoder — stream `proj.out` vs offline `proj.out` max\|d\|=**0.0** on Metal (single AND multi-feed 1000 ms chunks → cross-feed cache accumulation correct); CPU 6.8e-3/mean-7.6e-6 is matmul reduction-order noise. (2) **Formal CPU gate vs the reference `stream` oracle**: gen0 max\|d\|=0.160 (tol 0.26), gen8 0.199 (tol 0.31), argmax matches both → PASS. (3) Transcript byte-exact at 500/1000/2000/20000 ms; partials grow monotonically. (4) **Unbounded length via a 750-frame ring** (the reference `StaticSlidingWindowLayer`): the encoder cache is k_enc_ring_ctx=1536 slots, compacted (last 750 kept) so memory is constant for any stream length; decoder KV grows once to the 8192 sliding window. Validated on whole-earth (84 s, 4416 frames, multiple compactions): byte-exact transcript vs the offline whole-clip encoder, `proj.out` max\|d\|=1.2e-2 (f32 reduction-order noise, tol 2.2e-1); and dots-full (306 s, ~15300 frames, ~20 compactions) transcribes correctly at 3.6× realtime. (5) **Decoder sliding-window ring** (>8192 tokens / >10.9 min): the decoder KV is a modulo ring of `dec_sliding_window` (8192, read from the GGUF) — token `cur` writes slot `cur % n_ctx`, evicting `cur−8192` (one past the window); the step mask reveals `[cur−8191, cur]` mapped through `% n_ctx`; RoPE positions stay absolute (cap = `max_position_embeddings`=131072). Matches the reference `DynamicSlidingWindowLayer` (keep last swin−1 + new). Validated: ≤8192 byte-identical to the pre-ring path (jfk); wrap A/B byte-identical (ring 8192 vs a non-wrapping 16384, 10 259-token clip); and **byte-identical to the reference `cmd_stream`** (real `DynamicSlidingWindowLayer`) over **10 269 tokens crossing the wrap** — both free-running, zero argmax divergence. (6) **Incremental frontend (conv padding cache)**: only new mel frames feed the conv stem against carried caches (conv0 ← last 2 mel; conv1 ← last 1 conv0-out; reference `VoxtralRealtimeConv1dCacheLayer`) instead of recomputing the whole buffer (was O(N²/chunk)); `proj.out` max\|d\|=**0.0** vs offline on jfk, byte-exact transcript on whole-earth (with encoder compaction) and the 10 259-token clip. (7) **Streaming mel**: the log-mel is computed over a sliding window `[committed−GUARD, end]` (GUARD=4 discards the centre-pad reflect-contaminated frames; pad/hop=1.25) rather than the whole buffer; `proj.out` max\|d\|=**0.0** vs offline. Measured per-component wall (dots-full @ 500 ms chunks, `TRANSCRIBE_VOXTRAL_REALTIME_STREAM_TIMING=1`): mel **30.6%→0.2%** (38.4 s→0.14 s), conv 0.3% (cached, flat), so the **frontend is O(N) at ~0.5% of wall** (down from O(N²/chunk)); total 125.7 s→84.5 s (−33%) at low latency. Remaining cost is the decoder (81%) and per-feed encoder graph overhead (18.6%). (8) **Frontend memory bounding** (matches the reference's bounded streaming): the buffer is built only for the mel window `[ws_sample, end)` and PCM older than the window is erased (`stream_pcm_drop` keeps absolute indexing); consumed audio-embeds are released after each decode step (`stream_audio_base`; skipped under dump so the proj.out history stays whole). Demonstrated on dots-full (306 s, 3873 tokens): retained **PCM 1.0 s** (not 306 s) and **audio-embeds 0 frames** at finalize — both O(1) in length (~2.3 GB saved at the 131072-token max), byte-exact transcript. Frontend compute *and* memory are now O(N)/O(1). **Encoder per-feed profiling** (env `TRANSCRIBE_VOXTRAL_REALTIME_STREAM_TIMING`, dots-full, single-run): the CPU graph-build/alloc overhead is small (~0.6–1.3 s total, 2–8 ms/feed); the per-feed cost is GPU-side `graph_compute` (~50–65 ms/feed, the 32-layer encoder over ~24–48 new frames re-reading the 750-window, kernel-launch-bound) — lever is larger batches (latency trade) or encoder flash, not graph reuse. |
| Configurable transcription delay | realtime | `--stream-voxtral-delay N` (default 6 = 480 ms) on the streaming path | transcript produced at the requested delay; offline↔stream equivalence holds at the configured delay | MUST PASS | **PASS** — jfk on Metal at `--stream-voxtral-delay {2,6,10}` all produce the byte-exact reference transcript; the knob threads through the audio right-pad, the `[BOS]+[STREAMING_PAD]*(32+delay)` prompt, and the per-layer ada-norm scales (recomputed for the active delay). Exposed via `transcribe_voxtral_realtime_stream_ext.num_delay_tokens` (`include/transcribe/voxtral_realtime.h`). |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model .../voxtral-mini-4b-realtime-2602-BF16.gguf --list <4 short clips> --batch-sizes 2,4,8 --backend cpu --golden-out tests/golden/batch/voxtral-mini-4b-realtime-2602.cpu.json` | byte-identical hypotheses | MUST PASS (forced — batching is not optional) | **PASS** — real parallel `run_batch()` implemented (`src/arch/voxtral_realtime/`), a 5-pass path following the HF reference's batched audio tower. (1) **Batched whole-clip encoder** (`build_encoder_graph_batched` + `mha_batched`): each clip's mel is right-padded to the batch max and stacked on `ne[2]`; because the encoder is causal (left-pad conv + sliding-window attention) a real frame attends only to `≤t`, so right-padding never contaminates a row's real `[0,T_enc_b)` outputs and a **single shared positions + sliding-window(750) mask** serves every row — exactly as the reference runs the tower with no audio `attention_mask`. Dual-path (flash + manual softmax) so the default CPU source-of-truth (`encoder_use_flash=false`) batches too; the caller slices each row's real `n_audio_b`. (2) **Shared ada `ffn_scale`** (one compute; `num_delay` uniform per batch). (3) **Batched rectangular prefill** (`build_prefill_graph_batched`): uniform prompt length ⇒ no ragged pad/gather; pure additive fusion; TIED head. (4) **Batched GQA step loop** (`build_step_graph_batched` + `run_batch_step_loop`) over `causal_lm::block_*_batched`, adding **per-step audio injection** (a fresh audio embed per row/position) and a **per-row `n_audio` clamp** (each row stops at EOS or its own audio length — deliberately tighter than the reference's batch-wide `max_length`, for serial parity). Gated on `decoder_use_flash && n>1 && !debug` (serial fallback otherwise). Validated: `batch_parity.py` **PARITY OK** at sizes 2,4 — byte-identical to serial AND the frozen golden on **both CPU and Metal**; `validate.py all` 44/44 tensors unmoved (single path additive-only). The forced-batching policy is met with a true parallel path (no `ACCEPTED GAP`). See `reports/porting/voxtral_realtime/batching-plan.md`. (user-directed 2026-06-05) |
| Transcribe | explicit language hint | `--language <l>` | (processor is auto-detect only; hint ignored) | OUT OF SCOPE — streaming processor is auto-detect only (`TranscriptionRequest language=None`); `--language` accepted but ignored | SKIP — auto-detect only; runtime exposes no language hint to the realtime prompt |
| Transcribe (multilingual) | non-English audio | `build/bin/transcribe-cli -m .../-BF16.gguf samples/<non-en>.wav` | plausible non-English transcript | OUT OF SCOPE — English-only acceptance (user); multilingual WER not gated this port | SKIP — not gated (English-only acceptance); multilingual is demonstrable but out of scope |
| Language detection | — | n/a | detected language not separately surfaced | OUT OF SCOPE — auto-detect happens internally; not exposed as an observable by the transcribe CLI | SKIP — not exposed by runtime |
| Translate (speech→text) | — | n/a | — | OUT OF SCOPE — `translation=false` for this model | SKIP — not exposed by runtime (`translation=false`) |
| Segment timestamps | — | n/a | — | OUT OF SCOPE — model emits no timestamp tokens | SKIP — model emits no timestamp tokens |
| Word timestamps | — | n/a | — | OUT OF SCOPE — model emits no timestamp tokens | SKIP — model emits no timestamp tokens |

## Notes

- For jfk (`T_enc=748 < sliding_window=750`) the offline whole-clip encoder is
  exact ⇒ offline == streaming numerics; the `stream` oracle confirms this.
- `num_delay_tokens` default 6 (480 ms) is the publisher-recommended setting and
  the Oracle dump value.
