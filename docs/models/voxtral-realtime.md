# Voxtral Realtime (2602)

Mistral's [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
ported to transcribe.cpp. A **streaming** audio-LLM (~970 M causal audio
encoder + ~3.4 B Ministral decoder): a left-pad causal conv stem + 32-layer
causal, sliding-window (750), RoPE transformer encoder feeds a 4-frame-group
projector whose audio embeddings are **added** onto the decoder's input
embeddings; a 26-layer Ministral decoder (GQA 32/8, sliding-window 8192) with
delay-token latency conditioning emits one text token per 80 ms audio slot
(12.5 Hz).

Architecturally distinct from the offline [Voxtral 2507](voxtral.md) family
(own arch, streaming frontend with a fixed global log-mel max, causal encoder,
additive fusion, ada-norm FFN scaling) — it shares only the projector shape
and the tekken tokenizer. Licensed Apache-2.0.

## What it's for

Real-time and offline speech-to-text from a 16 kHz mono WAV.

- **Streaming transcription** — incremental emission with a configurable
  latency/quality trade-off. `--stream-chunk-ms <N>` feeds audio in N-ms
  chunks; the final transcript is byte-equal to the offline path.
- **Configurable delay** — `--stream-voxtral-delay <N>` (default 6 = 480 ms;
  range 80 ms–2.4 s) sets the transcription delay.
- Auto language detection (the streaming processor is auto-detect only).

## Download

`handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF`:

| Quantization | Download | Size |
| --- | --- | ---: |
| BF16   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-BF16.gguf)   | 8.87 GB |
| F16    | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-F16.gguf)    | 8.88 GB |
| Q8_0   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf)   | 4.73 GB |
| Q6_K   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q6_K.gguf)   | 3.66 GB |
| Q5_K_M | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf) | 3.28 GB |
| Q4_K_M | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf) | 2.83 GB |

BF16 WER on the full LibriSpeech `test-clean` split (2620 utterances, Whisper
English normalizer): **2.09%**, matching the same-machine HuggingFace
`transformers` reference (**2.08%**) within rounding. Per-quant WER is a
follow-up; authoritative per-preset numbers land at the release WER sweep.

## Quick Start

```bash
cmake -B build && cmake --build build

# offline transcription
build/bin/transcribe-cli \
  -m models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf \
  samples/jfk.wav

# streaming transcription (500 ms chunks)
build/bin/transcribe-cli \
  -m models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf \
  --stream-chunk-ms 500 samples/jfk.wav
```

CLI flags:

- `--stream-chunk-ms <N>` — incremental streaming at N-ms chunk granularity.
- `--stream-voxtral-delay <N>` — transcription delay in audio slots (default
  6 = 480 ms).
- `--spec-k-drafts <N>` — offline-path 1-gram-lookup speculative decoding
  draft length. `-1` (default) uses the family default (`2`). `0` disables
  spec (plain autoregression). `1..8` selects an explicit K. Speculation
  applies to `transcribe_run` / `transcribe-cli` only — the streaming path
  is unaffected.

## Speculative decoding

The offline decoder runs 1-gram-lookup speculative decoding by default. Each
verify pass processes K+1 positions in parallel: position 0 is the model's
true next-token decision; positions 1..K verify K draft tokens read from the
1-gram suffix lookup over the already-decoded prefix. Drafts are accepted as
long as the model's argmax matches the drafted token; the first mismatch ends
the accepted prefix. Because ~60–70% of audio slots emit `STREAMING_PAD` (id
32), the 1-gram lookup hits high acceptance during silence and during repeated
phrases.

The transcript is byte-identical to the K=0 (no-spec) path; only wall-clock
time changes.

Measured offline-decode wall reduction on an AMD Ryzen 7 PRO 4750U (Vega 7
iGPU, LPDDR4-3200 shared memory) at the family default `K=2`:

| Backend | Sample | K=0 decode | K=2 decode | Δ      |
| ------- | ------ | ---------: | ---------: | -----: |
| Vulkan  | jfk    |    10.3 s  |     8.5 s  | −17 %  |
| CPU     | jfk    |    11.5 s  |     7.6 s  | −34 %  |

The CPU win is larger because compute is the tightest budget on CPU; the
verify-N pass amortizes the single-load weight bandwidth over more tokens
without spilling compute headroom. On hardware with more compute relative to
bandwidth (discrete GPUs, M-series, etc.) the curve shifts toward larger K;
plausible sweet spots are K=4..8 there. K=1 is essentially tied with K=2 on
this iGPU; K≥3 regresses because per-call cost grows faster than tokens per
call. Tune via `--spec-k-drafts` or `transcribe_run_params::spec_k_drafts`.

Capability gate: `transcribe_capabilities::supports_spec_decode = true`.
Families that do not advertise this bit silently ignore the field.

## Validation

The streaming scheduler is validated against the upstream `transformers`
reference for true incremental equivalence: the offline and streaming paths
produce byte-equal transcripts, and the encoder StaticCache / decoder
sliding-window rings are gated bit-exact (or within matmul reduction-order
noise) against the reference. See the family port notes in
`docs/porting/families/voxtral_realtime.md` for the full streaming contract.

Upstream: [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).
