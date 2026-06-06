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

## Validation

The streaming scheduler is validated against the upstream `transformers`
reference for true incremental equivalence: the offline and streaming paths
produce byte-equal transcripts, and the encoder StaticCache / decoder
sliding-window rings are gated bit-exact (or within matmul reduction-order
noise) against the reference. See the family port notes in
`docs/porting/families/voxtral_realtime.md` for the full streaming contract.

Upstream: [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).
