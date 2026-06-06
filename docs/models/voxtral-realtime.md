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

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-BF16.gguf)   | 8.87 GB | 2.08% |
| F16    | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-F16.gguf)    | 8.88 GB | 2.09% |
| Q8_0   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf)   | 4.73 GB | 2.07% |
| Q6_K   | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q6_K.gguf)   | 3.66 GB | 2.08% |
| Q5_K_M | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf) | 3.28 GB | 2.08% |
| Q4_K_M | [GGUF](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf) | 2.83 GB | 2.08% |

WER measured on the full LibriSpeech `test-clean` split (2620 utterances)
with the Whisper-style English text normalizer, offline path, batch size 8
on an NVIDIA L40S. The same-machine HuggingFace `transformers` reference
(`VoxtralForConditionalGeneration`, BF16, greedy) lands at **2.08%**, and the
BF16 GGUF matches it (**2.08%**). Every shipped quant stays within bootstrap
noise of the reference (2.07–2.09%; 95% CI ≈ ±0.18), so the quantization
ladder is WER-neutral down to Q4_K_M.

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

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places). Measured on the offline path at the family-default `K=1`
speculative decoding.

### Apple M4 Max

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 1.22 s (9.0×)   | 1.14 s (9.7×)   |
| Metal   | dots (35.3s) | 4.34 s (8.1×)   | 3.91 s (9.0×)   |
| CPU     | jfk (11.0s)  | 4.43 s (2.5×)   | 4.69 s (2.3×)   |
| CPU     | dots (35.3s) | 13.65 s (2.6×)  | 13.12 s (2.7×)  |

macOS 15, transcribe.cpp `483c122`. Metal device: Apple M4 Max.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 12.62 s (0.87×) | 10.97 s (1.00×) |
| Vulkan  | dots (35.3s) | 39.29 s (0.90×) | 33.51 s (1.05×) |
| CPU     | jfk (11.0s)  | 19.54 s (0.56×) | 13.80 s (0.80×) |
| CPU     | dots (35.3s) | 58.00 s (0.61×) | 41.54 s (0.85×) |

Fedora 43, transcribe.cpp `483c122`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Voxtral-Mini-4B-Realtime-2602 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name voxtral-mini-4b-realtime-2602-publication
```

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
