# Moonshine

Useful Sensors' [Moonshine](https://github.com/usefulsensors/moonshine)
family ported to transcribe.cpp. An English-only encoder-decoder
transformer that consumes raw 16 kHz PCM directly via a three-layer
Conv1d stem — no STFT, no mel filterbank — and emits transcript-only
output (no language tokens, no `<|translate|>`, no timestamps).

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/moonshine.md`](../porting/families/moonshine.md).

## Choosing a variant

- **Smallest footprint, near-realtime CPU.** `moonshine-tiny` (27M
  params) at Q8_0 is 34 MB and decodes well above realtime on commodity
  hardware. WER is on par with `whisper-tiny.en` while running on raw
  audio (no mel frontend in the load path).
- **Higher accuracy, still small.** `moonshine-base` (62M params) at
  Q8_0 is 74 MB and lands inside `whisper-small.en` accuracy territory
  for English audio.
- **Streaming workloads.** Moonshine is **not** streaming-first — the
  encoder is global, the decoder runs on the whole utterance. If you
  need chunked / real-time decoding, see
  [`moonshine-streaming`](moonshine-streaming.md), which is a separate
  port from the same publisher.
- **Non-English audio.** Not supported on this family (English-only,
  no language detection, no translation). Use Whisper or Parakeet v3
  for multilingual.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset (the default
recommended quant), measured by transcribe.cpp's WER pipeline with
greedy decode (`num_beams=1`) and `max_length=194` matching the upstream
`generation_config`. See each per-variant doc for the full F32 / F16 /
Q8_0 matrix and the comparison to Useful Sensors' self-reported numbers.
The K-tier presets (Q6_K / Q5_K_M / Q4_K_M) are intentionally skipped
for this family — at moonshine's hidden / intermediate / vocab sizes,
none of the dimensions divide the k-quant super-block size of 256, so
those presets fall back to Q8_0 storage and would be near-duplicates.

| Variant | Params | Q8_0 size | WER (Q8_0) | Doc |
| --- | ---: | ---: | ---: | --- |
| `moonshine-tiny` |  27M | 34 MB | 4.60% | [moonshine-tiny.md](moonshine-tiny.md) |
| `moonshine-base` |  62M | 74 MB | 3.26% | [moonshine-base.md](moonshine-base.md) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-base/moonshine-base-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream Useful Sensors checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Moonshine variants support:

- **Transcription** of 16 kHz mono WAV input directly from raw PCM.
- **Single-utterance decode** — the encoder is global; there's no
  windowing or chunking layer.

What's not supported (consistent across the family): translation,
language detection, multilingual transcription (English only),
timestamps, real-time streaming (see `moonshine-streaming`), VAD,
speaker diarization. See the family doc for the full runtime contract.
