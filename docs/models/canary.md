# Canary

NVIDIA's [Canary](https://huggingface.co/collections/nvidia/canary)
family ported to transcribe.cpp. A FastConformer encoder paired with a
Transformer decoder in a multitask AED setup — every variant does both
**ASR** and **speech translation** between supported language pairs.
The variants differ in encoder/decoder depth (and thus
accuracy/latency) and in how many languages they cover.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/canary.md`](../porting/families/canary.md).

## Choosing a variant

- **Broadest language coverage.** `canary-1b-v2` — 25 European
  languages, deeper 8-layer decoder for higher-quality translation.
  Apache-2.0 weights.
- **Best 4-language accuracy.** `canary-1b` (the original release;
  CC-BY-NC-4.0) uses a 24-layer decoder, trading decode speed for
  accuracy on en/de/es/fr.
- **Faster decode, same 4 languages.** `canary-1b-flash` (883M, 4-layer
  decoder) — the speed-tuned sibling at ~1.6% WER on LibriSpeech
  test-clean.
- **Smallest footprint.** `canary-180m-flash` (182M, 208 MB at Q8_0) —
  the ultralight variant; same 4-language coverage as the flash 1B.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix and per-language WER/BLEU tables.

| Variant | Decoder depth | Params | Q8_0 size | WER (Q8_0) | Languages | Doc |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `canary-1b`        | 24 | 1.0B | 1.1 GB | 1.55% | en, de, es, fr  | [canary-1b.md](canary-1b.md) |
| `canary-1b-v2`     |  8 | 978M | 1.1 GB | 1.91% | 25 European     | [canary-1b-v2.md](canary-1b-v2.md) |
| `canary-1b-flash`  |  4 | 883M | 1.0 GB | 1.62% | en, de, es, fr  | [canary-1b-flash.md](canary-1b-flash.md) |
| `canary-180m-flash`|  4 | 182M | 208 MB | 1.93% | en, de, es, fr  | [canary-180m-flash.md](canary-180m-flash.md) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

Every variant accepts up to about **6.7 minutes (400 s)** of 16 kHz mono audio
per call — the encoder's positional table is the binding limit, shared across
the family. Longer audio is rejected up front with
`TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated; split it into
shorter segments. See the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/canary-1b-flash/canary-1b-flash-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream NVIDIA `.nemo` checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Canary variants support:

- **Transcription** of 16 kHz mono WAV input across all supported
  languages (language hint required — no auto-detect).
- **Translation** between supported language pairs per the upstream
  model cards.

What's not supported (consistent across the family): real-time
streaming, VAD, speaker diarization, auto language detection. See the
family doc for the full runtime contract.
