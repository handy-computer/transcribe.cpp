# Moonshine

Useful Sensors' [Moonshine](https://github.com/usefulsensors/moonshine)
family ported to transcribe.cpp. An encoder-decoder transformer that
consumes raw 16 kHz PCM directly via a three-layer Conv1d stem — no
STFT, no mel filterbank — and emits transcript-only output (no language
tokens, no `<|translate|>`, no timestamps). The base English variants
(`moonshine-tiny`, `moonshine-base`) ship alongside 12 language-specific
fine-tunes published by Useful Sensors (vi / uk / zh / ko / ar / ja at
both sizes); each fine-tune is single-language, same architecture, same
runtime contract.

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
- **Non-English audio.** Useful Sensors publishes per-language fine-tunes
  for Vietnamese, Ukrainian, Mandarin, Korean, Arabic, and Japanese at
  both tiny and base sizes. Each is single-language (no auto-detect, no
  translation), same architecture as the corresponding English variant.
  See the **All variants** table below. For models that auto-detect
  language or translate, use Whisper or Parakeet v3.

## All variants

Numbers are for the **Q8_0** preset (the default recommended quant),
measured by transcribe.cpp's WER pipeline with greedy decode
(`num_beams=1`) and `max_length=194` matching the upstream
`generation_config`. CJK rows report **CER** (character error rate) on
FLEURS; everything else reports **WER**. The K-tier presets (Q6_K /
Q5_K_M / Q4_K_M) are intentionally skipped for this family — at
moonshine's hidden / intermediate / vocab sizes, none of the dimensions
divide the k-quant super-block size of 256, so those presets fall back
to Q8_0 storage and would be near-duplicates.

### English

| Variant | Params | Q8_0 size | WER (LibriSpeech test-clean) | Doc |
| --- | ---: | ---: | ---: | --- |
| `moonshine-tiny` |  27M | 34 MB | 4.60% | [moonshine-tiny.md](moonshine-tiny.md) |
| `moonshine-base` |  62M | 74 MB | 3.26% | [moonshine-base.md](moonshine-base.md) |

### Language-specific (Useful Sensors fine-tunes)

Per-language fine-tunes of the same architecture. Acceptance was measured
against the **Transformers F32 reference on the same FLEURS test split**
because Useful Sensors does not publish per-language WER/CER for these
variants. See each repo's `README.md` on Hugging Face for the full
F32 / F16 / Q8_0 table and the reference baseline.

| Variant | Lang | Params | Q8_0 size | Metric (FLEURS test) | Q8_0 | Repo |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `moonshine-tiny-vi` | Vietnamese (vi) | 27M | 34 MB | WER | 13.16% | [handy-computer/moonshine-tiny-vi-gguf](https://huggingface.co/handy-computer/moonshine-tiny-vi-gguf) |
| `moonshine-tiny-uk` | Ukrainian (uk)  | 27M | 34 MB | WER | 18.89% | [handy-computer/moonshine-tiny-uk-gguf](https://huggingface.co/handy-computer/moonshine-tiny-uk-gguf) |
| `moonshine-tiny-zh` | Mandarin (zh)   | 27M | 34 MB | CER | 13.78% | [handy-computer/moonshine-tiny-zh-gguf](https://huggingface.co/handy-computer/moonshine-tiny-zh-gguf) |
| `moonshine-tiny-ko` | Korean (ko)     | 27M | 34 MB | CER |  8.98% | [handy-computer/moonshine-tiny-ko-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ko-gguf) |
| `moonshine-tiny-ar` | Arabic (ar)     | 27M | 34 MB | WER | 26.79% | [handy-computer/moonshine-tiny-ar-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ar-gguf) |
| `moonshine-tiny-ja` | Japanese (ja)   | 27M | 34 MB | CER | 13.36% | [handy-computer/moonshine-tiny-ja-gguf](https://huggingface.co/handy-computer/moonshine-tiny-ja-gguf) |
| `moonshine-base-vi` | Vietnamese (vi) | 62M | 74 MB | WER |  9.79% | [handy-computer/moonshine-base-vi-gguf](https://huggingface.co/handy-computer/moonshine-base-vi-gguf) |
| `moonshine-base-uk` | Ukrainian (uk)  | 62M | 74 MB | WER | 14.39% | [handy-computer/moonshine-base-uk-gguf](https://huggingface.co/handy-computer/moonshine-base-uk-gguf) |
| `moonshine-base-zh` | Mandarin (zh)   | 62M | 74 MB | CER | 17.00% | [handy-computer/moonshine-base-zh-gguf](https://huggingface.co/handy-computer/moonshine-base-zh-gguf) |
| `moonshine-base-ko` | Korean (ko)     | 62M | 74 MB | CER |  8.13% | [handy-computer/moonshine-base-ko-gguf](https://huggingface.co/handy-computer/moonshine-base-ko-gguf) |
| `moonshine-base-ar` | Arabic (ar)     | 62M | 74 MB | WER | 24.50% | [handy-computer/moonshine-base-ar-gguf](https://huggingface.co/handy-computer/moonshine-base-ar-gguf) |
| `moonshine-base-ja` | Japanese (ja)   | 62M | 74 MB | CER | 10.53% | [handy-computer/moonshine-base-ja-gguf](https://huggingface.co/handy-computer/moonshine-base-ja-gguf) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant repo's `README.md` has direct download links and the
full F32 / F16 / Q8_0 measurement table.

## Input limits

Moonshine has no input-length limit, but its decoder is capped at a short output
window — about **48 seconds** of typical speech. A clip whose transcript reaches
that cap is returned with the hard status `TRANSCRIBE_ERR_OUTPUT_TRUNCATED`, the
partial text retained and `transcribe_was_truncated()` set — never silently cut.
It is built for short utterances; segment longer audio (e.g. with VAD). See the
[input-length contract](../input-limits.md).

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
language detection (each fine-tune is single-language; pick the variant
that matches your audio), timestamps, real-time streaming (see
`moonshine-streaming`), VAD, speaker diarization. See the family doc
for the full runtime contract.
