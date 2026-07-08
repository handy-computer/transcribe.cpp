# Cohere Transcribe

Cohere's [Transcribe](https://huggingface.co/CohereLabs) family ported to
transcribe.cpp. A large Conformer encoder paired with a lightweight
Transformer decoder (cross-attention, tied token embedding); decoding is
autoregressive with a language-conditioned prompt. The variants share one
architecture (2.0B parameters, identical tensor shapes) and differ in
training focus and language coverage.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/cohere.md`](../porting/families/cohere.md).

## Choosing a variant

- **Broad multilingual coverage.** `cohere-transcribe-03-2026` — 14
  languages (English, French, German, Spanish, Italian, Portuguese,
  Dutch, Polish, Greek, Arabic, Japanese, Chinese, Vietnamese, Korean).
- **Arabic-focused.** `cohere-transcribe-arabic-07-2026` — retrained for
  Arabic, including dialects and Arabic-English code-switching, with
  English as a secondary language. Prefer it over the base model for
  Arabic audio.

## All variants

WER is for the **Q8_0** preset, measured by transcribe.cpp's WER
pipeline; each variant is evaluated on the dataset that matches its
focus. See each per-variant doc for the full quant matrix and
methodology.

| Variant | Params | Q8_0 size | WER (Q8_0) | Eval set | Languages | Doc |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `cohere-transcribe-03-2026`        | 2.0B | 2.41 GB | 1.27%  | LibriSpeech test-clean | 14 languages | [cohere-transcribe-03-2026.md](cohere-transcribe-03-2026.md) |
| `cohere-transcribe-arabic-07-2026` | 2.0B | 2.41 GB | 11.06% | FLEURS Arabic test     | ar, en       | [cohere-transcribe-arabic-07-2026.md](cohere-transcribe-arabic-07-2026.md) |

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

Pick a variant and run (pass the audio's language with `-l`):

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/cohere-transcribe-arabic-07-2026/cohere-transcribe-arabic-07-2026-Q8_0.gguf \
  -l ar \
  input.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream Cohere checkpoint via the per-variant doc's reproduction
section (the upstream repos are gated; accept the license first).

## Capabilities

All Cohere Transcribe variants support:

- **Transcription** of 16 kHz mono WAV input across the variant's
  supported languages (language hint required — no auto-detect).
- **Punctuation and capitalization** by default.

What's not supported (consistent across the family): real-time
streaming, translation, timestamps, VAD, speaker diarization, auto
language detection. See the family doc for the full runtime contract.
