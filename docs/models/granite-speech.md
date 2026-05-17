# Granite Speech 4 / 4.1

IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech)
family ported to transcribe.cpp. A Conformer audio encoder with block-local
Shaw attention, a windowed projector, and the Granite-4.0-1b LLM as the
text head. Variants differ in (a) the LLM decoding mode — autoregressive
(AR) versus single-pass non-autoregressive (NAR) editor — and (b) which
extras the head exposes: translation, word-level timestamps, etc.

For the architecture deep-dive, validation contract, and porting notes see
the per-decoder family docs:
[`docs/porting/families/granite.md`](../porting/families/granite.md)
covers the AR variants and
[`docs/porting/families/granite_nar.md`](../porting/families/granite_nar.md)
covers the NAR variant. The split exists because the two share an encoder
but their decoder pipelines are structurally different; from a user
perspective they are one family of GGUFs.

## Choosing a variant

- **Best WER, slowest decode.** `granite-speech-4.1-2b` — 4.1 generation
  LLM head, autoregressive; 1.31% WER on LibriSpeech test-clean. Improved
  punctuation and casing over 4.0-1b.
- **Most features.** `granite-speech-4.1-2b-plus` — same 4.1 base plus
  word-level timestamps (`[SS:N]` centisecond markers). Encoder
  concatenates a mid-layer and the final layer (`cat_hidden_layers=[3]`),
  doubling projector K/V width.
- **Fastest, ASR-only.** `granite-speech-4.1-2b-nar` — non-autoregressive
  editor: one forward pass through encoder + projector + bidirectional
  LLM + CTC decode replaces the AR token loop. ~1.5× faster than the AR
  siblings on Metal at Q8_0. No translation, no timestamps. 1.29% WER
  matches the upstream model card exactly.
- **Smallest 4.0-generation model.** `granite-4.0-1b-speech` — the 4.0
  baseline. Same architecture as 4.1-2b. 1.42% WER on LibriSpeech
  test-clean; covers Japanese in addition to en/fr/de/es/pt (4.1
  variants drop Japanese except for NAR).

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix.

| Variant | Decode mode | Params | Q8_0 size | WER (Q8_0) | Languages | Extras | Doc |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `granite-4.0-1b-speech`      | AR (audio-LLM) | ~3B† | 2.56 GB | 1.44% | en, fr, de, es, pt, ja | translate (en ↔ each)       | [granite-4.0-1b-speech.md](granite-4.0-1b-speech.md) |
| `granite-speech-4.1-2b`      | AR (audio-LLM) | ~3B† | 2.56 GB | 1.32% | en, fr, de, es, pt, ja | translate (en ↔ each)       | [granite-speech-4.1-2b.md](granite-speech-4.1-2b.md) |
| `granite-speech-4.1-2b-plus` | AR (audio-LLM) | ~3B† | 2.35 GB | 1.50% | en, fr, de, es, pt     | translate (en ↔ each), word timestamps | [granite-speech-4.1-2b-plus.md](granite-speech-4.1-2b-plus.md) |
| `granite-speech-4.1-2b-nar`  | NAR (editor)   | ~3B† | 2.50 GB | 1.29% | en, fr, de, es, pt     | (ASR only)                  | [granite-speech-4.1-2b-nar.md](granite-speech-4.1-2b-nar.md) |

† Parameter counts include the Conformer audio encoder, the projector,
and the Granite-4.0-1b text LM. The "1b" / "2b" in IBM's variant names
refers to the speech-front-end LoRA delta size, not the fused-stack
total.

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream IBM checkpoint via the per-variant doc's reproduction
section.

## Capabilities

All variants:
- **Transcription** of 16 kHz mono WAV input across the variant's
  supported languages.

AR variants (`granite-4.0-1b-speech`, `granite-speech-4.1-2b`,
`granite-speech-4.1-2b-plus`):
- **Translation** between English and each of the variant's other
  languages, in either direction (en ↔ fr, en ↔ de, en ↔ es, en ↔ pt,
  and en ↔ ja for 4.0-1b and 4.1-2b). There is no direct fr↔de etc. —
  translation always involves English on one side. Use
  `--translate --target-language <bcp47>`; the source language is
  inferred from the audio.

Plus only (`granite-speech-4.1-2b-plus`):
- **Word-level timestamps** as `[SS:N]` centisecond markers interleaved
  with words (`--timestamps word`).

NAR only (`granite-speech-4.1-2b-nar`):
- **Single-pass non-autoregressive decode** — fastest of the four.

What's not exposed by the v1 transcribe.cpp runtime: speaker diarization
(advertised on `-plus`), keyword/hotword biasing (advertised on AR
variants), real-time streaming, VAD. See the per-variant docs for
status.
