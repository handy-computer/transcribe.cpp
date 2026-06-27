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
  LLM head, autoregressive; 1.31% BF16 / 1.32% Q8_0 WER on LibriSpeech
  test-clean. Improved punctuation and casing over 4.0-1b.
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
  baseline. Same architecture as 4.1-2b. 1.42% BF16 / 1.44% Q8_0 WER on
  LibriSpeech test-clean; covers Japanese in addition to en/fr/de/es/pt —
  as does the base `granite-speech-4.1-2b`. The `granite-speech-4.1-2b-plus`
  and `granite-speech-4.1-2b-nar` variants drop Japanese (en/fr/de/es/pt only).

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix.

| Variant | Decode mode | Params | Q8_0 size | WER (Q8_0) | Languages | Extras | Doc |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `granite-4.0-1b-speech`      | AR (audio-LLM) | ~3B† | 2.56 GB | 1.44% | en, fr, de, es, pt, ja | translate (en ↔ ASR langs; en → it/zh) | [granite-4.0-1b-speech.md](granite-4.0-1b-speech.md) |
| `granite-speech-4.1-2b`      | AR (audio-LLM) | ~3B† | 2.56 GB | 1.32% | en, fr, de, es, pt, ja | translate (en ↔ ASR langs; en → it/zh) | [granite-speech-4.1-2b.md](granite-speech-4.1-2b.md) |
| `granite-speech-4.1-2b-plus` | AR (audio-LLM) | ~3B† | 2.35 GB | 1.50% | en, fr, de, es, pt     | word timestamps (ASR only)  | [granite-speech-4.1-2b-plus.md](granite-speech-4.1-2b-plus.md) |
| `granite-speech-4.1-2b-nar`  | NAR (editor)   | ~3B† | 2.33 GB | 1.29% | en, fr, de, es, pt     | (ASR only)                  | [granite-speech-4.1-2b-nar.md](granite-speech-4.1-2b-nar.md) |

† Parameter counts include the Conformer audio encoder, the projector,
and the Granite-4.0-1b text LM. The "1b" / "2b" in IBM's variant names
refers to the speech-front-end LoRA delta size, not the fused-stack
total.

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

Every variant accepts up to about **6 minutes** of 16 kHz mono audio per call —
the Granite-4.0-1b LLM's 4,096-token context is the binding limit. Longer audio
is rejected up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently
truncated; split it into shorter segments. (The autoregressive variants also
return `TRANSCRIBE_ERR_OUTPUT_TRUNCATED` if a transcript itself runs into the
budget; the NAR editor produces its output in one pass and cannot.) See the
[input-length contract](../input-limits.md).

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

Translation (`granite-4.0-1b-speech`, `granite-speech-4.1-2b`):
- **Translation** between English and each ASR language in either direction
  (en ↔ fr, en ↔ de, en ↔ es, en ↔ pt, en ↔ ja), plus English-to-Italian
  and English-to-Mandarin (`--target-language it` / `zh`). There is no direct
  fr↔de etc. — translation always involves English on one side. Use
  `--translate --target-language <bcp47>`; the source language is inferred
  from the audio. The `-plus` variant is ASR-only and does not translate.

Plus only (`granite-speech-4.1-2b-plus`):
- **Word-level timestamps** as `[SS:N]` centisecond markers interleaved
  with words (`--timestamps word`).

NAR only (`granite-speech-4.1-2b-nar`):
- **Single-pass non-autoregressive decode** — fastest of the four.

What's not exposed by the v1 transcribe.cpp runtime: speaker diarization
(advertised on `-plus`), keyword/hotword biasing (advertised on AR
variants), real-time streaming, VAD. See the per-variant docs for
status.
