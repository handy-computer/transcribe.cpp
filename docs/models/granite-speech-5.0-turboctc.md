# Granite Speech 5.0 TurboCTC

IBM's Granite Speech 5.0 TurboCTC ported to transcribe.cpp. A 470M-parameter
Granite Conformer encoder with a self-conditioned CTC head.

Offline speech-to-text. Takes a 16 kHz mono WAV and produces a transcript. Not
a streaming model, and it does not translate. The Apache-2.0 variant is
multilingual (en, fr, de, es, pt, it, ja) with punctuation and capitalization;
the NC variant is English-only and lowercase.

Unlike Granite Speech 4 / 4.1, there is no LLM head here — just an encoder and
a CTC layer, so decoding is a single pass and there is no timestamp surface.

For the architecture, validation contract and porting notes see
[`docs/porting/families/granite5_ctc.md`](../porting/families/granite5_ctc.md).

## Choosing a variant

Two variants, same architecture and the same speed. They no longer ship the
same kind of model:

- **`granite-speech-5.0-470m-turboctc`** — Apache-2.0. The multilingual
  release IBM published on 2026-10-21 (English, French, German, Spanish,
  Portuguese, Italian, Japanese; cased and punctuated; ~153,000 h of training
  audio). It replaced the original English-only weights under the same name.
- **`granite-speech-5.0-470m-turboctc-nc`** — CC-BY-NC-SA-4.0, research and
  non-commercial only. English-only and lowercase, unchanged upstream since
  August 2026. ShareAlike means anything you derive from it carries the same
  terms.

Take the Apache one unless you specifically want the NC weights.

## All variants

<!-- catalog:family variants=granite-speech-5.0-470m-turboctc,granite-speech-5.0-470m-turboctc-nc -->
| Variant | Params | Languages   | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `granite-speech-5.0-470m-turboctc` |   473M | 7 languages |    506 MB | LibriSpeech test-clean (WER) |     - | -            | [granite-speech-5.0-470m-turboctc.md](granite-speech-5.0-470m-turboctc.md) |
| `granite-speech-5.0-470m-turboctc-nc` |   473M | en          |    506 MB | LibriSpeech test-clean (WER) | 1.29% | -            | [granite-speech-5.0-470m-turboctc-nc.md](granite-speech-5.0-470m-turboctc-nc.md) |
<!-- /catalog -->

Both ship BF16, F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M, from 948 MB down to 279 MB.
Download links are on the per-variant cards.

## Quick start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  samples/jfk.wav
```

## Capabilities

Transcription only. No streaming, no translation, no language detection, and no
timestamps — CTC emission peaks are not word boundaries, so timestamp requests
are rejected rather than answered with a guess. Neither variant has a language
input: `--language` is accepted for the listed languages but reaches nothing.
