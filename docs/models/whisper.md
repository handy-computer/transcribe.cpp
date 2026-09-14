# Whisper

OpenAI's [Whisper](https://github.com/openai/whisper) family ported to
transcribe.cpp. All twelve OpenAI checkpoints — `tiny`, `base`, `small`,
`medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo`, plus the four
English-only `*.en` siblings — share the same encoder-decoder transformer
architecture and 30-second windowing, so most of the porting and runtime
contract is the same across variants. Pick a size based on the
accuracy/cost tradeoff you want; pick a `.en` variant only if you know the
audio is English.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/whisper.md`](../porting/families/whisper.md).

## Choosing a variant

- **English-only audio?** Prefer the `.en` checkpoint at your chosen size.
  They are typically a touch more accurate than the multilingual sibling
  and slightly cheaper to decode (no language-detection step). They cannot
  transcribe other languages and cannot translate.
- **Other languages, or auto-detect across languages?** Use the
  multilingual checkpoints (no `.en`). They cover 99 languages
  (100 in the v3 family, which adds Cantonese), do automatic language
  identification, and can produce English translations of non-English
  audio when invoked with `task=translate`.
- **Throughput vs accuracy.** WER drops as you go up the size ladder, but
  decode latency scales with parameter count. `tiny` and `base` run in
  near-realtime on CPU; `large` typically wants Metal or a recent CUDA
  GPU. `large-v3-turbo` is large-v3 quality with a much smaller decoder —
  the best accuracy/speed tradeoff for most multilingual workloads.
- **v3 family quirks.** `large-v3` and `large-v3-turbo` use a 128-bin mel
  input (the rest use 80) and add a Cantonese (`yue`) language token. The
  reference dtype shipped is F16, not F32 — they were released in F16
  upstream, so transcribe.cpp follows suit.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset (the default
recommended quant), measured by transcribe.cpp's WER pipeline with
timestamps off (`scripts/wer/run.py --timestamps none`, the WER harness
default). See each per-variant doc for the full quant
matrix (F32/F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M) and a discussion of how our
numbers compare to OpenAI's self-reported figures. Numbers come from single Metal-backed runs; Metal's non-deterministic parallel reductions add ~0.1pp run-to-run variance on the noise floor.

<!-- catalog:family variants=breeze-asr-25,whisper-tiny,whisper-tiny.en,whisper-base,whisper-base.en,whisper-small,whisper-small.en,whisper-medium,whisper-medium.en,whisper-large,whisper-large-v2,whisper-large-v3,whisper-large-v3-turbo -->
| Variant                  | Params | Languages                   | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities                  | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `breeze-asr-25`          |   1.5B | zh, en + auto-detect        |   1.67 GB | LibriSpeech test-clean (WER) | 2.27% | translate, segment timestamps | [handy-computer/Breeze-ASR-25-gguf](https://huggingface.co/handy-computer/Breeze-ASR-25-gguf) |
| `whisper-tiny`           |    38M | 99 languages + auto-detect  |     46 MB | LibriSpeech test-clean (WER) | 7.52% | translate, segment timestamps | [whisper-tiny.md](whisper-tiny.md) |
| `whisper-tiny.en`        |    38M | en                          |     46 MB | LibriSpeech test-clean (WER) | 5.72% | segment timestamps            | [whisper-tiny.en.md](whisper-tiny.en.md) |
| `whisper-base`           |    73M | 99 languages + auto-detect  |     85 MB | LibriSpeech test-clean (WER) | 5.12% | translate, segment timestamps | [whisper-base.md](whisper-base.md) |
| `whisper-base.en`        |    73M | en                          |     85 MB | LibriSpeech test-clean (WER) | 4.16% | segment timestamps            | [whisper-base.en.md](whisper-base.en.md) |
| `whisper-small`          |   242M | 99 languages + auto-detect  |    270 MB | LibriSpeech test-clean (WER) | 3.33% | translate, segment timestamps | [whisper-small.md](whisper-small.md) |
| `whisper-small.en`       |   242M | en                          |    270 MB | LibriSpeech test-clean (WER) | 3.09% | segment timestamps            | [whisper-small.en.md](whisper-small.en.md) |
| `whisper-medium`         |   764M | 99 languages + auto-detect  |    832 MB | LibriSpeech test-clean (WER) | 2.64% | translate, segment timestamps | [whisper-medium.md](whisper-medium.md) |
| `whisper-medium.en`      |   764M | en                          |    831 MB | LibriSpeech test-clean (WER) | 2.72% | segment timestamps            | [whisper-medium.en.md](whisper-medium.en.md) |
| `whisper-large`          |   1.5B | 99 languages + auto-detect  |   1.67 GB | LibriSpeech test-clean (WER) | 2.71% | translate, segment timestamps | [whisper-large.md](whisper-large.md) |
| `whisper-large-v2`       |   1.5B | 99 languages + auto-detect  |   1.67 GB | LibriSpeech test-clean (WER) | 2.97% | translate, segment timestamps | [whisper-large-v2.md](whisper-large-v2.md) |
| `whisper-large-v3`       |   1.5B | 100 languages + auto-detect |   1.67 GB | LibriSpeech test-clean (WER) | 1.82% | translate, segment timestamps | [whisper-large-v3.md](whisper-large-v3.md) |
| `whisper-large-v3-turbo` |   809M | 100 languages + auto-detect |    886 MB | LibriSpeech test-clean (WER) | 2.01% | segment timestamps            | [whisper-large-v3-turbo.md](whisper-large-v3-turbo.md) |
<!-- /catalog -->

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

No practical per-call length limit (`transcribe_capabilities.max_audio_ms == 0`):
Whisper slices long audio into 30-second windows internally and stitches the
results, so you can pass arbitrarily long recordings. See the
[input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-base.en/whisper-base.en-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream OpenAI checkpoint via the per-variant doc's reproduction
section.

## Capabilities

All Whisper variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Long-form audio** via 30-second chunked decoding with the
  prev-context window assembly described in the family doc.
- **Segment timestamps** — the finest granularity the library emits for
  Whisper (`max_timestamp_kind = segment`). Word-level timestamps are not
  currently exposed.
- **Translation** (any supported language → English) on multilingual
  checkpoints — `.en` variants are transcribe-only.

What's not supported (consistent across the family): real-time
streaming (whisper is not streaming-first; chunked 30-second windows
only), VAD, speaker diarization. See the family doc for the full
runtime contract.
