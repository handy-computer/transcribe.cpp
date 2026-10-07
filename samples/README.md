# Bench and smoke-test samples

Audio here is fixture, not test data: a published xRT figure is tied to the
exact file that produced it, so a clip named in `catalog/_benchmark_profiles.json`
must never be regenerated, re-encoded, or swapped for a different take. Add a
new file under a new name instead.

All files are 16 kHz mono 16-bit PCM WAV, the only format the CLI accepts.

## Per-language bench clips (FLEURS)

Pulled from the FLEURS test split already on disk under `samples/wer/`, two per
language: a short clip at the same 11 s as `jfk.wav`, and the longest available
utterance, to stand in for `dots.wav`. A single-language fine-tune decodes
English out of distribution and can loop until its position cap, so these exist
to give those variants a benchmark that measures transcription rather than a
repetition loop.

Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs), test
split, licensed **CC-BY-4.0**. The FLEURS utterance id is the original filename
and is recorded here so each clip can be traced back.

| file | duration | config | FLEURS utterance id |
| --- | ---: | --- | --- |
| `ar-short.wav` | 11.00 s | `ar_eg` | 13811922940508003061 |
| `ar-long.wav` | 25.74 s | `ar_eg` | 12943222207631713208 |
| `ja-short.wav` | 10.98 s | `ja_jp` | 8296538110626558656 |
| `ja-long.wav` | 28.20 s | `ja_jp` | 9518252661993015549 |
| `ko-short.wav` | 10.98 s | `ko_kr` | 11859537746411417197 |
| `ko-long.wav` | 25.80 s | `ko_kr` | 15152963524414515048 |
| `ru-short.wav` | 10.98 s | `ru_ru` | 2668014690611039917 |
| `ru-long.wav` | 33.84 s | `ru_ru` | 10388523902227354213 |
| `uk-short.wav` | 10.98 s | `uk_ua` | 7177130321767122387 |
| `uk-long.wav` | 28.92 s | `uk_ua` | 12340201221281017924 |
| `vi-short.wav` | 10.98 s | `vi_vn` | 9897090359729012443 |
| `vi-long.wav` | 25.10 s | `vi_vn` | 16802725889904809484 |
| `zh-short.wav` | 11.00 s | `cmn_hans_cn` | 17639585860488007329 |
| `zh-long.wav` | 31.12 s | `cmn_hans_cn` | 5655534691025010514 |

Regenerate the source pool, not the clips themselves, with
`uv run scripts/wer/ingest.py fleurs <lang>`.

## Everything else

These predate this file and arrived inside unrelated commits, so their source
and licence were never recorded. Treat the provenance column as a known gap to
resolve, not as a statement that the files are unencumbered.

| file | duration | first appeared in | source |
| --- | ---: | --- | --- |
| `jfk.wav` | 11.0 s | `785fe3e3` working parakeet impl | unrecorded |
| `dots.wav` | 35.3 s | `785fe3e3` working parakeet impl | unrecorded |
| `dots-full.wav` | 305.9 s | `f3d68a6b` add more reference audio | unrecorded |
| `german.wav` | 29.3 s | `d46b961d` init cohere support | unrecorded |
| `ja.wav` | 7.2 s | `3a8aa207` basic working sensevoice | unrecorded |
| `ko.wav` | 4.6 s | `3a8aa207` basic working sensevoice | unrecorded |
| `zh.wav` | 5.6 s | `3a8aa207` basic working sensevoice | unrecorded |
| `yue.wav` | 5.2 s | `3a8aa207` basic working sensevoice | unrecorded |
| `ru.wav` | 2.2 s | `c6a93377` working gigaam | unrecorded |
| `death.wav`, `love-loss.wav`, `whole-earth.wav`, `jobs-silence.wav`, `noise.wav` | 15 s - 233 s | `f3d68a6b` add more reference audio | unrecorded |
| `product-names.wav` | 56.1 s | `060e7afa` finish up parity with whisper | unrecorded |
| `cj-swimming-drop.wav` | 70.1 s | `98715301` add iOS example (#36) | unrecorded |
| `multitalker-2spk-mix.wav`, `sortformer-2spk-mix.wav`, `diar/` | various | diarization fixtures | unrecorded |

The bare `ja.wav`, `ko.wav`, `zh.wav`, `yue.wav` and `ru.wav` clips are
referenced by family-doc smoke tests and by gigaam's profile override; they stay
where they are. The `-short` / `-long` pairs above are the bench fixtures.
