# Parakeet

NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet)
family ported to transcribe.cpp. A FastConformer encoder paired with one
of three decoder heads — TDT (transducer with a duration prediction
head), classical RNN-T, or CTC — and a TDT+CTC hybrid that ships both
heads in one checkpoint. All variants take 16 kHz mono PCM through an
80-bin mel frontend; English-only across the family, except
`parakeet-tdt-0.6b-v3` and its German fine-tune `parakeet-primeline`,
which cover 25 European languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/parakeet.md`](../porting/families/parakeet.md).

## Choosing a variant

Most users want one of three:

- **English transcription → `parakeet-tdt-0.6b-v2`.** The default pick:
  small, fast, near top-of-family accuracy on English.
- **Multilingual (25 European languages) → `parakeet-tdt-0.6b-v3`.** The
  only multilingual variant; same size as v2, broader coverage at a
  small English-WER cost.
- **German → `parakeet-primeline`.** primeLine's German fine-tune of
  v3. Same size and speed; tuned for German while keeping the other 24
  v3 languages usable.
- **Streaming / real-time → Nemotron streaming.** Most Parakeet variants here
  are offline-only. For low-latency streaming use the FastConformer-lineage
  [`nemotron-3.5-asr-streaming-0.6b`](nemotron-3.5-asr-streaming-0.6b.md)
  (multilingual) or
  [`nemotron-speech-streaming-en-0.6b`](nemotron-speech-streaming-en-0.6b.md)
  (English). Within Parakeet itself, `parakeet-unified-en-0.6b` is the only
  streaming-capable variant.

If you specifically need the lowest WER or a different decoder:

- **Lowest WER, English.** `parakeet-tdt-1.1b` (1.38% Q8_0) is the most
  accurate in the family, narrowly ahead of `parakeet-rnnt-1.1b` (1.46%).
  TDT also decodes faster — the duration head lets the decoder skip frames.
- **Fastest decode at any size.** Use a CTC variant
  (`parakeet-ctc-0.6b` / `parakeet-ctc-1.1b`). Single-pass greedy
  alignment, no transducer loop — at a ~0.2pp WER cost vs the
  same-size RNN-T.
- **Tiny footprint.** `parakeet-tdt_ctc-110m` is the smallest Parakeet. The 1.1B `tdt_ctc` ships both heads but is
  primarily useful when you want TDT speed with CTC as a fallback at
  runtime.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix and the comparison to NVIDIA's self-reported numbers.

<!-- catalog:family variants=parakeet-tdt-0.6b-v2,parakeet-tdt-0.6b-v3,parakeet-primeline,parakeet-tdt-1.1b,parakeet-tdt_ctc-110m,parakeet-tdt_ctc-1.1b,parakeet-rnnt-0.6b,parakeet-rnnt-1.1b,parakeet-ctc-0.6b,parakeet-ctc-1.1b,parakeet-unified-en-0.6b -->
| Variant                    | Params | Languages                  | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities                | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `parakeet-tdt-0.6b-v2`     |   618M | en                         |    730 MB | LibriSpeech test-clean (WER) | 1.69% | token timestamps            | [parakeet-tdt-0.6b-v2.md](parakeet-tdt-0.6b-v2.md) |
| `parakeet-tdt-0.6b-v3`     |   627M | 25 languages + auto-detect |    740 MB | LibriSpeech test-clean (WER) | 1.94% | token timestamps            | [parakeet-tdt-0.6b-v3.md](parakeet-tdt-0.6b-v3.md) |
| `parakeet-primeline`       |   627M | 25 languages + auto-detect |    740 MB | FLEURS de (WER)              | 5.98% | token timestamps            | [parakeet-primeline.md](parakeet-primeline.md) |
| `parakeet-tdt-1.1b`        |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.38% | token timestamps            | [parakeet-tdt-1.1b.md](parakeet-tdt-1.1b.md) |
| `parakeet-tdt_ctc-110m`    |   114M | en                         |    135 MB | LibriSpeech test-clean (WER) | 2.43% | token timestamps            | [parakeet-tdt_ctc-110m.md](parakeet-tdt_ctc-110m.md) |
| `parakeet-tdt_ctc-1.1b`    |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.87% | token timestamps            | [parakeet-tdt_ctc-1.1b.md](parakeet-tdt_ctc-1.1b.md) |
| `parakeet-rnnt-0.6b`       |   617M | en                         |    730 MB | LibriSpeech test-clean (WER) | 1.62% | token timestamps            | [parakeet-rnnt-0.6b.md](parakeet-rnnt-0.6b.md) |
| `parakeet-rnnt-1.1b`       |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.46% | token timestamps            | [parakeet-rnnt-1.1b.md](parakeet-rnnt-1.1b.md) |
| `parakeet-ctc-0.6b`        |   609M | en                         |    722 MB | LibriSpeech test-clean (WER) | 1.87% | token timestamps            | [parakeet-ctc-0.6b.md](parakeet-ctc-0.6b.md) |
| `parakeet-ctc-1.1b`        |   1.1B | en                         |   1.26 GB | LibriSpeech test-clean (WER) | 1.85% | token timestamps            | [parakeet-ctc-1.1b.md](parakeet-ctc-1.1b.md) |
| `parakeet-unified-en-0.6b` |   618M | en                         |    731 MB | LibriSpeech test-clean (WER) | 1.60% | streaming, token timestamps | [parakeet-unified-en-0.6b.md](parakeet-unified-en-0.6b.md) |
<!-- /catalog -->

\* `parakeet-primeline` is scored on FLEURS German (862 utterances),
not LibriSpeech test-clean, so its number is not comparable to the rest
of the column. Its NeMo reference on the same manifest is 5.98%.

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

No practical per-call length limit (`transcribe_capabilities.max_audio_ms == 0`):
the Conformer encoder's positional encoding is recomputed per call, so audio of
any length is processed in a single pass — pass arbitrarily long recordings. See
the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream NVIDIA `.nemo` checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Parakeet variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Token-level timestamps** at the encoder frame rate (TDT and RNN-T;
  CTC also exposes frame-level alignment).

**Buffered streaming** is supported on `parakeet-unified-en-0.6b`
across all six published `(L, C, R)` configurations from the model's
training menu (lookahead latency from 160ms at `(70, 1, 1)` through
2.08s at the default `(70, 13, 13)`). See
[parakeet-unified-en-0.6b.md](parakeet-unified-en-0.6b.md#streaming)
for the per-config WER and the `--stream-buf-{left,chunk,right}-ms`
CLI surface. Other Parakeet variants run offline only.

What's not supported (consistent across the family): translation,
VAD, speaker diarization. Language coverage is English-only except
`parakeet-tdt-0.6b-v3` and `parakeet-primeline` (25 European languages,
no auto-detect — language hint required). Note that the v3 lineage,
including `parakeet-primeline`, writes German `ss` where standard
orthography uses `ß`; see
[parakeet-primeline.md](parakeet-primeline.md#orthography-ß-vs-ss) for
why and what to do about it. See the family doc for the full runtime
contract.
