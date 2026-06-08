# Parakeet

NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet)
family ported to transcribe.cpp. A FastConformer encoder paired with one
of three decoder heads — TDT (transducer with a duration prediction
head), classical RNN-T, or CTC — and a TDT+CTC hybrid that ships both
heads in one checkpoint. All variants take 16 kHz mono PCM through an
80-bin mel frontend; English-only across the family, with the single
exception of `parakeet-tdt-0.6b-v3` which extends to 25 European
languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/parakeet.md`](../porting/families/parakeet.md).

## Choosing a variant

- **Best accuracy, English.** `parakeet-rnnt-1.1b` and
  `parakeet-tdt-1.1b` land at the top of the family on LibriSpeech
  test-clean. RNN-T is marginally more accurate; TDT decodes faster
  because the duration head lets the decoder skip frames.
- **Best accuracy/speed tradeoff, English.** `parakeet-tdt-0.6b-v2`
  (English-only) — small, fast, near top-line accuracy.
- **Multilingual (25 European languages).** `parakeet-tdt-0.6b-v3` is
  the only multilingual variant; same size as v2, broader coverage at a
  small English-WER cost.
- **Fastest decode at any size.** Use a CTC variant
  (`parakeet-ctc-0.6b` / `parakeet-ctc-1.1b`). Single-pass greedy
  alignment, no transducer loop — at a ~0.2pp WER cost vs the
  same-size RNN-T.
- **Tiny footprint.** `parakeet-tdt_ctc-110m` (135 MB at Q8_0) is the
  smallest Parakeet. The 1.1B `tdt_ctc` ships both heads but is
  primarily useful when you want TDT speed with CTC as a fallback at
  runtime.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix and the comparison to NVIDIA's self-reported numbers.

| Variant | Decoder | Params | Q8_0 size | WER (Q8_0) | Languages | Doc |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `parakeet-tdt-0.6b-v2`     | TDT       | 0.6B | 730 MB  | 1.69% | English | [parakeet-tdt-0.6b-v2.md](parakeet-tdt-0.6b-v2.md) |
| `parakeet-tdt-0.6b-v3`     | TDT       | 0.6B | 740 MB  | 1.94% | 25 European | [parakeet-tdt-0.6b-v3.md](parakeet-tdt-0.6b-v3.md) |
| `parakeet-tdt-1.1b`        | TDT       | 1.1B | 1.27 GB | 1.38% | English | [parakeet-tdt-1.1b.md](parakeet-tdt-1.1b.md) |
| `parakeet-tdt_ctc-110m`    | TDT+CTC   | 110M | 135 MB  | 2.43% | English | [parakeet-tdt_ctc-110m.md](parakeet-tdt_ctc-110m.md) |
| `parakeet-tdt_ctc-1.1b`    | TDT+CTC   | 1.1B | 1.27 GB | 1.87% | English | [parakeet-tdt_ctc-1.1b.md](parakeet-tdt_ctc-1.1b.md) |
| `parakeet-rnnt-0.6b`       | RNN-T     | 0.6B | 730 MB  | 1.62% | English | [parakeet-rnnt-0.6b.md](parakeet-rnnt-0.6b.md) |
| `parakeet-rnnt-1.1b`       | RNN-T     | 1.1B | 1.27 GB | 1.46% | English | [parakeet-rnnt-1.1b.md](parakeet-rnnt-1.1b.md) |
| `parakeet-ctc-0.6b`        | CTC       | 0.6B | 722 MB  | 1.87% | English | [parakeet-ctc-0.6b.md](parakeet-ctc-0.6b.md) |
| `parakeet-ctc-1.1b`        | CTC       | 1.1B | 1.26 GB | 1.85% | English | [parakeet-ctc-1.1b.md](parakeet-ctc-1.1b.md) |
| `parakeet-unified-en-0.6b` | RNN-T     | 0.6B | 731 MB  | 1.60% | English | [parakeet-unified-en-0.6b.md](parakeet-unified-en-0.6b.md) |

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
`parakeet-tdt-0.6b-v3` (25 European languages, no auto-detect —
language hint required). See the family doc for the full runtime
contract.
