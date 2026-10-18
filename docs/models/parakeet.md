# Parakeet

NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet)
family ported to transcribe.cpp. A FastConformer encoder paired with one
of three decoder heads — TDT (transducer with a duration prediction
head), classical RNN-T, or CTC — and a TDT+CTC hybrid that ships both
heads in one checkpoint. All variants take 16 kHz mono PCM through a
mel frontend; English-only across the family, except
`parakeet-tdt-0.6b-v3`, Moondream's post-trained `parakeet-ultra`, Oruk's
retune `orukeet` and the German fine-tune `parakeet-primeline`, which
cover 25 European languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/parakeet.md`](../porting/families/parakeet.md).

## Choosing a variant

Most users want one of three:

- **English transcription → `parakeet-tdt-0.6b-v2`.** The default pick:
  small, fast, near top-of-family accuracy on English.
- **Multilingual (25 European languages) → `parakeet-ultra`.** Moondream's
  post-trained v3: same size, languages and API, more accurate on English
  and on every FLEURS language. A drop-in replacement for
  `parakeet-tdt-0.6b-v3`, which remains available.
  See [parakeet-ultra.md](parakeet-ultra.md) for how it segments long audio.
- **Multilingual, alternative → `orukeet`.** Oruk AI's retune of v3. Same
  size and speed, a bit more accurate than v3, less accurate than
  `parakeet-ultra`. CC-BY-SA-4.0 weights. See [orukeet.md](orukeet.md).
- **German → `parakeet-primeline`.** primeLine's German fine-tune of
  v3. Same size and speed; tuned for German while keeping the other 24
  v3 languages usable.
- **English, alternative → `phonon-2`.** Fermion Research's
  five-value-quantized repack of v3, expanded back to dense weights here.
  English only, accuracy on par with `parakeet-tdt-0.6b-v2`. CC-BY-4.0
  weights. See [phonon-2.md](phonon-2.md).
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

<!-- catalog:family variants=parakeet-tdt-0.6b-v2,parakeet-ultra,parakeet-tdt-0.6b-v3,orukeet,phonon-2,parakeet-primeline,parakeet-tdt-1.1b,parakeet-tdt_ctc-110m,parakeet-tdt_ctc-1.1b,parakeet-rnnt-0.6b,parakeet-rnnt-1.1b,parakeet-ctc-0.6b,parakeet-ctc-1.1b,parakeet-unified-en-0.6b -->
| Variant                    | Params | Languages                  | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities                | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `parakeet-tdt-0.6b-v2`     |   618M | en                         |    730 MB | LibriSpeech test-clean (WER) | 1.69% | token timestamps            | [parakeet-tdt-0.6b-v2.md](parakeet-tdt-0.6b-v2.md) |
| `parakeet-ultra`           |   627M | 25 languages + auto-detect |    740 MB | LibriSpeech test-clean (WER) | 1.80% | token timestamps            | [parakeet-ultra.md](parakeet-ultra.md) |
| `parakeet-tdt-0.6b-v3`     |   627M | 25 languages + auto-detect |    740 MB | LibriSpeech test-clean (WER) | 1.94% | token timestamps            | [parakeet-tdt-0.6b-v3.md](parakeet-tdt-0.6b-v3.md) |
| `orukeet`                  |   627M | 25 languages + auto-detect |    740 MB | LibriSpeech test-clean (WER) | 1.86% | token timestamps            | [orukeet.md](orukeet.md) |
| `phonon-2`                 |   627M | en                         |    740 MB | LibriSpeech test-clean (WER) | 2.15% | token timestamps            | [phonon-2.md](phonon-2.md) |
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
any length is processed in a single pass — pass arbitrarily long recordings. The
exception is `parakeet-ultra`, which cuts audio over 30 s into segments at pauses
found by its voice-activity head; length is still unbounded. See
the [input-length contract](../input-limits.md).

### Long-form audio

Variants whose checkpoint uses full self-attention (`parakeet-tdt-0.6b-v2`,
`parakeet-tdt-0.6b-v3`, `orukeet`, `parakeet-primeline`, the 1.1B and 110M
TDT/RNN-T/CTC variants) run NeMo's `rel_pos_local_attn` window of
`[256, 256]` encoder frames (20.5 s each side). NVIDIA documents this window
as the family's long-audio mode (`change_attention_model("rel_pos_local_attn",
[256, 256])`); NeMo itself defaults to full attention, which it rates for up
to 24 minutes on an 80 GB A100. Clips up to 20.5 s run the full-attention
graph, so their output is identical to full attention; beyond that, encoder
time and memory grow linearly with length instead of quadratically, so a
25-minute talk runs on a 16 GB laptop and multi-hour recordings fit.
`parakeet-tdt_ctc-1.1b` ships its own `[128, 128]` window and
`parakeet-ultra` keeps full attention inside its 30 s segments; the streaming
variants keep their chunked attention.

The port matches NeMo with the same window at tensor level (encoder blocks
and `enc.final` within the family tolerances; six variants on a 70 s clip,
the bounded-window graph, and v3 on a 35 s clip, the dense local graph), so
the accuracy below is the model's rather than the port's.

WER, full attention -> window (Q8_0):

| set | v3 | v2 | tdt-1.1b | orukeet | primeline |
| --- | --- | --- | --- | --- | --- |
| TED-LIUM 3 full talks (10-25 min) | 2.87 -> 2.71 | 3.20 -> 3.11 | 4.98 -> 3.94 | 2.94 -> 3.06 | 3.57 -> 3.82 |
| TED-LIUM 3 talks cut into 1-5 min pieces | 3.19 -> 2.35 | 3.06 -> 3.09 | 4.09 -> 3.91 | 3.37 -> 3.01 | 3.89 -> 4.00 |
| LibriSpeech test-clean, 1-5 min clips | 2.52 -> 2.16 | 1.63 -> 1.74 | 1.31 -> 1.26 | 1.90 -> 1.86 | 2.69 -> 2.65 |

On the other LibriSpeech-clip variants the window is equal or better
(rnnt-0.6b 1.62 -> 1.51, rnnt-1.1b 1.31 -> 1.31, ctc-0.6b 1.86 -> 1.79,
ctc-1.1b 2.11 -> 1.95, tdt_ctc-110m 2.51 -> 2.42).

Non-English (Multilingual TEDx test talks): on 1-5 min clips the window is
equal or better for `orukeet` in every language except Portuguese
(10.1 -> 10.8), and for `parakeet-tdt-0.6b-v3` except French (15.5 -> 16.4)
and Spanish; `parakeet-primeline` loses on French (10.7 -> 11.8),
Portuguese and Spanish. On full talks v3's French drops from 12.4% to 17.7%.
`parakeet-tdt-0.6b-v3` and its fine-tunes can drift into English on accented
spontaneous speech in either mode; under full attention that drift can take
over a whole talk, under the window it is confined to the window.

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
speaker diarization, and VAD as a user-facing feature (`parakeet-ultra` uses
its voice-activity head internally, only to segment long audio). Language
coverage is English-only except `parakeet-tdt-0.6b-v3`, `parakeet-ultra` and
`parakeet-primeline` (25 European languages,
auto-detected; the language hint is accepted but does not change the
output). Note that the v3 lineage,
including `parakeet-primeline`, writes German `ss` where standard
orthography uses `ß`; see
[parakeet-primeline.md](parakeet-primeline.md#orthography-ß-vs-ss) for
why and what to do about it. See the family doc for the full runtime
contract.
