# GigaAM-v3

ai-sage's [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
family ported to transcribe.cpp. Russian-only ASR built around a shared
16-layer Conformer encoder (768-d, 16 heads, rotary positional
embeddings), paired with one of two decoder heads (RNN-T or CTC) and
trained either end-to-end on cased+punctuated text or on
lowercased no-punctuation text with a 33-entry character vocabulary.
The four shipped variants are the four cells of that 2×2.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/gigaam.md`](../porting/families/gigaam.md).

## Choosing a variant

- **Cased + punctuated Russian, best accuracy.** `gigaam-v3-e2e-rnnt`
  — RNN-T head trained end-to-end with the 1024-piece SentencePiece
  vocab.
- **Cased + punctuated Russian, fastest decode.** `gigaam-v3-e2e-ctc`
  — same training data and tokenizer as `e2e-rnnt`, single-pass CTC
  alignment instead of the transducer loop.
- **Lowercased no-punct (charwise output).** `gigaam-v3-rnnt` and
  `gigaam-v3-ctc` — 33-entry character vocabulary (space + а–я),
  output is normalized for downstream ASR scoring pipelines that expect
  this convention. Higher raw WER than the e2e variants because errors
  on case/punctuation are no longer absorbed by tokenization.
- The encoder is structurally identical across the four variants but
  the weights are per-variant fine-tuned — you cannot swap heads at
  runtime.

## All variants

WER is on FLEURS Russian (`fleurs-ru`) for the **Q8_0** preset,
measured by transcribe.cpp's WER pipeline. See each per-variant doc
for the full quant matrix.

| Variant | Decoder | Output | Params | Q8_0 size | WER (Q8_0) | Doc |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `gigaam-v3-e2e-rnnt` | RNN-T | cased + punctuated         | ~180M | 261 MB | 5.36% | [gigaam-v3-e2e-rnnt.md](gigaam-v3-e2e-rnnt.md) |
| `gigaam-v3-e2e-ctc`  | CTC   | cased + punctuated         | ~180M | 260 MB | 5.50% | [gigaam-v3-e2e-ctc.md](gigaam-v3-e2e-ctc.md) |
| `gigaam-v3-rnnt`     | RNN-T | lowercased, no-punctuation | ~180M | 260 MB | 8.08% | [gigaam-v3-rnnt.md](gigaam-v3-rnnt.md) |
| `gigaam-v3-ctc`      | CTC   | lowercased, no-punctuation | ~180M | 259 MB | 8.40% | [gigaam-v3-ctc.md](gigaam-v3-ctc.md) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

GigaAM is trained for utterances up to about **25 seconds**. Longer audio is
accepted, but the library logs a `WARN` and accuracy may degrade past that
window — it is not rejected (upstream GigaAM rejects outright; transcribe.cpp
leaves the choice to you). Segment long recordings (e.g. with VAD) for best
results. See the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream `ai-sage/GigaAM-v3` checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All GigaAM-v3 variants support:

- **Transcription** of 16 kHz mono WAV input, Russian only.
- **Token-level timestamps** at the encoder frame rate (40 ms
  granularity).

What's not supported (consistent across the family): translation, real-time
streaming, VAD, speaker diarization, languages other than Russian,
long-form input beyond 25 seconds per utterance (upstream
`transcribe_longform` PyAnnote-VAD chunking is intentionally not
ported). See the family doc for the full runtime contract.
