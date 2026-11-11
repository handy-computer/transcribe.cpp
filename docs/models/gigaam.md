# GigaAM-v3

ai-sage's [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
family ported to transcribe.cpp. Russian-only ASR built around a shared
16-layer Conformer encoder (768-d, 16 heads, rotary positional
embeddings), paired with one of two decoder heads (RNN-T or CTC) and
trained either end-to-end on cased+punctuated text or on
lowercased no-punctuation text with a 33-entry character vocabulary.
The four shipped variants are the four cells of that 2×2.

A fifth variant, `gigaam-multilingual-large-ctc`, comes from
[`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual),
transcribes five languages, and is not published yet; see
[GigaAM Multilingual Large CTC](#gigaam-multilingual-large-ctc).

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/gigaam.md`](../porting/families/gigaam.md).

## Choosing a variant

- **Cased + punctuated Russian, best accuracy.** `gigaam-v3-e2e-rnnt`
  — RNN-T head trained end-to-end with a 1024-piece SentencePiece
  vocab.
- **Cased + punctuated Russian, fastest decode.** `gigaam-v3-e2e-ctc`
  — same Conformer encoder and training data as `e2e-rnnt`, but a more
  compact 256-piece SentencePiece vocab, with single-pass CTC alignment
  instead of the transducer loop.
- **Lowercased no-punct (charwise output).** `gigaam-v3-rnnt` and
  `gigaam-v3-ctc` — 33-entry character vocabulary (space + а–я),
  output is normalized for downstream ASR scoring pipelines that expect
  this convention. Higher raw WER than the e2e variants because errors
  on case/punctuation are no longer absorbed by tokenization.
- **Russian, English, Kazakh, Kyrgyz, or Uzbek.**
  `gigaam-multilingual-large-ctc` — lowercase charwise output with no
  punctuation. No pre-built GGUF; convert it yourself, see
  [GigaAM Multilingual Large CTC](#gigaam-multilingual-large-ctc).
- The encoder is structurally identical across the four v3 variants but
  the weights are per-variant fine-tuned — you cannot swap heads at
  runtime.

## All variants

WER is on FLEURS Russian (`fleurs-ru`) for the **Q8_0** preset,
measured by transcribe.cpp's WER pipeline. See each per-variant doc
for the full quant matrix.

<!-- catalog:family variants=gigaam-v3-e2e-rnnt,gigaam-v3-e2e-ctc,gigaam-v3-rnnt,gigaam-v3-ctc -->
| Variant              | Params | Languages | Q8_0 size | Benchmark       |  Q8_0 | Capabilities     | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `gigaam-v3-e2e-rnnt` |   223M | ru        |    274 MB | FLEURS ru (WER) | 5.35% | token timestamps | [gigaam-v3-e2e-rnnt.md](gigaam-v3-e2e-rnnt.md) |
| `gigaam-v3-e2e-ctc`  |   221M | ru        |    272 MB | FLEURS ru (WER) | 5.53% | token timestamps | [gigaam-v3-e2e-ctc.md](gigaam-v3-e2e-ctc.md) |
| `gigaam-v3-rnnt`     |   222M | ru        |    273 MB | FLEURS ru (WER) | 8.07% | token timestamps | [gigaam-v3-rnnt.md](gigaam-v3-rnnt.md) |
| `gigaam-v3-ctc`      |   221M | ru        |    272 MB | FLEURS ru (WER) | 8.42% | token timestamps | [gigaam-v3-ctc.md](gigaam-v3-ctc.md) |
<!-- /catalog -->

Pre-built GGUFs for every v3 variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## GigaAM Multilingual Large CTC

`gigaam-multilingual-large-ctc` is the `large_ctc` branch of
[`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual).
It transcribes Russian, English, Kazakh, Kyrgyz, and Uzbek into lowercase
text with no punctuation. The encoder has 24 layers at d_model=1024, larger
than the v3 encoder, and the CTC head decodes a 70-character vocabulary.

The variant is not in the table above: no GGUF is published, and its WER
and benchmarks have not been measured. Conversion and numerical validation
at F32 pass; the [family doc](../porting/families/gigaam.md#gigaam-multilingual-large-ctc)
has the validation results.

Convert, quantize, and run:

```bash
uv run --project scripts/envs/gigaam \
  scripts/convert-gigaam.py ai-sage/GigaAM-Multilingual \
  --repo-id gigaam-multilingual-large-ctc

build/bin/transcribe-quantize \
  models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-F32.gguf \
  models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-Q8_0.gguf \
  --quant Q8_0

build/bin/transcribe-cli \
  -m models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-Q8_0.gguf \
  samples/fleurs-en.wav
```

The converter downloads a 2.3 GB snapshot at the commit pinned in
`scripts/lib/gigaam_hf.py` and writes a 2.3 GB (2341378624 bytes) F32 GGUF.
It accepts only the repo id `ai-sage/GigaAM-Multilingual`, not a local
directory.

The model takes no language input. `--language` accepts `ru`, `en`, `kk`,
`ky`, and `uz` and leaves the transcript unchanged; any other code fails
with `run: unsupported language`. On the clips checked with Metal, Q5_K_M
and larger quantizations match the reference transcript; Q4_K_M changes one
letter on one Kazakh clip.

## Input limits

GigaAM is trained for utterances up to about **25 seconds**. Longer audio is
encoded in overlapping windows of at most 25 s, and the encoder frames are
stitched into one sequence for a single greedy decode. There is no practical
length limit, and the encoder workspace does not grow with input length. Audio
up to 25 s is encoded in one pass. See the [input-length contract](../input-limits.md).

The runtime applies the same 25 s window to `gigaam-multilingual-large-ctc`.
The port records no training length for this variant, and no clip longer
than 25 s has been compared with the reference.

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
streaming, VAD, speaker diarization, languages other than Russian.
Long-form input is windowed internally rather than through upstream's
`transcribe_longform` PyAnnote-VAD chunking. See the family doc for the full
runtime contract.

`gigaam-multilingual-large-ctc` has the same capabilities and gaps, except
that it transcribes the five languages listed in
[its section](#gigaam-multilingual-large-ctc).
