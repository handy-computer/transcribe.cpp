# Parakeet primeLine (German-tuned)

<!-- catalog:intro -->
Upstream: [`primeline/parakeet-primeline`](https://huggingface.co/primeline/parakeet-primeline) at [`3f1a9bc`](https://huggingface.co/primeline/parakeet-primeline/commit/3f1a9bc).

primeLine's German fine-tune of NVIDIA's parakeet-tdt-0.6b-v3. A
FastConformer encoder with a TDT/RNNT transducer decoder, taking 16 kHz
mono WAV and producing a punctuated, cased transcript with optional
token-level timestamps. Tuned for German, but the fine-tune did not
collapse the base model's multilingual ability: it still transcribes the
other 24 v3 languages with correct per-language casing and punctuation.
Not a streaming model and does not translate.
<!-- /catalog -->

## What it's for

Offline German speech-to-text. The model takes a 16 kHz mono WAV and
produces a punctuated, cased transcript with optional token-level
timestamps. It is not a streaming model and does not translate.

The fine-tune targets German but did not collapse the base model's
multilingual coverage. On 150-utterance FLEURS subsets it scores 4.24%
(English), 3.24% (Spanish), 7.89% (Russian) and 8.06% (Ukrainian), with
correct per-language casing and punctuation, so all 25 v3 languages
remain usable. Pick this variant when German is your primary workload
and `parakeet-tdt-0.6b-v3` when it is not.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`3f1a9bc`](https://huggingface.co/primeline/parakeet-primeline/commit/3f1a9bc), pinned 2026-08-16. Validated against the NeMo reference at transcribe.cpp commit [`856d7c1`](https://github.com/handy-computer/transcribe.cpp/tree/856d7c1) on 2026-08-16.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (FLEURS de) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-primeline-F32.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-F32.gguf) | 2.51 GB | 6.00% |
| F16          | [parakeet-primeline-F16.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-F16.gguf) | 1.26 GB | 6.00% |
| Q8_0         | [parakeet-primeline-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q8_0.gguf) |  740 MB | 5.98% |
| Q6_K         | [parakeet-primeline-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q6_K.gguf) |  610 MB | 5.96% |
| Q5_K_M       | [parakeet-primeline-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q5_K_M.gguf) |  549 MB | 5.99% |
| Q4_K_M       | [parakeet-primeline-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q4_K_M.gguf) |  485 MB | 5.98% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full FLEURS de split (862 utterances), batch sizes 1 and 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, no external LM.

primeLine's published figures (2.95% average over Tuda-De, Multilingual
LibriSpeech, and Common Voice 19.0) are on different corpora and are not
comparable to these numbers. As a like-for-like baseline we ran primeLine's
own NeMo checkpoint over the identical manifest: **5.98% WER**. The C++
numbers above match that reference within bootstrap-CI noise, and the quant
spread is 0.04pp end to end with no monotonic degradation.

Orthography note: this checkpoint writes Swiss `ss` forms (`grosse`,
`heisst`) almost everywhere instead of `ß`, which appears just 5 times
across the 862 hypotheses. The upstream SentencePiece vocabulary carries
only 4 pieces containing `ß` against 58 containing `ss`, so this is a
property of the v3-family tokenizer, not of the port — the NeMo reference
produces the same spellings on the same utterances. FLEURS references use
`ß` throughout, which costs roughly 1.05pp: folding `ß`→`ss` on both sides
gives 4.92% for the reference and 4.94% for F32.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    | 16.33% |
| cs       | WER    | 15.05% |
| da       | WER    | 20.52% |
| el       | WER    | 34.76% |
| en       | WER    |  4.82% |
| es       | WER    |  3.85% |
| et       | WER    | 17.17% |
| fi       | WER    | 13.39% |
| fr       | WER    |  6.35% |
| hr       | WER    | 13.68% |
| hu       | WER    | 17.52% |
| it       | WER    |  3.17% |
| lt       | WER    | 23.08% |
| lv       | WER    | 28.83% |
| mt       | WER    | 24.74% |
| nl       | WER    |  8.49% |
| pl       | WER    |  8.19% |
| pt       | WER    |  5.17% |
| ro       | WER    | 13.80% |
| ru       | WER    |  7.81% |
| sk       | WER    | 12.36% |
| sl       | WER    | 51.07% |
| sv       | WER    | 16.42% |
| uk       | WER    |  8.11% |

**LibriSpeech test-clean**

| Language | Metric |   F32 |   F16 |  Q8_0 |  Q6_K | Q5_K_M | Q4_K_M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en       | WER    | 2.20% | 2.19% | 2.20% | 2.19% |  2.20% |  2.26% |
<!-- /catalog -->

primeLine's published 2.95% average is over Tuda-De, Multilingual
LibriSpeech, and Common Voice 19.0. Those corpora are not in this repo's
WER pipeline and the number is not comparable to the table above.

## Orthography: `ß` vs `ss`

This checkpoint almost never writes `ß`. It appears 5 times across the
862 hypotheses (`verstieß`, `stieß`, `sechsunddreißig`,
`siebenunddreißig`) against 214 occurrences in the references;
everywhere else it emits Swiss `ss` spellings: `grosse`, `heisst`,
`einschliesslich`, `dass`.

This is not a port defect. The NeMo reference produces the same
spellings on the same utterances, with the same count. The cause is the
upstream SentencePiece vocabulary, which carries only 4 pieces
containing `ß` (`iß`, `▁weiß`, `ßen`, `ß`) against 58 containing `ss`,
so `ss` spellings fall out of common merged pieces while `ß` requires a
rare standalone piece. It is a likelihood preference rather than an
impossibility: `▁weiß` exists as a piece, yet the model still writes
`weiss`. That tokenizer is byte-identical to `parakeet-tdt-0.6b-v3`'s,
so the behaviour is inherited by the whole v3 lineage.

Against FLEURS references this costs roughly 1.05pp: folding `ß`→`ss` on
both sides gives 4.92% for the reference and 4.94% for F32. The table
above reports the unfolded numbers.

If your downstream consumer needs standard German orthography, apply a
lexicon-based normalizer. Do not blanket-rewrite `ss`→`ß`: the mapping
is not one-to-one (`Masse` vs `Maße`, `Busse` vs `Buße`, and `dass` is
correct as written).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-primeline/parakeet-primeline-Q8_0.gguf \
  audio-de.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Not separately benchmarked. The checkpoint is a weights-only fine-tune
whose encoder, decoder, joint, and preprocessor configuration is
identical to `parakeet-tdt-0.6b-v3`, so throughput is unchanged; see
[that variant's numbers](parakeet-tdt-0.6b-v3.md#performance). For
reference, the WER run above sustained 5.1 utterances/s on an Apple M4
via Metal.

## Numerical Validation

This variant is validated end-to-end against NeMo rather than
tensor-by-tensor. The `.nemo` checkpoint's `model_config.yaml` is
identical to `parakeet-tdt-0.6b-v3`'s across every structural section
(`encoder`, `decoder`, `joint`, `decoding`, `preprocessor`, `tokenizer`,
`model_defaults`, `loss`), differing only in training-time keys
(`train_ds`, `validation_ds`, `test_ds`, `optim`, `spec_augment`,
`nemo_version`). The SentencePiece model and vocabulary are
byte-identical (sha256 `eacec2b0…` and `41130ff4…`). It is a weights-only
fine-tune, so the graph is already covered by v3's tensor manifest and
the C++ implementation needed no changes.

| Field | Value |
| --- | --- |
| Reference | NeMo, `primeline/parakeet-primeline` |
| Reference runner | `scripts/wer/run_reference_parakeet_nemo.py` |
| Manifest | `samples/wer/fleurs-de.manifest.jsonl` (862 utterances) |
| Reference WER | 5.9792% |
| C++ F32 WER | 5.9952% |
| Report | `reports/wer/parakeet-primeline.fleurs-de.summary.md` |

The +0.016pp gap resolves to 38 of 862 hypotheses differing, all
single-token near-ties in greedy transducer decoding: proper nouns
(`Schapan`/`Chapan`, `Danielle`/`Daniel`), one comma, one `zwanzig%`
spacing. At 18715 reference words the entire gap is 3 words.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py \
    "$(huggingface-cli download primeline/parakeet-primeline 2_95_WER.nemo)" \
    --repo-id primeline/parakeet-primeline
```

### Quantize

```bash
uv run scripts/quantize-all.py \
  models/parakeet-primeline/parakeet-primeline-F32.gguf
```

### WER

```bash
uv run scripts/wer/ingest.py fleurs --lang de

# reference arm
uv run --project scripts/envs/parakeet \
  scripts/wer/run_reference_parakeet_nemo.py \
    --model /path/to/2_95_WER.nemo \
    --manifest samples/wer/fleurs-de.manifest.jsonl \
    --out reports/wer/parakeet-primeline-REF.fleurs-de.jsonl \
    --batch-size 8

# transcribe.cpp arm
uv run scripts/wer/run.py \
  --model models/parakeet-primeline/parakeet-primeline-F32.gguf \
  --manifest samples/wer/fleurs-de.manifest.jsonl

# score both; --language de is required, see below
uv run scripts/wer/score.py <report>.jsonl --language de
```

`--language de` is not optional. Without it `score.py` falls back to the
`EnglishTextNormalizer`, which applies English contraction and
number-word rules to German text and reports a misleadingly low 4.87%.
