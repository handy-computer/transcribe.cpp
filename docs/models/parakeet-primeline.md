# Parakeet primeLine (German-tuned)

primeLine's [`primeline/parakeet-primeline`](https://huggingface.co/primeline/parakeet-primeline)
ported to transcribe.cpp. A German fine-tune of NVIDIA's
[`parakeet-tdt-0.6b-v3`](parakeet-tdt-0.6b-v3.md): a 0.6B-parameter
FastConformer encoder with a TDT/RNNT transducer decoder.

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

Licensed CC-BY-4.0. Ported from upstream commit
[`3f1a9bc`](https://huggingface.co/primeline/parakeet-primeline/commit/3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e),
pinned 2026-08-16.

## Download

| Quantization | Download | Size | WER (FLEURS de test) |
| --- | --- | ---: | ---: |
| F32    | [parakeet-primeline-F32.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-F32.gguf)       | 2.51 GB | 6.00% |
| F16    | [parakeet-primeline-F16.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-F16.gguf)       | 1.26 GB | 6.00% |
| Q8_0   | [parakeet-primeline-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q8_0.gguf)     | 740 MB  | 6.00% |
| Q6_K   | [parakeet-primeline-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q6_K.gguf)     | 610 MB  | 5.96% |
| Q5_K_M | [parakeet-primeline-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q5_K_M.gguf) | 549 MB  | 5.99% |
| Q4_K_M | [parakeet-primeline-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-primeline-gguf/resolve/main/parakeet-primeline-Q4_K_M.gguf) | 485 MB  | 5.98% |

WER is measured on the full FLEURS German test split (862 utterances)
with greedy transducer decoding and no external LM. The reference
baseline, primeLine's own NeMo checkpoint over the identical manifest,
is 5.98%. The quant matrix spans 0.04pp with no monotonic degradation;
Q4_K_M scores marginally better than F32 (5.9845% vs 5.9952%), which is
noise, not an improvement. Any preset is safe to ship.

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
