# Orukeet

<!-- catalog:intro -->
Upstream: [`oruk/orukeet`](https://huggingface.co/oruk/orukeet) at [`b59c13a`](https://huggingface.co/oruk/orukeet/commit/b59c13a).

Oruk AI's fine-tune of NVIDIA's parakeet-tdt-0.6b-v3. Offline multilingual
speech-to-text over the same 25 European languages. A FastConformer encoder
with a TDT transducer decoder, in which half of the encoder's temporal
depthwise filters (12,288 nine-tap kernels) were replaced by frozen fitted
Gabor functions before the rest of the network was retrained around them.
The frozen taps are stored as ordinary convolution weights, so the runtime
is exactly v3's. Takes 16 kHz mono WAV and produces a punctuated, cased
transcript with optional token-level timestamps. A drop-in replacement for
v3: same size, languages and API. Not a streaming model and does not
translate. Weights are CC-BY-SA-4.0.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text over the same 25 European languages as
`parakeet-tdt-0.6b-v3`. Oruk AI fine-tuned v3 without changing its
architecture, tokenizer or frontend, so it drops into any v3 setup by
swapping the GGUF: same size, same languages, same API, same speed.

The twist is inside the encoder. Each of the 24 FastConformer blocks has
1,024 nine-tap temporal depthwise filters; Oruk fitted a Gabor function
(a Gaussian-windowed cosine) to every one of them and froze the half that
fit best, 12,288 kernels in all, then retrained the rest of the network
around them. The frozen taps are stored as ordinary convolution weights, so
the runtime has nothing Gabor-specific in it.

Not a streaming model and does not translate. See Oruk's
[model card](https://huggingface.co/oruk/orukeet) and
[technical report](https://huggingface.co/oruk/orukeet/blob/main/orukeet-technical-report.pdf)
for the training recipe and their own evaluation.

One caveat on the upstream numbers: the final adaptation pass and
checkpoint selection used LibriSpeech test-other, so treat Oruk's test-other
figure as not held out. test-clean, which we score below, was not used for
training.

<!-- catalog:pin -->
Licensed CC-BY-SA-4.0. Ported from upstream commit [`b59c13a`](https://huggingface.co/oruk/orukeet/commit/b59c13a), pinned 2026-10-06. Validated against the NeMo reference at transcribe.cpp commit [`65856fac`](https://github.com/handy-computer/transcribe.cpp/tree/65856fac) on 2026-10-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [orukeet-F32.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-F32.gguf) | 2.51 GB | 1.86% |
| F16          | [orukeet-F16.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-F16.gguf) | 1.26 GB | 1.86% |
| Q8_0         | [orukeet-Q8_0.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q8_0.gguf) |  740 MB | 1.86% |
| Q6_K         | [orukeet-Q6_K.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q6_K.gguf) |  610 MB | 1.86% |
| Q5_K_M       | [orukeet-Q5_K_M.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q5_K_M.gguf) |  549 MB | 1.83% |
| Q4_K_M       | [orukeet-Q4_K_M.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q4_K_M.gguf) |  485 MB | 1.95% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `37273409`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, no external LM. The reference is NeMo
loading Oruk's published `.nemo` archive, the same pipeline behind the
numbers on their card. On this manifest NeMo scores 1.87%.

Oruk's card reports 1.46% on the same split with their own normalizer,
against 1.53% for v3 on that pipeline. Their final adaptation and
checkpoint selection used LibriSpeech test-other, so test-other is not a
held-out split for this model; test-clean was not used for training.

Against our own sweeps of the sibling checkpoints at Q8_0: lower WER
than parakeet-tdt-0.6b-v3 on test-clean and on 20 of the 25 FLEURS
languages (the exceptions are German, French, Spanish, Italian and
Russian, by 0.1 to 0.9 points), with the largest gains on the Baltic,
Finno-Ugric and South Slavic languages. parakeet-ultra scores lower than
Orukeet on test-clean and on 23 of 25 FLEURS languages (Orukeet leads on
Czech and Croatian).
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    | 11.14% |
| cs       | WER    | 10.22% |
| da       | WER    | 16.58% |
| de       | WER    |  5.64% |
| el       | WER    | 33.12% |
| en       | WER    |  4.43% |
| es       | WER    |  3.75% |
| et       | WER    | 15.16% |
| fi       | WER    | 11.88% |
| fr       | WER    |  6.17% |
| hr       | WER    | 11.26% |
| hu       | WER    | 13.53% |
| it       | WER    |  3.11% |
| lt       | WER    | 17.72% |
| lv       | WER    | 19.53% |
| mt       | WER    | 17.18% |
| nl       | WER    |  6.93% |
| pl       | WER    |  7.35% |
| pt       | WER    |  4.56% |
| ro       | WER    | 10.86% |
| ru       | WER    |  6.70% |
| sk       | WER    |  8.99% |
| sl       | WER    | 24.00% |
| sv       | WER    | 13.26% |
| uk       | WER    |  6.53% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/orukeet/orukeet-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

Pass `--language de` (or any of the 25 codes) to pin the language; without
a hint the model detects it.

## Performance

Same encoder, decoder and tensor shapes as `parakeet-tdt-0.6b-v3`, so the
[v3](parakeet-tdt-0.6b-v3.md) and [`parakeet-ultra`](parakeet-ultra.md)
tables are representative. This variant's own profile run on the two
publication rigs is pending; the tables land here when it has run.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models orukeet
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo loading the
published `.nemo` archive, using the parakeet family's finalized tolerances
unchanged: the checkpoint is a weights-only fine-tune, so every tensor in the
v3 contract applies as-is. All 18 contract tensors pass on `jfk` and the
transcript matches NeMo verbatim. Batched decoding is byte-identical to
single-stream at batch sizes 2, 4 and 8.

| Field | Value |
| --- | --- |
| Reference | NeMo (`nemo_toolkit[asr]`), `oruk/orukeet` `orukeet-v0.1.0.nemo` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/orukeet.manifest.json` |
| Tolerances | `tests/tolerances/parakeet.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant orukeet --model models/orukeet/orukeet-v0.1.0.nemo` |

## Reproduction

### Convert

Oruk publishes a NeMo `.nemo` archive alongside Transformers, ONNX, Core ML
and native exports. The converter reads the archive directly (no NeMo
install needed for this step) with the `orukeet` profile.

```bash
hf download oruk/orukeet orukeet-v0.1.0.nemo --local-dir models/orukeet
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py models/orukeet/orukeet-v0.1.0.nemo \
    models/orukeet/orukeet-F32.gguf --repo-id oruk/orukeet
```

Oruk also ships a `transcribe-cpp/orukeet-Q8_0.gguf` in their repo, built
with this same converter and profile at an earlier commit. It loads and
transcribes on current builds; the matrix here is regenerated from the F32
so every quant shares one provenance.

### Quantize

```bash
uv run scripts/quantize-all.py models/orukeet/orukeet-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant orukeet \
  --model models/orukeet/orukeet-v0.1.0.nemo
```

### WER

```bash
uv run scripts/wer/ingest.py librispeech

# reference arm (NeMo); `org/repo:file.nemo` pins the archive inside the HF repo
uv run --project scripts/envs/parakeet \
  scripts/wer/run_reference_parakeet_nemo.py \
    --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
    --model oruk/orukeet:orukeet-v0.1.0.nemo \
    --out reports/wer/orukeet-REF.librispeech-test-clean.jsonl

# transcribe.cpp arm
uv run scripts/wer/run.py \
  --model models/orukeet/orukeet-F32.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/orukeet-F32.librispeech-test-clean.b1.jsonl

uv run scripts/wer/score.py <report>.jsonl
```

## Known Limitations

- Offline only. No streaming decode; the publisher's "live" mode is window
  re-decoding, not a cache-aware encoder.
- No translation.
- Token-level timestamps only, as for every TDT variant in the family.
- Weights are CC-BY-SA-4.0 (ShareAlike), unlike the CC-BY-4.0 of the rest of
  the parakeet family. Derivatives of the weights inherit the licence.
- Long audio runs through the family's unbounded single-pass path; there is
  no voice-activity segmenter (that is `parakeet-ultra`'s addition, not
  this model's).
