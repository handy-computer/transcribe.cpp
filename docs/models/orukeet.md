# Orukeet

<!-- catalog:intro -->
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
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
<!-- /catalog -->

<!-- catalog:recipe -->
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
<!-- /catalog -->

<!-- catalog:accuracy -->
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
v3 and `parakeet-ultra` numbers are representative until this variant has
its own profile run.

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
<!-- /catalog -->

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
