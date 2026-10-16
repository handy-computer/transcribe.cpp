# Phonon-2

<!-- catalog:intro -->
Upstream: [`FermionResearch/Phonon-2`](https://huggingface.co/FermionResearch/Phonon-2) at [`7e153e4`](https://huggingface.co/FermionResearch/Phonon-2/commit/7e153e4).

Offline English speech-to-text. Fermion Research's Phonon-2 is a
five-value-quantized repack of NVIDIA's parakeet-tdt-0.6b-v3; these GGUFs
expand the packed weights back to dense tensors and run them through
transcribe.cpp's Parakeet engine. Takes a 16 kHz mono WAV and produces a
transcript with optional token-level timestamps. Not a streaming model;
English only, no translation or language detection.

Weights are CC-BY-4.0 (Fermion Research, derived from NVIDIA); preserve
the upstream NOTICE and license with any redistribution.
<!-- /catalog -->

English-only Parakeet TDT variant from
[Fermion Research](https://huggingface.co/FermionResearch/Phonon-2). This is a
conversion-first native port: the learned five-value, integer and FP16 records
are expanded to a dense F32 GGUF, then run through the existing Parakeet
engine. It does not retain the publisher's 164 MB download size, packed
compute, or published speed claims.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`7e153e4`](https://huggingface.co/FermionResearch/Phonon-2/commit/7e153e4), pinned 2026-10-15. Validated against the Transformers (publisher loader) reference at transcribe.cpp commit [`66ad945d`](https://github.com/handy-computer/transcribe.cpp/tree/66ad945d) on 2026-10-15.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [phonon-2-F32.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-F32.gguf) | 2.51 GB | 2.15% |
| F16          | [phonon-2-F16.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-F16.gguf) | 1.26 GB | 2.16% |
| Q8_0         | [phonon-2-Q8_0.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-Q8_0.gguf) |  740 MB | 2.15% |
| Q6_K         | [phonon-2-Q6_K.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-Q6_K.gguf) |  610 MB | 2.15% |
| Q5_K_M       | [phonon-2-Q5_K_M.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-Q5_K_M.gguf) |  549 MB | 2.15% |
| Q4_K_M       | [phonon-2-Q4_K_M.gguf](https://huggingface.co/handy-computer/phonon-2-gguf/resolve/main/phonon-2-Q4_K_M.gguf) |  485 MB | 2.20% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `66ad945d`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy TDT decoding, no external LM. Measured against the publisher's own
Transformers reference loader on the full LibriSpeech test-clean split:
reference 2.16%, this port's F32 2.15%, so the port matches the reference
within rounding. Shipped quants sit within 0.05 of F32.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 5.49% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/phonon-2/phonon-2-Q8_0.gguf \
  samples/jfk.wav
```

Audio has to be 16 kHz mono WAV. Convert anything else first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

English only; `--language en` is accepted but not needed.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  54 ms (204.37×) |  56 ms (197.85×) |
| Metal   | dots (35.3s) | 159 ms (222.32×) | 161 ms (219.69×) |
| CPU     | jfk (11.0s)  |  208 ms (53.00×) |  264 ms (41.66×) |
| CPU     | dots (35.3s) |  715 ms (49.41×) |  899 ms (39.32×) |

Apple M4 Max: transcribe.cpp `66ad945d` on 2026-10-16.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 462 ms (23.82×) | 462 ms (23.80×) |
| Vulkan  | dots (35.3s) | 1.36 s (25.95×) | 1.38 s (25.64×) |
| CPU     | jfk (11.0s)  | 718 ms (15.32×) | 676 ms (16.27×) |
| CPU     | dots (35.3s) | 2.87 s (12.29×) | 2.41 s (14.64×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `66ad945d` on 2026-10-16.
<!-- /catalog -->

Rerun the benchmark yourself:

```bash
uv run scripts/bench/run.py --profile --models phonon-2
```

## How we checked it

Compared tensor by tensor against the publisher's independently implemented,
hash-checked Transformers loader (`reference_transformers.py` at the pinned
revision, Transformers 5.20.0 / Torch 2.10.0, CPU F32). All 18 gate tensors
are within tolerance and the JFK transcript matches exactly. Batches of 2, 3,
4 and 6 give the same text as one at a time (`scripts/batch_parity.py`), and
the full test-clean sweep at batch size 8 lands at the reference WER.

| Field | Value |
| --- | --- |
| Reference | Publisher Transformers loader, `FermionResearch/Phonon-2` `7e153e4` |
| Dump script | `scripts/dump_reference_parakeet_phonon.py` |
| Manifest | `tests/golden/parakeet/phonon-2.manifest.json` |
| Tolerances | `tests/tolerances/phonon-2.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant phonon-2` |

The numerical reference uses the publisher's `reference_transformers.py` and
`fermion_container.py`, both pinned and SHA256-checked. The dumper observes
the canonical forward through hooks rather than reimplementing the encoder.
Reference scripts are executed only by this explicit validation path.
Dependency versions are recorded in the `scripts/envs/parakeet-phonon`
lockfile.

### Full test-clean comparison

The acceptance corpus is the complete LibriSpeech test-clean split (2,620
utterances). The reference runner uses the publisher loader and stock greedy
Transformers generation on CPU F32, one whole clip per call. Its output is
compatible with the repository scorer; it refuses to overwrite an existing
reference report.

```bash
uv run scripts/wer/ingest.py librispeech
uv run --project scripts/envs/parakeet-phonon \
  scripts/wer/run_reference_parakeet_phonon.py \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/phonon-2-REF.librispeech-test-clean.jsonl
uv run scripts/wer/run.py \
  --model models/phonon-2/phonon-2-F32.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --language en --backend cpu --batch-size 1 \
  --out reports/wer/phonon-2-F32.librispeech-test-clean.cpu.b1.jsonl
uv run scripts/wer/score.py reports/wer/phonon-2-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/phonon-2-F32.librispeech-test-clean.cpu.b1.jsonl
```

For batch validation, repeat the native run with `--batch-size 8
--sort-by-length` and a distinct output path. Metal uses a separate build:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DTRANSCRIBE_METAL=ON -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON \
  -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build-metal
```

Pass `--cli build-metal/bin/transcribe-cli --backend metal` to `run.py`,
using distinct `.metal.b1.jsonl` and `.metal.b8.jsonl` output paths. GPU
device access is required. F32 GGUF storage does not guarantee F32 Metal
arithmetic: the vendored Metal matrix kernel uses half-precision operands.
Keep transcript/WER acceptance separate from the strict CPU-reference tensor
gate, and retain numerical failures rather than widening tolerances.

## Build it yourself

```bash
uv run --project scripts/envs/parakeet-phonon scripts/convert-parakeet.py \
  FermionResearch/Phonon-2 models/phonon-2/phonon-2-F32.gguf \
  --repo-id FermionResearch/Phonon-2 \
  --revision 7e153e4054c0a10db6d47d6fff5c062d4c77c154
```

The converter checks the archive and container SHA256, extracts only regular
files with expected names, and rejects malformed/duplicate records and
non-finite inference weights. An extracted local bundle containing
`model.fermion` may be passed instead of the repo ID; its container hash is
checked too. No downloaded Python code is executed during conversion.

Architecture, tokenizer and processor files come from
`nvidia/parakeet-tdt-0.6b-v3@541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
The teacher's weights are never downloaded. The F32 GGUF has 697 tensors;
the source's 723 records also include unused BatchNorm counters and separate
LSTM biases, which the normal converter combines.

Then make the smaller files and validate:

```bash
uv run scripts/quantize-all.py models/phonon-2/phonon-2-F32.gguf

cmake -B build -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON \
  -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build --target transcribe-cli transcribe_parakeet_real_smoke transcribe_decoder_smoke

uv run scripts/validate.py all --family parakeet --variant phonon-2 --report
uv run --project scripts/envs/parakeet-phonon scripts/lib/test_phonon.py
TRANSCRIBE_PARAKEET_GGUF="$PWD/models/phonon-2/phonon-2-F32.gguf" \
  ctest --test-dir build -R '^transcribe_(parakeet_real_smoke|decoder_smoke)$' --output-on-failure
```

`stt.parakeet.encoder.length_masking=transformers` masks the pointwise bias
on invalid subsampler frames. The existing `kestrel` mode retains that bias;
`none` retains legacy NeMo behavior. The setting is semantic metadata, with
no variant-name branch in the native runtime. Global TDT budget behavior is
reused from the existing decoder.

## Limits

- English only; no translation, no language detection, no streaming.
- Timestamps are per token, not per word.
- No publisher-CLI energy-based long-audio segmentation, hotword bias,
  CoreML/MLX, or custom packed kernels.
- Whole-clip attention memory grows with audio length; long files go through
  in one pass. Nothing cuts them up for you.
- Metal transcribes correctly (WER parity), but exceeds the strict
  CPU-reference tensor tolerances; CPU is the numerically gated backend.

## Attribution

Weights: Fermion Research, derived from NVIDIA Parakeet TDT 0.6B v3,
CC-BY-4.0. Changes here expand the publisher's quantized weights, rename
tensors, and combine predictor biases for native inference; they do not
retrain the model. Preserve the upstream
[`NOTICE`](https://huggingface.co/FermionResearch/Phonon-2/blob/7e153e4054c0a10db6d47d6fff5c062d4c77c154/NOTICE)
and weight license with any redistributed artifact. The container format
reference and publisher validation scripts are Apache-2.0, copyright 2026
Fermion Research.

See the [porting log](../../reports/porting/parakeet/phonon-2/_porting-log.md)
for the detailed evidence trail.
