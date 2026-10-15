# Phonon-2

English-only Parakeet TDT variant from
[Fermion Research](https://huggingface.co/FermionResearch/Phonon-2), pinned at
`7e153e4054c0a10db6d47d6fff5c062d4c77c154`. This is a conversion-first native
port: the learned five-value, integer and FP16 records are expanded to an F32
GGUF, then run through the existing Parakeet engine. It does not retain the
publisher's 164 MB download size, packed compute, or published speed claims.

## Convert

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

## Run and validate

```bash
cmake -B build -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON \
  -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build --target transcribe-cli transcribe_parakeet_real_smoke transcribe_decoder_smoke
build/bin/transcribe-cli --backend cpu \
  -m models/phonon-2/phonon-2-F32.gguf samples/jfk.wav
uv run scripts/validate.py all --family parakeet --variant phonon-2 --report
uv run --project scripts/envs/parakeet-phonon scripts/lib/test_phonon.py
TRANSCRIBE_PARAKEET_GGUF="$PWD/models/phonon-2/phonon-2-F32.gguf" \
  ctest --test-dir build -R '^transcribe_(parakeet_real_smoke|decoder_smoke)$' --output-on-failure
```

The numerical reference uses the publisher's independently implemented
`reference_transformers.py` and `fermion_container.py`, both pinned and
SHA256-checked, with Transformers 5.20.0 and Torch 2.10.0 on CPU in F32.
The dumper observes the canonical forward through hooks rather than
reimplementing the encoder. Reference scripts are executed only by this
explicit validation path. Dependency versions are recorded in the environment
lockfile.

`stt.parakeet.encoder.length_masking=transformers` masks the pointwise bias
on invalid subsampler frames. The existing `kestrel` mode retains that bias;
`none` retains legacy NeMo behavior. The setting is semantic metadata, with
no variant-name branch in the native runtime. Global TDT budget behavior is
reused from the existing decoder.

### Full test-clean comparison

The acceptance corpus is the complete LibriSpeech test-clean split (2,620
utterances). The reference runner uses the publisher loader and stock greedy
Transformers generation on CPU F32, one whole clip per call. Its output is
compatible with the repository scorer; it refuses to overwrite an existing
reference report.

```bash
uv run scripts/wer/ingest.py librispeech --split test-clean
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

## Scope and release status

This candidate supports local whole-clip transcription in English, the
existing Parakeet timestamp and batch interfaces, and the standard CPU
backend. It does not implement translation, language detection, streaming,
the publisher CLI's energy-based long-audio segmentation, hotword bias,
CoreML/MLX, or custom packed kernels. Whole-clip attention memory grows with
audio length; no bounded-memory long-form claim is made.

No canonical GGUF has been uploaded. Local full test-clean reference, CPU
serial and Metal serial/batch-8 results are recorded in
`reports/wer/phonon-2.librispeech-test-clean.summary.md`; full CPU batch-8 was
skipped at the user's request. Metal completed transcription and lifecycle
checks, but exceeds the strict CPU-reference tensor tolerances. See the
[porting log](../../reports/porting/parakeet/phonon-2/_porting-log.md) for scope
and evidence. Deployment quantization, backend publication benchmarks and
download roundtrip remain publication work.

## Attribution

Weights: Fermion Research, derived from NVIDIA Parakeet TDT 0.6B v3,
CC-BY-4.0. Changes here expand the publisher's quantized weights, rename
tensors, and combine predictor biases for native inference; they do not
retrain the model. Preserve the upstream
[`NOTICE`](https://huggingface.co/FermionResearch/Phonon-2/blob/7e153e4054c0a10db6d47d6fff5c062d4c77c154/NOTICE)
and weight license with any redistributed artifact. The container format
reference and publisher validation scripts are Apache-2.0, copyright 2026
Fermion Research.
