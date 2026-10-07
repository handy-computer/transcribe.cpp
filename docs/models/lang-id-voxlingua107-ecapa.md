# VoxLingua107 ECAPA-TDNN (language ID)

<!-- catalog:intro -->
Upstream: [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) at [`0253049`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049).

Spoken language identification over 107 languages: SpeechBrain's
ECAPA-TDNN trained on VoxLingua107. NOT a transcription model: a run
returns the model's language labels ranked by probability, optionally
restricted to a caller-chosen set. Takes 16 kHz mono WAV; scores up to
the last 30 s of a clip.
<!-- /catalog -->

## What it's for

Spoken language identification over 107 languages. This is **not a
transcription model**. It serves the LANGID role
(`include/transcribe/langid.h`, [`docs/langid.md`](../langid.md)): a run
returns the model's language labels ranked by probability, optionally
restricted to a caller-chosen set, plus `allowed_mass`, the unrestricted
probability that set captured.

See SpeechBrain's [model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
for training data and the label list.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`0253049`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049), pinned 2026-10-05. Validated against the SpeechBrain reference at transcribe.cpp commit [`62522202`](https://github.com/handy-computer/transcribe.cpp/tree/62522202) on 2026-10-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |  Size | Top-1 accuracy (FLEURS multilingual) |
| --- | --- | ---: | ---: |
| F32          | [lang-id-voxlingua107-ecapa-F32.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-F32.gguf) | 85 MB | 85.23% |
| F16          | [lang-id-voxlingua107-ecapa-F16.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-F16.gguf) | 45 MB | 85.17% |
| Q8_0         | [lang-id-voxlingua107-ecapa-Q8_0.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-Q8_0.gguf) | 27 MB | 86.27% |
<!-- /catalog -->

<!-- catalog:recipe -->
Top-1 accuracy on FLEURS multilingual (3,000 utterances), scored on cpu. Measured at transcribe.cpp `16d46bc2` on 2026-10-06.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Open-set top-1 accuracy over all 107 labels, the mean over 15 FLEURS
languages, on the first 5 s of each clip without silence trimming.
Agreement with the SpeechBrain reference, per GGUF, is on the
transcribe.cpp model page.
<!-- /catalog -->

## Accuracy

<!-- catalog:agreement -->
| GGUF | Top-1 accuracy (95% CI) | Top-1 agreement with SpeechBrain | Max abs logit difference |
| --- | ---: | ---: | ---: |
| F32  | 85.23% (84.07-86.47) | 12000 / 12000 | 7.4e-05 |
| F16  | 85.17% (84.00-86.43) | 11984 / 12000 | 0.094 |
| Q8_0 | 86.27% (85.13-87.47) | 11704 / 12000 | 2.9 |

Measured at transcribe.cpp `16d46bc2` on 2026-10-06.
<!-- /catalog -->

Agreement counts the scored sweep's top-1 decisions that match the
SpeechBrain reference on the same audio, over the 3 / 5 / 10 s / full crops.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-Q8_0.gguf \
  --allow en,de,fr --top 3 audio.wav
# language: de index=18 p=0.999998
#   ...
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

From the C API, use the LANGID role (`include/transcribe/langid.h`):
`transcribe_langid_session_init`, `transcribe_langid_run`, then
`transcribe_langid_get_result` / `transcribe_langid_get_candidate`. Labels are
the model's own codes (`iw`, `jw`, `tl`, `no`; `he`, `jv`, `fil`, `nb` are
accepted as aliases).

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode), speedup over realtime in parentheses.

| Backend | Sample               |               F32 |                F16 |               Q8_0 |
| ------- | -------------------- | ----------------: | -----------------: | -----------------: |
| Metal   | long-45s-10s (10.0s) | 14.3 ms (699.79×) |  14.1 ms (709.72×) |  14.2 ms (703.23×) |
| Metal   | long-45s-30s (30.0s) | 37.1 ms (808.41×) |  35.6 ms (841.51×) |  36.2 ms (828.04×) |
| CPU     | long-45s-10s (10.0s) | 147.1 ms (67.97×) |  82.5 ms (121.26×) |  74.9 ms (133.58×) |
| CPU     | long-45s-30s (30.0s) | 450.9 ms (66.53×) | 251.5 ms (119.27×) | 229.5 ms (130.70×) |

Apple M4 Max: transcribe.cpp `62522202` on 2026-10-05.
<!-- /catalog -->

Benchmark reproduction (`tools/transcribe-bench` is ASR-only):

```bash
uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py --profile \
  --library build-shared/src/libtranscribe.dylib
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against SpeechBrain on eight
FLEURS clips (`samples/fleurs-*.wav`): every stage tensor gated in
`tests/tolerances/ecapa_tdnn.json` (front end, every encoder block, pooling,
embedding, logits) falls within tolerance, and the top-1 label matches the
reference on every clip.

| Field | Value |
| --- | --- |
| Reference | SpeechBrain 1.1.1, `speechbrain/lang-id-voxlingua107-ecapa` |
| Dump script | `scripts/dump_reference_ecapa_tdnn_speechbrain.py` |
| Manifest | `tests/golden/ecapa_tdnn/lang-id-voxlingua107-ecapa.manifest.json` |
| Command | `uv run scripts/validate.py all --family ecapa_tdnn` |
| Dataset gate | `scripts/langid/compare.py` (FLEURS, every crop) |

## Known Limitations

- **Closed set.** There is no "unknown" or "no speech" label: silence, music
  and out-of-set languages still get a top candidate.
- **No Cantonese label**: Cantonese is classified as `zh`.
- **Minimum 500 ms** of scored audio (`TRANSCRIBE_ERR_INPUT_TOO_SHORT`).
- **Scores at most the last 30 s** by default
  (`transcribe_langid_session_params::max_audio_ms`).
- **Q8_0 weights are widened to F16 at load**, so Q8_0 computes and uses
  memory like F16.

## Reproduction

### Convert

Loads the SpeechBrain checkpoint through SpeechBrain from the Hugging Face
cache and writes F32 only.

```bash
uv run --project scripts/envs/ecapa_tdnn \
  scripts/convert-ecapa_tdnn.py speechbrain/lang-id-voxlingua107-ecapa
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \
  models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-Q8_0.gguf \
  --quant Q8_0
# repeat with F16
```

### Validate

```bash
uv run scripts/validate.py all --family ecapa_tdnn
```

### Accuracy acceptance

One reference sweep, then one C++ sweep per shipped GGUF (shown for F32;
repeat with F16 and Q8_0, passing `--report-only` to `compare.py`, which gates
F32 only).

```bash
uv run --project scripts/envs/ecapa_tdnn scripts/langid/ingest.py fleurs --lang all
uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py --engine speechbrain \
  --model speechbrain/lang-id-voxlingua107-ecapa \
  --manifest samples/langid/fleurs-*.manifest.jsonl --crops 3,5,10,full \
  --out reports/langid/ref-speechbrain-untrimmed.jsonl
uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py --engine cpp \
  --model models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \
  --library build-shared/src/libtranscribe.dylib \
  --manifest samples/langid/fleurs-*.manifest.jsonl --crops 3,5,10,full \
  --out reports/langid/cpp-f32-untrimmed.jsonl
uv run scripts/langid/compare.py reports/langid/ref-speechbrain-untrimmed.jsonl \
  reports/langid/cpp-f32-untrimmed.jsonl \
  --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.agreement.json
uv run scripts/langid/score.py reports/langid/cpp-f32-untrimmed.jsonl \
  --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.score.json
uv run scripts/catalog/ingest_accuracy.py --models lang-id-voxlingua107-ecapa
uv run scripts/catalog/render.py
```
