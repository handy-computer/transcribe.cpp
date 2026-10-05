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

Spoken language identification: which of 107 languages is this clip in?
This is **not a transcription model**. It serves the LANGID role
(`include/transcribe/langid.h`): a run returns the model's language labels
ranked by probability, optionally restricted to the languages your user has
enabled, plus `allowed_mass`, the signal that the speech is outside that set.
Use it to pick the `language` for an ASR model, or to route audio.

How to get good answers out of it (trim silence, pass the allowed set, decide
at 5 s, re-decide at 10 s) is in [`docs/langid.md`](../langid.md). See
SpeechBrain's [model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
for training data and the label list.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`0253049`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049), pinned 2026-10-05. Validated against the SpeechBrain reference at transcribe.cpp commit [`62522202`](https://github.com/handy-computer/transcribe.cpp/tree/62522202) on 2026-10-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |  Size | Top-1 accuracy (FLEURS multilingual) |
| --- | --- | ---: | ---: |
| F32          | [lang-id-voxlingua107-ecapa-F32.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-F32.gguf) | 85 MB | 85.20% |
| F16          | [lang-id-voxlingua107-ecapa-F16.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-F16.gguf) | 45 MB | 85.20% |
| Q8_0         | [lang-id-voxlingua107-ecapa-Q8_0.gguf](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-Q8_0.gguf) | 27 MB | 86.30% |
<!-- /catalog -->

<!-- catalog:recipe -->
Top-1 accuracy on FLEURS multilingual (3,000 utterances), scored on cpu. Measured at transcribe.cpp `62522202` on 2026-10-05.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Open-set top-1 accuracy over all 107 labels, mean of 15 FLEURS languages
(200 test utterances each), on the first 5 s of each clip without silence
trimming. Accuracy is higher with an allowed set and with trimmed speech;
see docs/langid.md. The C++ F32 port makes the same decision as the
SpeechBrain reference on all 12000 FLEURS decisions (3 / 5 / 10 s / full).
<!-- /catalog -->

Only near-reference tiers ship (F32 / F16 / Q8_0): the decision is an argmax
over 107 logits, and the k-quant tiers would save a few MB at most.

## Accuracy

FLEURS `test`, 15 languages x 200 utterances, crops from the start of each
clip (no silence trimming), C++ F32 on CPU. Top-1 accuracy, %:

| decision space | 3 s | 5 s | 10 s | full |
|---|---|---|---|---|
| open set (all 107 labels) | 67.0 | 85.2 | 91.1 | 91.4 |
| a 40-language dictation list | 77.4 | 92.3 | 96.6 | 96.6 |
| en+ru | 98.8 | 100.0 | 100.0 | 100.0 |
| en+fr+de+es | 88.6 | 94.1 | 98.9 | 99.4 |
| cs+sk | 83.0 | 92.2 | 96.8 | 96.5 |
| id+ms | 87.8 | 89.5 | 89.5 | 88.8 |

Per-language tables, confidence / coverage and confusions:
`uv run scripts/langid/score.py <run>.jsonl --md <run>.md`.

Agreement with the SpeechBrain reference (top-1, the same 12000 decisions:
3000 clips x 3 / 5 / 10 s / full):

| GGUF | agreement | max abs logit difference | open set 3 / 5 / 10 s / full |
|---|---|---|---|
| F32 | 12000 / 12000 | 6.5e-05 | 67.0 / 85.2 / 91.1 / 91.4 |
| F16 | 11991 / 12000 | 0.098 | 67.0 / 85.2 / 91.1 / 91.5 |
| Q8_0 | 11703 / 12000 | 2.8 | 67.2 / 86.3 / 92.0 / 92.3 |

F16's nine flips are all near-ties (top-two margin under 0.025 logit). Q8_0
is a download format: transcribe.cpp widens its weights to F16 at load, so it
computes like F16 (45 MB resident) and its accuracy matches F32 within the
confidence interval. Its flips are the 8-bit weights alone, on decisions F32
also finds uncertain. Computed in Q8_0, as langid.cpp does, ggml also rounds
the activations to 8 bits; that moves 6.9% of decisions (93.1% agreement,
65.4 / 84.3 / 90.9 / 91.4).

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
accepted as aliases). Match them against your ASR model's
`transcribe_capabilities::languages` yourself.

## Performance

### Apple M4 Max

Compute latency (mel + encode), median of 20 warm runs, transcribe.cpp
`62522202`, `scripts/langid/bench.py`. CPU uses the library default thread
count (8).

| Backend | Audio | F32 | F16 | Q8_0 |
|---|---|---:|---:|---:|
| Metal | 3 s | 6.6 ms | 6.4 ms | 6.1 ms |
| Metal | 5 s | 9.4 ms | 9.5 ms | 9.3 ms |
| Metal | 10 s | 14.3 ms | 14.1 ms | 14.2 ms |
| Metal | 30 s | 37.1 ms | 35.7 ms | 36.2 ms |
| CPU | 3 s | 40.1 ms | 25.1 ms | 22.9 ms |
| CPU | 5 s | 70.1 ms | 41.4 ms | 37.7 ms |
| CPU | 10 s | 147.1 ms | 82.5 ms | 74.9 ms |
| CPU | 30 s | 450.9 ms | 251.5 ms | 229.5 ms |

`AUTO` resolves like every other family (the GPU when there is one). Metal is
5-10x faster here; CPU is still well under real time and avoids contending
with an ASR model on the GPU, so load with `TRANSCRIBE_BACKEND_CPU` when that
matters. Not yet measured on the AMD Ryzen 7 PRO 4750U publication rig, or on
Vulkan / CUDA.

```bash
uv run --project scripts/envs/ecapa_tdnn scripts/langid/bench.py \
  --library build-shared/src/libtranscribe.dylib \
  --gguf models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-{F32,F16,Q8_0}.gguf \
  --backends cpu,metal --durations 3,5,10,30
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against SpeechBrain 1.1.1 on
eight committed FLEURS clips (`samples/fleurs-*.wav`): 17 stage tensors
(front end, every encoder block, pooling, embedding, logits) within family
tolerance, and the top-1 label equal to the reference on every clip. The
dataset gate is the FLEURS decision parity above.

| Field | Value |
| --- | --- |
| Reference | SpeechBrain 1.1.1, `speechbrain/lang-id-voxlingua107-ecapa` |
| Dump script | `scripts/dump_reference_ecapa_tdnn_speechbrain.py` |
| Manifest | `tests/golden/ecapa_tdnn/lang-id-voxlingua107-ecapa.manifest.json` |
| Command | `uv run scripts/validate.py all --family ecapa_tdnn` |
| Dataset gate | `scripts/langid/compare.py` (FLEURS, 12000 decisions) |

## Known Limitations

- **Closed set.** There is no "unknown" or "no speech" label: silence, music
  and out-of-set languages still get a confident-looking top candidate. Gate
  with VAD and watch `allowed_mass`.
- **Confusable pairs.** `ru`/`be`, `no`/`nn`, `id`/`jw`/`ms`, `cs`/`sk`; `id`
  vs `ms` stays near 89% even restricted to the pair.
- **No Cantonese label**: Cantonese is classified as `zh`.
- **Minimum 500 ms** of scored audio (`TRANSCRIBE_ERR_INPUT_TOO_SHORT`);
  answers under ~3 s of speech are weak.
- **Scores at most the last 30 s** by default
  (`transcribe_langid_session_params::max_audio_ms`).
- **Q8_0 runs at F16 speed.** langid.cpp computes Q8_0 with ggml's ARM
  weight-repacking kernels, about 1.3x faster than F16 on an Apple CPU, at
  the accuracy cost above; transcribe.cpp widens Q8_0 to F16 instead.

## Reproduction

### Convert

Loads the SpeechBrain checkpoint through SpeechBrain (the only implementation)
from the Hugging Face cache, and writes F32 only.

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
  reports/langid/cpp-f32-untrimmed.jsonl
```
