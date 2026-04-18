# WER

Word Error Rate is the user-facing acceptance gate. The `scripts/wer/`
tools turn a reference corpus into transcripts, score them, and compare
runs.

## Pipeline

```text
  LibriSpeech (or similar) dir
            │
            ▼ scripts/wer/ingest.py        flac → 16-bit PCM wav + manifest.jsonl
            │
            ▼ scripts/wer/run.py           transcribe-cli --batch → hypothesis JSONL
            │
            ▼ scripts/wer/score.py         jiwer + bootstrap CI → .score.json
            │
            ▼ scripts/wer/compare.py       delta table across variants
```

Each stage is a separate script because the expensive one (`run.py`) is
slow, and the cheap ones (`score.py`, `compare.py`) get re-run often
while iterating on normalization and aggregation.

## `ingest.py` — prepare a corpus

```bash
uv run scripts/wer/ingest.py
# defaults:
#   --raw      samples/wer/raw/LibriSpeech/test-clean
#   --out-dir  samples/wer/test-clean
#   --manifest samples/wer/test-clean.manifest.jsonl
```

Walks the extracted LibriSpeech tree, decodes each `.flac` to 16-bit
PCM mono 16 kHz wav (the format `transcribe-cli` expects), and writes
a one-line-per-utterance manifest. Idempotent — existing wavs are
skipped by existence check.

## `run.py` — generate hypotheses

```bash
uv run scripts/wer/run.py \
  --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl
```

Invokes `transcribe-cli --batch` (one model load, N utterances in one
process) and writes:

```text
reports/wer/<model-stem>.<dataset>.jsonl
  {"id":..., "ref_text":..., "hyp_text":..., "mel_ms":..., "encode_ms":..., "decode_ms":...}
```

Output path is auto-derived from model name and manifest name; override
with `--out`.

## `score.py` — compute WER + CI

```bash
uv run scripts/wer/score.py \
  reports/wer/parakeet-tdt-0.6b-v2-F32.test-clean.jsonl
```

Uses `jiwer` for WER and applies `whisper_normalizer`'s
`EnglishTextNormalizer` before comparison. Emits a `.score.json`
alongside the input:

```json
{
  "wer": 0.0312, "wer_ci_lo": 0.0285, "wer_ci_hi": 0.0340,
  "n": 2620, "substitutions": ..., "deletions": ..., "insertions": ...,
  "per_utterance": [ { "id": "...", "ref": "...", "hyp": "...", "wer": ... } ]
}
```

`wer_ci_lo` / `wer_ci_hi` are a 95% bootstrap CI (1000 resamples,
seed=42 — deterministic).

## `compare.py` — delta table

```bash
uv run scripts/wer/compare.py \
  reports/wer/parakeet-tdt-0.6b-v2-F32.test-clean.score.json \
  reports/wer/parakeet-tdt-0.6b-v2-Q8_0.test-clean.score.json \
  reports/wer/parakeet-tdt-0.6b-v2-Q4_K_M.test-clean.score.json
```

First file is baseline. Prints:

```text
variant     WER%   delta   CI low  CI high  lat_p50
f32          3.12   —       2.85    3.40     720
q8_0         3.14  +0.02    2.87    3.42     620
q4_k_m       3.38  +0.26    3.10    3.66     580
```

Use this to decide whether a quant is shippable. A delta inside the
baseline's CI is indistinguishable from noise; a delta outside the CI
is a real regression.

## Relationship to validate

`validate.py` is the dev gate: per-tensor tolerances, runs in seconds.
WER is the user-facing gate: runs in minutes-to-hours, measures what
the user actually experiences. Required before enabling a new quant
preset as shippable (see
[`../porting/3-conversion.md`](../porting/3-conversion.md#quantization-policy)).
