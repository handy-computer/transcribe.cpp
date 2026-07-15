# WER and DER

Word Error Rate is the user-facing acceptance gate. The `scripts/wer/`
tools turn a reference corpus into transcripts, score text with WER/CER,
score timed speaker attribution with Diarization Error Rate (DER), and compare
runs.

## Pipeline

```text
  LibriSpeech (or similar) dir
            │
            ▼ scripts/wer/ingest.py        flac → 16-bit PCM wav + manifest.jsonl
            │
            ▼ scripts/wer/run.py           transcribe-cli --batch → hypothesis JSONL
            ├── scripts/wer/score.py       jiwer + bootstrap CI → .score.json
            └── scripts/wer/der.py         timed speaker intervals → .der.json
            │
            ▼ scripts/wer/compare.py       delta table across variants
```

Each stage is a separate script because the expensive one (`run.py`) is
slow, and the cheap ones (`score.py`, `compare.py`) get re-run often
while iterating on normalization and aggregation.

## Methodology (pinned recipe)

WER is only comparable across runs, backends, and against external
numbers when the decode recipe is identical. The published transcribe.cpp
tables use this recipe, which matches the field standard for **short-form**
LibriSpeech WER:

| Knob | Value | Why |
| --- | --- | --- |
| Timestamps | **`none`** (`<\|notimestamps\|>`) | OpenAI's own LibriSpeech eval and the HF Open ASR Leaderboard report short-form WER with timestamps off; on the leaderboard timestamps are only used in the long-form track. |
| Language | **forced** (`en` for LibriSpeech) | LibriSpeech manifests carry `"language":"en"`, so `run.py` forces it instead of letting multilingual models auto-detect. Matches `DecodingOptions(language="en")`. |
| Decoding | **greedy** | No beam search / no sampling at temperature 0 (transcribe.cpp default). |
| Temperature fallback | ladder `0.0, 0.2, …, 1.0` | Library default (`temperature=0`, `temperature_inc=0.2`); `seed=0` keeps any T>0 tier deterministic. |
| Fallback thresholds | compression `2.4`, logprob `-1.0`, no-speech `0.6` | Library defaults (`transcribe_whisper_run_ext_init`). |
| Condition on prev | **off** | Library default; long-form conditioning is not part of short-form WER. |
| Normalization | `EnglishTextNormalizer` (en) / `BasicTextNormalizer` (other) | Applied to both ref and hyp at score time (`score.py`). |
| Dataset | full LibriSpeech `test-clean` (2620 utts) | — |

The recipe is **stamped into the hyp JSONL `batch_header`** (`recipe` field)
by `run.py`, so every artifact is self-describing and a methodology drift
shows up in the file rather than silently shifting the number.

**What does and doesn't move WER (measured on whisper-medium F16):**

- **Timestamps move it ~0.2pp.** `segment` → 2.63%, `none` → 2.81%. This is
  the single biggest knob. Earlier published whisper tables were measured
  with `segment` enabled (non-standard); they have been re-baselined to
  `none` to match OpenAI/Open ASR Leaderboard. `segment` is *better* here
  because timestamp constraints suppress short-clip hallucination, but it is
  not how the field reports WER.
- **Backend is WER-neutral.** CUDA `none` = 2.81% vs Metal `none` = 2.82%.
- **Batching is WER-neutral.** batch-1 vs batch-8 differ by ≤0.08% across the
  whole whisper family — under the ~0.1pp Metal run-to-run noise floor.

> If you intentionally want timestamped WER (a product-usage question, not a
> benchmark one), pass `--timestamps segment`; the output filename and the
> stamped `recipe` will record it so it never gets confused with the
> standard number.

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
process). By default the WER harness explicitly runs `--timestamps none` for
text-only scoring (the library and CLI defaults are `auto`) and writes the
timestamp mode into the auto-derived output name:

```text
reports/wer/<model-stem>.<dataset>-timestamps_none.jsonl
  {"id":..., "ref_text":..., "hyp_text":..., "mel_ms":..., "encode_ms":..., "decode_ms":...}
```

Output path is auto-derived from model name and manifest name; override
with `--out`. Pass `--timestamps segment` or another supported mode
when you intentionally want timestamped WER; the auto path will use the
same `-timestamps_<mode>` suffix.

For a diarization run, pass `--diarize`. The input manifest must provide timed
reference speaker intervals, and `run.py` retains both sides plus the WAV
duration in the report:

```json
{"id":"meeting-1","audio":"samples/meeting-1.wav","ref_text":"...",
 "ref_speaker_segments":[
   {"t0_ms":0,"t1_ms":920,"speaker_id":"alice"},
   {"t0_ms":920,"t1_ms":1800,"speaker_id":"bob"}]}
```

```bash
uv run scripts/wer/run.py \
  --model models/MOSS-Transcribe-Diarize/moss-transcribe-diarize-BF16.gguf \
  --manifest samples/wer/meetings.manifest.jsonl \
  --diarize
```

`--diarize` changes the harness timestamp default from `none` to `auto`, since
DER needs timed speaker rows. An explicit `--timestamps ...` still wins.

The resulting utterance record adds `duration_ms`,
`ref_speaker_segments`, and `hyp_speaker_segments`. The ordinary LibriSpeech /
FLEURS ingest adapters do not invent diarization references; use a corpus with
speaker-time annotations when preparing a DER manifest.

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

Diarizing models may emit inline timestamps and speaker labels. Remove those
spans explicitly when the evaluation contract treats them as metadata:

```bash
uv run scripts/wer/score.py reports/wer/<report>.jsonl --dediarize
```

This is opt-in because bracketed annotations such as `[laughter]` or `<unk>`
may be meaningful scoring content in other datasets.

## `der.py` — compute DER + CI

```bash
uv run scripts/wer/der.py reports/wer/<diarized-report>.jsonl
```

DER is computed from intervals, never by reparsing transcript text. For each
utterance the scorer optimally matches hypothesis speakers to reference
speakers, then aggregates missed speaker time, false-alarm speaker time, and
speaker-confusion time over reference speaker time. Speaker IDs are opaque:
0-based or 1-based integers and string labels all have identical semantics
after matching.

The default contract scores overlap and applies no boundary collar. Both are
explicitly configurable:

```bash
uv run scripts/wer/der.py reports/wer/<report>.jsonl --collar-ms 250
uv run scripts/wer/der.py reports/wer/<report>.jsonl --ignore-overlap
```

`--collar-ms` is the *total* width of a collar centered on every reference
boundary, pyannote's parameterization: `--collar-ms 250` excludes 125 ms on
each side. NIST md-eval's `-c` is *per side* — to reproduce a published
"0.25 s collar" (md-eval `-c 0.25`), pass `--collar-ms 500`.
The output records the collar and overlap policy alongside corpus DER, a
deterministic bootstrap confidence interval, component times, per-utterance
scores, and the selected speaker mapping.

Reference intervals are trusted annotations and validated strictly; a
malformed reference fails the run. Hypothesis intervals are model output, so
geometric defects the metric already penalizes are sanitized instead of
aborting the corpus: rows are clamped to `[0, duration_ms]` and rows that are
empty after clamping (zero-length or fully out of range) are dropped.

DER fundamentally requires timing. Granite's SAA prompt currently attributes
text turns but emits no time intervals; an all-`[0, 0]` hypothesis is that
task's explicit untimed sentinel, and `der.py` rejects it with `DER is
unavailable` instead of reporting a bogus score. Timed diarization outputs
such as MOSS can be scored directly.

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
