# Artifacts And Goldens

The repo policy is:

```text
Commit golden contracts, not golden payloads.
```

A committed contract says what should exist and how to reproduce it. The
binary payloads are generated or restored into the build tree or artifact
cache, then verified against the contract.

## Locations

Committed contracts:

```text
tests/golden/<family>/<variant>.manifest.json
tests/tolerances/<family>.json
docs/porting/families/<family>.md
```

Generated or restored payloads:

```text
build/goldens/<family>/<case>/
build/validate/<family>/<variant>/<case>/ref/
build/validate/<family>/<variant>/<case>/cpp/
```

Reports:

```text
reports/validate/<family>/<case>.json
reports/reference/<family>/<case>.json
reports/bench/<machine>/<family>.<backend>.json
reports/porting/<family>/<variant>/intake.json
reports/porting/<family>/<variant>/_porting-log.md
```

Release references live as a sentence in `docs/models/<family>.md` and
in the HF README rendered by `scripts/hf_cards/generate.py` — no
separate release artifact is written.

Optional local cache:

```text
$TRANSCRIBE_ARTIFACT_CACHE/
~/Library/Caches/transcribe.cpp/
~/.cache/transcribe.cpp/
```

The source tree should not rely on ignored `.f32`, `.bin`, `.npy`,
`.gguf`, or model files being present.

## Manifest

Two committed artifacts carry port-level state:

1. **Golden manifest** — immutable validation provenance.
   `tests/golden/<family>/<variant>.manifest.json`. Never mutated by upload
   events.
2. **Intake** — pinned research packet.
   `reports/porting/<family>/<variant>/intake.json`. Captured once before
   converter work.

Validation runs (`validate.py --report` bundles) are ephemeral and
gitignored — see `CONTRIBUTING.md`. Release references are a sentence in
the model card, not a separate artifact.

### Golden manifest

```json
{
  "schema": "transcribe-golden-manifest-v1",
  "family": "cohere",
  "variant": "cohere-transcribe-03-2026",
  "source_model": {
    "hf_repo": "CohereLabs/cohere-transcribe-03-2026",
    "hf_revision": "76b8b23e8607f35f0265a23d481b338fb0e26aea"
  },
  "reference": {
    "kind": "transformers",
    "source": "https://github.com/huggingface/transformers",
    "revision": "v5.5.3",
    "entrypoint": "scripts/dump_reference_cohere_transformers.py"
  },
  "expected_dtype": "bfloat16",
  "dtype_source": "weights_header",
  "frontend": {
    "sample_rate": 16000,
    "n_mels": 128,
    "hop_length": 160,
    "fft_size": 512,
    "win_length": 400,
    "window": "hann_symmetric",
    "normalization": "per_feature",
    "preemphasis": 0.97,
    "dither": 0.0
  },
  "tokenizer_summary": {
    "type": "bpe",
    "vocab_size": null,
    "special_tokens": {}
  },
  "capabilities": {
    "languages": ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"],
    "language_detection": false,
    "translation": false,
    "timestamps": [],
    "streaming": false,
    "voice_activity_detection": false,
    "speaker_diarization": false
  },
  "tolerance_file": "tests/tolerances/cohere.json",
  "cases": ["jfk"]
}
```

`validate.py` interprets the manifest by convention:

- `schema`: current value is `transcribe-golden-manifest-v1`.
- `family`: maps to `scripts/envs/<family>/`,
  `tests/tolerances/<family>.json`, and `models/<family>/`.
- `variant`: selects `build/validate/<family>/<variant>/...`.
- `source_model.hf_repo` / `source_model.hf_revision`: HF model repo id +
  commit SHA pinned at intake. The revision is the single most important
  model-weight reproducibility field; do not leave it `null` for new ports.
- `reference.kind`: one of `nemo`, `transformers`, `author_repo_<name>`, or
  another short implementation key chosen during reference research.
- `reference.source`: canonical source for the reference implementation. Use
  a repo URL when possible; for non-repo implementations, use the most stable
  source identifier available and document the rationale in the family note.
- `reference.revision`: commit SHA, release tag, or other immutable revision
  for `reference.source`. If the current reference runs from a PyPI package
  and no commit was captured during historical backfill, use the corresponding
  package release tag and tighten to a commit on the next validation refresh.
- `reference.entrypoint`: relative path to the local dump script that
  instruments this reference. `validate.py ref` uses this field directly.
- `expected_dtype` + `dtype_source`: from intake. `expected_dtype` maps from
  `dtype.expected`. `dtype_source` is one of `config`, `weights_header`,
  `manual`, or `unresolved`.
- `frontend`: sample_rate, n_mels, hop_length, fft_size, window,
  preemphasis, dither, and any other fields the family's preprocessor
  exposes. Mirrors intake.
- `tokenizer_summary`: tokenizer type, vocab size, special token IDs. Not
  the full tokenizer — that lives in the GGUF.
- `capabilities`: what the family claims to support (languages,
  translation, timestamps, streaming, VAD, diarization). Mirrors intake.
  Preflight cross-checks `capabilities.languages` against the GGUF's
  `general.languages` array and `capabilities.language_detection` against
  `stt.capability.lang_detect`.
- `tolerance_file`: relative path to the family's tolerance file.
- `cases`: sample basenames under `samples/`, such as `jfk`.

Null values are acceptable for provenance fields when backfilling a
historical manifest, but every new intake should populate
`source_model.hf_revision` and `reference.revision` at capture time.

### Release references

No separate release-record artifact is written. The release pin is a
single sentence in the family's model card — see `docs/models/<family>.md`
and the rendered HF README at `handy-computer/<slug>-gguf` — pointing at
the transcribe.cpp commit that was validated before upload:

> Validated against the {reference} reference at transcribe.cpp commit
> `<short>` on `<date>`.

The rendered HF README carries the same sentence via
`scripts/hf_cards/<variant>.yaml`'s `validation` block, so each HF
revision preserves the validation pin that was current at its upload.

This replaces a heavier release-record design — per-upload JSON bundles
with validation/wer/bench snapshots — that duplicated git history and
HF revision state. The single-sentence pin plus HF's own revision log is
enough: a user or reviewer clicks the commit link and can run
`uv run scripts/validate.py all --family <f>` at that commit to
regenerate the full evidence.

Generated payload directories contain tensor pairs plus any behavioral
artifacts such as `transcript.json`. A future index/report can list every
tensor, shape, dtype, sha256, and non-tensor artifact for a run; the
current validation gate reads the manifest and tolerance file directly.

## Tensor Layout

Reference dumpers emit tensors in the validation contract shape: raw
little-endian fp32, row-major, with only the leading batch dim squeezed
for single-sample cases unless a family note explicitly documents a
layout adapter. The contract shape should be decided before the C++ model
exists and then matched by the C++ debug dump path.

For a new family, generate these reference dumps before the C++ model
exists. The C++ debug dump path should then be implemented to match the
committed reference contract. If a layout adapter is unavoidable, make it
explicit in the dumper metadata and document it in the family note.

Behavioral outputs such as transcripts are separate JSON artifacts, for
example `transcript.json`. They are generated by the reference path and
by `validate.py cpp`; `validate.py compare` checks transcript text for an
exact character-by-character match when the reference transcript exists.

## Generation

Use `scripts/validate.py` for validation workflows. It reads the
manifest to find the model and reference type, resolves dump paths by
convention, and drives the reference dump, C++ dump, tensor comparison,
and exact transcript comparison.

```bash
# Generate reference dumps (downloads model from HF if needed)
uv run scripts/validate.py ref --family cohere

# Generate C++ dumps
uv run scripts/validate.py cpp --family cohere

# Compare with tolerances
uv run scripts/validate.py compare --family cohere

# All three in one shot
uv run scripts/validate.py all --family cohere

# Override model source (local path instead of HF)
uv run scripts/validate.py ref --family parakeet --model /local/path/to/model

# Override GGUF source for C++ dumps
uv run scripts/validate.py cpp --family cohere --gguf /local/path/to/model.gguf

# Specify variant when a family has multiple models
uv run scripts/validate.py all --family qwen3-asr --variant qwen3-asr-0.6b
```

The dump scripts can also be called directly for debugging:

```bash
uv run --project scripts/envs/cohere \
  scripts/dump_reference_cohere_transformers.py decode \
  --model ../models/cohere-transcribe-03-2026 \
  --audio samples/jfk.wav \
  --out build/validate/cohere/cohere-transcribe-03-2026/jfk/ref \
  --torch-threads 1
```

## Default Tests

Default `ctest` must pass on a clean checkout without real model files.

If a test needs a generated golden, one of these must be true:

- CMake builds the golden first from repo-local or fetched inputs.
- The test skips with return code `77` when the manifest exists but the
  payload cannot be generated or restored.
- The payload is a deliberately committed tiny exception.

Do not register a test that unconditionally fails because an ignored
source-tree golden payload is absent.

## Cache Keys

Artifact cache keys should be derived from:

- family
- case
- manifest schema
- reference implementation and revision
- source model id and revision
- audio sha256
- dump-point list
- dtype and quantization policy

If the cache key or output sha256 does not match, regenerate or fail with
a clear command. Do not silently accept stale payloads.
