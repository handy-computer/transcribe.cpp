# Remote WER (Modal)

Dispatch transcribe.cpp WER runs to Modal GPUs. One container per
`(model, quant)` cell, parallel fanout, hyp JSONLs returned to the laptop
for local scoring.

Scoring runs locally (`scripts/wer/score.py`), not on Modal.

## One-time setup

1. Install Modal at a pinned version and authenticate:
   ```
   uv tool install 'modal==1.4.3' && modal token new
   ```
   The dispatcher uses specific Modal API surfaces (per-GPU function registration,
   Volume read/commit semantics). Bumping Modal is intentional. Before upgrading,
   smoke `modal run scripts/wer/remote/modal_sweep.py::run --model moonshine-base
   --quant Q8_0 --n-utts 32` to confirm the API still matches.
2. Add an HF token secret to Modal:
   ```
   modal secret create huggingface-secret HF_TOKEN="$(cat ~/.cache/huggingface/token)"
   ```
3. Verify CUDA wiring is on the branch you'll dispatch from
   (`TRANSCRIBE_CUDA` in `CMakeLists.txt`, etc.).

## Sweep one or more models

```
modal run scripts/wer/remote/modal_sweep.py::sweep --models moonshine-base,moonshine-tiny
```

Defaults (visible in `--help`):
- `--dataset librispeech:test-clean`
- `--quants ""` (all quants)
- `--gpu L4`
- `--n-utts -1` (full manifest, 2620 utts for test-clean)
- `--clean false`

Each `--models` entry is one of:

| Form | Example | Quant set |
|---|---|---|
| **hf_card slug** | `moonshine-base` | pinned by `scripts/hf_cards/moonshine-base.yaml` |
| **HF repo path** | `handy-computer/granite-4.0-1b-speech-gguf` | discovered via the HF API |

Use slugs for shipped models (an hf_card exists). Use repo paths for
in-progress models without a card yet (e.g. on a porting branch).

## Quick check (one model, one quant, subset)

```
modal run scripts/wer/remote/modal_sweep.py::sweep \
    --models moonshine-base --quants Q8_0 --n-utts 128
```

`--quants` is a substring filter against the model's `.gguf` filenames;
`--n-utts` caps the manifest. Useful for validating "does this model
load and produce sensible hyps" before paying for a full sweep.

## Output layout

```
reports/wer/<gguf-stem>.<dataset-id>.jsonl          hyp JSONL per cell (run.py format)
reports/wer/remote_sweep.<dataset-id>.summary.tsv   per-cell wall / RTF
```

Score locally (the sweep prints the exact one-liner with the right glob):
```
for f in reports/wer/*.librispeech-test-clean.jsonl; do
    uv run scripts/wer/score.py "$f"
done
```

## Datasets

| `--dataset` | Manifest | Notes |
|---|---|---|
| `librispeech:test-clean` | English | 2620 utts, default. Downloads via `scripts/wer/setup.sh`. |
| `librispeech:<split>` | English | Other splits work if `setup.sh` supports them. |
| `fleurs:<bcp47>` | Per-language | One language at a time. Downloads via `datasets`. |

## What's cached on Modal Volumes

| Volume | Mount | Holds |
|---|---|---|
| `transcribe-build` | `/build` | Per-(source-fingerprint, GPU arch) build trees + transcribe-cli binary |
| `hf-cache` | `/root/.cache/huggingface` | All HF model downloads |
| `transcribe-data` | `/data` | LibriSpeech raw + extracted, FLEURS, manifests, cached hyp JSONLs |

Two fingerprints govern caching, scoped to what each cache actually depends on:

- **Build cache** keyed by `(SRC_FP, GPU arch)` where `SRC_FP = hash(CMakeLists
  content + C++ source path list)`. A `.cpp` / `.cu` / `.h` / `CMakeLists.txt`
  change or a `--gpu` change rotates the dir; nothing else does.
- **Hyp cache** keyed by `HYP_FP = hash(SRC_FP + scripts/wer/{run.py,ingest.py})`.
  Any binary or pipeline-script change invalidates it; the C++ source pieces
  are folded in via SRC_FP.

Crucially, edits to the dispatcher (`scripts/wer/remote/modal_sweep.py`) and to
local-only scripts (`score.py`) do NOT rotate either fingerprint, because they
cannot change what the cell produces. Branch switches and source edits rebuild
correctly without `--clean`; iterating on dispatcher logic costs nothing.

A single-arch build finishes nvcc in ~2 min vs ~6 min for a fat multi-arch
build. The Volume grows as new (SRC_FP, arch) combinations land; periodic
manual cleanup is fine.

Re-running the same sweep returns immediately with `[CACHED]` entries; an
interrupted sweep can be relaunched and only the cells that never completed
will run.

## Cost rule of thumb (L4, $0.80/hr)

For ~6 hours of audio at ~20× RTF:
- Per cell: ~16 min wall, ~$0.21
- 18-cell sweep (3 models × 6 quants), parallel: ~30 min wall, ~$5-7

L40S is ~1.6× faster than L4 for ~2.4× the cost. A10G is dominated by L4
on $/perf for this workload.

## Live progress

Per-cell `run.py` output is unbuffered and streams to Modal logs. Watch
live via the dashboard URL printed at the top of each `modal run` (the
`ap-XXX` link). Each cell shows `[N/total] X.X utt/s, ETA Ys` every 200
utterances.

## When to pass `--clean`

Almost never — the source-fingerprint cache key auto-rotates on any C++ or
CMakeLists change. Use `--clean` only if you suspect the Volume itself is
corrupted (very rare), or to free disk inside `/build/cuda-<old-fp>` after
many fingerprint rotations.

## Extending CUDA arch list

Default in `modal_sweep.py` is `75;86;89` (T4, A10G, L4, L40S). Add:
- `80` for A100 (40/80 GB)
- `90` for H100 / H200
- `100` for B200

Wider arch list means longer compile per `--clean` (each arch adds ~2 min
of nvcc work).
