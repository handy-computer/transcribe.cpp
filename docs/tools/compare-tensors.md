# Compare tensors

`scripts/compare_tensors.py` is the symmetric directory diff used by the
per-stage numerical accuracy harness. Given two dirs of dumped tensors,
it reports element-wise stats per tensor and returns a pass/fail exit
code gated on tolerances.

## CLI

```bash
uv run scripts/compare_tensors.py <LEFT_DIR> <RIGHT_DIR> \
  [--max-abs 1e-3] [--mean-abs 1e-4] \
  [--tolerances tests/tolerances/<family>.json]
```

Convention: left is the C++ dump dir, right is the reference dump dir.
The comparison is symmetric, so the order only affects the sign in any
future verbose output, not pass/fail.

## Input format

Both dirs contain pairs of files, one per tensor:

```text
<name>.f32    raw little-endian float32, row-major (C order)
<name>.json   { "shape": [...], "dtype": "float32", "stage": "...", "source": "..." }
```

See [`reference-dumps.md`](reference-dumps.md) for how these are
produced.

## What it reports

For every tensor that appears in **either** dir:

- `max_abs`  — `max(|L - R|)`
- `mean_abs` — `mean(|L - R|)`
- first divergent flat index (or `-` if exact)
- shape on each side

A tensor that appears on only one side is reported as `MISSING`. It
fails the run **only** if it has an explicit tolerance entry — which
marks it as a contract tensor. Debug-only tensors that only one side
dumps are reported for visibility but don't fail.

The `<name>.json` sidecars also carry per-tensor `min`, `max`, and
`mean` fields when produced by the current C++ and reference dumpers.
Those fields are useful for checking activation scale, while
`compare_tensors.py` remains the element-wise pass/fail gate.

## C++ intermediate lifetime

For ggml graphs using the scheduler, intermediate buffers may be reused
unless the tensor is marked as a graph output. A family that stores
intermediate tensor handles and dumps them after `graph_compute` must
call `transcribe::debug::mark_tensor_for_dump(t)` for every contract
dump tensor during graph construction. Otherwise the dump can read stale
or reused storage and produce misleading tolerances. If unrelated
tensors are byte-exact equal in the C++ dump dir, check this first.

## Tolerances

Global:

```bash
--max-abs  1e-3    # default: max |diff| allowed
--mean-abs 1e-4    # default: mean |diff| allowed
```

Per-tensor overrides via `--tolerances`:

```json
{
  "enc.pre_encode.out": { "max_abs": 5e-4, "mean_abs": 5e-5 },
  "enc.final":          { "max_abs": 5e-3, "mean_abs": 5e-4 }
}
```

Per-family defaults live at `tests/tolerances/<family>.json` and are
used by `validate.py` automatically.

## Exit codes

- `0` — every contract tensor within tolerance.
- `1` — one or more contract tensors out of tolerance or missing.

## Typical usage

Standalone:

```bash
export TRANSCRIBE_DUMP_DIR=build/validate/parakeet/v2/jfk/cpp
build/bin/transcribe-cli --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf samples/jfk.wav

uv run scripts/compare_tensors.py \
  build/validate/parakeet/v2/jfk/cpp \
  build/validate/parakeet/v2/jfk/ref \
  --tolerances tests/tolerances/parakeet.json
```

Via orchestrator (preferred):

```bash
uv run scripts/validate.py compare --family parakeet
```

## When it fails

If the comparator reports a FAIL, use the `numerical-debugger` agent or
walk the output from the first failing tensor. Earlier stages cascade into
later ones, so the first divergence is usually the root cause. See
[`docs/porting/4a-numerical-troubleshooting.md`](../porting/4a-numerical-troubleshooting.md)
for common failure modes such as mel/STFT off-by-one errors, dtype drift,
layout bugs, op mismatches, attention mask or position errors, KV-cache
issues, tokenization problems, and log-softmax infinities.
