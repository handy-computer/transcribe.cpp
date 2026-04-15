# Benchmarking

`scripts/bench/run.py` drives `transcribe-bench` across a matrix of
(backend, family, quant, sample) cells, auto-detects the host, and
aggregates results into per-(family, backend) JSON reports.

## What it measures

`transcribe-bench` times per-stage latency (mel frontend, encoder,
decoder) and overall RTFx on a fixed audio sample, with configurable
warmup + measurement iterations. It loads the model once per cell and
runs the full stack — not just matmul microbenchmarks.

## CLI

```bash
uv run scripts/bench/run.py                               # full matrix, all backends
uv run scripts/bench/run.py --family parakeet --quants F16
uv run scripts/bench/run.py --samples jfk --iters 20 --warmup 5
uv run scripts/bench/run.py --backends metal,cpu,vulkan
uv run scripts/bench/run.py --backends cpu --name pre-refactor
uv run scripts/bench/run.py --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf --samples jfk
```

### Flags

| Flag           | Purpose                                                              |
|----------------|----------------------------------------------------------------------|
| `--family`     | `parakeet` / `cohere` / `all` (default `all`)                        |
| `--quants`     | Quant presets to include (default `F16,Q8_0,Q4_K_M`)                 |
| `--samples`    | Sample stems under `samples/` (default `jfk,dots`)                   |
| `--iters`      | Measured iterations per cell (default `2`)                           |
| `--warmup`     | Warmup iterations per cell (default `1`)                             |
| `--backends`   | `metal,cpu,vulkan` or `all` (default: auto-detect)                   |
| `--name`       | Stable label for named baselines (otherwise timestamp)               |
| `--model`      | Escape hatch: bypass discovery, bench one specific GGUF              |
| `--bench-bin`  | Legacy override: single-backend, explicit binary path                |
| `--out-dir`    | Override reports output root                                         |
| `--dry-run`    | Print selected backends + matrix without running                     |

## Backend resolution

Each run passes `--backend <name>` to the bench binary so the library
selector is exercised end-to-end (not just the binary's default).

| Backend | Binary                                   | Flag                |
|---------|------------------------------------------|---------------------|
| metal   | `build/bin/transcribe-bench`             | `--backend metal`   |
| cpu     | `build/bin/transcribe-bench`             | `--backend cpu`     |
| vulkan  | `build-vulkan/bin/transcribe-bench`      | `--backend vulkan`  |

Auto-detection (when `--backends` is unset or `all`):

- `metal` — `build/bin/transcribe-bench` exists **and** `sys.platform == 'darwin'`
- `cpu` — `build/bin/transcribe-bench` exists
- `vulkan` — `build-vulkan/bin/transcribe-bench` exists

## Output

One aggregated JSON per (family, backend):

```text
reports/perf/<machine-slug>/<timestamp-or-name>_<family>_<backend>.json
```

`<machine-slug>` is derived from hostname + CPU model so reports from
different machines don't overwrite each other.

## Comparing runs

`scripts/bench/compare.py` diffs two report files and prints a
per-cell delta table. Use `--name pre-refactor` / `--name post-refactor`
for stable named baselines that overwrite on re-run, then compare the
named pair.

## Quant coverage

The default `--quants F16,Q8_0,Q4_K_M` covers the canonical tiers.
Add legacy quants or K-quants as they stabilize in
[`quantization.md`](quantization.md). The family must have a GGUF
present in `models/<family>/` for each listed quant — missing files
are skipped with a warning, not an error.
