# Benchmarks

Benchmarks are part of the port. They provide both a reference baseline
and the C++ performance record for later regression checks.

## When Benchmarks Happen

Run benchmarks at three points:

1. After the reference run works, capture a reference implementation
   baseline.
2. After the first accuracy GGUF passes end-to-end, capture a C++ CPU
   baseline.
3. After accelerator and quantized variants work, capture the backend
   matrix.

Benchmarks should not replace numerical validation. They should include
enough accuracy metadata to reject performance wins that changed output.

## Required Fields

Benchmark reports should include:

- schema version
- git SHA and dirty flag
- family, variant, quant, backend, sample
- model path or model id
- audio path and sha256
- host OS, CPU, RAM, GPU, backend runtime
- compiler and build type
- thread count and backend options
- load, mel, encode, decode timings
- `rtf_compute_mean`
- `rtf_wall_mean`
- transcript hash or token hash
- optional golden/validation status

Reference baseline rows should identify their runtime, for example:

```text
runtime = "nemo"
runtime = "transformers"
runtime = "mlx-audio"
runtime = "parakeet-mlx"
runtime = "transcribe.cpp"
```

## Commands

Current C++ benchmark shape:

```bash
uv run scripts/bench/run.py \
  --family <family> \
  --samples jfk \
  --out reports/bench/<machine>/<family>.json
```

Candidate comparison shape:

```bash
uv run scripts/bench/compare.py \
  --baseline reports/bench/<machine>/<family>.baseline.json \
  --candidate reports/bench/<machine>/<family>.candidate.json
```

Reference implementation benchmarks may need a family-specific runner
until a unified runner exists. Put the command in
`docs/porting/families/<family>.md` and save the result under:

```text
reports/reference/<family>/<case>.json
```

## Regression Gates

Bench comparison should fail on configured regressions, not on incidental
noise. Thresholds should be per family/backend/quant where needed.

Suggested initial gates:

- wall RTF regression above threshold: fail
- compute RTF regression above threshold: fail
- transcript/token hash changed: fail or require explicit override
- missing benchmark row from candidate: fail

Per-op or per-block timing is optional for general CI, but required when
optimizing encoder/decoder internals.

