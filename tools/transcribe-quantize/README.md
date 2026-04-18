# transcribe-quantize

C++ binary that requantizes an input GGUF to any supported preset
(`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`). This is the only place
derived quantized GGUFs are produced in the project; Python converters
only emit source/reference-dtype GGUFs.

## Usage

```bash
build/bin/transcribe-quantize INPUT.gguf OUTPUT.gguf --quant PRESET
```

## Documentation

Full reference — preset table, bucket rules, per-family policy, loader
allowlist, and the Python↔C++ separation — is in
[`../../docs/tools/quantization.md`](../../docs/tools/quantization.md).
