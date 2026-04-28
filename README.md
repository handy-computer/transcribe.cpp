# transcribe.cpp

C/C++ speech-to-text inference library. Runs diverse STT model families via [GGUF](https://github.com/ggerganov/gguf) models on the [ggml](https://github.com/ggml-org/ggml) runtime, targeting Metal and Vulkan for fast GPU inference everywhere.

**Supported models:**

| Family | Variant | Docs |
| --- | --- | --- |
| Parakeet TDT | `parakeet-tdt-0.6b-v2` | [docs/models/parakeet-tdt-0.6b-v2.md](docs/models/parakeet-tdt-0.6b-v2.md) |
| Parakeet TDT | `parakeet-tdt-0.6b-v3` | [docs/models/parakeet-tdt-0.6b-v3.md](docs/models/parakeet-tdt-0.6b-v3.md) |
| Cohere Transcribe | `cohere-transcribe-03-2026` | [docs/models/cohere-transcribe-03-2026.md](docs/models/cohere-transcribe-03-2026.md) |
| Qwen3-ASR | `qwen3-asr-0.6b` | [docs/models/qwen3-asr-0.6b.md](docs/models/qwen3-asr-0.6b.md) |
| Qwen3-ASR | `qwen3-asr-1.7b` | [docs/models/qwen3-asr-1.7b.md](docs/models/qwen3-asr-1.7b.md) |
| Whisper | 12 variants (`tiny` through `large-v3-turbo`, plus `.en` siblings) | [docs/models/whisper.md](docs/models/whisper.md) |

More families planned (Moonshine, Canary, SenseVoice, GigaAM).

## Build

```bash
cmake -B build
cmake --build build
```

Metal is enabled automatically on Apple Silicon. For Vulkan (Linux/Windows):

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libvulkan-dev glslc libopenblas-dev

cmake -B build -DTRANSCRIBE_VULKAN=ON
cmake --build build
```

`libopenblas-dev` is optional but recommended — it accelerates the host-side decoder ~10-15x. Without it the build falls back to a scalar path automatically.

To build the quantization tool:

```bash
cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON
cmake --build build
```

## Models

Pre-built GGUFs for all supported models are hosted on Hugging Face under
[`handy-computer`](https://huggingface.co/handy-computer). Each per-model doc
(linked in the table above) includes direct download links for every quant.
Convert from source only if you need a different dtype or a checkpoint that
isn't pre-built.

### Convert to GGUF

The converter loads directly from NVIDIA's NeMo checkpoints via
`ASRModel.from_pretrained`. Requires [uv](https://docs.astral.sh/uv/);
the parakeet env ships NeMo and its deps.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2
```

This writes `models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf` following
the llama.cpp-style `<slug>-<QUANT>.gguf` naming convention. Pass a local
`.nemo` path or extracted directory for offline conversion.

### Quantize

The `transcribe-quantize` tool produces smaller models from the
reference GGUF. Available presets: `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`,
`Q4_K_M`.

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q4_K_M.gguf \
  --quant Q4_K_M
```

## Usage

```bash
build/bin/transcribe-cli -m models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf samples/jfk.wav
```

Input must be 16 kHz mono WAV. Use `ffmpeg` or `sox` to convert other formats:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Tests

```bash
cd build && ctest
```

Some tests require a real model file. Enable them with:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build
TRANSCRIBE_REAL_PARAKEET_GGUF=path/to/model.gguf ctest --test-dir build
```

For the model-family smoke-test, numerical-validation, and benchmark
pattern expected of new ports, see
[`docs/model-family-testing.md`](docs/model-family-testing.md).

## Project layout

```
include/transcribe.h       Public C API (single header)
src/                       Library internals (C++17)
src/arch/parakeet/         Parakeet family implementation
src/arch/cohere/           Cohere Transcribe family implementation
examples/cli/              CLI binary source
tools/transcribe-quantize/ Quantization tool source
docs/                      Porting and validation guidance
scripts/                   Python converter + test tooling
ggml/                      Vendored ggml (see ggml/UPSTREAM for pinned SHA)
samples/                   Test audio files
tests/                     Unit and smoke tests
```

## License

transcribe.cpp is MIT-licensed. See [LICENSE](LICENSE) for details.
