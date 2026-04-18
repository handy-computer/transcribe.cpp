# transcribe.cpp

C/C++ speech-to-text inference library. Runs diverse STT model families via [GGUF](https://github.com/ggerganov/gguf) models on the [ggml](https://github.com/ggml-org/ggml) runtime, targeting Metal and Vulkan for fast GPU inference everywhere.

**Supported models:** Parakeet TDT (v2, v3), Cohere Transcribe (03-2026). More families planned (Moonshine, Canary, SenseVoice, GigaAM, Whisper).

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
reference GGUF. Available presets: `F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`.

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
examples/cli/              CLI binary source
tools/transcribe-quantize/ Quantization tool source
docs/                      Porting and validation guidance
scripts/                   Python converter + test tooling
ggml/                      Vendored ggml (see ggml/UPSTREAM for pinned SHA)
samples/                   Test audio files
tests/                     Unit and smoke tests
```

## License

TODO
