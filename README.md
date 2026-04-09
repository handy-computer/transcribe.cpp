# transcribe.cpp

C/C++ speech-to-text inference library. Runs diverse STT model families via [GGUF](https://github.com/ggerganov/gguf) models on the [ggml](https://github.com/ggml-org/ggml) runtime, targeting Metal and Vulkan for fast GPU inference everywhere.

**Supported models:** Parakeet TDT (v2, v3). More families planned (Cohere, Moonshine, Canary, SenseVoice, GigaAM, Whisper).

## Build

```bash
cmake -B build
cmake --build build
```

Metal is enabled automatically on Apple Silicon. For Vulkan:

```bash
cmake -B build -DTRANSCRIBE_VULKAN=ON
cmake --build build
```

To build the quantization tool:

```bash
cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON
cmake --build build
```

## Models

### Download

Parakeet weights are available as MLX safetensors:

- **v2:** [mlx-community/parakeet-tdt-0.6b-v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2)
- **v3:** [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)

Clone or download whichever variant you want. Each directory should contain `config.json`, `tokenizer.model`, and `model.safetensors`.

### Convert to GGUF

Requires [uv](https://docs.astral.sh/uv/) (dependencies are inline in the script).

```bash
uv run scripts/convert-parakeet.py path/to/parakeet-tdt-0.6b-v2 parakeet-v2.f32.gguf
```

For f16 at conversion time:

```bash
uv run scripts/convert-parakeet.py path/to/parakeet-tdt-0.6b-v2 parakeet-v2.f16.gguf --quant f16
```

### Quantize

The `transcribe-quantize` tool produces smaller models from an f32 or f16 GGUF. Available presets: `q8_0`, `q5_k_m`, `q4_k_m`.

```bash
build/bin/transcribe-quantize parakeet-v2.f32.gguf parakeet-v2.q4_k_m.gguf q4_k_m
```

## Usage

```bash
build/bin/transcribe-cli -m parakeet-v2.f32.gguf samples/jfk.wav
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

## Project layout

```
include/transcribe.h       Public C API (single header)
src/                       Library internals (C++17)
src/arch/parakeet/         Parakeet family implementation
examples/cli/              CLI binary source
tools/transcribe-quantize/ Quantization tool source
scripts/                   Python converter + test tooling
ggml/                      Vendored ggml (see ggml/UPSTREAM for pinned SHA)
samples/                   Test audio files
tests/                     Unit and smoke tests
```

## License

TODO
