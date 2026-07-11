# transcribe.cpp

C/C++ speech-to-text inference library. Runs diverse STT model families via [GGUF](https://github.com/ggerganov/gguf) models on the [ggml](https://github.com/ggml-org/ggml) runtime, with Metal, Vulkan, and CUDA backends for fast GPU inference plus a tinyBLAS-accelerated CPU path.

16 model families and 60+ variants, streaming and batch. Every model we publish under [`handy-computer`](https://huggingface.co/handy-computer) is numerically verified and WER-tested against its reference implementation

**Supported models:**

| Family | Variants | Docs |
| --- | --- | --- |
| Parakeet | 10 variants: TDT, RNN-T, CTC, TDT+CTC (110M–1.1B) | [docs/models/parakeet.md](docs/models/parakeet.md) |
| Canary | `canary-1b`, `canary-1b-v2`, `canary-1b-flash`, `canary-180m-flash` | [docs/models/canary.md](docs/models/canary.md) |
| Canary-Qwen | `canary-qwen-2.5b` (FastConformer + Qwen3-1.7B SALM) | [docs/models/canary-qwen-2.5b.md](docs/models/canary-qwen-2.5b.md) |
| Whisper | 12 variants (`tiny` through `large-v3-turbo`, plus `.en` siblings) | [docs/models/whisper.md](docs/models/whisper.md) |
| GigaAM | `gigaam-v3-{e2e-rnnt,e2e-ctc,rnnt,ctc}` | [docs/models/gigaam.md](docs/models/gigaam.md) |
| Moonshine | `moonshine-tiny`, `moonshine-base` | [docs/models/moonshine.md](docs/models/moonshine.md) |
| Moonshine Streaming | `moonshine-streaming-{tiny,small,medium}` | [docs/models/moonshine-streaming.md](docs/models/moonshine-streaming.md) |
| Qwen3-ASR | `qwen3-asr-0.6b`, `qwen3-asr-1.7b` | [docs/models/qwen3-asr.md](docs/models/qwen3-asr.md) |
| Cohere Transcribe | `cohere-transcribe-03-2026` | [docs/models/cohere-transcribe-03-2026.md](docs/models/cohere-transcribe-03-2026.md) |
| SenseVoice | `sensevoice-small` | [docs/models/sensevoice-small.md](docs/models/sensevoice-small.md) |
| FunASR Nano | `fun-asr-nano-2512`, `fun-asr-mlt-nano-2512` | [docs/models/fun-asr-nano.md](docs/models/fun-asr-nano.md) |
| Nemotron Speech Streaming | `nemotron-speech-streaming-en-0.6b` | [docs/models/nemotron-speech-streaming-en-0.6b.md](docs/models/nemotron-speech-streaming-en-0.6b.md) |
| Nemotron 3.5 ASR Streaming | `nemotron-3.5-asr-streaming-0.6b` (multilingual, 40 locales) | [docs/models/nemotron-3.5-asr-streaming-0.6b.md](docs/models/nemotron-3.5-asr-streaming-0.6b.md) |
| Granite Speech 4 / 4.1 | `granite-4.0-1b-speech`, `granite-speech-4.1-2b{,-plus,-nar}` | [docs/models/granite-speech.md](docs/models/granite-speech.md) |
| Voxtral | `voxtral-mini-3b-2507`, `voxtral-small-24b-2507` (audio-LLM; transcription + translation) | [docs/models/voxtral.md](docs/models/voxtral.md) |
| Voxtral Realtime | `voxtral-mini-4b-realtime-2602` (streaming audio-LLM) | [docs/models/voxtral-realtime.md](docs/models/voxtral-realtime.md) |
| MedASR | `medasr` (Conformer + CTC, English medical-dictation, gated) | [docs/models/medasr.md](docs/models/medasr.md) |

Per-variant model cards live under [`docs/models/`](docs/models/).

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

On Windows, see the [complete build guide](docs/build-windows.md) for Vulkan
SDK setup, Visual Studio commands, and the short-build-root fallback for
unusually deep checkouts.

For CUDA (Linux + NVIDIA GPU):

```bash
# requires the CUDA toolkit (nvcc) on PATH
cmake -B build -DTRANSCRIBE_CUDA=ON
cmake --build build
```

`libopenblas-dev` is optional but recommended. It accelerates the host-side decoder ~10-15x. Without it the build falls back to a scalar path automatically.

tinyBLAS (Justine Tunney's `llamafile_sgemm` kernels) is on by default.

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

## Bindings

Official bindings wrap the C API for other languages:

| Language | Path |
| --- | --- |
| Python | [bindings/python](bindings/python) |
| TypeScript / JavaScript | [bindings/typescript](bindings/typescript) |
| Rust | [bindings/rust/transcribe-cpp](bindings/rust/transcribe-cpp) |
| Swift / ObjC | [bindings/swift](bindings/swift) |

See [`docs/bindings.md`](docs/bindings.md) for how the bindings are generated
and kept in sync with the header.

## Tests

```bash
cd build && ctest
```

Some tests require a real model file. Enable them with:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build
TRANSCRIBE_PARAKEET_GGUF=path/to/model.gguf ctest --test-dir build
```

For the model-family smoke-test, numerical-validation, and benchmark
pattern expected of new ports, see
[`docs/model-family-testing.md`](docs/model-family-testing.md).

## Sponsors & Supporting Organizations

### Mozilla AI & BiR Program

A huge thanks to [Mozilla AI](https://www.mozilla.ai/) and their [BiR Program](https://www.mozilla.ai/company/bir).
This whole project started out as an idea, not even an implementation direction. It was a research project in how
to accelerate transcription models across all platforms as easily as possible. The BiR program and Davide helped
support the research, and my eventual direction to choose to implement and inference engine backed by ggml. And
also experimenting with automated model porting using agentic programming tools.

### Hugging Face

[Hugging Face](https://huggingface.co/) provided the project extra storage so we can host all of the models
which we support. We want to provide canonical references for as many models as reasonably possible,
the support from Hugging Face helps to enable this.

### Modal

[Modal](https://modal.com/) helped to provide GPU credits so the project can test and validate the projects
implementations match the transformers or nemo reference source. This is critical to ensuring that we have 
as close to a production grade inference engine that works everywhere. We believe it is critical to have
accurate transcriptions and the only way to ensure this is through long running WER checks which Modal
helps to provide. Every model published under [handy-computer](https://huggingface.co/handy-computer) 
on hugggingface has had the WER checked, so you can trust the results. And if there are any regressions, you
bet we will be fixing them.

### Blacksmith

[Blacksmith](https://www.blacksmith.sh/) provides many of the CI runners for this project. That helps to keep
transcribe.cpp well tested and ensure our releases are as smooth as possible. The CI is quick and a drop 
in replacement for the standard Github Actions runners. I ran into limits very fast with them and super happy
upon reaching out to Blacksmith they were able to provide runners for the project. 

## Project layout

```
include/transcribe.h       Public C API (single header)
src/                       Library internals (C++17)
src/arch/parakeet/         Parakeet family implementation
src/arch/cohere/           Cohere Transcribe family implementation
examples/cli/              CLI binary source
tools/transcribe-quantize/ Quantization tool source
bindings/                  Python, TypeScript, Rust, and Swift bindings
docs/                      Porting and validation guidance
scripts/                   Python converter + test tooling
ggml/                      Vendored ggml (see ggml/UPSTREAM for pinned SHA)
src/third_party/miniz/     Vendored miniz deflate codec (see its UPSTREAM file)
samples/                   Test audio files
tests/                     Unit and smoke tests
```

## License

transcribe.cpp is MIT-licensed. See [LICENSE](LICENSE) for details. Vendored
third-party components (ggml, miniz — both MIT) are attributed in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
