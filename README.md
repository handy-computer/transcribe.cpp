# transcribe.cpp

C/C++ speech-to-text inference library. Runs diverse STT model families via [GGUF](https://github.com/ggerganov/gguf) models on the [ggml](https://github.com/ggml-org/ggml) runtime, with Metal, Vulkan, and CUDA backends for fast GPU inference plus a tinyBLAS-accelerated CPU path.

16 model families and 60+ variants, streaming and batch. Every model we publish under [`handy-computer`](https://huggingface.co/handy-computer) is numerically verified and WER-tested against its reference implementation

**Supported models:**

<!-- catalog:family-index -->
| Family | Variants | Available capabilities | Docs |
| --- | --- | --- | --- |
| Canary | `canary-180m-flash`, `canary-1b`, `canary-1b-flash`, `canary-1b-v2` | translate | [docs/models/canary.md](docs/models/canary.md) |
| Canary-Qwen 2.5B | `canary-qwen-2.5b` | - | [docs/models/canary-qwen-2.5b.md](docs/models/canary-qwen-2.5b.md) |
| Cohere Transcribe | `cohere-transcribe-03-2026`, `cohere-transcribe-arabic-07-2026` | - | [docs/models/cohere.md](docs/models/cohere.md) |
| Fun-ASR-Nano | `fun-asr-mlt-nano-2512`, `fun-asr-nano-2512` | - | [docs/models/fun-asr-nano.md](docs/models/fun-asr-nano.md) |
| GigaAM-v3 | `gigaam-v3-ctc`, `gigaam-v3-e2e-ctc`, `gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt` | token timestamps | [docs/models/gigaam.md](docs/models/gigaam.md) |
| Granite Speech 4 / 4.1 | `granite-4.0-1b-speech`, `granite-speech-4.1-2b`, `granite-speech-4.1-2b-nar`, `granite-speech-4.1-2b-plus` | diarize, translate, word timestamps | [docs/models/granite-speech.md](docs/models/granite-speech.md) |
| Granite Speech 5.0 TurboCTC | `granite-speech-5.0-470m-turboctc`, `granite-speech-5.0-470m-turboctc-nc` | - | [docs/models/granite-speech-5.0-turboctc.md](docs/models/granite-speech-5.0-turboctc.md) |
| MedASR | `medasr` | token timestamps | [docs/models/medasr.md](docs/models/medasr.md) |
| Moonshine | `moonshine-base`, `moonshine-base-ar`, `moonshine-base-ja`, `moonshine-base-ko`, `moonshine-base-uk`, `moonshine-base-vi`, `moonshine-base-zh`, `moonshine-tiny`, `moonshine-tiny-ar`, `moonshine-tiny-ja`, `moonshine-tiny-ko`, `moonshine-tiny-uk`, `moonshine-tiny-vi`, `moonshine-tiny-zh` | - | [docs/models/moonshine.md](docs/models/moonshine.md) |
| Moonshine Streaming | `moonshine-streaming-medium`, `moonshine-streaming-small`, `moonshine-streaming-tiny` | streaming | [docs/models/moonshine-streaming.md](docs/models/moonshine-streaming.md) |
| MOSS-Transcribe-Diarize | `moss-transcribe-diarize` | diarize, segment timestamps | [docs/models/moss-transcribe-diarize.md](docs/models/moss-transcribe-diarize.md) |
| Multitalker Parakeet Streaming 0.6B v1 | `multitalker-parakeet-streaming-0.6b-v1` | diarize, streaming, token timestamps | [docs/models/multitalker-parakeet-streaming-0.6b-v1.md](docs/models/multitalker-parakeet-streaming-0.6b-v1.md) |
| Nemotron 3.5 ASR Streaming 0.6B | `nemotron-3.5-asr-streaming-0.6b` | streaming, token timestamps | [docs/models/nemotron-3.5-asr-streaming-0.6b.md](docs/models/nemotron-3.5-asr-streaming-0.6b.md) |
| Nemotron Speech Streaming EN 0.6B | `nemotron-speech-streaming-en-0.6b` | streaming, token timestamps | [docs/models/nemotron-speech-streaming-en-0.6b.md](docs/models/nemotron-speech-streaming-en-0.6b.md) |
| Parakeet | `parakeet-ctc-0.6b`, `parakeet-ctc-1.1b`, `parakeet-primeline`, `parakeet-rnnt-0.6b`, `parakeet-rnnt-1.1b`, `parakeet-tdt-0.6b-v2`, `parakeet-tdt-0.6b-v3`, `parakeet-tdt-1.1b`, `parakeet-tdt_ctc-1.1b`, `parakeet-tdt_ctc-110m`, `parakeet-unified-en-0.6b` | streaming, token timestamps | [docs/models/parakeet.md](docs/models/parakeet.md) |
| Qwen3-ASR | `qwen3-asr-0.6b`, `qwen3-asr-1.7b` | - | [docs/models/qwen3-asr.md](docs/models/qwen3-asr.md) |
| SenseVoice Small | `sensevoice-small` | - | [docs/models/sensevoice-small.md](docs/models/sensevoice-small.md) |
| Voxtral (2507) | `voxtral-mini-3b-2507`, `voxtral-small-24b-2507` | translate | [docs/models/voxtral.md](docs/models/voxtral.md) |
| Voxtral Realtime (2602) | `voxtral-mini-4b-realtime-2602` | streaming | [docs/models/voxtral-realtime.md](docs/models/voxtral-realtime.md) |
| Whisper | `breeze-asr-25`, `whisper-base`, `whisper-base.en`, `whisper-large`, `whisper-large-v2`, `whisper-large-v3`, `whisper-large-v3-turbo`, `whisper-medium`, `whisper-medium.en`, `whisper-small`, `whisper-small.en`, `whisper-tiny`, `whisper-tiny.en` | segment timestamps, translate | [docs/models/whisper.md](docs/models/whisper.md) |
<!-- /catalog -->

**Speaker diarization models** (no transcription; verified by DER/JER rather than WER):

<!-- catalog:family-index transcribe=false -->
| Family | Variants | Available capabilities | Docs |
| --- | --- | --- | --- |
| Streaming Sortformer Diarizer 4spk v2.1 | `diar_streaming_sortformer_4spk-v2.1` | diarize, streaming | [docs/models/diar_streaming_sortformer_4spk-v2.1.md](docs/models/diar_streaming_sortformer_4spk-v2.1.md) |
<!-- /catalog -->

Per-variant model cards live under [`docs/models/`](docs/models/).

## Model catalog

[`catalog/`](catalog/) is the source of truth for model metadata, accuracy, and
performance. Each release includes a queryable
[`catalog.db`](https://github.com/handy-computer/transcribe.cpp/releases/latest/download/catalog.db)
and [`SHA-256 checksum`](https://github.com/handy-computer/transcribe.cpp/releases/latest/download/catalog.db.sha256).
Rebuild it locally with `uv run scripts/catalog/db.py --out catalog.db`.
The exact accuracy and published speed matrices and standard benchmark recipes
required for publication live in
[`catalog/_benchmark_profiles.json`](catalog/_benchmark_profiles.json). Run
`uv run scripts/catalog/check.py --publication-profile` to enforce it; the
ordinary catalog check reports the migration backlog without failing.

Published tables are generated from it rather than hand-written. A model doc
delegates a region with a marker, and `scripts/catalog/render.py` rewrites
only what sits between the pair:

```markdown
<!-- catalog:downloads -->
| Quantization | Download | Size | WER (LibriSpeech test-clean) |
...
<!-- /catalog -->
```

The Hugging Face card specs under [`scripts/hf_cards/`](scripts/hf_cards/)
hold editorial copy only (summary, tags, validation pin, prose notes).
`scripts/hf_cards/generate.py` reads the spec and the catalog record together,
so repos, licence, languages, capabilities, the quant table, and per-rig
speedups are never written into a YAML by hand.

```bash
uv run scripts/catalog/format.py --check  # canonical record layout
uv run scripts/catalog/check.py           # schema, integrity, pairing
uv run scripts/catalog/render.py          # rewrite the marked doc regions
uv run scripts/catalog/render.py --check
```

## Build

```bash
cmake -B build
cmake --build build
```

Metal is enabled automatically on Apple Silicon. For Vulkan (Linux/Windows):

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libvulkan-dev glslc libopenblas-dev

# Fedora
sudo dnf install vulkan-headers openblas-devel glslc spirv-headers-devel

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

For HIP/ROCm (Linux + AMD GPU), which needs ROCm 6.1 or newer:

```bash
cmake -B build -DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1201
cmake --build build
```

Replace `gfx1201` with your GPU architecture — `rocminfo | grep gfx` prints it.
Pass a semicolon-separated list for several architectures. ROCm devices report
as the `rocm` backend kind, are picked up by the default `auto` backend, and can
be required explicitly with `--backend rocm`.

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
and kept in sync with the header. Upgrading from 0.1? Read the
[0.2 migration guide](docs/migrating-to-0.2.md), including the new exact-device
selection API and the changed meaning of CLI `--device 0`.

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
ggml/                      Vendored ggml (see ggml/UPSTREAM for its recipe)
patches/ggml/              Downstream patches applied by scripts/sync-ggml.sh
src/third_party/miniz/     Vendored miniz deflate codec (see its UPSTREAM file)
samples/                   Test audio files
tests/                     Unit and smoke tests
```

## License

transcribe.cpp is MIT-licensed. See [LICENSE](LICENSE) for details. Vendored
third-party components (ggml, miniz — both MIT) are attributed in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
