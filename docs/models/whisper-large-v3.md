# Whisper large-v3

<!-- catalog:intro -->
Upstream: [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) at [`06f233f`](https://huggingface.co/openai/whisper-large-v3/commit/06f233f).

OpenAI Whisper large-v3 — converted to GGUF for transcribe.cpp. Multilingual transcription, language detection, and speech translation (audio in any supported language → English text). v3 family adds Cantonese (yue) and uses a 128-bin mel input. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (100 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3) for training data, intended
use, and the original evaluation methodology.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`06f233f`](https://huggingface.co/openai/whisper-large-v3/commit/06f233f), pinned 2026-04-25. Validated against the transformers reference at transcribe.cpp commit [`0a26478`](https://github.com/handy-computer/transcribe.cpp/tree/0a26478) on 2026-09-13.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16          | [whisper-large-v3-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-F16.gguf) | 3.11 GB | 1.81% |
| Q8_0         | [whisper-large-v3-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q8_0.gguf) | 1.67 GB | 1.82% |
| Q6_K         | [whisper-large-v3-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q6_K.gguf) | 1.30 GB | 1.83% |
| Q5_K_M       | [whisper-large-v3-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q5_K_M.gguf) | 1.16 GB | 1.84% |
| Q4_K_M       | [whisper-large-v3-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-gguf/resolve/main/whisper-large-v3-Q4_K_M.gguf) |  997 MB | 1.86% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
OpenAI's self-reported number on the same split is 2.01%. Both are
short-form WER decoded without timestamps; OpenAI does not publish its exact
evaluation configuration, so small differences are expected. Single-run
figures: GPU reductions can shift corpus WER by about 0.1pp between runs,
mostly on short-clip hallucination outcomes at the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  32.43% |
| am       | WER    | 132.70% |
| ar       | WER    |  14.92% |
| as       | WER    | 104.77% |
| az       | WER    |  21.10% |
| be       | WER    |  43.78% |
| bg       | WER    |  12.82% |
| bn       | WER    |  55.03% |
| bs       | WER    |  13.50% |
| ca       | WER    |   4.97% |
| cs       | WER    |  10.50% |
| cy       | WER    |  30.67% |
| da       | WER    |  12.48% |
| de       | WER    |   4.13% |
| el       | WER    |  11.53% |
| en       | WER    |   4.03% |
| es       | WER    |   2.70% |
| et       | WER    |  18.30% |
| fa       | WER    |  30.11% |
| fi       | WER    |   7.73% |
| fil      | WER    |  11.82% |
| fr       | WER    |   5.39% |
| gl       | WER    |  13.27% |
| gu       | WER    |  66.38% |
| ha       | WER    |  85.65% |
| he       | WER    |  26.73% |
| hi       | WER    |  17.06% |
| hr       | WER    |  10.94% |
| hu       | WER    |  13.40% |
| hy       | WER    |  43.64% |
| id       | WER    |   6.08% |
| is       | WER    |  31.85% |
| it       | WER    |   2.54% |
| ja       | CER    |   4.81% |
| jv       | WER    |  64.72% |
| ka       | WER    |  93.97% |
| kk       | WER    |  33.07% |
| km       | CER    | 101.09% |
| kn       | WER    |  31.99% |
| ko       | CER    |   4.89% |
| lb       | WER    |  85.67% |
| ln       | WER    |  72.22% |
| lo       | CER    | 100.28% |
| lt       | WER    |  24.55% |
| lv       | WER    |  19.21% |
| mi       | WER    |  38.78% |
| mk       | WER    |  15.09% |
| ml       | WER    | 100.17% |
| mn       | WER    |  85.60% |
| mr       | WER    |  34.30% |
| ms       | WER    |   7.59% |
| mt       | WER    |  68.79% |
| my       | CER    | 143.39% |
| nb       | WER    |   8.19% |
| ne       | WER    |  40.44% |
| nl       | WER    |   5.42% |
| oc       | WER    |  69.15% |
| pa       | WER    |  57.39% |
| pl       | WER    |   4.69% |
| ps       | WER    |  89.14% |
| pt       | WER    |   3.88% |
| ro       | WER    |   9.20% |
| ru       | WER    |   4.96% |
| sd       | WER    | 184.20% |
| sk       | WER    |   9.25% |
| sl       | WER    |  19.15% |
| sn       | WER    | 115.38% |
| so       | WER    |  91.10% |
| sr       | WER    |  28.49% |
| sv       | WER    |   7.80% |
| sw       | WER    |  34.57% |
| ta       | WER    |  20.04% |
| te       | WER    |  65.84% |
| tg       | WER    |  80.86% |
| th       | CER    |   8.78% |
| tr       | WER    |   6.51% |
| uk       | WER    |   6.28% |
| ur       | WER    |  21.75% |
| uz       | WER    |  86.28% |
| vi       | WER    |   8.74% |
| yo       | WER    |  97.26% |
| yue      | CER    |  22.06% |
| zh       | CER    |   7.98% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 744.8 ms (14.77×) | 616.5 ms (17.84×) |
| Metal   | dots (35.3s) |   1.79 s (19.74×) |   1.58 s (22.41×) |
| CPU     | jfk (11.0s)  |    3.73 s (2.95×) |    4.14 s (2.66×) |
| CPU     | dots (35.3s) |    8.07 s (4.38×) |    8.72 s (4.05×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v3
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  6.60 s (1.67×) |  6.63 s (1.66×) |
| Vulkan  | dots (35.3s) | 15.20 s (2.32×) | 15.05 s (2.35×) |
| CPU     | jfk (11.0s)  | 13.60 s (0.81×) | 12.95 s (0.85×) |
| CPU     | dots (35.3s) | 31.24 s (1.13×) | 28.10 s (1.26×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models whisper-large-v3
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 23 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v3.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v3.json).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v3.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v3.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v3` |

The C++ mel frontend (Slaney filterbank + Hann periodic window +
whisper-style log-mel compression) drives `enc.mel.in` to fp32-vs-fp64
STFT precision drift; downstream tensors stay within budget. KV-cached
decoder runs through F16 self/cross caches by default — flip with
`--kv-type f32` for tighter parity.

## Reproduction

### Convert

The whisper converter loads from a Hugging Face checkpoint and emits a
reference-dtype GGUF.

```bash
uv run --project scripts/envs/whisper \
  scripts/convert-whisper.py openai/whisper-large-v3 \
  --revision 06f233f
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v3/whisper-large-v3-F16.gguf \
  models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v3
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-large-v3/whisper-large-v3-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
