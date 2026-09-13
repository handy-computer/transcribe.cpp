# Whisper large-v3-turbo

<!-- catalog:intro -->
Upstream: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) at [`41f01f3`](https://huggingface.co/openai/whisper-large-v3-turbo/commit/41f01f3).

OpenAI Whisper large-v3-turbo — converted to GGUF for transcribe.cpp. Multilingual transcription and language detection; unlike the full large-v3 model, this turbo variant does not support speech translation. The v3 family adds Cantonese (yue) and uses a 128-bin mel input. Encoder-decoder transformer; 30-second windows with chunked long-form decoding.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text and any-language → English speech translation. The model auto-detects the audio's language (100 languages covered) and emits a transcript in that language; passing `language="<code>"` and `task="translate"` to the underlying `whisper_full_params` produces an English translation instead. `transcribe-cli` reads a 16 kHz mono WAV and returns the transcript text. Long audio is handled via 30-second chunked decoding. v3 family adds Cantonese (yue) on top of v2's 99 languages and switches to a 128-bin mel input.

See the [upstream model card](https://huggingface.co/openai/whisper-large-v3-turbo) for training data, intended
use, and the original evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`41f01f3`](https://huggingface.co/openai/whisper-large-v3-turbo/commit/41f01f3),
pinned 2026-04-25. Validated against the transformers reference at
transcribe.cpp commit
[`5.6.1`](https://github.com/handy-computer/transcribe.cpp/tree/5.6.1)
on 2026-04-26.

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F16          | [whisper-large-v3-turbo-F16.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-F16.gguf) | 1.63 GB | 2.01% |
| Q8_0         | [whisper-large-v3-turbo-Q8_0.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf) |  886 MB | 2.01% |
| Q6_K         | [whisper-large-v3-turbo-Q6_K.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q6_K.gguf) |  693 MB | 2.01% |
| Q5_K_M       | [whisper-large-v3-turbo-Q5_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q5_K_M.gguf) |  620 MB | 2.03% |
| Q4_K_M       | [whisper-large-v3-turbo-Q4_K_M.gguf](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q4_K_M.gguf) |  536 MB | 2.04% |
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances) with the transcribe.cpp default decode (greedy, suppress_tokens, temperature fallback, segment timestamps enabled). OpenAI's self-reported number on the same split is 2.10%. We don't know upstream's exact eval config, but the most likely cause of any divergence is that OpenAI's `model.generate()` defaults to `<|notimestamps|>` while transcribe.cpp's pipeline runs with timestamps enabled. Numbers come from a single Metal-backed run; Metal's non-deterministic parallel reductions can shift corpus WER by ~0.1pp between runs, mostly driven by short-clip hallucination outcomes on the noise floor.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| af       | WER    |  36.06% |
| am       | WER    | 146.29% |
| ar       | WER    |  15.48% |
| as       | WER    | 101.22% |
| az       | WER    |  23.15% |
| be       | WER    |  50.65% |
| bg       | WER    |  13.58% |
| bn       | WER    |  67.53% |
| bs       | WER    |  14.77% |
| ca       | WER    |   5.42% |
| cs       | WER    |  11.81% |
| cy       | WER    |  36.42% |
| da       | WER    |  13.60% |
| de       | WER    |   4.54% |
| el       | WER    |  13.26% |
| en       | WER    |   4.38% |
| es       | WER    |   3.12% |
| et       | WER    |  18.44% |
| fa       | WER    |  30.56% |
| fi       | WER    |   8.29% |
| fil      | WER    |  12.08% |
| fr       | WER    |   5.51% |
| gl       | WER    |  12.76% |
| gu       | WER    |  78.95% |
| ha       | WER    |  97.24% |
| he       | WER    |  29.71% |
| hi       | WER    |  18.85% |
| hr       | WER    |  12.54% |
| hu       | WER    |  15.07% |
| hy       | WER    |  45.62% |
| id       | WER    |   7.20% |
| is       | WER    |  21.39% |
| it       | WER    |   2.77% |
| ja       | CER    |   4.82% |
| ka       | WER    | 109.21% |
| kk       | WER    |  21.27% |
| km       | CER    |  95.20% |
| kn       | WER    |  32.57% |
| ko       | CER    |   5.24% |
| lb       | WER    |  87.21% |
| ln       | WER    |  75.39% |
| lo       | CER    | 115.41% |
| lt       | WER    |  25.11% |
| lv       | WER    |  19.53% |
| mi       | WER    |  48.91% |
| mk       | WER    |  17.85% |
| ml       | WER    |  98.75% |
| mn       | WER    | 101.49% |
| mr       | WER    |  36.12% |
| ms       | WER    |   8.64% |
| mt       | WER    |  70.92% |
| my       | CER    | 121.67% |
| nb       | WER    |   9.10% |
| ne       | WER    |  43.15% |
| nl       | WER    |   5.98% |
| oc       | WER    |  70.94% |
| pa       | WER    |  99.53% |
| pl       | WER    |   5.81% |
| ps       | WER    |  91.81% |
| pt       | WER    |   4.17% |
| ro       | WER    |  10.90% |
| ru       | WER    |   5.93% |
| sd       | WER    | 122.10% |
| sk       | WER    |  10.21% |
| sl       | WER    |  20.56% |
| sn       | WER    | 110.94% |
| so       | WER    | 101.29% |
| sr       | WER    |  32.36% |
| sv       | WER    |   8.72% |
| sw       | WER    |  33.96% |
| ta       | WER    |  27.41% |
| te       | WER    |  63.03% |
| tg       | WER    | 106.06% |
| th       | CER    |  13.15% |
| tr       | WER    |   6.97% |
| uk       | WER    |   7.31% |
| ur       | WER    |  23.19% |
| uz       | WER    | 102.52% |
| vi       | WER    |   9.48% |
| yo       | WER    |  99.38% |
| yue      | CER    |  34.62% |
| zh       | CER    |   8.50% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are compute latency (mel + encode + decode, mean over the recorded
iterations after warmup), with speedup over realtime in parentheses. Units:
`ms` below 1 s, `s` above (2 decimal places). Decode latency dominates as
model size grows; the encoder is only run once per 30-second window.

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 288.3 ms (38.16×) | 288.9 ms (38.07×) |
| Metal   | dots (35.3s) | 649.5 ms (54.40×) | 666.0 ms (53.05×) |
| CPU     | jfk (11.0s)  |    7.60 s (1.45×) |    5.89 s (1.87×) |
| CPU     | dots (35.3s) |   15.34 s (2.30×) |   11.87 s (2.98×) |
<!-- /catalog -->

macOS 26.4.1, transcribe.cpp `e0fa0f6`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3-turbo \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-turbo-publication
```

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  4.77 s (2.31×) |  4.92 s (2.24×) |
| Vulkan  | dots (35.3s) | 10.16 s (3.48×) | 10.26 s (3.44×) |
| CPU     | jfk (11.0s)  | 19.85 s (0.55×) | 15.74 s (0.70×) |
| CPU     | dots (35.3s) | 40.18 s (0.88×) | 32.22 s (1.10×) |
<!-- /catalog -->

Fedora 43, transcribe.cpp `2ab01b8`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models whisper-large-v3-turbo \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends cpu,vulkan \
  --iters 3 --warmup 1 \
  --name whisper-large-v3-turbo-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the transformers reference (`WhisperForConditionalGeneration`, fp32 CPU) on the manifest's case (`samples/jfk.wav`). All 22 checkpointed tensors fall within per-variant tolerance, and the transcript matches the HF reference verbatim. Tolerance budget lives at
[`tests/tolerances/whisper-large-v3-turbo.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/whisper-large-v3-turbo.json). Last validated at commit [`1854f57`](https://github.com/handy-computer/transcribe.cpp/tree/1854f57).

| Field | Value |
| --- | --- |
| Reference | transformers 5.6.1 (`WhisperForConditionalGeneration`, CPU fp32) |
| Manifest | `tests/golden/whisper/whisper-large-v3-turbo.manifest.json` |
| Tolerance file | `tests/tolerances/whisper-large-v3-turbo.json` |
| Command | `uv run scripts/validate.py all --family whisper --variant whisper-large-v3-turbo` |

Selected tensors (worst observed across cases; see tolerance file for per-tensor budgets):

| Tensor                 | Max abs diff | Mean abs diff | Notes |
| ---------------------- | ---: | ---: | --- |
| `enc.mel.in`           |  `2.229e-05` |   `4.055e-08` | fp32 mixed-radix FFT vs torch fp64 frontend |
| `enc.conv1.out`        |  `1.258e-03` |   `4.264e-05` | fp32 conv stem |
| `enc.conv2.out`        |  `2.696e-03` |   `5.033e-05` | stride-2 conv stem (matches enc.embed.out) |
| `enc.block.0.out`      |  `1.439e-02` |   `1.334e-03` | first encoder block |
| `enc.block.31.out`     |  `2.335e+00` |   `3.895e-03` | final encoder block (peak signal grows with depth) |
| `enc.final`            |  `1.343e+00` |   `7.386e-03` | post-LN encoder output |
| `dec.token_emb`        |  `0.000e+00` |   `0.000e+00` | exact zero-drift (`ggml_get_rows` on the F32 GGUF) |
| `dec.block.0.out`      |  `8.217e-03` |   `1.484e-04` | first decoder block, prompt pass |
| `dec.block.3.out`      |  `1.453e-02` |   `9.290e-04` | final decoder block (accumulated) |
| `dec.out_before_head`  |  `2.122e-01` |   `2.041e-02` | post final LN, pre-vocab projection |
| `dec.logits_raw`       |  `1.462e-01` |   `2.968e-02` | vocab projection (raw logits) |
| `dec.logits`           |  `1.771e-01` |   `3.328e-02` | log-softmax over vocab |
| `dec.logits_raw.gen20` |  `3.806e-02` |   `6.097e-03` | step-20 logits (KV-cached path) |

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
  scripts/convert-whisper.py openai/whisper-large-v3-turbo \
  --revision 41f01f3
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0;
repeat for the other shipped presets:

```bash
build/bin/transcribe-quantize \
  models/whisper-large-v3-turbo/whisper-large-v3-turbo-F16.gguf \
  models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family whisper --variant whisper-large-v3-turbo
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_WHISPER_GGUF=$PWD/models/whisper-large-v3-turbo/whisper-large-v3-turbo-Q8_0.gguf \
  ctest --test-dir build --output-on-failure -R whisper
```
