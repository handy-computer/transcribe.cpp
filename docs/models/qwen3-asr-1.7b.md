# Qwen3-ASR 1.7B

Alibaba's [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
ported to transcribe.cpp. Architecture is the same audio-LLM pattern as
the 0.6B variant (24-layer bidirectional audio encoder + Qwen3 causal LM
with audio-token injection); the 1.7B is wider: encoder `d_model=1024`
(16 heads), LM `hidden_size=2048`, `intermediate_size=6144`.

## What it's for

Same contract as the 0.6B: offline multilingual STT, 30-language
auto-detect, 16 kHz mono WAV in → transcript text out. Targets the same
use cases as Qwen3-ASR-0.6B with more parameters for accuracy headroom.

See the
[Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
for training data and upstream evaluation.

Licensed Apache-2.0 (weights) / Apache-2.0 (author `qwen_asr` package).
Ported from upstream commit
[`7278e1e`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/commit/7278e1e70fe206f11671096ffdd38061171dd6e5).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Qwen3-ASR-1.7B-BF16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-BF16.gguf)     | 3894 MB | 1.62% |
| F16    | [Qwen3-ASR-1.7B-F16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-F16.gguf)       | 3902 MB | 1.62% |
| Q8_0   | [Qwen3-ASR-1.7B-Q8_0.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q8_0.gguf)     | 2084 MB | 1.61% |
| Q6_K   | [Qwen3-ASR-1.7B-Q6_K.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q6_K.gguf)     | 1614 MB | 1.65% |
| Q5_K_M | [Qwen3-ASR-1.7B-Q5_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q5_K_M.gguf) | 1396 MB | 1.67% |
| Q4_K_M | [Qwen3-ASR-1.7B-Q4_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q4_K_M.gguf) | 1191 MB | 1.84% |

WER measured on LibriSpeech `test-clean` (2620 utterances), Whisper-style
English text normalizer, jiwer 3.x, metal backend on Apple M4. Reproduce
with `scripts/wer/run.py` + `scripts/wer/score.py`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Public API caveat — language hints

Same contract as the 0.6B: `params.language == NULL` runs auto-detect;
any explicit hint returns `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`. See the
0.6B doc and the family note at `docs/porting/families/qwen3_asr.md` for
the rationale and the planned follow-up.

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  256 ms (43×)  |  213 ms (52×)  |
| Metal   | dots (35.3s) |  978 ms (36×)  |  803 ms (44×)  |
| CPU     | jfk (11.0s)  | 1.40 s (8×)    | 1.10 s (10×)   |
| CPU     | dots (35.3s) | 4.46 s (8×)    | 4.04 s (9×)    |

macOS 26.3.1, transcribe.cpp `0c88a71`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  | 2.66 s (4.1×)  | 2.29 s (4.8×)  |
| Vulkan  | dots (35.3s) | 9.87 s (3.6×)  | 8.37 s (4.2×)  |
| CPU     | jfk (11.0s)  | 5.19 s (2.1×)  | 3.58 s (3.1×)  |
| CPU     | dots (35.3s) | 18.53 s (1.9×) | 12.93 s (2.7×) |

Fedora 43, transcribe.cpp `3d16f74`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Qwen3-ASR-1.7B \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name qwen3-asr-1.7b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the author
reference implementation (`qwen_asr` 0.0.6 / transformers 4.57.6) on
`samples/jfk.wav`. All 13 checkpointed tensors fall within family
tolerance, and the transcript matches the reference verbatim.
Tolerances are set in `tests/tolerances/qwen3_asr-1.7b.json`. Last
validated at commit
[`3f61df7`](https://github.com/handy-computer/transcribe.cpp/tree/3f61df7).

| Field | Value |
| --- | --- |
| Reference | `qwen_asr` 0.0.6 (Alibaba author package) |
| Dump script | `scripts/dump_reference_qwen3_asr_author.py` |
| Manifest | `tests/golden/qwen3_asr/qwen3-asr-1.7b.manifest.json` |
| Command | `uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-1.7b` |

Selected tensors (observed on CPU; see tolerance file for budgets):

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`         | `3.906e-03` | `6.608e-04` | fp64 vs fp32 STFT precision gap |
| `enc.subsample.out`  | `2.574e-02` | `2.473e-03` | After the 4× conv subsampler |
| `enc.block.0.out`    | `4.970e-02` | `3.068e-03` | Early encoder |
| `enc.block.23.out`   | `6.846e-01` | `1.125e-02` | Final encoder block |
| `enc.proj.out`       | `2.351e-03` | `1.595e-04` | Audio→LM width projection |
| `dec.audio_injected` | `2.351e-03` | `1.443e-04` | Fused audio+text sequence |
| `dec.token_emb`      | `0.000e+00` | `0.000e+00` | Exact match |
| `dec.block.0.out`    | `1.439e-01` | `3.223e-03` | Early LM |
| `dec.block.27.out`   | `2.052e+02` | `9.783e-01` | Final LM block (accumulated) |
| `dec.out_before_head`| `1.828e+01` | `7.569e-02` | Pre-head hidden state |
| `dec.logits_raw`     | `1.266e+00` | `2.131e-01` | Raw logits |

## Reproduction

### Convert

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-Q4_K_M.gguf \
  --quant Q4_K_M
```

### Validate

```bash
uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-1.7b
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_QWEN3_ASR_1_7B_GGUF=$PWD/models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
TRANSCRIBE_QWEN3_ASR_GGUF=$PWD/models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
  ctest --test-dir build --output-on-failure -R 'qwen3_asr'
```
