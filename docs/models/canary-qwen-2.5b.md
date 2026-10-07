# Canary-Qwen 2.5B

<!-- catalog:intro -->
Upstream: [`nvidia/canary-qwen-2.5b`](https://huggingface.co/nvidia/canary-qwen-2.5b) at [`b1469e1`](https://huggingface.co/nvidia/canary-qwen-2.5b/commit/b1469e1).

Offline English speech-to-text. NeMo SALM (Speech-Augmented Language
Model): a FastConformer audio encoder (32 layers, `d_model=1024`) feeds
audio embeddings into a Qwen3-1.7B causal LM (28 layers,
`hidden_size=2048`) via audio-token injection at a sentinel position in
the prompt. English only. Takes a 16 kHz mono WAV and produces a
transcript via greedy decoding.
<!-- /catalog -->

## What it's for

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript via greedy decoding. English only; no translation, no
explicit PnC toggle (SALM applies punctuation and capitalization
implicitly when the audio supports it).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-qwen-2.5b)
for training data, intended use, and upstream evaluation.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`b1469e1`](https://huggingface.co/nvidia/canary-qwen-2.5b/commit/b1469e1), pinned 2026-05-15. Validated against the NeMo SALM 2.7.3 reference at transcribe.cpp commit [`6f6c699`](https://github.com/handy-computer/transcribe.cpp/tree/6f6c699) on 2026-05-16.
<!-- /catalog -->

## Input limits

Accepts up to about **54 minutes** of 16 kHz mono audio per call — the
40,960-token Qwen3 decoder context is the binding limit. That ceiling bounds
memory and is far longer than any normal clip; audio past it is rejected up front
with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated. Lowering
`--n-ctx` lowers the limit, and `transcribe_session_get_limits()` reports the
exact per-session value. See the [input-length contract](../input-limits.md).

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [canary-qwen-2.5b-BF16.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-BF16.gguf) | 5.08 GB | 1.63% |
| F16          | [canary-qwen-2.5b-F16.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-F16.gguf) | 5.08 GB | 1.63% |
| Q8_0         | [canary-qwen-2.5b-Q8_0.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q8_0.gguf) | 2.80 GB | 1.63% |
| Q6_K         | [canary-qwen-2.5b-Q6_K.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q6_K.gguf) | 2.21 GB | 1.63% |
| Q5_K_M       | [canary-qwen-2.5b-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q5_K_M.gguf) | 1.98 GB | 1.63% |
| Q4_K_M       | [canary-qwen-2.5b-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q4_K_M.gguf) | 1.74 GB | 1.63% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored with the Whisper-style English text normalizer and jiwer 3.x on an Apple M4.
The same-machine NeMo SALM reference run (CPU torch, dither=0.0, greedy
`model.generate`) lands at **1.61%** with 95% bootstrap CI [1.47%, 1.75%]: `0.01`
above NVIDIA's published 1.60% but well within statistical noise. All six GGUF
presets land at exactly 1.63% (`+0.02` over our reference run, same CI band).
Investigation of the worst per-utterance differences shows scattered token-level
noise consistent with BF16 weight precision (homophones, word-boundary flips,
function-word substitutions). Reproduce with `scripts/wer/run.py` +
`scripts/wer/score.py`.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 3.58% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/canary-qwen-2.5b/canary-qwen-2.5b-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags:

- `-l en` (or omit): English is the only supported language; passing
  any other code returns `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`.
- No `--task`, `--target-language`, or `--pnc` toggles. SALM is
  ASR-only, English-only, and applies PnC implicitly.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 228.6 ms (48.11×) | 204.7 ms (53.73×) |
| Metal   | dots (35.3s) | 961.4 ms (36.75×) | 830.8 ms (42.53×) |
| CPU     | jfk (11.0s)  |   1.02 s (10.83×) |   1.04 s (10.55×) |
| CPU     | dots (35.3s) |    3.83 s (9.22×) |    3.81 s (9.27×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.47 s (4.45×) |  2.15 s (5.12×) |
| Vulkan  | dots (35.3s) | 10.09 s (3.50×) |  8.71 s (4.06×) |
| CPU     | jfk (11.0s)  |  3.87 s (2.84×) |  3.46 s (3.18×) |
| CPU     | dots (35.3s) | 16.24 s (2.18×) | 13.87 s (2.55×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models canary-qwen-2.5b
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo SALM (`nemo.collections.speechlm2.SALM` 2.7.3) on
`samples/jfk.wav` with the strict CPU backend, BF16 weights promoted to F32 at load time. All 16 checkpointed
tensors fall within family tolerance, and the BF16 transcript matches the reference verbatim (`And so my
fellow Americans ask not what your country can do for you ask what you can do for your country`). Tolerances
are pinned in `tests/tolerances/canary_qwen.json` with a detailed `_comment` block naming the precision
regime, the two implementation gotchas (BF16 mel filterbank in NeMo's preprocessor, forced F32 promotion of
F16 depthwise conv kernels on CPU), and the mechanism behind every widened entry.

| Field | Value |
| --- | --- |
| Reference | NeMo SALM 2.7.3 (`nvidia/canary-qwen-2.5b`) |
| Dump script | `scripts/dump_reference_canary_qwen_nemo.py` |
| Manifest | `tests/golden/canary_qwen/canary-qwen-2.5b.manifest.json` |
| Tolerances | `tests/tolerances/canary_qwen.json` |
| Command | `uv run scripts/validate.py all --family canary_qwen --variant canary-qwen-2.5b` |

For the full porting writeup including the SALM trace, the
audio-injection scatter contract, and the BF16-vs-F32 weight precision
investigation, see
[`docs/porting/families/canary_qwen.md`](../porting/families/canary_qwen.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/canary_qwen \
  scripts/convert-canary-qwen.py nvidia/canary-qwen-2.5b \
  --revision b1469e1bba1cfe140205529c79c434ca47180960
```

### Quantize

```bash
uv run scripts/quantize-all.py models/canary-qwen-2.5b/canary-qwen-2.5b-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary_qwen --variant canary-qwen-2.5b
```

### Score WER

```bash
PRESET=BF16
uv run scripts/wer/run.py \
  --model models/canary-qwen-2.5b/canary-qwen-2.5b-${PRESET}.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/canary-qwen-2.5b-${PRESET}.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/canary-qwen-2.5b-${PRESET}.librispeech-test-clean.jsonl
```

### Score WER against the NeMo SALM reference

```bash
uv run --project scripts/envs/canary_qwen \
  scripts/wer/run_reference_canary_qwen_nemo.py \
    --model nvidia/canary-qwen-2.5b \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/canary-qwen-2.5b-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/canary-qwen-2.5b-REF.librispeech-test-clean.jsonl
```
