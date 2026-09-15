# Fun-ASR-Nano

<!-- catalog:intro -->
Upstream: [`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) at [`a7088d6`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/commit/a7088d6).

Offline speech-to-text in Chinese, English, and Japanese, plus 7 Chinese
dialects (Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin) and 26 regional
Mandarin accents. ~800M trainable parameters wrapping a frozen
SenseVoiceEncoderSmall (50 SAN-M main blocks + 20 transformer blocks),
a 2-layer audio adaptor (512 → 1024), and a bundled Qwen3-0.6B LLM
(28 layers, 16/8 GQA, BF16) that produces the transcript autoregressively.
Takes a 16 kHz mono WAV and emits text. Not a streaming model, no
translation, no built-in long-form chunking, no timestamps. ITN
(inverse text normalization) is supported by the model and exposed
via the `--itn` CLI flag and `transcribe_funasr_nano_params { use_itn }`
in the library API.
<!-- /catalog -->

## What it's for

Offline speech-to-text in **Chinese, English, and Japanese**, plus 7
Chinese dialects (Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin) and 26
regional Mandarin accents. The model takes a 16 kHz mono WAV and produces
a transcript. Not a streaming model, no translation, no built-in long-form
chunking, no timestamps.

ITN (inverse text normalization — digits, capitalization, punctuation)
is supported by the model. Pass `--itn` on the CLI, or set
`transcribe_funasr_nano_params { use_itn = true }` via the library API.

For multilingual coverage beyond zh/en/ja, see the sibling
[Fun-ASR-MLT-Nano](fun-asr-mlt-nano-2512.md) (31 languages).

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). Ported from upstream commit [`a7088d6`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/commit/a7088d6), pinned 2026-05-06. Validated against the FunASR reference at transcribe.cpp commit [`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28) on 2026-05-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Fun-ASR-Nano-2512-BF16.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-BF16.gguf) | 1.67 GB | 1.78% |
| F16          | [Fun-ASR-Nano-2512-F16.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-F16.gguf) | 1.67 GB | 1.79% |
| Q8_0         | [Fun-ASR-Nano-2512-Q8_0.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q8_0.gguf) |  891 MB | 1.79% |
| Q6_K         | [Fun-ASR-Nano-2512-Q6_K.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q6_K.gguf) |  691 MB | 1.78% |
| Q5_K_M       | [Fun-ASR-Nano-2512-Q5_K_M.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q5_K_M.gguf) |  631 MB | 1.82% |
| Q4_K_M       | [Fun-ASR-Nano-2512-Q4_K_M.gguf](https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q4_K_M.gguf) |  557 MB | 1.92% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy LLM decoding via the bundled Qwen3-0.6B head. Publisher reports 1.76% on this
split (model card "Open-Source Dataset Performance" table). Our FunASR 1.3.1
reference run scores 1.79% (95% CI [1.63%, 1.95%]), within bootstrap noise of the
publisher's number. transcribe.cpp's BF16 port matches that baseline within -0.01
percentage-points. LibriSpeech is an English-only benchmark; Chinese (AISHELL-1,
WenetSpeech) and Japanese (CommonVoice JA) are the recommended complementary checks.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 5.49% |
| ja       | CER    | 8.50% |
| zh       | CER    | 8.59% |
<!-- /catalog -->

LibriSpeech is an English benchmark; Fun-ASR-Nano's strongest case is
Mandarin. **FLEURS-zh** (945 utterances) CER: 8.61% on our FunASR 1.3.1
reference run, 8.59% on the Q8_0 port (95% CI [7.70%, 9.43%]); within
bootstrap noise. Reproduce with
`uv run scripts/wer/run.py --model … --dataset fleurs:zh`; reference run
via `uv run --project scripts/envs/funasr_nano scripts/wer/run_reference_funasr_nano.py`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-Q8_0.gguf \
  --language en \
  samples/jfk.wav
```

Pass `--language zh` / `ja` (or omit for auto-detection) for the other
supported languages. Add `--itn` to render digits, capitalization, and
punctuation in formal form:

```bash
build/bin/transcribe-cli --itn -m … samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 132 ms (83.13×) | 124 ms (88.83×) |
| Metal   | dots (35.3s) | 483 ms (73.13×) | 449 ms (78.65×) |
| CPU     | jfk (11.0s)  | 365 ms (30.12×) | 362 ms (30.40×) |
| CPU     | dots (35.3s) | 1.36 s (26.00×) | 1.32 s (26.86×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 905 ms (12.16×) | 838 ms (13.12×) |
| Vulkan  | dots (35.3s) |  3.80 s (9.30×) | 3.06 s (11.54×) |
| CPU     | jfk (11.0s)  |  1.23 s (8.94×) |  1.15 s (9.60×) |
| CPU     | dots (35.3s) |  5.08 s (6.95×) |  4.66 s (7.59×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models fun-asr-nano-2512
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1 on
`samples/jfk.wav`. All 22 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
("And so my fellow Americans ask not what your country can do for you ask
what you can do for your country.").

| Field | Value |
| --- | --- |
| Reference | FunASR 1.3.1, `FunAudioLLM/Fun-ASR-Nano-2512` (rev `a7088d6`) |
| Dump script | `scripts/dump_reference_funasr_nano_funasr.py` |
| Manifest | `tests/golden/funasr_nano/fun-asr-nano-2512.manifest.json` |
| Tolerances | `tests/tolerances/funasr_nano.json` |
| Command | `uv run scripts/validate.py all --family funasr_nano --variant fun-asr-nano-2512` |

The deeper SAN-M residual blocks (`enc.encoders.48.out`, magnitudes
~10⁴) accumulate fp32 reduction-order drift up to ~1e+2 max abs; the
trailing tier-boundary LayerNorm (`enc.after_norm`, `enc.tp_norm`)
collapses it back to O(0.02). The Qwen3 LM prefill logits drift is
~5e-2 max with F16 KV cache; mid-generation logits (`dec.logits_raw.gen8`)
drift slightly more (~1e-1 max) because the F16 KV cache rounds K/V at
every step write. Argmax decisions agree with the reference at every
step on this manifest.

## Reproduction

### Convert

Loads directly from FunASR's `model.pt` pickle via `funasr.AutoModel`.
The converter extracts both encoder + audio adaptor + Qwen3 LM tensors
in one pass and writes a mixed-precision GGUF (encoder F32, LLM BF16,
norms/biases F32).

```bash
uv run --project scripts/envs/funasr_nano \
  scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-Nano-2512 \
  --repo-id FunAudioLLM/Fun-ASR-Nano-2512 \
  --variant fun-asr-nano-2512
```

### Quantize

Run `transcribe-quantize` once per target quant.

```bash
for Q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf \
    models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-${Q}.gguf \
    --quant ${Q}
done
```

### Validate

```bash
uv run scripts/validate.py all --family funasr_nano --variant fun-asr-nano-2512
```

### Score WER

```bash
# Reference baseline (FunASR; ~80 min on a 12-thread CPU for 2620 utts).
uv run --project scripts/envs/funasr_nano \
  scripts/wer/run_reference_funasr_nano.py \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-nano-2512-REF.test-clean.jsonl \
    --torch-threads 12
uv run scripts/wer/score.py reports/wer/fun-asr-nano-2512-REF.test-clean.jsonl

# transcribe.cpp ports (one preset shown; loop in the family doc).
uv run scripts/wer/run.py \
  --model models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out      reports/wer/fun-asr-nano-2512-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/fun-asr-nano-2512-BF16.test-clean.jsonl
```
