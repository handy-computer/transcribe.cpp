# Fun-ASR-Nano

Alibaba / FunAudioLLM's [`FunAudioLLM/Fun-ASR-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
ported to transcribe.cpp. ~800M trainable parameters wrapping a frozen
**SenseVoiceEncoderSmall** (50 SAN-M main blocks + 20 transformer blocks),
a 2-layer audio adaptor (512 → 1024), and a bundled **Qwen3-0.6B** LLM
(28 layers, 16/8 GQA, BF16) that produces the transcript autoregressively.

## What it's for

Offline speech-to-text in **Chinese, English, and Japanese**, plus 7
Chinese dialects (Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin) and 26
regional Mandarin accents. The model takes a 16 kHz mono WAV and produces
a transcript. Not a streaming model, no translation, no built-in long-form
chunking, no timestamps.

ITN (inverse text normalization — digits, capitalization, punctuation) is
supported by the model.

For multilingual coverage beyond zh/en/ja, see the sibling
[Fun-ASR-MLT-Nano](fun-asr-mlt-nano-2512.md) (31 languages).

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
for training data, intended use, and upstream evaluation methodology.

Licensed under the **FunASR Model Open Source License Agreement v1.1**
([MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Fun-ASR-Nano-2512-BF16.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-BF16.gguf)     | 1590 MB | 1.78% |
| F16    | [Fun-ASR-Nano-2512-F16.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-F16.gguf)       | 1590 MB | 1.79% |
| Q8_0   | [Fun-ASR-Nano-2512-Q8_0.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q8_0.gguf)     |  850 MB | 1.79% |
| Q6_K   | [Fun-ASR-Nano-2512-Q6_K.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q6_K.gguf)     |  659 MB | 1.78% |
| Q5_K_M | [Fun-ASR-Nano-2512-Q5_K_M.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q5_K_M.gguf) |  602 MB | 1.82% |
| Q4_K_M | [Fun-ASR-Nano-2512-Q4_K_M.gguf](https://huggingface.co/handy-computer/fun-asr-nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q4_K_M.gguf) |  531 MB | 1.92% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy LLM decoding via the bundled Qwen3-0.6B head. Publisher
reports 1.76% on this split (model card "Open-Source Dataset Performance"
table). Our FunASR 1.3.1 reference run scores 1.79% (95% CI [1.63%, 1.95%]),
within bootstrap noise of the publisher's number. transcribe.cpp's BF16
port matches that baseline within -0.01 percentage-points; F16/Q8_0/Q6_K
are numerically indistinguishable.

LibriSpeech is an English benchmark; Fun-ASR-Nano's strongest case is
Mandarin.

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
supported languages. If your audio is not already 16 kHz mono WAV,
convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |   134 ms (82×)  |   129 ms (86×)  |
| Metal   | dots (35.3s) |   486 ms (73×)  |   433 ms (82×)  |
| CPU     | jfk (11.0s)  |   379 ms (29×)  |   358 ms (31×)  |
| CPU     | dots (35.3s) |   1.40 s (25×)  |   1.31 s (27×)  |

macOS 26.4.1, transcribe.cpp `f094d28`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Fun-ASR-Nano-2512 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name fun-asr-nano-2512-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1
on `samples/jfk.wav`. All 22 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
("And so my fellow Americans ask not what your country can do for you ask
what you can do for your country."). Last validated at commit
[`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28).

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
