# SenseVoice Small

Alibaba / FunAudioLLM's [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
ported to transcribe.cpp. A 234M-parameter SAN-M encoder with a single CTC
head over a 25,055-token SentencePiece vocabulary covering Chinese, Cantonese,
English, Japanese, and Korean.

## What it's for

Offline multilingual speech-to-text in zh / yue / en / ja / ko. The model takes
a 16 kHz mono WAV (capped at **30 seconds per call**) and produces a transcript.
It is not a streaming model and does not translate.
Long-form audio is the caller's responsibility.

The same CTC head also emits language ID, simple emotion labels (`<|HAPPY|>`,
`<|NEUTRAL|>`, `<|SAD|>`, `<|ANGRY|>`, `<|EMO_UNKNOWN|>`), audio-event tags
(`<|Speech|>`, `<|BGM|>`, `<|Applause|>`, …), and an inverse-text-normalization
flag (`<|withitn|>` / `<|woitn|>`). These are stripped from the transcript by
default; pass `--raw-tokens` to keep them, and `--itn` to enable ITN.

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
for training data, intended use, and upstream evaluation methodology.

Licensed under the **FunASR Model Open Source License Agreement** —
the legacy "model-license" form
([MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)).

## Input limits

SenseVoice runs on short segments — up to about **30 seconds** per call (the
window its upstream pipeline feeds via VAD). Longer audio is accepted, but the
library logs a `WARN` and accuracy may degrade; it is not rejected. Segment long
recordings (e.g. with VAD) for best results. See the
[input-length contract](../input-limits.md).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32    | [SenseVoiceSmall-F32.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-F32.gguf)       | 893 MB | 3.13% |
| F16    | [SenseVoiceSmall-F16.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-F16.gguf)       | 449 MB | 3.13% |
| Q8_0   | [SenseVoiceSmall-Q8_0.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf)     | 241 MB | 3.13% |
| Q6_K   | [SenseVoiceSmall-Q6_K.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q6_K.gguf)     | 187 MB | 3.14% |
| Q5_K_M | [SenseVoiceSmall-Q5_K_M.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q5_K_M.gguf) | 164 MB | 3.18% |
| Q4_K_M | [SenseVoiceSmall-Q4_K_M.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q4_K_M.gguf) | 139 MB | 3.45% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
with greedy CTC decoding. The publisher does not report a numerical
LibriSpeech WER, so the
gate baseline is **our own FunASR 1.3.1 reference run** on the same manifest:
3.13% (95% CI [2.93%, 3.34%]). transcribe.cpp's F32 port matches that
baseline within +0.002 percentage-points. Q4_K_M is the only quant with a
visible regression (+0.32 pp); F16 / Q8_0 / Q6_K / Q5_K_M are within
bootstrap noise of F32.

LibriSpeech is an English benchmark; SenseVoice's strongest case is
Mandarin. **FLEURS-zh** (945 utterances) CER: 10.20% on our FunASR 1.3.1
reference run, 10.11% on the Q8_0 port (95% CI [9.18%, 11.02%]); within
bootstrap noise. Reproduce with
`uv run scripts/wer/run.py --model … --dataset fleurs:zh`; reference run
via `uv run --project scripts/envs/sensevoice scripts/wer/run_reference_sensevoice.py`.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/SenseVoiceSmall/SenseVoiceSmall-Q8_0.gguf \
  --language en \
  samples/jfk.wav
```

Pass `--language zh` / `yue` / `ja` / `ko` (or omit for auto-detection) for
the other supported languages. Raw control tokens and ITN are opt-in:

```bash
# Keep <|en|><|HAPPY|><|Speech|><|woitn|>… in the output text:
build/bin/transcribe-cli --raw-tokens -m … samples/jfk.wav

# Render numbers/punctuation in formal form:
build/bin/transcribe-cli --itn -m … samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

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
| Metal   | jfk (11.0s)  |    42 ms (260×) |    44 ms (250×) |
| Metal   | dots (35.3s) |   111 ms (319×) |   137 ms (258×) |
| CPU     | jfk (11.0s)  |   208 ms (53×)  |   213 ms (52×)  |
| CPU     | dots (35.3s) |   700 ms (50×)  |   727 ms (49×)  |

macOS 26.4.1, transcribe.cpp `811fe2a`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  329 ms (33×)  |  332 ms (33×)  |
| Vulkan  | dots (35.3s) | 1.11 s (32×)   | 1.12 s (31×)   |
| CPU     | jfk (11.0s)  |  687 ms (16×)  |  590 ms (19×)  |
| CPU     | dots (35.3s) | 2.31 s (15×)   | 2.03 s (17×)   |

Fedora 43, transcribe.cpp `8635bd1`. Vulkan device: `AMD Radeon Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models SenseVoiceSmall \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name sensevoice-small-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1
on `samples/jfk.wav`. All 16 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
(both spelled `… laled out …` on token `1089-134686-0000` — a quirk of
SenseVoice, not a port defect). Last validated at commit
[`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28).

| Field | Value |
| --- | --- |
| Reference | FunASR 1.3.1, `FunAudioLLM/SenseVoiceSmall` (rev `3eb3b4e`) |
| Dump script | `scripts/dump_reference_sensevoice_funasr.py` |
| Manifest | `tests/golden/sensevoice/sensevoice-small.manifest.json` |
| Command | `uv run scripts/validate.py compare --family sensevoice --variant sensevoice-small` |

Selected tensors:

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `frontend.fbank.lfr.cmvn.out` | `3.13e-03` | `6.34e-04` | fp32 FFT vs C++ fp64 STFT round-off |
| `enc.input.with_prefix`       | `3.13e-03` | `6.20e-04` | frontend drift carried by concat (no compute) |
| `enc.embed.out`               | `7.08e-02` | `1.40e-02` | frontend drift × √d_model after sinusoidal PE |
| `enc.encoders0.0.out`         | `1.53e+02` | `2.52e+00` | first SAN-M block (560→512 projection) |
| `enc.encoders.0.out`          | `9.63e+01` | `2.78e+00` | main-tier block 0 |
| `enc.encoders.24.out`         | `4.60e+02` | `1.41e+01` | mid-tier (block 24); reference values ~4.5k |
| `enc.encoders.48.out`         | `4.52e+04` | `7.74e+01` | last main block; reference values ~46k |
| `enc.after_norm.out`          | `6.66e+00` | `4.71e-01` | tier-boundary LayerNorm renormalises |
| `enc.tp_encoders.{0,10,19}.out` | `≤ 1.10e+04` | `≤ 9.07e+00` | tp-tier 20-block stack |
| `enc.tp_norm.out`             | `1.53e+01` | `5.86e-01` | final encoder output, post-LN |
| `ctc.logits.raw`              | `3.23e+01` | `1.87e+00` | CTC logits — argmax positions identical |
| `ctc.log_probs`               | `3.07e+01` | `2.90e+00` | log-softmax CTC distribution |

The expected divergence is fp32 reduction-order drift accumulated through
70 SAN-M blocks. SenseVoice's encoder has **no inter-layer normalization**
(only `after_norm` between the two tiers and `tp_norm` at the end), so
absolute magnitudes grow ~170× through the main-tier stack — and so does
the absolute drift. After `tp_norm` re-renormalises, the final output is
within ~0.6 mean / 15 max. Both reference and C++ produce argmax-equivalent
CTC outputs; the transcript is a verbatim match.

## Reproduction

### Convert

Loads directly from FunASR's `model.pt` pickle via `funasr.AutoModel`.

```bash
uv run --project scripts/envs/sensevoice \
  scripts/convert-sensevoice.py FunAudioLLM/SenseVoiceSmall
```

### Quantize

Run `transcribe-quantize` once per target quant.

```bash
for Q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/SenseVoiceSmall/SenseVoiceSmall-F32.gguf \
    models/SenseVoiceSmall/SenseVoiceSmall-${Q}.gguf \
    --quant ${Q}
done
```

### Validate

```bash
uv run scripts/validate.py all --family sensevoice --variant sensevoice-small
```

### Score WER

```bash
# Reference baseline (FunASR; ~25 min on a single CPU thread for 2620 utts).
uv run --project scripts/envs/sensevoice \
  scripts/wer/run_reference_sensevoice.py \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/sensevoice-small-REF.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/sensevoice-small-REF.test-clean.jsonl

# transcribe.cpp ports (one preset shown; loop in the family doc).
uv run scripts/wer/run.py \
  --model models/SenseVoiceSmall/SenseVoiceSmall-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out      reports/wer/sensevoice-small-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/sensevoice-small-F32.test-clean.jsonl
```
