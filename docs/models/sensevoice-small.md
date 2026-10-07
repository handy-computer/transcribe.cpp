# SenseVoice Small

<!-- catalog:intro -->
Upstream: [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) at [`3eb3b4e`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/commit/3eb3b4e).

Offline multilingual speech-to-text in Chinese, Cantonese, English, Japanese,
and Korean. A SAN-M encoder with a single CTC head over a 25,055-token
SentencePiece vocabulary. Takes a 16 kHz mono WAV (capped at 30 seconds per
call, per upstream's direct-inference contract) and produces a transcript. Not
a streaming model, no translation, no built-in long-form chunking. The same CTC
head also emits language-ID, simple emotion labels, audio-event tags, and
inverse-text-normalization control tags. These tags are hidden unless
`--raw-tokens` is passed. ITN is on by default for readable casing,
punctuation, and digits; pass `--no-itn` for upstream's spoken-form output.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text in zh / yue / en / ja / ko. The model takes
a 16 kHz mono WAV (capped at **30 seconds per call**) and produces a transcript.
It is not a streaming model and does not translate.
Long-form audio is the caller's responsibility.

The same CTC head also emits language ID, simple emotion labels (`<|HAPPY|>`,
`<|NEUTRAL|>`, `<|SAD|>`, `<|ANGRY|>`, `<|EMO_UNKNOWN|>`), audio-event tags
(`<|Speech|>`, `<|BGM|>`, `<|Applause|>`, …), and an inverse-text-normalization
flag (`<|withitn|>` / `<|woitn|>`). These are stripped from the transcript by
default; pass `--raw-tokens` to keep them.

**ITN is on by default.** SenseVoice has no separate punctuation/capitalization
control — the ITN flag is what produces casing, punctuation, and digits — so
transcribe.cpp resolves the run-time default to on rather than following
upstream's `itn=False`. Pass `--no-itn` (library: `itn = TRANSCRIBE_ITN_MODE_OFF`)
for upstream's verbatim spoken form.

ITN changes the CTC decode, not just the rendering, so it moves accuracy — and
the direction depends on the language:

- **English costs a little.** LibriSpeech test-clean (512 utts, F32/CPU):
  **+0.110pp WER** (2.556% → 2.666%), diffuse single-word corruption
  (`arcadian` → `arrcadian`). This is upstream behavior, not a port artifact —
  the FunASR 1.3.1 reference shows a *larger* penalty on the same data
  (+0.147pp) and mangles the same words byte-for-byte. The cost is inside the
  table's bootstrap CI.
- **Chinese gains a lot.** FLEURS-zh (945 utts, F32/CPU): **−2.030pp CER**
  (10.100% → 8.070%). The FLEURS-zh reference is digit-normalized, so ITN-on's
  `2011年8月` matches it where ITN-off's `二零一一年八月` does not. This is a
  scoring-convention match, not measured evidence of better recognition.

Readable output is judged the better default for interactive use; `--no-itn` is
the right choice for a pipeline that scores or post-processes text. The WER
numbers below are measured with ITN **off**, matching the reference runs; see
[WER methodology](../tools/wer.md).

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed [model-license (FunASR MODEL_LICENSE)](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). Ported from upstream commit [`3eb3b4e`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/commit/3eb3b4e), pinned 2026-05-06. Validated against the FunASR reference at transcribe.cpp commit [`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28) on 2026-05-06.
<!-- /catalog -->

This is the legacy `model-license` form of the agreement.

## Input limits

SenseVoice runs on short segments — up to about **30 seconds** per call (the
window its upstream pipeline feeds via VAD). Longer audio is accepted, but the
library logs a `WARN` and accuracy may degrade; it is not rejected. Segment long
recordings (e.g. with VAD) for best results. See the
[input-length contract](../input-limits.md).

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [SenseVoiceSmall-F32.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-F32.gguf) | 937 MB | 3.13% |
| F16          | [SenseVoiceSmall-F16.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-F16.gguf) | 470 MB | 3.13% |
| Q8_0         | [SenseVoiceSmall-Q8_0.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf) | 253 MB | 3.13% |
| Q6_K         | [SenseVoiceSmall-Q6_K.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q6_K.gguf) | 196 MB | 3.14% |
| Q5_K_M       | [SenseVoiceSmall-Q5_K_M.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q5_K_M.gguf) | 172 MB | 3.18% |
| Q4_K_M       | [SenseVoiceSmall-Q4_K_M.gguf](https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q4_K_M.gguf) | 146 MB | 3.45% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy CTC decoding. The publisher does not report a numerical LibriSpeech WER (the
model card publishes scores only as PNG figures), so the gate baseline is our own
FunASR 1.3.1 reference run on the same manifest: 3.13% (95% CI [2.93%, 3.34%]).
transcribe.cpp's F32 port matches that baseline within +0.002 percentage-points.
LibriSpeech is an English benchmark; SenseVoice's strongest case is Mandarin, and
AISHELL-1 (CER) is the recommended complementary check. These table values were
measured with ITN off, matching the FunASR reference; `scripts/wer/run.py` pins
`--no-itn` so the benchmark does not inherit the runtime default.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| en       | WER    |  7.14% |
| ja       | CER    |  7.63% |
| ko       | CER    |  8.27% |
| yue      | CER    | 37.44% |
| zh       | CER    | 10.12% |
<!-- /catalog -->

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
the other supported languages.

```bash
# Keep <|en|><|HAPPY|><|Speech|><|withitn|>… in the output text:
build/bin/transcribe-cli --raw-tokens -m … samples/jfk.wav

# Default (ITN on):
#   And so my fellow Americans ask not what your country can do for you, ask
#   what you can do for your country.
build/bin/transcribe-cli -m … samples/jfk.wav

# Upstream spoken form — lowercase, unpunctuated, numbers as words:
#   and so my fellow americans ask not what your country can do for you ask
#   what you can do for your country
build/bin/transcribe-cli --no-itn -m … samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  45 ms (245.28×) |  46 ms (238.72×) |
| Metal   | dots (35.3s) | 122 ms (288.69×) | 123 ms (286.71×) |
| CPU     | jfk (11.0s)  |  215 ms (51.15×) |  220 ms (49.92×) |
| CPU     | dots (35.3s) |  737 ms (47.92×) |  757 ms (46.69×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 395 ms (27.85×) | 398 ms (27.63×) |
| Vulkan  | dots (35.3s) | 1.28 s (27.53×) | 1.30 s (27.27×) |
| CPU     | jfk (11.0s)  | 531 ms (20.71×) | 627 ms (17.55×) |
| CPU     | dots (35.3s) | 1.88 s (18.76×) | 2.15 s (16.40×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models sensevoice-small
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1 on
`samples/jfk.wav`. All 16 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
(both spelled `… laled out …` on token `1089-134686-0000` — a quirk of
SenseVoice, not a port defect).

| Field | Value |
| --- | --- |
| Reference | FunASR 1.3.1, `FunAudioLLM/SenseVoiceSmall` (rev `3eb3b4e`) |
| Dump script | `scripts/dump_reference_sensevoice_funasr.py` |
| Manifest | `tests/golden/sensevoice/sensevoice-small.manifest.json` |
| Command | `uv run scripts/validate.py compare --family sensevoice --variant sensevoice-small` |

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
