# Fun-ASR-MLT-Nano

<!-- catalog:intro -->
Upstream: [`FunAudioLLM/Fun-ASR-MLT-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) at [`cf67a93`](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/commit/cf67a93).

Offline speech-to-text covering 31 languages, with focused optimization
on East and Southeast Asian languages: Chinese, English, Cantonese,
Japanese, Korean, Vietnamese, Indonesian, Thai, Malay, Filipino, plus
Arabic, Hindi, and 19 European languages (Bulgarian, Croatian, Czech,
Danish, Dutch, Estonian, Finnish, Greek, Hungarian, Irish, Latvian,
Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian,
Swedish). Same architecture as Fun-ASR-Nano-2512 (~800M trainable
parameters: frozen SenseVoiceEncoderSmall + 2-layer audio adaptor +
bundled Qwen3-0.6B LLM); trained on a smaller multilingual corpus
("hundreds of thousands of hours" per the model card, vs Nano's
"tens of millions"). Takes a 16 kHz mono WAV and emits text. Not
streaming, no translation, no timestamps. ITN (inverse text
normalization) is supported by the model and exposed via the
`--itn` CLI flag and `transcribe_funasr_nano_params { use_itn }`
in the library API.
<!-- /catalog -->

## What it's for

Offline speech-to-text covering **31 languages**, with focused
optimization on East and Southeast Asian languages plus broad European
coverage:

- East / SE Asian: zh, en, **yue, ja, ko, vi, id, th, ms, tl** (10)
- Other Asian / MENA: **ar, hi** (2)
- European: **bg, hr, cs, da, nl, et, fi, el, hu, ga, lv, lt, mt, pl,
  pt, ro, sk, sl, sv** (19)

The model takes a 16 kHz mono WAV and produces a transcript. Not a
streaming model, no translation, no built-in long-form chunking, no
timestamps. The README explicitly lists timestamps, speaker
diarization, and training as upstream TODOs.

ITN (inverse text normalization — digits, capitalization, punctuation)
is supported by the model. Pass `--itn` on the CLI, or set
`transcribe_funasr_nano_params { use_itn = true }` via the library API.

For Mandarin-only / dialect-heavy use, the regular
[Fun-ASR-Nano](fun-asr-nano-2512.md) was trained on a much larger
zh/en/ja corpus and may give better Chinese accuracy.

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed [FunASR Model Open Source License Agreement v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE). Ported from upstream commit [`cf67a93`](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/commit/cf67a93), pinned 2026-05-06. Validated against the FunASR reference at transcribe.cpp commit [`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28) on 2026-05-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Fun-ASR-MLT-Nano-2512-BF16.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-BF16.gguf) | 1.67 GB | 1.74% |
| F16          | [Fun-ASR-MLT-Nano-2512-F16.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-F16.gguf) | 1.67 GB | 1.74% |
| Q8_0         | [Fun-ASR-MLT-Nano-2512-Q8_0.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q8_0.gguf) |  891 MB | 1.74% |
| Q6_K         | [Fun-ASR-MLT-Nano-2512-Q6_K.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q6_K.gguf) |  691 MB | 1.69% |
| Q5_K_M       | [Fun-ASR-MLT-Nano-2512-Q5_K_M.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q5_K_M.gguf) |  631 MB | 1.77% |
| Q4_K_M       | [Fun-ASR-MLT-Nano-2512-Q4_K_M.gguf](https://huggingface.co/handy-computer/Fun-ASR-MLT-Nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q4_K_M.gguf) |  557 MB | 1.89% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy LLM decoding via the bundled Qwen3-0.6B head. The publisher does not report a
numerical LibriSpeech WER for the MLT variant specifically (the shared README's
per-model table covers Fun-ASR-Nano only). Gate baseline is our own FunASR 1.3.1
reference run on the same manifest: 1.76% (95% CI [1.60%, 1.93%]). transcribe.cpp's
BF16 port matches that baseline within -0.02 percentage-points. LibriSpeech is
English only; the strength of the MLT variant is multilingual coverage, not English
accuracy. For the other 30 languages, run your own representative manifest.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |    Q8_0 |
| --- | --- | ---: |
| ar       | WER    |  25.79% |
| bg       | WER    |  84.98% |
| cs       | WER    |  53.56% |
| da       | WER    |  69.93% |
| el       | WER    | 103.55% |
| en       | WER    |   4.90% |
| et       | WER    |  64.22% |
| fi       | WER    |  68.16% |
| fil      | WER    |  15.62% |
| ga       | WER    | 100.08% |
| hi       | WER    |  43.96% |
| hr       | WER    |  61.07% |
| hu       | WER    | 113.21% |
| id       | WER    |   7.52% |
| ja       | CER    |   2.32% |
| ko       | CER    |   5.20% |
| lt       | WER    |  78.42% |
| lv       | WER    |  56.71% |
| ms       | WER    |   9.92% |
| mt       | WER    |  91.71% |
| nl       | WER    |  42.97% |
| pl       | WER    |  59.34% |
| pt       | WER    |  28.24% |
| ro       | WER    |  74.39% |
| sk       | WER    |  64.01% |
| sl       | WER    |  75.22% |
| sv       | WER    |  75.36% |
| th       | CER    |   7.99% |
| vi       | WER    |   8.32% |
| yue      | CER    |  12.72% |
| zh       | CER    |   8.64% |
<!-- /catalog -->

LibriSpeech is English only and is not the strength of this model. For
the other 30 languages, run your own representative manifest. CommonVoice
splits per language are a reasonable starting point.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-Q8_0.gguf \
  --language en \
  samples/jfk.wav
```

Pass `--language ko` / `vi` / `th` / etc. for any of the 31 supported
languages, or omit for auto-detection. Add `--itn` to render digits,
capitalization, and punctuation in formal form:

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
| Metal   | jfk (11.0s)  | 146 ms (75.38×) | 136 ms (80.64×) |
| Metal   | dots (35.3s) | 546 ms (64.69×) | 489 ms (72.22×) |
| CPU     | jfk (11.0s)  | 533 ms (20.65×) | 537 ms (20.50×) |
| CPU     | dots (35.3s) | 1.92 s (18.43×) | 1.93 s (18.28×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |           Q8_0 |          Q4_K_M |
| ------- | ------------ | -------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 1.13 s (9.74×) | 1.04 s (10.61×) |
| Vulkan  | dots (35.3s) | 4.45 s (7.93×) |  4.03 s (8.78×) |
| CPU     | jfk (11.0s)  | 1.79 s (6.16×) |  1.77 s (6.20×) |
| CPU     | dots (35.3s) | 7.40 s (4.78×) |  6.94 s (5.09×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `cd0ea568` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models fun-asr-mlt-nano-2512
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1 on
`samples/jfk.wav`. All 22 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
("and so my fellow americans ask not what your country can do for you ask
what you can do for your country").

| Field | Value |
| --- | --- |
| Reference | FunASR 1.3.1, `FunAudioLLM/Fun-ASR-MLT-Nano-2512` (rev `cf67a93`) |
| Dump script | `scripts/dump_reference_funasr_nano_funasr.py` |
| Manifest | `tests/golden/funasr_nano/fun-asr-mlt-nano-2512.manifest.json` |
| Tolerances | `tests/tolerances/funasr_nano.json` (shared with Fun-ASR-Nano) |
| Command | `uv run scripts/validate.py all --family funasr_nano --variant fun-asr-mlt-nano-2512` |

MLT exposed two minor issues in the family-shared validation regime
that did not surface on Nano:

- **Tolerance widening:** `enc.encoders.48.out` and
  `enc.tp_encoders.0.out` drift on MLT exceeded the Nano-tuned
  family tolerances by 5–46%. Mechanism is BLAS reduction-order
  noise on different per-checkpoint weight magnitudes; widened to
  cover both variants. Drift is spread, not localized.
- **Off-by-one fix:** the C++ `dec.logits_raw.gen8` dump captured
  the 10th lm_head call instead of the 9th to match REF. Masked on
  Nano (adjacent gen-step logits are similar in magnitude), but
  exposed on MLT (the tied-lm_head distribution is heavily shifted,
  REF mean ~13.4 vs Nano's ~0.27). Fixed in
  `src/arch/funasr_nano/model.cpp` — `gen_dump_step 8 → 7`.

Both fixes shipped in commit `f094d28`. See
`tests/tolerances/funasr_nano.json` `_comment` block for the
full mechanism notes.

## Reproduction

### Convert

The converter is the same one used for Fun-ASR-Nano — only `--repo-id`
and `--variant` change. The 31-language list in `general.languages` is
populated automatically from the per-variant table in
`scripts/convert-funasr_nano.py`'s `VARIANT_LANGUAGES`.

```bash
uv run --project scripts/envs/funasr_nano \
  scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
  --revision cf67a938bf2829959d08fdfb84e186eff02a67ff \
  --repo-id FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
  --variant fun-asr-mlt-nano-2512
```

### Quantize

```bash
for Q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf \
    models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-${Q}.gguf \
    --quant ${Q}
done
```

### Validate

```bash
uv run scripts/validate.py all --family funasr_nano --variant fun-asr-mlt-nano-2512
```

### Score WER

```bash
# Reference baseline (FunASR; ~80 min on a 12-thread CPU for 2620 utts).
uv run --project scripts/envs/funasr_nano \
  scripts/wer/run_reference_funasr_nano.py \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-mlt-nano-2512-REF.test-clean.jsonl \
    --model    FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --revision cf67a938bf2829959d08fdfb84e186eff02a67ff \
    --torch-threads 12
uv run scripts/wer/score.py reports/wer/fun-asr-mlt-nano-2512-REF.test-clean.jsonl

# transcribe.cpp ports (one preset shown; loop the rest similarly).
uv run scripts/wer/run.py \
  --model models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out      reports/wer/fun-asr-mlt-nano-2512-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/fun-asr-mlt-nano-2512-BF16.test-clean.jsonl
```
