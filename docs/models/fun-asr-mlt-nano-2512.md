# Fun-ASR-MLT-Nano

Alibaba / FunAudioLLM's [`FunAudioLLM/Fun-ASR-MLT-Nano-2512`](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
ported to transcribe.cpp — the multilingual sibling of
[Fun-ASR-Nano](fun-asr-nano-2512.md). Identical architecture
(SenseVoiceEncoderSmall + 2-layer audio adaptor + bundled Qwen3-0.6B LLM,
~800M trainable parameters), trained on a smaller corpus
("hundreds of thousands of hours" per the model card, vs Nano's
"tens of millions") with broad multilingual coverage instead of
Mandarin-dialect depth.

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

ITN (inverse text normalization — digits, capitalization, punctuation) is
supported by the model.

For Mandarin-only / dialect-heavy use, the regular
[Fun-ASR-Nano](fun-asr-nano-2512.md) was trained on a much larger
zh/en/ja corpus and may give better Chinese accuracy.

See FunAudioLLM's [model card](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
for training data, intended use, and upstream evaluation methodology.

Licensed under the **FunASR Model Open Source License Agreement v1.1**
([MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Fun-ASR-MLT-Nano-2512-BF16.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-BF16.gguf)     | 1590 MB | 1.74% |
| F16    | [Fun-ASR-MLT-Nano-2512-F16.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-F16.gguf)       | 1590 MB | 1.74% |
| Q8_0   | [Fun-ASR-MLT-Nano-2512-Q8_0.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q8_0.gguf)     |  850 MB | 1.74% |
| Q6_K   | [Fun-ASR-MLT-Nano-2512-Q6_K.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q6_K.gguf)     |  659 MB | 1.69% |
| Q5_K_M | [Fun-ASR-MLT-Nano-2512-Q5_K_M.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q5_K_M.gguf) |  602 MB | 1.77% |
| Q4_K_M | [Fun-ASR-MLT-Nano-2512-Q4_K_M.gguf](https://huggingface.co/handy-computer/fun-asr-mlt-nano-2512-gguf/resolve/main/Fun-ASR-MLT-Nano-2512-Q4_K_M.gguf) |  531 MB | 1.89% |

WER is measured on the full LibriSpeech test-clean split (2620
utterances) with greedy LLM decoding via the bundled Qwen3-0.6B head.
The publisher does **not** report a numerical LibriSpeech WER for the MLT
variant specifically (the shared Fun-ASR README's per-model table covers
the regular Fun-ASR-Nano only). Gate baseline is our own FunASR 1.3.1
reference run on the same manifest: 1.76% (95% CI [1.60%, 1.93%]).
transcribe.cpp's BF16 port matches that baseline within -0.02
percentage-points; F16/Q8_0 are numerically indistinguishable. Q4_K_M is
the only quant with a visible regression (+0.13 pp); F16/Q8_0/Q6_K/Q5_K_M
are within bootstrap noise of BF16.

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
languages, or omit for auto-detection. If your audio is not already
16 kHz mono WAV, convert it first:

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
| Metal   | jfk (11.0s)  |   156 ms (70×)  |   144 ms (76×)  |
| Metal   | dots (35.3s) |   539 ms (66×)  |   499 ms (71×)  |
| CPU     | jfk (11.0s)  |   661 ms (17×)  |   575 ms (19×)  |
| CPU     | dots (35.3s) |   2.36 s (15×)  |   2.12 s (17×)  |

macOS 26.4.1, transcribe.cpp `f094d28`. MLT is ~10–15% slower than
Fun-ASR-Nano on the same hardware; the gap is from per-step LLM
generation cost (different decoded transcript lengths between the two
variants).

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Fun-ASR-MLT-Nano-2512 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu \
  --iters 3 --warmup 1 \
  --name fun-asr-mlt-nano-2512-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against FunASR 1.3.1
on `samples/jfk.wav`. All 22 checkpointed tensors fall within family
tolerance, and the final transcript matches the FunASR reference verbatim
("and so my fellow americans ask not what your country can do for you ask
what you can do for your country"). Last validated at commit
[`f094d28`](https://github.com/handy-computer/transcribe.cpp/tree/f094d28).

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
