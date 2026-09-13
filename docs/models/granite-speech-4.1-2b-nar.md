# Granite Speech 4.1-2b NAR

<!-- catalog:intro -->
Upstream: [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) at [`99a4df9`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/commit/99a4df9).

Offline multilingual speech-to-text in a single non-autoregressive editor
pass. IBM Granite Speech 4.1-2b NAR shares the Conformer audio encoder
with the AR Granite-Speech family but pairs it with a custom MLP-with-
attention projector and the Granite-4.0-1b LLM used as a bidirectional
editor (causal mask disabled). One forward pass produces logits over the
full transcript; CTC decode yields the final text. No token-by-token loop.
Takes a 16 kHz mono WAV and produces a transcript. English plus French,
German, Spanish, and Portuguese; ASR only (no translation, no
timestamps).
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text in a single non-autoregressive editor
pass. Covers English plus French, German, Spanish, and Portuguese. ASR
only — no translation, no timestamps, no diarization.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`99a4df9`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/commit/99a4df9007ac5682f9daa093fb7008ff606e9a5d),
pinned 2026-05-24 (single-file `modeling_granite_speech_nar.py` snapshot —
the README's canonical inference target).

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [granite-speech-4.1-2b-nar-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-BF16.gguf) | 4.51 GB | 1.29% |
| F16          | [granite-speech-4.1-2b-nar-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-F16.gguf) | 4.52 GB | 1.29% |
| Q8_0         | [granite-speech-4.1-2b-nar-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q8_0.gguf) | 2.50 GB | 1.29% |
| Q6_K         | [granite-speech-4.1-2b-nar-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q6_K.gguf) | 1.98 GB | 1.29% |
| Q5_K_M       | [granite-speech-4.1-2b-nar-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q5_K_M.gguf) | 1.78 GB | 1.28% |
| Q4_K_M       | [granite-speech-4.1-2b-nar-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q4_K_M.gguf) | 1.56 GB | 1.34% |
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
WER measured on the full LibriSpeech test-clean split (2620 utterances).
BF16 reference baseline (transformers `model.transcribe`, MPS, re-run
locally): 1.28% — matches the upstream model card's 1.29% to within
sampling noise. Text normalizer: Whisper `EnglishTextNormalizer`, the
same normalizer Open ASR Leaderboard uses. Reference reproduction
follows the model card path verbatim (`AutoProcessor` +
`AutoModel.transcribe` + `processor.batch_decode`) at HF revision
`99a4df9` (single-file `modeling_granite_speech_nar.py` snapshot, the
README's canonical target); no mask patching is required because the
NAR LM uses `create_bidirectional_mask()` natively. F16, Q8_0, and
Q6_K all match BF16's 1.29%; Q5_K_M dips slightly to 1.25% (within
overlapping CIs).
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

NAR is single-task ASR; there is no `--translate` flag or `--timestamps`
mode for this variant. The `--language` flag is accepted but ignored — the
editor handles language detection implicitly.

## Performance

Cells are compute latency (mel + encode + decode), with speedup over realtime in parentheses.
NAR is faster than the AR variants on GPU backends because there is no
autoregressive step loop — a single bidirectional forward through 40 LLM
layers replaces the per-token decode graph.

### Apple M4 Max

Mean over 3 iterations after 1 warmup.

<!-- catalog:perf machine=m4-max -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 151 ms (72.70×) | 163 ms (67.43×) |
| Metal   | dots (35.3s) | 491 ms (71.91×) | 518 ms (68.21×) |
| CPU     | jfk (11.0s)  |  1.89 s (5.81×) |  1.78 s (6.17×) |
| CPU     | dots (35.3s) |  7.38 s (4.79×) |  6.21 s (5.69×) |
<!-- /catalog -->

macOS 26.4, transcribe.cpp `de05c43`.

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

Mean over 3 iterations after 1 warmup.

<!-- catalog:perf machine=ryzen-4750u -->
| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  2.68 s (4.10×) |  2.75 s (4.00×) |
| Vulkan  | dots (35.3s) |  8.32 s (4.25×) |  8.53 s (4.14×) |
| CPU     | jfk (11.0s)  |  6.75 s (1.63×) |  5.46 s (2.01×) |
| CPU     | dots (35.3s) | 23.77 s (1.49×) | 19.55 s (1.81×) |
<!-- /catalog -->

Linux 6.18 (Fedora 43), transcribe.cpp `dbe5814`. NAR's Vulkan RTF stays
flat across short and long samples (jfk and dots both ~3.6×) because the
single bidirectional LLM pass dominates over the encoder; on CPU the
encoder dominates so RTF tapers slightly with sequence length.

## Capabilities

| Capability                  | Status |
|-----------------------------|--------|
| Transcribe (English)        | Yes    |
| Transcribe (fr/de/es/pt)    | Yes    |
| Translation                 | No (not supported by the NAR family; use the AR variants) |
| Word/segment timestamps     | No (the NAR encoder pools to per-window output; per-token timing is lost) |

## Numerical Validation

Tensor-level parity with the transformers reference on `samples/jfk.wav`.
Per-tensor `max_abs` / `mean_abs` budgets in
[`tests/tolerances/granite_nar.json`](https://github.com/handy-computer/transcribe.cpp/blob/main/tests/tolerances/granite_nar.json).
Drift on `dec.text_logits` is dominated by BF16 reduction-order noise over
40 bidirectional LLM layers; absolute magnitudes run O(100) at confident
positions, observed drift ~3% relative — the editor is argmax-stable on
this drift band, hence the BF16/F16/Q8_0/Q6_K all match WER.

## Reproduction

### Convert

```bash
uv run --project scripts/envs/granite_nar \
  scripts/convert-granite_nar.py ibm-granite/granite-speech-4.1-2b-nar \
  --repo-id ibm-granite/granite-speech-4.1-2b-nar
```

### Quantize

```bash
for PRESET in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-BF16.gguf \
    models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-${PRESET}.gguf \
    --quant ${PRESET}
done
```

### Validate

```bash
uv run scripts/validate.py all --family granite_nar --variant granite-speech-4.1-2b-nar
```

### Reproduce WER

```bash
uv run scripts/wer/run.py \
  --model models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/granite-speech-4.1-2b-nar-BF16.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/granite-speech-4.1-2b-nar-BF16.test-clean.jsonl
```
