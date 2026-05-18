# Granite Speech 4.1-2b NAR

IBM's [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
ported to transcribe.cpp. The non-autoregressive editor variant of
Granite-Speech. Shares the Conformer audio encoder with the AR Granite-
Speech family but pairs it with a custom MLP-with-attention projector and
the Granite-4.0-1b LLM used as a bidirectional editor (causal mask
disabled). One forward pass produces logits over the full transcript;
CTC decode yields the final text — no token-by-token loop.

## What it's for

Offline multilingual speech-to-text in a single non-autoregressive editor
pass. Covers English plus French, German, Spanish, and Portuguese. ASR
only — no translation, no timestamps, no diarization.

See IBM's [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
for training data, intended use, and upstream evaluation methodology.

Licensed Apache-2.0. Ported from upstream commit
[`7d20732`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/commit/7d20732df04d097262c4ecd8fe7f34ec2b3e6c42),
pinned 2026-05-17.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-4.1-2b-nar-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-BF16.gguf)     | 4.51 GB | 1.29% |
| F16    | [granite-speech-4.1-2b-nar-F16.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-F16.gguf)       | 4.52 GB | 1.29% |
| Q8_0   | [granite-speech-4.1-2b-nar-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q8_0.gguf)     | 2.50 GB | 1.29% |
| Q6_K   | [granite-speech-4.1-2b-nar-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q6_K.gguf)     | 1.98 GB | 1.29% |
| Q5_K_M | [granite-speech-4.1-2b-nar-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q5_K_M.gguf) | 1.78 GB | 1.28% |
| Q4_K_M | [granite-speech-4.1-2b-nar-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-4.1-2b-nar-gguf/resolve/main/granite-speech-4.1-2b-nar-Q4_K_M.gguf) | 1.56 GB | 1.34% |

WER measured on the full LibriSpeech test-clean split (2620 utterances).
BF16 reference baseline (transformers, re-run locally): 1.29% — matches the
upstream model card exactly. Reference reproduction on a host without
`flash-attn-2` requires monkey-patching
`transformers.models.granite.modeling_granite.create_causal_mask` to `None`;
the transcribe.cpp runtime implements bidirectional attention natively, so
this caveat only matters if you reproduce the reference. Text normalizer:
Whisper `EnglishTextNormalizer`. F16, Q8_0, and Q6_K all score bit-for-bit
the same WER as BF16, indicating the editor is very robust to weight
quantization down to Q5_K_M.

## Quick start

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

Cells are wall-clock latency, with speedup over realtime in parentheses.
NAR is faster than the AR variants on GPU backends because there is no
autoregressive step loop — a single bidirectional forward through 40 LLM
layers replaces the per-token decode graph.

### Apple M4

Mean over 5 iterations after 2 warmups. Q8_0.

| Backend | Sample      |       Q8_0        |
| ------- | ----------- | ----------------: |
| Metal   | jfk (11.0s) |    614 ms (18×)   |
| CPU     | jfk (11.0s) |   2.55 s (4×)     |

macOS 26.1, transcribe.cpp `275332d`.

### AMD Ryzen 7 PRO 4750U (Vega 8 iGPU)

Mean over 3 iterations after 1 warmup.

**Vulkan (RADV)**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |   3.16 s (3.5×)  |   3.06 s (3.6×)  |
| dots (35.3s) |   9.85 s (3.6×)  |   9.57 s (3.7×)  |

**CPU**

| Sample       |     Q4_K_M       |       Q8_0       |
| ------------ | ---------------: | ---------------: |
| jfk (11.0s)  |   5.71 s (1.9×)  |   7.05 s (1.6×)  |
| dots (35.3s) |  20.39 s (1.7×)  |  24.81 s (1.4×)  |

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

## Numerical validation

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
