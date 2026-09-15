# Voxtral Mini 3B (2507)

<!-- catalog:intro -->
Upstream: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) at [`3060fe3`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/commit/3060fe3).

Offline audio-LLM speech-to-text and speech translation. A Whisper-large-v3
bidirectional audio encoder feeds a 4-frame-group projector (375 audio tokens
per 30 s chunk) into a Ministral-3B causal LM (30 layers, GQA 32/8, NEOX RoPE,
SwiGLU) via audio-token injection. Takes a 16 kHz mono WAV and produces a
transcript via greedy decoding; speech translation runs through the
mistral-common instruct template. The smaller sibling of Voxtral Small 24B —
same encoder, projector, log-mel frontend, and tekken tokenizer, with a 3B
decoder in place of Mistral-Small-24B.
<!-- /catalog -->

## What it's for

Offline speech-to-text and speech-to-text translation. Takes a 16 kHz
mono WAV and produces a transcript via greedy decoding.

- **Transcription** — auto language detection, or an explicit `--language`
  hint. Voxtral advertises English, French, German, Spanish, Italian,
  Portuguese, Dutch, and Hindi.
- **Translation** — `--translate --target-language <code>` runs the
  mistral-common instruct template ("Translate this to {Language}.") to
  translate non-English speech into the target language's text.

See Mistral's [model card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
for training data, intended use, and upstream evaluation.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`3060fe3`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/commit/3060fe3), pinned 2026-06-06. Validated against the Transformers reference at transcribe.cpp commit [`483c122`](https://github.com/handy-computer/transcribe.cpp/tree/483c122) on 2026-06-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Voxtral-Mini-3B-2507-BF16.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-BF16.gguf) | 9.37 GB | 1.88% |
| F16          | [Voxtral-Mini-3B-2507-F16.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-F16.gguf) | 9.38 GB | 1.89% |
| Q8_0         | [Voxtral-Mini-3B-2507-Q8_0.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-Q8_0.gguf) | 5.00 GB | 1.87% |
| Q6_K         | [Voxtral-Mini-3B-2507-Q6_K.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-Q6_K.gguf) | 3.87 GB | 1.87% |
| Q5_K_M       | [Voxtral-Mini-3B-2507-Q5_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-Q5_K_M.gguf) | 3.46 GB | 1.91% |
| Q4_K_M       | [Voxtral-Mini-3B-2507-Q4_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf/resolve/main/Voxtral-Mini-3B-2507-Q4_K_M.gguf) | 2.98 GB | 1.94% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored with the Whisper English text normalizer on an NVIDIA L40S. Same-machine
HuggingFace transformers reference (VoxtralForConditionalGeneration, BF16,
attn_implementation=eager, greedy): 1.87%; the BF16 GGUF matches within rounding.
The BF16-vs-reference parity is the family's tensor-level numerical gate — 43
checkpointed tensors within tolerance, transcript byte-exact.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 4.71% |
| en       | WER    | 3.89% |
| es       | WER    | 3.52% |
| fr       | WER    | 4.51% |
| hi       | WER    | 8.93% |
| it       | WER    | 2.56% |
| nl       | WER    | 6.57% |
| pt       | WER    | 3.84% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

# transcription (auto language)
build/bin/transcribe-cli \
  -m models/Voxtral-Mini-3B-2507/Voxtral-Mini-3B-2507-Q8_0.gguf \
  samples/jfk.wav

# transcription with an explicit language hint
build/bin/transcribe-cli \
  -m models/Voxtral-Mini-3B-2507/Voxtral-Mini-3B-2507-Q8_0.gguf \
  --language de samples/german.wav

# speech translation (non-English audio -> English text)
build/bin/transcribe-cli \
  -m models/Voxtral-Mini-3B-2507/Voxtral-Mini-3B-2507-Q8_0.gguf \
  --translate --target-language en samples/german.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags:

- `--language <code>` — BCP-47 hint. Omit for auto-detection.
- `--translate --target-language <code>` — speech translation via the
  instruct template.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max dp_ms=1 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |              Q8_0 |            Q4_K_M |
| ------- | ------------ | ----------------: | ----------------: |
| Metal   | jfk (11.0s)  | 862.3 ms (12.76×) | 769.4 ms (14.30×) |
| Metal   | dots (35.3s) |   2.51 s (14.08×) |   2.09 s (16.90×) |
| CPU     | jfk (11.0s)  |    5.83 s (1.89×) |    6.12 s (1.80×) |
| CPU     | dots (35.3s) |   14.94 s (2.36×) |   14.18 s (2.49×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 11.11 s (0.99×) | 10.58 s (1.04×) |
| Vulkan  | dots (35.3s) | 30.04 s (1.18×) | 27.46 s (1.29×) |
| CPU     | jfk (11.0s)  | 22.37 s (0.49×) | 20.88 s (0.53×) |
| CPU     | dots (35.3s) | 55.99 s (0.63×) | 49.49 s (0.71×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models voxtral-mini-3b-2507
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the HuggingFace
`transformers` reference (`VoxtralForConditionalGeneration`, BF16,
`attn_implementation=eager`) on `samples/jfk.wav` with the strict CPU
backend. All 43 checkpointed tensors fall within family tolerance, and
the BF16 transcript matches the reference verbatim
(`And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.`).

The encoder mel frontend is computed in-process (not injected from the
reference), so `enc.mel.in` is the real frontend-parity gate and matches
the reference `WhisperFeatureExtractor` to `2.2e-5` max / `4.1e-8` mean.
The dominant remaining drift is **not** a bug: the reference casts the mel
to BF16 before `conv1` and keeps BF16 activations through the whole stack,
while the C++ ggml graph runs F32 activations with BF16 weights — so the
C++ is the *more* accurate path, and the cpp-vs-BF16-reference drift is the
reference's own BF16 activation rounding (which compounds with depth). A
three-way decomposition confirmed this: cpp-vs-F32-reference is 3–4× tighter
than cpp-vs-BF16-reference on the encoder, and the transcript is byte-exact
against both the BF16 and F32 references. Tolerances and the full mechanism
are pinned in `tests/tolerances/voxtral.json`.

| Field | Value |
| --- | --- |
| Reference | HuggingFace `transformers` v4.57.6 (`mistralai/Voxtral-Mini-3B-2507`) |
| Dump script | `scripts/dump_reference_voxtral_transformers.py` |
| Manifest | `tests/golden/voxtral/voxtral-mini-3b-2507.manifest.json` |
| Tolerances | `tests/tolerances/voxtral.json` |
| Command | `uv run scripts/validate.py all --family voxtral --variant voxtral-mini-3b-2507` |

