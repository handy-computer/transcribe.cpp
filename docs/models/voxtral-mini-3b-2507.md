# Voxtral Mini 3B (2507)

Mistral's [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
ported to transcribe.cpp. An offline audio-LLM: a Whisper-large-v3
bidirectional audio encoder (32 layers, `d_model=1280`, 20 heads) feeds
a 4-frame-group projector (375 audio tokens per 30 s chunk) into a
Ministral-3B causal LM (30 layers, `hidden_size=3072`,
`intermediate_size=8192`, GQA 32 q / 8 kv heads, NEOX RoPE, SwiGLU) via
audio-token injection at the `audio_token_id=24` positions in the prompt.

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

Licensed Apache-2.0. Ported from upstream commit
[`3060fe3`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/commit/3060fe34b35ba5d44202ce9ff3c097642914f8f3).

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Voxtral-Mini-3B-2507-BF16.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-BF16.gguf)     | 9.37 GB | 1.88% |
| F16    | [Voxtral-Mini-3B-2507-F16.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-F16.gguf)       | 9.38 GB | 1.89% |
| Q8_0   | [Voxtral-Mini-3B-2507-Q8_0.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-Q8_0.gguf)     | 5.00 GB | 1.87% |
| Q6_K   | [Voxtral-Mini-3B-2507-Q6_K.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-Q6_K.gguf)     | 3.87 GB | 1.87% |
| Q5_K_M | [Voxtral-Mini-3B-2507-Q5_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-Q5_K_M.gguf) | 3.46 GB | 1.91% |
| Q4_K_M | [Voxtral-Mini-3B-2507-Q4_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-Q4_K_M.gguf) | 2.98 GB | 1.94% |

WER measured on the full LibriSpeech `test-clean` split (2620 utterances)
with the Whisper-style English text normalizer, batch size 8 on an NVIDIA
L40S. The same-machine HuggingFace `transformers` reference run
(`VoxtralForConditionalGeneration`, BF16, `attn_implementation=eager`,
greedy) lands at **1.87%**, and the BF16 GGUF matches it (1.87% at batch
1.

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places).

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  | 727.3 ms (15.1×) | 656.8 ms (16.7×) |
| Metal   | dots (35.3s) |   2.40 s (14.7×) |   1.90 s (18.6×) |
| CPU     | jfk (11.0s)  |   6.06 s (1.8×)  |   6.76 s (1.6×)  |
| CPU     | dots (35.3s) |  16.60 s (2.1×)  |  15.31 s (2.3×)  |

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Voxtral-Mini-3B-2507 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name voxtral-mini-3b-2507-publication
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

Selected tensors (observed on CPU, strict backend; see tolerance file
for budgets):

| Tensor | Shape | Max abs diff | Mean abs diff | Notes |
| --- | --- | ---: | ---: | --- |
| `enc.mel.in` | `[128,3000]` | `2.229e-05` | `4.060e-08` | In-process log-mel vs reference `WhisperFeatureExtractor` — the frontend-parity gate |
| `enc.out`    | `[1500,1280]` | `1.414e+01` | `1.121e-02` | Final encoder LayerNorm; the drift here is the reference's BF16 activation rounding (cpp-vs-F32-ref is ~4× tighter) |
| `proj.out`   | `[375,3072]`  | `3.493e-01` | `3.158e-03` | Projector output (the audio embeddings injected into the LM) |
