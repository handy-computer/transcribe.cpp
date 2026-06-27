# Voxtral Small 24B (2507)

Mistral's [`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)
ported to transcribe.cpp. An offline audio-LLM: a Whisper-large-v3
bidirectional audio encoder (32 layers, `d_model=1280`, 20 heads) feeds
a 4-frame-group projector (375 audio tokens per 30 s chunk) into a
Mistral-Small-24B causal LM (40 layers, `hidden_size=5120`,
`intermediate_size=32768`, GQA 32 q / 8 kv heads, NEOX RoPE, SwiGLU) via
audio-token injection at the `audio_token_id=24` positions in the prompt.

It is the larger sibling of [Voxtral Mini 3B](voxtral-mini-3b-2507.md):
the audio encoder, projector pattern, log-mel frontend, and tekken
tokenizer are identical — only the text decoder is scaled up (Mistral-Small-24B
in place of Ministral-3B).

## What it's for

Offline speech-to-text and speech-to-text translation. Takes a 16 kHz
mono WAV and produces a transcript via greedy decoding.

- **Transcription** — auto language detection, or an explicit `--language`
  hint. Voxtral advertises English, French, German, Spanish, Italian,
  Portuguese, Dutch, and Hindi.
- **Translation** — `--translate --target-language <code>` runs the
  mistral-common instruct template ("Translate this to {Language}.") to
  translate non-English speech into the target language's text.

See Mistral's [model card](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)
for training data, intended use, and upstream evaluation.

Licensed Apache-2.0. Ported from upstream commit
[`da5b424`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507/commit/da5b42409f279fdd92febee0511a6c32828569c1),
pinned 2026-06-05.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [Voxtral-Small-24B-2507-BF16.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-BF16.gguf)     | 48.54 GB | 1.56% |
| F16    | [Voxtral-Small-24B-2507-F16.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-F16.gguf)       | 48.55 GB | 1.57% |
| Q8_0   | [Voxtral-Small-24B-2507-Q8_0.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q8_0.gguf)     | 25.81 GB | 1.56% |
| Q6_K   | [Voxtral-Small-24B-2507-Q6_K.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q6_K.gguf)     | 19.94 GB | 1.58% |
| Q5_K_M | [Voxtral-Small-24B-2507-Q5_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q5_K_M.gguf) | 17.14 GB | 1.60% |
| Q4_K_M | [Voxtral-Small-24B-2507-Q4_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q4_K_M.gguf) | 14.30 GB | 2.11% |

WER measured on the full LibriSpeech `test-clean` split (2620 utterances)
with the Whisper-style English text normalizer, batch size 8 on an NVIDIA
A100 80 GB. The same-split HuggingFace `transformers` reference run
(`VoxtralForConditionalGeneration`, BF16, greedy) lands at **1.57%**, and
the BF16 GGUF matches it at **1.56%**.

## Quick Start

```bash
cmake -B build
cmake --build build

# transcription (auto language)
build/bin/transcribe-cli \
  -m models/Voxtral-Small-24B-2507/Voxtral-Small-24B-2507-Q8_0.gguf \
  samples/jfk.wav

# transcription with an explicit language hint
build/bin/transcribe-cli \
  -m models/Voxtral-Small-24B-2507/Voxtral-Small-24B-2507-Q8_0.gguf \
  --language de samples/german.wav

# speech translation (non-English audio -> English text)
build/bin/transcribe-cli \
  -m models/Voxtral-Small-24B-2507/Voxtral-Small-24B-2507-Q8_0.gguf \
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
- `--batch-size <N>` — batched offline transcription. **Use `N ≤ 8` for
  this model** (see Notes).

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places).

### Apple M4 Max

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |   3.36 s (3.3×)  |   2.62 s (4.2×)  |
| Metal   | dots (35.3s) |  11.20 s (3.2×)  |   8.95 s (3.9×)  |

A 24B is a GPU-class model; on Apple Silicon it runs at **~3–4× realtime**
on Metal (the 3B sibling is ~15–18×). CPU is impractical at this size and is
not benchmarked. transcribe.cpp `96adddb`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models Voxtral-Small-24B-2507 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal \
  --iters 3 --warmup 1 \
  --name voxtral-small-24b-2507-publication
```

## Notes

- **Batched inference: use batch size ≤ 8.** transcribe.cpp ships a true
  parallel `run_batch()` fast path for the Voxtral family, and it is
  WER-neutral (byte-identical to single-stream). At the 24B's memory
  footprint, however, a batch composed entirely of long (>30 s, multi-chunk)
  clips can exceed the compute headroom on an 80 GB GPU and fail to allocate
  — so the practical ceiling is batch size 8. Smaller batches and single-clip
  transcription are unaffected. The failure is loud (the run errors rather
  than silently producing a partial result), so an oversized batch never
  yields a quietly-wrong transcript.

- BF16 and F16 require ~50 GB of memory; on an 80 GB GPU they run
  comfortably at batch ≤ 8.

## Numerical Validation

This variant is validated **end-to-end by word error rate** against the
HuggingFace `transformers` reference, not by a separate tensor-by-tensor
sweep. The full-set BF16 WER (**1.56%**) matches the same-machine
`transformers` reference (**1.57%**) within rounding on all 2620 test-clean
utterances.

The family's per-tensor numerical correctness is established by the
[Voxtral Mini 3B](voxtral-mini-3b-2507.md) sibling, which passes a strict
CPU tensor-parity sweep against the reference. The 24B shares that exact
architecture (same encoder, projector, fusion, RoPE, and tokenizer; only
the decoder is scaled), and given its size it was accepted on the WER match
rather than re-running tensor parity — a deliberate scoping decision.

| Field | Value |
| --- | --- |
| Reference | HuggingFace `transformers` v4.57.6 (`mistralai/Voxtral-Small-24B-2507`) |
| Acceptance dataset | LibriSpeech `test-clean` (2620 utterances), Whisper English normalizer |
| Reference WER | 1.57% |
| BF16 GGUF WER | 1.56% |
| Tolerances (family) | `tests/tolerances/voxtral.json` |
