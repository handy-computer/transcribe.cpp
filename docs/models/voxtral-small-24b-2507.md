# Voxtral Small 24B (2507)

<!-- catalog:intro -->
Upstream: [`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) at [`da5b424`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507/commit/da5b424).

Offline audio-LLM speech-to-text and speech translation. A Whisper-large-v3
bidirectional audio encoder feeds a 4-frame-group projector (375 audio tokens
per 30 s chunk) into a Mistral-Small-24B causal LM (40 layers, GQA 32/8, NEOX
RoPE, SwiGLU) via audio-token injection. Takes a 16 kHz mono WAV and produces a
transcript via greedy decoding. The larger sibling of Voxtral Mini 3B — same
encoder, projector, frontend, and tokenizer, with a scaled-up decoder.
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

See Mistral's [model card](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)
for training data, intended use, and upstream evaluation.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`da5b424`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507/commit/da5b424), pinned 2026-06-05. Validated against the Transformers reference at transcribe.cpp commit [`dac22fa`](https://github.com/handy-computer/transcribe.cpp/tree/dac22fa) on 2026-06-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |     Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [Voxtral-Small-24B-2507-BF16.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-BF16.gguf) | 48.54 GB | 1.56% |
| F16          | [Voxtral-Small-24B-2507-F16.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-F16.gguf) | 48.55 GB | 1.57% |
| Q8_0         | [Voxtral-Small-24B-2507-Q8_0.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q8_0.gguf) | 25.81 GB | 1.56% |
| Q6_K         | [Voxtral-Small-24B-2507-Q6_K.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q6_K.gguf) | 19.94 GB | 1.58% |
| Q5_K_M       | [Voxtral-Small-24B-2507-Q5_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q5_K_M.gguf) | 17.14 GB | 1.60% |
| Q4_K_M       | [Voxtral-Small-24B-2507-Q4_K_M.gguf](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-gguf/resolve/main/Voxtral-Small-24B-2507-Q4_K_M.gguf) | 14.30 GB | 2.11% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy decoding scored with the Whisper English text normalizer on an NVIDIA A100 80
GB. Same-machine HuggingFace transformers reference
(VoxtralForConditionalGeneration, BF16, greedy): 1.57%; the BF16 GGUF matches at
1.56%. Validation for this variant is end-to-end by WER — the family's tensor-level
numerical parity is established by the Voxtral Mini 3B sibling (identical
architecture).
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| de       | WER    | 3.29% |
| en       | WER    | 3.55% |
| es       | WER    | 2.86% |
| fr       | WER    | 3.86% |
| hi       | WER    | 7.40% |
| it       | WER    | 2.69% |
| nl       | WER    | 5.12% |
| pt       | WER    | 3.74% |
<!-- /catalog -->

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

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  |  4.45 s (2.47×) |  3.96 s (2.77×) |
| Metal   | dots (35.3s) | 14.68 s (2.41×) | 12.93 s (2.73×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

A 24B is a GPU-class model; on Apple Silicon it runs at **~3–4× realtime**
on Metal (the 3B sibling is ~15–18×). CPU is impractical at this size and is
not benchmarked. transcribe.cpp `96adddb`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models voxtral-small-24b-2507
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
