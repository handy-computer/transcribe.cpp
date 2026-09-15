# MOSS-Transcribe-Diarize

<!-- catalog:intro -->
Upstream: [`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) at [`d7231bb`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/commit/d7231bb).

Offline English/Chinese speech-to-text with speaker diarization. A 0.9B
audio-LLM: a Whisper-Medium encoder (24 layers, d_model=1024) feeds a
4x temporal merge + VQAdaptor bridge into a Qwen3-0.6B decoder (28 layers)
via audio-token injection. The model emits `[start][Sxx]text[end]`; the
runtime parses those generated markers into clean text and segment rows.
Speaker attribution is opt-in (`--diarize`) and returns structured speaker
ids/turns. Not a streaming model.
<!-- /catalog -->

## What it's for

Offline English and Chinese speech-to-text with optional speaker attribution.
The model generates the canonical diarized format `[start][Sxx]text[end]`
(e.g. `[0.48][S01]Welcome[1.66]`) via greedy decoding. The runtime parses those
emergent text markers into clean `full_text`, segment rows, and—when
`diarize=ON`—speaker IDs and speaker-turn rows. Built for long-form,
multi-speaker audio. No translation; not a streaming model.

See OpenMOSS's [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
for training data, intended use, and upstream evaluation. All of OpenMOSS's
published metrics are Chinese multi-speaker diarization CER/cpCER; LibriSpeech
test-clean is used here only as an English acceptance set.

<!-- catalog:pin -->
Licensed Apache-2.0. Ported from upstream commit [`d7231bb`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/commit/d7231bb), pinned 2026-07-12. Validated against the MOSS author repo (OpenMOSS/MOSS-Transcribe-Diarize) reference at transcribe.cpp commit [`3f5e15c`](https://github.com/handy-computer/transcribe.cpp/tree/3f5e15c) on 2026-07-12.
<!-- /catalog -->

## Memory and length

MOSS keeps the whole recording in memory while it transcribes, so RAM use grows
with how long the audio is. Plan for roughly **85 MB of extra memory per minute
of audio**, on top of the model file itself — about **2.5 GB for a 30-minute
clip** and **~5 GB for an hour**. It can handle recordings up to a couple of
hours if you have the memory; anything longer is rejected with a clear error
instead of being silently cut off. If you run low on memory, split long audio
into shorter pieces.

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16         | [MOSS-Transcribe-Diarize-BF16.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-BF16.gguf) | 1.83 GB | 2.08% |
| F16          | [MOSS-Transcribe-Diarize-F16.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-F16.gguf) | 1.83 GB | 2.07% |
| Q8_0         | [MOSS-Transcribe-Diarize-Q8_0.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q8_0.gguf) |  987 MB | 1.93% |
| Q6_K         | [MOSS-Transcribe-Diarize-Q6_K.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q6_K.gguf) |  768 MB | 1.96% |
| Q5_K_M       | [MOSS-Transcribe-Diarize-Q5_K_M.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q5_K_M.gguf) |  700 MB | 1.99% |
| Q4_K_M       | [MOSS-Transcribe-Diarize-Q4_K_M.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q4_K_M.gguf) |  617 MB | 2.59% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 1, timestamps none. Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored with the Whisper-style English text normalizer and jiwer 3.x. MOSS emits the
diarized format `[start][Sxx]text[end]`; the bracket spans are metadata and are
de-diarized to a space (for both hypothesis and reference) before scoring, matching
the author-repo reference runner. These values describe this dataset only, not a
general quality ranking: a quant that scores slightly better here is not necessarily
better in real-world use, because dataset-specific decoding near-ties can make
quantization noise help or hurt individual utterances. The same-manifest MOSS
author-repo reference (bf16, greedy) lands at **2.07%** with 95% bootstrap CI
[1.82%, 2.40%]. The BF16 port lands at 2.08% (within +0.01 of the reference, well
inside the CI band); the lower-bit presets sit between 1.93% and 1.99% (statistical
noise) except Q4_K_M at 2.59%, whose excess is a handful of 4-bit tail failures (6
empty outputs, 5 English->Chinese language-drift utterances, 1 timestamp-token
repetition loop) rather than broad degradation. Prefer Q5_K_M or higher if those
tail failures matter. Reproduce with `scripts/wer/run.py` + `scripts/wer/score.py
--dediarize`; public `full_text` applies equivalent marker removal.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |  Q8_0 |
| --- | --- | ---: |
| en       | WER    | 5.13% |
| zh       | CER    | 9.23% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-Q8_0.gguf \
  --diarize \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags:

- `-l en` / `-l zh` (or omit for `auto`): English and Chinese are supported.
- No `--task` / `--target-language`: the model is ASR-only, no translation.
- Speaker attribution is off by default. Pass `--diarize` to populate
  `speaker_id` and speaker-turn rows; `--no-diarize` is the explicit off form.
- Timestamp selection is independent: `--timestamps segment` or `auto` keeps
  parsed turn timing; `--timestamps none` returns attribution with zero times.
- `full_text` is always clean marker-free text after a successful parse.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Metal   | jfk (11.0s)  | 393 ms (27.98×) | 382 ms (28.82×) |
| Metal   | dots (35.3s) | 1.40 s (25.17×) | 1.22 s (29.01×) |
| CPU     | jfk (11.0s)  |  2.08 s (5.28×) |  2.20 s (5.00×) |
| CPU     | dots (35.3s) |  5.43 s (6.50×) |  5.47 s (6.46×) |

Apple M4 Max: transcribe.cpp `77b0c93` on 2026-09-14.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  |  3.73 s (2.95×) |  3.48 s (3.16×) |
| Vulkan  | dots (35.3s) | 11.09 s (3.18×) |  9.95 s (3.55×) |
| CPU     | jfk (11.0s)  |  7.54 s (1.46×) |  6.90 s (1.59×) |
| CPU     | dots (35.3s) | 21.08 s (1.68×) | 19.24 s (1.84×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `218aeae3` on 2026-09-14.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models moss-transcribe-diarize
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the MOSS author repo
(`scripts/dump_reference_moss_author.py`, `trust_remote_code`) on
`samples/jfk.wav` with the strict CPU backend. The reference runs BF16 (torch,
eager attention); the C++ path dequantizes BF16 weights to F32 and computes in
F32, so C++ is the *more* precise side and the residual gap is a constant ~1-3%
relative bf16-vs-f32 drift, not a bug. The transcript compare is `dediarized`
(bracket metadata stripped to a space). Confirmed WER-neutral: on the first 100
test-clean utterances the C++ ref-dtype WER (1.40%) is bit-identical to the
Oracle reference on the same subset (1.40%). Tolerances are pinned in
`tests/tolerances/moss.json` with a `_comment` block naming the precision
regime, the large-pre-normalization-activation maxes, and the encoder
padding-trim contract.

| Field | Value |
| --- | --- |
| Reference | MOSS author repo (`OpenMOSS-Team/MOSS-Transcribe-Diarize`) |
| Dump script | `scripts/dump_reference_moss_author.py` |
| Manifest | `tests/golden/moss/moss-transcribe-diarize.manifest.json` |
| Tolerances | `tests/tolerances/moss.json` |
| Command | `uv run scripts/validate.py all --family moss --variant moss-transcribe-diarize` |

For the full porting writeup, see
[`docs/porting/families/moss.md`](../porting/families/moss.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/moss \
  scripts/convert-moss.py OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision d7231bbae2587a4af278735eb765b318c4f64edd
```

### Quantize

```bash
uv run scripts/quantize-all.py models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family moss --variant moss-transcribe-diarize
```

### Score WER

```bash
PRESET=BF16
uv run scripts/wer/run.py \
  --model models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-${PRESET}.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/MOSS-Transcribe-Diarize-${PRESET}.librispeech-test-clean.jsonl
uv run scripts/wer/score.py \
  reports/wer/MOSS-Transcribe-Diarize-${PRESET}.librispeech-test-clean.jsonl \
  --dediarize
```

### Score WER against the MOSS author-repo reference

```bash
uv run --project scripts/envs/moss \
  scripts/wer/run_reference_moss_author.py \
    --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --revision d7231bbae2587a4af278735eb765b318c4f64edd \
    --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
    --out reports/wer/moss-transcribe-diarize-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py \
  reports/wer/moss-transcribe-diarize-REF.librispeech-test-clean.jsonl \
  --dediarize
```
