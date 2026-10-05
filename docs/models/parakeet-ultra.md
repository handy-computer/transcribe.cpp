# Parakeet Ultra

<!-- catalog:intro -->
Upstream: [`moondream/parakeet-ultra`](https://huggingface.co/moondream/parakeet-ultra) at [`73175eb`](https://huggingface.co/moondream/parakeet-ultra/commit/73175eb).

Moondream's post-trained parakeet-tdt-0.6b-v3. Offline multilingual
speech-to-text over the same 25 European languages, with lower error rates
than v3 on English and on every FLEURS language we measured. A FastConformer
encoder with a TDT transducer decoder, plus a small voice-activity head that
the model uses to cut long recordings at pauses into segments of at most
30 seconds. Takes 16 kHz mono WAV and produces a punctuated, cased
transcript with optional token-level timestamps. A drop-in replacement for
v3: same size, languages and API. Not a streaming model and does not
translate.
<!-- /catalog -->

## What it's for

Offline multilingual speech-to-text, and the default pick over
`parakeet-tdt-0.6b-v3` for new work. Moondream post-trained v3 without
changing its architecture, tokenizer or frontend, so it covers the same 25
European languages and drops into any v3 setup by swapping the GGUF.

The one addition is a small voice-activity head on the encoder's input
layers. The model uses it to segment long audio, described below. It is not
a streaming model and does not translate.

See Moondream's [model card](https://huggingface.co/moondream/parakeet-ultra)
for training details and their own evaluation.

## Long-form audio

Audio up to 30 seconds is transcribed in one pass, exactly like v3. Longer
audio is segmented the way Moondream's runtime does it:

1. The voice-activity head scores every 80 ms frame, scanning the file in
   120 s blocks.
2. Frames above 0.5 are speech; gaps under 0.1 s are bridged and runs under
   0.1 s dropped. A pause is any silence of at least 0.2 s.
3. Each segment ends at the midpoint of the last pause that falls between
   1 s and 30 s into it, or at 30 s if there is none. Segments without speech
   are skipped.
4. Each segment is transcribed independently; the texts are joined with a
   space and timestamps are shifted to file time.

The thresholds are stored in the GGUF, not hard-coded. Cut points match
Moondream's runtime sample for sample. Two consequences when moving from v3:
segment boundaries and timestamps on long files differ from v3's output,
and audio over 30 s pays for one extra voice-activity pass.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`73175eb`](https://huggingface.co/moondream/parakeet-ultra/commit/73175eb), pinned 2026-10-04. Validated against the kestrel 0.9.1 (Photon) reference at transcribe.cpp commit [`08f9e6f0`](https://github.com/handy-computer/transcribe.cpp/tree/08f9e6f0) on 2026-10-05.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [parakeet-ultra-F32.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-F32.gguf) | 2.51 GB | 1.80% |
| F16          | [parakeet-ultra-F16.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-F16.gguf) | 1.26 GB | 1.80% |
| Q8_0         | [parakeet-ultra-Q8_0.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf) |  740 MB | 1.80% |
| Q6_K         | [parakeet-ultra-Q6_K.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q6_K.gguf) |  611 MB | 1.81% |
| Q5_K_M       | [parakeet-ultra-Q5_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q5_K_M.gguf) |  550 MB | 1.81% |
| Q4_K_M       | [parakeet-ultra-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q4_K_M.gguf) |  486 MB | 1.82% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `021aa17d`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, no external LM. The reference is Moondream's
own runtime (kestrel 0.9.1, the engine behind Photon), not transformers.
On this manifest kestrel scores 1.80% and the C++ F32 build matches it.

Moondream's card reports 1.41% on the same split, scored with the Open
ASR Leaderboard normalizer.

Long-form: on 11 full TED-LIUM 3 talks, segmented by the model's
voice-activity head exactly as kestrel does it, F32 scores 2.22%, identical
to kestrel, and every quant stays within 0.05pp.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    | 11.04% |
| cs       | WER    | 11.04% |
| da       | WER    | 15.98% |
| de       | WER    |  4.58% |
| el       | WER    | 32.25% |
| en       | WER    |  4.26% |
| es       | WER    |  3.06% |
| et       | WER    | 14.44% |
| fi       | WER    | 11.14% |
| fr       | WER    |  4.66% |
| hr       | WER    | 11.30% |
| hu       | WER    | 13.09% |
| it       | WER    |  2.34% |
| lt       | WER    | 17.06% |
| lv       | WER    | 18.04% |
| mt       | WER    | 17.01% |
| nl       | WER    |  6.70% |
| pl       | WER    |  6.16% |
| pt       | WER    |  4.16% |
| ro       | WER    | 10.12% |
| ru       | WER    |  5.63% |
| sk       | WER    |  7.83% |
| sl       | WER    | 18.98% |
| sv       | WER    | 12.83% |
| uk       | WER    |  5.71% |
<!-- /catalog -->

The voice-activity head is kept at F32 in every quant. Lower quants can still
move a cut point, because the head reads the quantized input layers' output:
on a 306 s test file, F16, Q8_0 and Q6_K cut exactly where F32 does, Q5_K_M
differs at one segment boundary and Q4_K_M produces 12 segments instead of
11.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-ultra/parakeet-ultra-Q8_0.gguf \
  samples/jfk.wav
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
| Metal   | jfk (11.0s)  |  53 ms (206.94×) |  54 ms (202.43×) |
| Metal   | dots (35.3s) | 206 ms (171.31×) | 212 ms (166.34×) |
| CPU     | jfk (11.0s)  |  283 ms (38.87×) |  311 ms (35.42×) |
| CPU     | dots (35.3s) |  1.13 s (31.20×) |  1.12 s (31.68×) |

Apple M4 Max: transcribe.cpp `a91a58db` on 2026-10-05.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 455 ms (24.18×) | 459 ms (23.98×) |
| Vulkan  | dots (35.3s) | 1.62 s (21.82×) | 1.64 s (21.55×) |
| CPU     | jfk (11.0s)  | 733 ms (15.01×) | 778 ms (14.13×) |
| CPU     | dots (35.3s) | 3.05 s (11.57×) | 3.18 s (11.12×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `08f90613` on 2026-10-05.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models parakeet-ultra
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against kestrel 0.9.1, the
runtime Moondream ships for this model, on three cases: `jfk` (encoder and
decode), and `dots` (35 s) and `dots-full` (306 s) through the long-form path.
The long-form cases compare every voice-activity scan block, the cut points
(exact), each segment's encoder output and the stitched transcript. All
transcripts match kestrel verbatim, and LibriSpeech test-clean and TED-LIUM
long-form WER at F32 equal kestrel's exactly.

| Field | Value |
| --- | --- |
| Reference | kestrel 0.9.1 / kestrel-kernels 0.7.4, `moondream/parakeet-ultra` |
| Dump script | `scripts/dump_reference_parakeet_kestrel.py` |
| Manifest | `tests/golden/parakeet/parakeet-ultra.manifest.json` |
| Tolerances | `tests/tolerances/parakeet-ultra.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant parakeet-ultra` |

The tensor comparison runs with `TRANSCRIBE_NO_FLASH=1` (set by the manifest),
because the flash-attention path rounds one attention term to F16. Flash
attention stays on in production; WER is the check there. Remaining drift is
float32 accumulation order, plus float32 STFT noise on kestrel's side of the
mel comparison.

To match kestrel, the model takes three runtime paths that the other parakeet
variants don't, each switched on by a GGUF key: masking after every strided
subsampling conv, F32 for the subsampler's pointwise convs, and a single
per-utterance symbol budget in the TDT decoder.

## Reproduction

### Convert

Moondream ships HF safetensors only; the converter maps them onto the NeMo
layout and adds the voice-activity head and segmenter constants.

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py moondream/parakeet-ultra
```

### Quantize

```bash
uv run scripts/quantize-all.py \
  models/parakeet-ultra/parakeet-ultra-F32.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant parakeet-ultra
```

### WER

```bash
uv run scripts/wer/ingest.py librispeech
uv run scripts/wer/ingest.py tedlium-longform

# reference arm (kestrel)
uv run --project scripts/envs/parakeet-kestrel \
  scripts/wer/run_reference_parakeet_kestrel.py \
    --manifest samples/wer/tedlium-longform.manifest.jsonl \
    --out reports/wer/parakeet-ultra-REF.tedlium-longform.jsonl

# transcribe.cpp arm
uv run scripts/wer/run.py \
  --model models/parakeet-ultra/parakeet-ultra-F32.gguf \
  --manifest samples/wer/tedlium-longform.manifest.jsonl \
  --out reports/wer/parakeet-ultra-F32.tedlium-longform.b1.jsonl

uv run scripts/wer/score.py <report>.jsonl
```
