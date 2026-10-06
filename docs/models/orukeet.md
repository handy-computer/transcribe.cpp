# Orukeet

<!-- catalog:intro -->
Upstream: [`oruk/orukeet`](https://huggingface.co/oruk/orukeet) at [`b59c13a`](https://huggingface.co/oruk/orukeet/commit/b59c13a).

A tuned version of parakeet-tdt-0.6b-v3 from Oruk AI. Turns speech into
text in 25 European languages, works offline, and runs as fast as v3. Give
it a 16 kHz mono WAV and it gives you back text with punctuation and
capitals. Swap it in anywhere you use v3. Does not stream and does not
translate.
<!-- /catalog -->

## What it does

Speech to text in 25 European languages, offline. It is
`parakeet-tdt-0.6b-v3` with new weights from Oruk AI: same size, same speed,
same 25 languages, slightly more accurate. If you already run v3, point at
this file instead and you are done.

It does not stream and it does not translate.

<!-- catalog:pin -->
Licensed CC-BY-SA-4.0. Ported from upstream commit [`b59c13a`](https://huggingface.co/oruk/orukeet/commit/b59c13a), pinned 2026-10-06. Validated against the NeMo reference at transcribe.cpp commit [`65856fac`](https://github.com/handy-computer/transcribe.cpp/tree/65856fac) on 2026-10-06.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [orukeet-F32.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-F32.gguf) | 2.51 GB | 1.86% |
| F16          | [orukeet-F16.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-F16.gguf) | 1.26 GB | 1.86% |
| Q8_0         | [orukeet-Q8_0.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q8_0.gguf) |  740 MB | 1.86% |
| Q6_K         | [orukeet-Q6_K.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q6_K.gguf) |  610 MB | 1.86% |
| Q5_K_M       | [orukeet-Q5_K_M.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q5_K_M.gguf) |  549 MB | 1.83% |
| Q4_K_M       | [orukeet-Q4_K_M.gguf](https://huggingface.co/handy-computer/orukeet-gguf/resolve/main/orukeet-Q4_K_M.gguf) |  485 MB | 1.95% |
<!-- /catalog -->

<!-- catalog:recipe -->
WER on the full LibriSpeech test-clean split (2,620 utterances), batch size 8, timestamps none, language hint `en`, decoded on cuda. Measured at transcribe.cpp `37273409`.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
A bit more accurate than v3 on English and on 20 of the 25 languages.
parakeet-ultra is more accurate than both on almost everything.
<!-- /catalog -->

<!-- catalog:accuracy -->
**FLEURS test**

| Language | Metric |   Q8_0 |
| --- | --- | ---: |
| bg       | WER    | 11.14% |
| cs       | WER    | 10.22% |
| da       | WER    | 16.58% |
| de       | WER    |  5.64% |
| el       | WER    | 33.12% |
| en       | WER    |  4.43% |
| es       | WER    |  3.75% |
| et       | WER    | 15.16% |
| fi       | WER    | 11.88% |
| fr       | WER    |  6.17% |
| hr       | WER    | 11.26% |
| hu       | WER    | 13.53% |
| it       | WER    |  3.11% |
| lt       | WER    | 17.72% |
| lv       | WER    | 19.53% |
| mt       | WER    | 17.18% |
| nl       | WER    |  6.93% |
| pl       | WER    |  7.35% |
| pt       | WER    |  4.56% |
| ro       | WER    | 10.86% |
| ru       | WER    |  6.70% |
| sk       | WER    |  8.99% |
| sl       | WER    | 24.00% |
| sv       | WER    | 13.26% |
| uk       | WER    |  6.53% |
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/orukeet/orukeet-Q8_0.gguf \
  samples/jfk.wav
```

Audio has to be 16 kHz mono WAV. Convert anything else first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

It figures out the language on its own. Add `--language de` (or any of the
25 codes) if you want to force one.

## Performance

Same speed as v3.

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  59 ms (185.68×) |  65 ms (170.12×) |
| Metal   | dots (35.3s) | 159 ms (221.73×) | 164 ms (215.85×) |
| CPU     | jfk (11.0s)  |  304 ms (36.17×) |  305 ms (36.03×) |
| CPU     | dots (35.3s) |  969 ms (36.45×) |  1.03 s (34.29×) |

Apple M4 Max: transcribe.cpp `d91374b3` on 2026-10-06.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses; profile `asr-publication-v2`: mean over 3 iterations after 1 warmup.

| Backend | Sample       |            Q8_0 |          Q4_K_M |
| ------- | ------------ | --------------: | --------------: |
| Vulkan  | jfk (11.0s)  | 454 ms (24.20×) | 455 ms (24.17×) |
| Vulkan  | dots (35.3s) | 1.36 s (25.99×) | 1.38 s (25.59×) |
| CPU     | jfk (11.0s)  | 730 ms (15.07×) | 776 ms (14.18×) |
| CPU     | dots (35.3s) | 2.90 s (12.17×) | 2.95 s (11.97×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `8bab590e` on 2026-10-06.
<!-- /catalog -->

Rerun the benchmark yourself:

```bash
uv run scripts/bench/run.py --profile --models orukeet
```

## How we checked it

Compared tensor by tensor against NeMo running Oruk's own `.nemo` file,
using the same tolerances as v3. Everything matches and the test transcript
is identical. Batches of 2, 4 and 8 give the same text as one at a time.

| Field | Value |
| --- | --- |
| Reference | NeMo, `oruk/orukeet` `orukeet-v0.1.0.nemo` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/orukeet.manifest.json` |
| Tolerances | `tests/tolerances/parakeet.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant orukeet --model models/orukeet/orukeet-v0.1.0.nemo` |

## Build it yourself

```bash
# get the source checkpoint
hf download oruk/orukeet orukeet-v0.1.0.nemo --local-dir models/orukeet

# convert to F32 GGUF
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py models/orukeet/orukeet-v0.1.0.nemo \
    models/orukeet/orukeet-F32.gguf --repo-id oruk/orukeet

# make the smaller files
uv run scripts/quantize-all.py models/orukeet/orukeet-F32.gguf

# check it against NeMo
uv run scripts/validate.py all --family parakeet --variant orukeet \
  --model models/orukeet/orukeet-v0.1.0.nemo

# measure WER
uv run scripts/wer/ingest.py librispeech
uv run scripts/wer/run.py \
  --model models/orukeet/orukeet-F32.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/orukeet-F32.librispeech-test-clean.b1.jsonl
uv run scripts/wer/score.py reports/wer/orukeet-F32.librispeech-test-clean.b1.jsonl
```

Oruk also ships a `transcribe-cpp/orukeet-Q8_0.gguf` in their own repo.
It was made with this same converter and works too; ours is just rebuilt
from the F32 so all the sizes come from one place.

## Limits

- No streaming.
- No translation.
- Timestamps are per token, not per word.
- Licence is CC-BY-SA-4.0. The rest of the parakeet family is CC-BY-4.0.
  If you make a derivative of the weights, it has to carry the same licence.
- Long files go through in one pass. Nothing cuts them up for you.
