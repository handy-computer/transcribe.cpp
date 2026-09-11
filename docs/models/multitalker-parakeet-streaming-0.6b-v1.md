# Multitalker Parakeet Streaming 0.6B v1

NVIDIA's [`nvidia/multitalker-parakeet-streaming-0.6b-v1`](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
ported to transcribe.cpp. A 0.6B-parameter cache-aware streaming
FastConformer encoder with an RNN-T transducer decoder, fine-tuned from
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b).

## What it's for

Offline and cache-aware **streaming** English speech-to-text with greedy
RNN-T decoding. Outputs cased, punctuated transcripts. Token- and
word-level timestamps are available.

Upstream this is a **multitalker (speaker-attributed)** checkpoint: it can
transcribe several overlapping speakers into per-speaker channels. This
port ships that path too, via **bundle GGUFs** that embed the
[`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
streaming diarizer alongside the ASR model. A plain (non-bundle) GGUF runs
the model's `single_speaker_mode` ASR path — a cache-aware streaming RNN-T
with the checkpoint's always-on layer-0 speaker-kernel injection. A bundle
GGUF with `--diarize` runs the full multitalker pipeline and emits a
speaker-tagged transcript (see
[Multitalker](#multitalker-speaker-attributed-asr)).

This port runs the model in both **offline** and **cache-aware streaming**
modes.

See NVIDIA's [model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
for training data, intended use, the multitalker methodology, and the full
latency-vs-accuracy table.

Licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Ported from upstream commit
[`8749fc7`](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/commit/8749fc71fd6e2d88ef230159bbf2aea69b524ee1),
pinned 2026-07-12.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean, offline) |
| --- | --- | ---: | ---: |
| F32    | [multitalker-parakeet-streaming-0.6b-v1-F32.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-F32.gguf)       | 2.49 GB | 2.19% |
| F16    | [multitalker-parakeet-streaming-0.6b-v1-F16.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-F16.gguf)       | 1.25 GB | 2.19% |
| Q8_0   | [multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf)     | 734 MB  | 2.18% |
| Q6_K   | [multitalker-parakeet-streaming-0.6b-v1-Q6_K.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-Q6_K.gguf)     | 604 MB  | 2.20% |
| Q5_K_M | [multitalker-parakeet-streaming-0.6b-v1-Q5_K_M.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-Q5_K_M.gguf) | 542 MB  | 2.18% |
| Q4_K_M | [multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf) | 478 MB  | 2.18% |

WER is measured on the full LibriSpeech test-clean split (2620 utterances)
in `single_speaker_mode` with greedy RNN-T decoding, whisper-normalizer
scoring (PnC-stripped), and no external LM. F32 reference baseline: 2.19%.
The measured NeMo `single_speaker_mode` reference on the same split is
2.19%, and NVIDIA's self-reported number is 2.19% (from the
[HF model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)).

### Multitalker bundles

Bundle GGUFs embed the streaming Sortformer diarizer. The tier names the
ASR half's dtype; the embedded diarizer is F32 for the F32 bundle, F16 for
F16, and Q8_0 for all k-quant tiers.

| Bundle | Download | Size |
| --- | --- | ---: |
| F32    | [bundle/multitalker-parakeet-streaming-0.6b-v1-F32.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-F32.gguf)       | 2.96 GB |
| F16    | [bundle/multitalker-parakeet-streaming-0.6b-v1-F16.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-F16.gguf)       | 1.48 GB |
| Q8_0   | [bundle/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf)     | 873 MB  |
| Q6_K   | [bundle/multitalker-parakeet-streaming-0.6b-v1-Q6_K.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-Q6_K.gguf)     | 743 MB  |
| Q5_K_M | [bundle/multitalker-parakeet-streaming-0.6b-v1-Q5_K_M.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-Q5_K_M.gguf) | 681 MB  |
| Q4_K_M | [bundle/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf](https://huggingface.co/handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf/resolve/main/bundle/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf) | 617 MB  |

## Streaming parity

Cache-aware streaming was validated tensor-by-tensor against NeMo's
`conformer_stream_step` reference via `scripts/validate_streaming.py`. On
`samples/jfk.wav` at `--backend cpu --threads 1`, the always-on layer-0
speaker-kernel injection is applied per chunk on the post-drop block-0
input (matching the offline path), and the harness reports **150/150
streaming-tensor pairs within tolerance and the transcript byte-exact vs
NeMo at R=13** (`att_context_size=[70, 13]`, 1.12s chunk). Streaming
`enc_out` drift (observed 5.5e-6) is tighter than the offline C++
`enc.final` drift (1.9e-3) on the same audio, because clean per-chunk
caches accumulate less error than a single 24-block offline pass.

Reproduce:

```bash
uv run --project scripts/envs/parakeet scripts/validate_streaming.py \
    --hf-model nvidia/multitalker-parakeet-streaming-0.6b-v1 \
    --gguf models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1-F32.gguf \
    --audio samples/jfk.wav \
    --out build/validate_streaming/multitalker/jfk \
    --right 13 6 1 0 \
    --backend cpu --threads 1 \
    --tolerances tests/tolerances/multitalker-parakeet-streaming-0.6b-v1.streaming.json
```

## Quick Start

```bash
cmake -B build
cmake --build build

# Plain GGUF: single-speaker transcription.
build/bin/transcribe-cli \
  -m models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf \
  samples/jfk.wav

# Bundle GGUF: speaker-attributed transcription.
build/bin/transcribe-cli --diarize \
  -m models/multitalker-parakeet-streaming-0.6b-v1/bundle/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf \
  meeting.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

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
| Metal   | jfk (11.0s)  |  67 ms (164×) |  69 ms (159×) |
| Metal   | dots (35.3s) | 184 ms (192×) | 185 ms (191×) |
| CPU     | jfk (11.0s)  | 310 ms (36×)  | 307 ms (36×)  |
| CPU     | dots (35.3s) | 1.05 s (34×)  | 1.03 s (34×)  |

macOS 26.5.1, transcribe.cpp `c55a09d`.

### AMD Ryzen 7 4750U Pro

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  466 ms (24×) |  475 ms (23×) |
| Vulkan  | dots (35.3s) |  1.36 s (26×) |  1.39 s (26×) |
| CPU     | jfk (11.0s)  |  751 ms (15×) |  816 ms (13×) |
| CPU     | dots (35.3s) |  2.99 s (12×) |  3.12 s (11×) |

Fedora 43, transcribe.cpp `c55a09d`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models multitalker-parakeet-streaming-0.6b-v1 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name multitalker-parakeet-streaming-0.6b-v1-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/jfk.wav` via `scripts/validate.py`. Per-tensor tolerances live
in a per-variant file
([`tests/tolerances/multitalker-parakeet-streaming-0.6b-v1.json`](../../tests/tolerances/multitalker-parakeet-streaming-0.6b-v1.json))
rather than the family-shared one because the unnormalised log-mel
(NeMo's `normalize="NA"` no-op) lands on a different magnitude scale than
the per-feature-normalised siblings, and because the layer-0
speaker-kernel injection is unique to this variant. The family-level
forward map at
[`reports/porting/parakeet/forward-map.md`](../../reports/porting/parakeet/forward-map.md)
documents the per-stage divergence sources.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/multitalker-parakeet-streaming-0.6b-v1` (`single_speaker_mode`) |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/multitalker-parakeet-streaming-0.6b-v1.manifest.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant multitalker-parakeet-streaming-0.6b-v1` |

## Batch

The model ships an explicit `run_batch()` parallel fast path. Batch output
is WER-neutral: byte-equal to the serial single-stream path at batch sizes
2 / 4 / 8 (golden frozen at
`tests/golden/batch/multitalker-parakeet-streaming-0.6b-v1.cpu.json`),
CPU same-length tensor parity is bit-exact (max_abs 0.0) at batch=4 on
`jfk.wav`, diverse-length flash tensor parity is bit-exact across arbitrary
length mixes, and full test-clean batch-8 WER equals batch-1 (2.19%).

## Multitalker (speaker-attributed ASR)

A **bundle GGUF** (composed by `scripts/compose-multitalker-bundle.py`)
embeds the streaming Sortformer diarizer. Running it with `--diarize`
produces a speaker-tagged transcript (speaker ids on segments, with words
linked through their segment; up to 4 speakers): the embedded diarizer
streams over the clip at the reference operating point (14-frame chunks,
spkcache/FIFO 188), per-chunk cache gating selects which speakers are
active, and each active speaker gets a private cache-aware streaming
encoder+decoder instance fed only its active chunks. Encoder/decoder
working state is constant in clip length (~5 MB per active speaker), while
total call memory remains O(T) for the input audio, whole-clip mel, and
accumulated diarizer predictions. It does not allocate the O(T²) attention
buffers used by the optional offline replay.

Two supervision modes, selected via `TRANSCRIBE_MULTITALKER_MODE`:

- **kernel** (default): trained speaker-kernel injection at conformer
  layer 0 (`masked_asr=False` upstream). Audio is unmodified; the model
  suppresses non-target speech.
- **masked**: mel feature masking (`masked_asr=True`, the NeMo example
  default). Non-target frames are zero-masked before pre-encode.

cpWER on ami-ihm-test (16 meetings, meeteval, refs from `edinburghcstr/ami`
segment transcripts; NeMo baseline = the pinned reference pipeline on
L40S, batch 1):

| system | kernel | masked |
| --- | ---: | ---: |
| transcribe.cpp (CUDA F32) | **19.35%** | 23.73% |
| NeMo reference            | 21.39%     | 24.00% |
| NVIDIA self-reported      | 21.26%     | — |

Kernel mode is the default because it wins on both implementations. The
C++ kernel number is *better* than the reference for a verified reason:
the reference harness has a background-slot indexing bug
(`get_active_speakers_info` builds `inactive_speaker_ids` from
`range(len(active))` minus a slot id), confirmed at tensor level from the
reference's own supervision dumps. With
`TRANSCRIBE_MULTITALKER_REF_BG_COMPAT=1` (parity tooling only) the port
reproduces that bug and scores 21.23% — matching the reference (and
NVIDIA's published 21.26%). Under matched configuration both modes are
transcript-exact vs the reference on CPU F32 (0 differing words on a
3-minute 4-speaker AMI slice); on long meetings a measured ~3e-5 F32 pred
drift eventually flips near-tie top-k picks inside the diarizer's
spkcache compression (a discontinuous selection both sides implement
bit-identically vs a neutral arbiter), after which trajectories diverge
benignly (~0.4 cpWER on the affected meeting).

`TRANSCRIBE_MULTITALKER_OFFLINE=1` selects the older offline whole-clip
replay instead of the streaming pass (parity/dump tooling; attention
buffers grow quadratically with clip length — do not use on meetings
longer than ~15 minutes).

## Known limitations

- **Multitalker is offline-API only.** `--diarize` on a bundle runs the
  streaming pipeline internally, but the public streaming API
  (`stream_feed`) still exposes single-speaker transcription only; a
  per-speaker streaming text ABI is future work. `run_batch` on a bundle
  warns and runs single-speaker.
- **English only.** The model is English-only by training.
- **No translation and no language detection.**

## Reproduction

### Convert

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/multitalker-parakeet-streaming-0.6b-v1
```

### Quantize

Run `transcribe-quantize` once per target quant. Example for Q8_0; repeat
with `F16`, `Q6_K`, `Q5_K_M`, `Q4_K_M`:

```bash
build/bin/transcribe-quantize \
  models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1-F32.gguf \
  models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf \
  --quant Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family parakeet --variant multitalker-parakeet-streaming-0.6b-v1
```
