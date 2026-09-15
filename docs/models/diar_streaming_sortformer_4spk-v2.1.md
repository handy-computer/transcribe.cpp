# Streaming Sortformer Diarizer 4spk v2.1

<!-- catalog:intro -->
Upstream: [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) at [`fafaab5`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/commit/fafaab5).

Streaming speaker diarization: who spoke when, for up to 4 speakers.
A FastConformer encoder with an 18-layer Transformer head emitting
per-frame speaker-activity probabilities, running online with an
Arrival-Order Speaker Cache (AOSC) + FIFO. NOT a transcription model:
a run produces speaker segments (start, end, speaker id in arrival
order), no text. Takes 16 kHz mono WAV.
<!-- /catalog -->

## What it's for

Speaker diarization: who spoke when. This is **not a transcription
model** — a run produces no text. It takes a 16 kHz mono WAV and emits
speaker segments (`t0_ms`, `t1_ms`, `speaker_id`), with speakers numbered
in order of first appearance. Hard architectural cap of 4 concurrent
speakers. It is the diarization supervisor for the Parakeet multitalker
speaker-attributed ASR path (future work).

See NVIDIA's [model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
for training data, intended use, and upstream evaluation methodology.

<!-- catalog:pin -->
Licensed NVIDIA Open Model License. Ported from upstream commit [`fafaab5`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/commit/fafaab5), pinned 2026-07-19. Validated against the NeMo reference at transcribe.cpp commit [`d42c3bb`](https://github.com/handy-computer/transcribe.cpp/tree/d42c3bb) on 2026-07-22.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |   Size | DER (AMI IHM test) |
| --- | --- | ---: | ---: |
| F32          | [diar_streaming_sortformer_4spk-v2.1-F32.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-F32.gguf) | 471 MB | 14.59% |
| F16          | [diar_streaming_sortformer_4spk-v2.1-F16.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-F16.gguf) | 237 MB | 14.23% |
| Q8_0         | [diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf) | 139 MB | 14.73% |
<!-- /catalog -->

<!-- catalog:recipe -->
DER on the full AMI IHM test split (16 meetings). Figures without a commit were published before provenance was recorded.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Scored against forced-alignment RTTMs with dihard3-dev post-processing, collar 0.0,
overlap scored, at the very_high_latency operating point. Measured NeMo reference
under the identical protocol: 14.83% DER / 19.89% JER; the C++ F32 port scores
14.59% / 19.51%. Published DER numbers vary with RTTM source and post-processing;
compare like with like. Only near-reference tiers ship for this family (k-quant
tiers withdrawn; see the transcribe.cpp family doc, "Quant policy (Stage 7)").
<!-- /catalog -->

Only near-reference tiers ship for this family. K-quant tiers were
evaluated and withdrawn: the model's output depends on discrete
speaker-cache decisions, and k-tier weight error can deterministically
flip a near-tie and permute speaker labels mid-stream (observed at
Q5_K_M on one AMI meeting; details in
`docs/porting/families/sortformer.md`, "Quant policy (Stage 7)").

## Quick Start

```bash
cmake -B build
cmake --build build

# Speaker segments as JSON (one line per file):
echo audio.wav > files.txt
build/bin/transcribe-cli \
  -m models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf \
  --batch files.txt --batch-jsonl
# {"file":"audio.wav","text":"","speakers":[{"t0_ms":320,"t1_ms":2400,"speaker_id":1},...]}
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

From the C API, read results via `transcribe_n_speaker_segments` /
`transcribe_get_speaker_segment` (the transcript accessors return empty
text). The streaming operating point (latency / accuracy trade-off) is
selected with the run extension in `include/transcribe/sortformer.h`:

```c
transcribe_sortformer_stream_ext ext;
transcribe_sortformer_stream_ext_init(&ext);        /* DEFAULT = model cfg */
ext.preset = TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY;
run_params.family = &ext.ext;
```

`VERY_HIGH_LATENCY` (~30 s lookahead) is the offline-file operating
point used for the DER numbers above; `LOW_LATENCY` (~1 s lookahead) is
the real-time point and costs substantially more compute per audio
second (many small windows).

## Performance

### Apple M4

<!-- catalog:perf machine=m4 -->
Compute latency (mel + encode + decode), speedup over realtime in parentheses.

| Backend | Sample       |              F16 |             Q8_0 |
| ------- | ------------ | ---------------: | ---------------: |
| Metal   | jfk (11.0s)  |  68 ms (161.23×) |  64 ms (172.42×) |
| Metal   | dots (35.3s) | 316 ms (111.81×) | 318 ms (111.16×) |
| CPU     | jfk (11.0s)  |  136 ms (80.68×) | 109 ms (101.09×) |
| CPU     | dots (35.3s) |  794 ms (44.49×) |  685 ms (51.59×) |

m4: transcribe.cpp `d42c3bb` on 2026-07-22.
<!-- /catalog -->

Benchmark reproduction:

```bash
uv run scripts/bench/run.py --profile --models diar_streaming_sortformer_4spk-v2.1
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/sortformer-2spk-mix.wav` (a committed deterministic 2-speaker
mix with a 1.5 s overlap). All 6 checkpointed tensors fall within family
tolerance, and the streaming AOSC cache-compression internals were
additionally verified bit-exact against NeMo at the index level on a
full 39-minute AMI meeting (87 compression calls).

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| Dump script | `scripts/dump_reference_sortformer_nemo.py` |
| Manifest | `tests/golden/sortformer/diar_streaming_sortformer_4spk-v2.1.manifest.json` |
| Command | `uv run scripts/validate.py compare --family sortformer` |

## Known Limitations

- **Maximum 4 speakers** (architectural cap). More than 4 concurrent
  speakers will be merged into the 4 arrival-order slots.
- **No transcript.** Pair with an ASR model if you need speaker-attributed
  text; native multitalker interop is planned, not shipped.
- **Speaker labels are arrival-order, not identities.** Labels are stable
  within a recording, but there is no cross-recording speaker matching.
- **Perturbation-sensitive label continuity.** The streaming speaker
  cache makes discrete near-tie decisions; different backends (Metal vs
  CPU float order) can occasionally resolve a near-tie differently on a
  given recording, changing label assignment mid-stream. Aggregate DER is
  unaffected in our measurements; the shipped F16/Q8_0 tiers showed no
  label-swap events on the 16-meeting acceptance set.
- **`LOW_LATENCY` is compute-heavy on CPU** (~1.2x realtime on Apple M4;
  the 0.5 s chunks rebuild the compute graph often). Use Metal or a
  higher-latency preset for offline files.
- **No batch fast path** (`run_batch`); multi-file CLI batches run
  serially.

## Reproduction

### Convert

Loads NVIDIA's NeMo checkpoint via `SortformerEncLabelModel.from_pretrained`.

```bash
uv run --project scripts/envs/sortformer \
  scripts/convert-sortformer.py nvidia/diar_streaming_sortformer_4spk-v2.1 \
  --out models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf
```

### Quantize

```bash
build/bin/transcribe-quantize \
  models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf \
  models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F16.gguf \
  --quant F16
# repeat with Q8_0
```

### Validate

```bash
uv run scripts/validate.py all --family sortformer
```

### DER acceptance

The full protocol (AMI ingest, forced-alignment RTTMs, reference run,
scoring) is documented with commands in
`docs/porting/families/sortformer.md`. Short form:

```bash
uv run --project scripts/envs/sortformer scripts/diar/run_cpp_sortformer.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --gguf models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf \
  --preset very_high_latency \
  --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
  --pred-dir reports/diar/pred/cpp-ami --out reports/diar/cpp-ami.jsonl
uv run scripts/diar/score_der.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --pred-dir reports/diar/pred/cpp-ami --out reports/diar/cpp-ami.score.json
```

### Run real-model tests

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_SORTFORMER_GGUF=models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf \
  ctest --test-dir build --output-on-failure -R sortformer
```
