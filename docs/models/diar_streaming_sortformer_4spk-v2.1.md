# Streaming Sortformer Diarizer 4spk v2.1

NVIDIA's [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
ported to transcribe.cpp. A FastConformer encoder with an 18-layer
Transformer head that emits per-frame speaker-activity probabilities for
up to 4 speakers, running online with an Arrival-Order Speaker Cache
(AOSC) + FIFO.

## What it's for

Speaker diarization: who spoke when. This is **not a transcription
model** — a run produces no text. It takes a 16 kHz mono WAV and emits
speaker segments (`t0_ms`, `t1_ms`, `speaker_id`), with speakers numbered
in order of first appearance. Hard architectural cap of 4 concurrent
speakers. It is the diarization supervisor for the Parakeet multitalker
speaker-attributed ASR path (future work).

See NVIDIA's [model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
for training data, intended use, and upstream evaluation methodology.

Licensed under the NVIDIA Open Model License. Ported from upstream commit
[`fafaab5`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/commit/fafaab5faa1617a0ca52d38dd3dc4bd636800d3d),
pinned 2026-07-19.

## Download

| Quantization | Download | Size | DER (AMI IHM test) |
| --- | --- | ---: | ---: |
| F32  | [diar_streaming_sortformer_4spk-v2.1-F32.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-F32.gguf)   | 471 MB | 14.59% |
| F16  | [diar_streaming_sortformer_4spk-v2.1-F16.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-F16.gguf)   | 237 MB | 14.23% |
| Q8_0 | [diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf](https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf) | 139 MB | 14.73% |

DER is measured on the full AMI IHM test set (16 meetings, ~9 h) against
forced-alignment RTTMs with dihard3-dev post-processing, collar 0.0,
overlap scored, at the `very_high_latency` operating point. Our measured
NeMo reference under the identical protocol is **14.83% DER / 19.89%
JER**; the C++ F32 port scores 14.59% / 19.51%. (Published DER numbers
for this model vary with the RTTM source and post-processing; manual
RTTMs score ~13 points worse than forced-alignment RTTMs on the same
system output. Compare like with like.)

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

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Default (model-config)
operating point.

### Apple M4

| Backend | Sample       |           F16 |          Q8_0 |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  69 ms (159×) |  65 ms (171×) |
| Metal   | dots (35.3s) | 318 ms (111×) | 320 ms (111×) |
| CPU     | jfk (11.0s)  |  137 ms (80×) | 110 ms (100×) |
| CPU     | dots (35.3s) | 796 ms (44×)  | 687 ms (51×)  |

macOS 25.5.0, transcribe.cpp `d42c3bb`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models diar_streaming_sortformer_4spk-v2.1 \
  --quants f16,q8_0 \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name diar_streaming_sortformer_4spk-v2.1-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo on
`samples/sortformer-2spk-mix.wav` (a committed deterministic 2-speaker
mix with a 1.5 s overlap). All 6 checkpointed tensors fall within family
tolerance, and the streaming AOSC cache-compression internals were
additionally verified bit-exact against NeMo at the index level on a
full 39-minute AMI meeting (87 compression calls). Last validated at
commit `d42c3bb`.

| Field | Value |
| --- | --- |
| Reference | NeMo, `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| Dump script | `scripts/dump_reference_sortformer_nemo.py` |
| Manifest | `tests/golden/sortformer/diar_streaming_sortformer_4spk-v2.1.manifest.json` |
| Command | `uv run scripts/validate.py compare --family sortformer` |

Selected tensors:

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`           | `0.000e+00` | `0.000e+00` | Exact (shape differs by NeMo pad_to=16, values identical) |
| `enc.fastconformer.out`| `3.263e-03` | `1.396e-04` | F32 accumulation over 17 Conformer blocks |
| `enc.encoder_proj.out` | `9.829e-04` | `1.242e-04` | Drift attenuates through the projection |
| `enc.transformer.out`  | `1.128e-03` | `1.323e-04` | 18-layer Transformer head |
| `diar.preds_offline`   | `2.961e-04` | `5.007e-06` | Final sigmoid probabilities (offline) |
| `diar.probs`           | `2.961e-04` | `5.007e-06` | Streaming path output (== offline on a single-chunk clip) |

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
