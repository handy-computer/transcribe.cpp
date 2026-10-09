# Silero VAD (voice activity detection)

Status: supported (VAD role). Independently implemented inference for the
Silero VAD v6.2 16 kHz model and published v5.1.2/v6.2.0 binary weights,
validated against upstream's own JIT (tag `v5.1.2` /
`silero-vad==6.2.3`); probability policies are translated from upstream.

Silero VAD scores 512-sample (32 ms) frames of 16 kHz audio with a speech
probability (architecture pattern `encoder-classifier`): a conv-STFT
magnitude front end, four 1-D convs, an LSTM cell carried across frames,
and a sigmoid head. It is not a transcription model. The family scores whole
frames and owns its recurrent state; the VAD role dispatcher
(`src/transcribe-vad.cpp`) owns input validation, frame buffering for
streaming, zero padding of the last frame, and the probabilities -> segments
step (a line-for-line port of upstream's `get_speech_timestamps_from_probs`).
Offline/live probability policies retain the Silero MIT notice in
`src/third_party/silero_vad/LICENSE` and distributed artifacts.
API and usage: `docs/vad.md`.

Acceptance: per-frame tensor parity on nine clips (`validate.py`), and exact
speech-segment parity with `get_speech_timestamps` (default parameters) on
those clips, on AMI IHM test (16 meetings, 9.1 h) and on LibriSpeech
test-clean plus four FLEURS languages (5505 clips). The segmentation port is
additionally pinned by 360 reference-generated vectors covering every
parameter (`tests/vad_dispatch_unit.cpp`). Shipped matrix: F32 only (1.2 MB;
quantization buys nothing for our GGUF). The upstream binary compatibility
path expands the published mixed F16/F32 payloads to F32 without recovering
lost precision. It is not a new engine or a canonical GGUF publication.

## Identity

- Family key: `silero_vad`
- Upstream: `silero_vad.jit` `VADRNNJITMerge._model` (16 kHz sub-model)
- Source: PyPI `silero-vad==6.2.3` (GitHub `snakers4/silero-vad` tag
  `v6.2.3`, commit `5cd7945`); `silero_vad/data/silero_vad.jit` sha256
  `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`,
  unchanged since tag `v6.2`. This original JIT is not distributed on Hugging Face.
- License: MIT
- Golden-manifest/GGUF variant: `silero-vad-v6.2` (unchanged; no new manifests).
- Binary runtime variants: `silero-vad-v5.1.2`, `silero-vad-v6.2.0`.
- Binary downloads and hashes: `docs/models/silero-vad-v6.2.md`.
- Stored-weight reference pins, including original v5 JIT: `scripts/lib/silero_bin_reference.py`.

## References

- Canonical reference: `silero_vad.load_silero_vad()` (TorchScript) scored
  exactly as `get_speech_timestamps` does (reset, one `model(chunk, 16000)`
  call per 512-sample frame, last frame zero padded), torch 2.8.0, fp32, CPU,
  one thread.
- Instrumented reference: `scripts/dump_reference_silero_vad_author.py`.
  TorchScript modules take no forward hooks, so every frame is scored twice:
  end to end through `model(chunk, 16000)`, and step by step through the
  scripted submodules in the order `VADRNNJIT.forward` calls them. The dumper
  asserts the two agree bit for bit, and that its probabilities segment to
  exactly `get_speech_timestamps(audio, model)`.
- whisper.cpp's Silero port was read for orientation only; nothing here is
  checked against it. One difference worth knowing: upstream reflect-pads 64
  samples on the right of each 576-sample chunk only, which gives 4 STFT
  windows per frame; whisper.cpp pads both sides.

### Binary stored-weight reference

`scripts/lib/silero_bin_reference.py` verifies official hashes and all 15
payloads against rounded upstream weights, then loads those values into the
original v5/v6 JIT. The existing dumper's bit-exact stepwise/end-to-end bridge
is retained. `bin_parity.py` checks all nine clips at 1/4 threads: intermediate
tensors, exact segments/live events and bit-identical offline/stream scores.

V6 uses `tests/tolerances/silero_vad.json`; v5's larger accumulated cell state
needs a separate `dec.lstm_c` bound. Its FP64 analysis and bound derivation are
in `silero_vad-bin-v5.1.2.json`; probability bounds remain unchanged.

## Forward pass

Per frame (`src/arch/silero_vad/silero_vad.h`):

```
chunk  = [context (64) | frame (512)]          context = previous chunk's last 64
padded = reflect_pad_right(chunk, 64)          640 samples, torch "reflect"
mag    = |basis . windows(padded, 256, hop 128)|   [129 bins, 4 windows]
enc    = 4 x (conv1d k3 p1 + ReLU), strides 1, 2, 2, 1 -> [128, 1]
h, c   = LSTMCell(enc, (h, c))                 gates i, f, g, o
p      = sigmoid(head . relu(h) + b)
```

`forward_basis_buffer` is the periodic-Hann real DFT (rows 0-128 cos, rows
129-257 -sin); the converter checks it to 1e-6. The last window of every
frame sees reflected samples, not the next frame's audio, so the STFT is per
frame and cannot be shared.

Implementation:

- Everything before the recurrence is state independent, so it runs as one
  ggml graph over a block of up to 256 frames
  (`src/arch/silero_vad/encoder.cpp`). Each frame's receptive field is one
  column; every stage is one `mul_mat` over the block. The four convs act on
  4, 4, 2 and 1 time steps, so the loader unrolls each into a dense matrix
  over the frame's time-major features (padding taps are zero), and the LSTM
  input projection plus both biases join the same graph. The STFT basis is
  padded from 129 to 136 bins per half (zero rows) so every dimension is a
  multiple of the SIMD width and ggml's tinyBLAS path is taken.
- The recurrence (`decoder.cpp`) is a 128 x 512 matvec and elementwise work
  per frame: it runs on the host from a transposed W_hh copy, with a
  polynomial `expf` (sigmoid / tanh within 9e-8 absolute of exact).
- Graphs shorter than 4 frames are zero padded to 4, so ggml always picks the
  same matmul kernel on CPU. That makes CPU streaming probabilities
  bit-identical to an offline run with the same thread count for any chunking
  (tested at 1 / 100 / 511 / 512 / 513 / ... samples, 1 and 4 threads).
- `TRANSCRIBE_BACKEND_AUTO` resolves to the strict CPU backend. ggml-metal's
  F32 `mul_mm` stages operands as half, about 2e-3 in probability (segments
  still match on jfk); an explicit GPU request is honored.

## GGUF contract

Keys: `stt.variant`, `stt.frontend.{sample_rate, n_fft, hop_length}`,
`stt.vad.frame_samples`, `stt.silero_vad.{context_samples, reflect_pad,
encoder_channels, encoder_strides, lstm_hidden, source_sha256}`,
`stt.capability.streaming = true`. The loader derives every time-step count
from them and rejects a geometry whose encoder does not end on one step.

Tensors (all F32): `frontend.stft_basis` ne `[256, 258]`,
`enc.{0..3}.conv.weight` ne `[3, IC, OC]` (torch layout) and `.bias`,
`lstm.{weight_ih, weight_hh, bias_ih, bias_hh}` (torch gate order), and
`head.{weight, bias}`.

## Commands

```bash
# Conversion (reads the TorchScript model from the pinned wheel)
uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py

# Cheap GGUF configuration gate (variant also makes model discovery explicit)
uv run scripts/preflight.py --family silero_vad --variant silero-vad-v6.2 --gate B

# Reference dumps, C++ dumps, tensor + segment parity
uv run scripts/validate.py all --family silero_vad --variant silero-vad-v6.2

# Segment parity on a corpus (WAVs or scripts/wer manifests)
uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
  --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf samples/diar/ami-ihm-test/*.wav

# Upstream speed, for comparison
uv run --project scripts/envs/silero_vad scripts/vad/bench_reference.py samples/love-loss.wav

# Segmentation test vectors (after a reference bump)
uv run --project scripts/envs/silero_vad scripts/vad/gen_segment_vectors.py

# Live iterator vectors (actual upstream with a probability stub)
uv run --project scripts/envs/silero_vad scripts/vad/gen_iterator_vectors.py

# Published binary stored-weight numerical parity (V5 JIT fetched/verified if absent)
uv run --project scripts/envs/silero_vad scripts/vad/bin_parity.py \
  --bin /path/ggml-silero-v5.1.2.bin
uv run --project scripts/envs/silero_vad scripts/vad/bin_parity.py \
  --bin /path/ggml-silero-v6.2.0.bin

# Binary public API smoke (each variable optional)
TRANSCRIBE_SILERO_VAD_V5_BIN=/path/ggml-silero-v5.1.2.bin \
TRANSCRIBE_SILERO_VAD_V6_BIN=/path/ggml-silero-v6.2.0.bin \
  ctest --test-dir build -R silero_vad

# GGUF real-model smoke
TRANSCRIBE_SILERO_VAD_GGUF=models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
  ctest --test-dir build -R silero_vad
```

## Capability Validation

| Capability | Mode | Command / test | Expected | Target | Status |
|---|---|---|---|---|---|
| Offline segments | default params | `validate.py all --family silero_vad`; `scripts/vad/parity.py` | segments identical to `get_speech_timestamps` | MUST PASS | PASS (9/9 clips, AMI 16/16, 5505 LibriSpeech + FLEURS clips) |
| Segment params | every field | `transcribe_vad_dispatch_unit` (360 vectors) | identical to `get_speech_timestamps_from_probs` | MUST PASS | PASS |
| Live START/END policy | upstream defaults and edge cases; arbitrary grouping/reset/EOF | `transcribe_vad_iterator_unit` (15 named cases); `scripts/vad/bin_parity.py` | exact `VADIterator` events on the same float32 probabilities | MUST PASS | PASS |
| Per-frame probabilities | CPU, 1 thread | `validate.py` (`vad.probs`) | within `tests/tolerances/silero_vad.json` | MUST PASS | PASS (max 6.9e-6) |
| Streaming | CPU, any chunking, 1 / 4 threads | `transcribe_silero_vad_smoke`, `_real_smoke` | bit-identical to offline | MUST PASS | PASS |
| GPU backend | explicit Metal | `transcribe_silero_vad_real_smoke` | within 5e-3, same segments | OUT OF SCOPE - CPU is the target; Metal is a smoke check | PASS (2.0e-3) |
| 8 kHz model | - | - | - | OUT OF SCOPE - library input is 16 kHz | SKIP |
| Quantized GGUFs | - | - | - | OUT OF SCOPE - 1.2 MB model, nothing to gain | SKIP |
