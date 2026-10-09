# Silero VAD (voice activity detection)

Status: supported (VAD role). Silero VAD v6.2 16 kHz model, plus whisper.cpp's
published v5.1.2 / v6.2.0 `.bin` weights. Usage: `docs/vad.md`; downloads and
measurements: `docs/models/silero-vad-v6.2.md`.

The model scores 512-sample (32 ms) frames with a speech probability
(architecture pattern `encoder-classifier`). The family scores whole frames
and owns its recurrent state. The VAD role dispatcher
(`src/transcribe-vad.cpp`) owns input validation, stream buffering, zero
padding of the last frame, and the probabilities -> segments step, a
line-for-line port of upstream's `get_speech_timestamps_from_probs`; the live
iterator is a port of `VADIterator`. Both keep the Silero MIT notice
(`src/third_party/silero_vad/`).

## Identity

- Family key: `silero_vad`; variant `silero-vad-v6.2`
- Upstream: `silero_vad.jit` `VADRNNJITMerge._model` (16 kHz sub-model),
  PyPI `silero-vad==6.2.3` (tag `v6.2.3`, commit `5cd7945`);
  `silero_vad/data/silero_vad.jit` sha256
  `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`
- License: MIT
- `.bin` runtime variants: `silero-vad-v5.1.2`, `silero-vad-v6.2.0`. Their
  mixed F16/F32 payloads are expanded to F32 at load.

## References

- Canonical: `silero_vad.load_silero_vad()` (TorchScript) scored as
  `get_speech_timestamps` does (reset, one `model(chunk, 16000)` per frame,
  last frame zero padded), torch 2.8.0, fp32, CPU, one thread.
- Instrumented: `scripts/dump_reference_silero_vad_author.py`. TorchScript
  takes no forward hooks, so each frame is also replayed through the scripted
  submodules in `VADRNNJIT.forward` order; the dumper asserts both paths agree
  bit for bit and segment to exactly `get_speech_timestamps(audio, model)`.
- whisper.cpp's port was read for orientation only. Upstream reflect-pads 64
  samples on the right of each chunk only (4 STFT windows per frame);
  whisper.cpp pads both sides.

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

Implementation:

- Everything before the recurrence is state independent and runs as one ggml
  graph over up to 256 frames (`encoder.cpp`). The convs act on 4, 4, 2 and 1
  time steps, so the loader unrolls each into a dense matrix; the STFT basis
  is padded from 129 to 136 bins per half so ggml's tinyBLAS path is taken.
- The LSTM recurrence and head (`decoder.cpp`) run on the host.
- Graphs shorter than 4 frames are zero padded to 4 so ggml picks the same
  matmul kernel on CPU; this makes streaming bit-identical to offline.
- `AUTO` resolves to the CPU: ggml-metal's F32 `mul_mm` stages operands as
  half (about 2e-3 in probability).

## GGUF contract

Keys: `stt.variant`, `stt.frontend.{sample_rate, n_fft, hop_length}`,
`stt.vad.frame_samples`, `stt.silero_vad.{context_samples, reflect_pad,
encoder_channels, encoder_strides, lstm_hidden, source_sha256}`,
`stt.capability.streaming = true`.

Tensors (all F32): `frontend.stft_basis` ne `[256, 258]`,
`enc.{0..3}.conv.weight` ne `[3, IC, OC]` and `.bias`,
`lstm.{weight_ih, weight_hh, bias_ih, bias_hh}` (torch gate order),
`head.{weight, bias}`.

## Commands

```bash
uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py
uv run scripts/preflight.py --family silero_vad --variant silero-vad-v6.2 --gate B
uv run scripts/validate.py all --family silero_vad --variant silero-vad-v6.2

# Segment parity on a corpus
uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
  --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf samples/diar/ami-ihm-test/*.wav

# Regenerate the segmentation and iterator test vectors
uv run --project scripts/envs/silero_vad scripts/vad/gen_vectors.py

# Real-model smoke (each variable optional)
TRANSCRIBE_SILERO_VAD_GGUF=$PWD/models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
TRANSCRIBE_SILERO_VAD_V5_BIN=/path/ggml-silero-v5.1.2.bin \
TRANSCRIBE_SILERO_VAD_V6_BIN=/path/ggml-silero-v6.2.0.bin \
  ctest --test-dir build -R silero_vad
```

## Capability Validation

| Capability | Command / test | Expected | Status |
|---|---|---|---|
| Offline segments | `validate.py`; `scripts/vad/parity.py` | identical to `get_speech_timestamps` | PASS (9/9 clips, AMI 16/16, 5505 LibriSpeech + FLEURS) |
| Segment params | `transcribe_vad_dispatch_unit` | identical to `get_speech_timestamps_from_probs` | PASS |
| Live iterator | `transcribe_vad_iterator_unit` | identical `VADIterator` events | PASS |
| Per-frame probabilities | `validate.py` | within `tests/tolerances/silero_vad.json` | PASS (max 6.9e-6) |
| Streaming | `transcribe_silero_vad_smoke` | bit-identical to offline on CPU | PASS |
| `.bin` loading | `transcribe_silero_vad_bin_smoke` | same probabilities as the equivalent GGUF | PASS |
| 8 kHz model, quantized GGUFs | - | OUT OF SCOPE | SKIP |
