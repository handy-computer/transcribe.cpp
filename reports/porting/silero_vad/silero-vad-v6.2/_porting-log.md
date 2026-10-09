# silero_vad / silero-vad-v6.2 porting log

Surprises worth feeding back into the docs and tooling.

- **No Hugging Face source.** Silero ships the model inside its PyPI wheel
  (TorchScript, ONNX, safetensors). `scripts/intake.py` and preflight are
  built around HF repos, so the intake was written by hand
  (`hf_revision: null`), provenance is the package version plus the `.jit`
  sha256, and the converter refuses any other version or hash.
- **TorchScript takes no forward hooks.** The dumper instead replays each
  frame through the scripted submodules in `VADRNNJIT.forward` order and
  asserts the result equals `model(chunk, 16000)` bit for bit, so the
  captured intermediates are provably the reference's.
- **New role.** VAD needed `TRANSCRIBE_ROLE_VAD` (bit 3), ABI ids 24-28,
  `include/transcribe/vad.h`, `Arch::vad`, a dispatcher and a CLI driver.
  The role dispatcher owns the segmentation so future VAD families only
  score frames.
- **The segmentation is part of the reference contract.** Ported line for
  line, including Python truthiness of sample position 0 and float
  thresholds; the public thresholds are `double` so values like 0.6 compare
  as the reference's Python floats do. Pinned by 360 vectors generated from
  the reference; mutation checks showed the first vector set missed the
  max-speech split paths, so the generator gained frame-aligned ms values and
  a long-speech generator with tied silences.
- **One mutant is equivalent.** In the max-speech split, the reference's
  `if next_start < prev_end + cur_sample` is always true once a silence was
  recorded (positions are non-negative); replacing it with `true` changes
  nothing over 20,000 random vectors. Kept verbatim.
- **ggml kernel choice changes reduction order.** The CPU matmul takes
  tinyBLAS only from 4 columns on NEON; 1-3 frame graphs used another
  kernel and streaming differed from offline at the 1e-7 level. Padding
  graphs to at least 4 frames made them bit-identical.
- **SIMD-friendly padding.** 129 STFT bins defeated tinyBLAS (m % 4 != 0);
  padding the basis to 136 bins per half (zero rows) cut the encoder time
  by 37%.
- **ggml-metal F32 mul_mm stages operands as half**: 2e-3 probability drift.
  AUTO resolves to CPU for this family.
- **Tolerances**: two entries (enc.3.out, dec.lstm_h) exceed the Stage-2
  magnitude budget by up to 1.3x; mechanism recorded in
  `tests/tolerances/silero_vad.json`.
