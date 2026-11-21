# sava / sava porting log

Surprises worth feeding back into the docs and tooling.

- **No HF config.** The configuration lives in the safetensors metadata
  (`format=sava`, `config=<json>`). `scripts/intake.py inspect` finds
  nothing, and HF's safetensors-metadata API reports no dtype, so the intake
  was written by hand. Gate A warns on dtype and front end (nothing to
  cross-check against), the same as Silero. Gate B passes.
- **bf16 source, fp32 reference.** Upstream upcasts to fp32 on CPU, so the
  intake declares `float32` (manual) and the F32 GGUF is exact. BF16 is a
  lossless storage option.
- **Hooks see the real run.** `sava.run` is plain PyTorch, so forward hooks
  and instance-method wrappers capture the intermediates (attention
  `forward_chunk`, `mixer.rows`, `_turn_input`, `ssd.chunked_scan`) of the
  same call. The dumper checks that the captured logits reproduce the curves
  bit for bit.
- **Run and stream differ in the reference** (up to 3.1e-6 in probability):
  chunked SSD and windowed attention offline, recurrence and per-frame turn
  head in a Session. Upstream's own golden test allows 1e-5 between them.
- **RoPE in fp32 with absolute positions.** The reference's outputs depend on
  stream age: 4e-4 in probability after an hour, 3e-3 after 11 h. The model
  is shift invariant in exact arithmetic. ggml_rope's iterated theta_scale
  does not reproduce torch's per-index frequencies, so `rope.freqs` ships in
  the GGUF.
- **Quant-policy name collisions.** The parakeet-ultra rule
  `vad.*.weight -> Conv` would have caught a `vad.` head prefix, hence
  `head.vad.*`. The Python mirror lacks policy.cpp's `_norm.weight` suffix
  rule, hence `norm_*` names. ggml_ssm_conv and ggml_ssm_scan need F32 conv
  kernels and A, so new F32 rules for `*.ssm.{a,d,conv1d.weight}`,
  `*.speech_embed.weight`, `frontend.dft_basis` and `rope.freqs` were added
  to policy.cpp, gguf_common.py and the sync test.
- **Thresholds as float64 KVs.** 0.9 in fp32 is 0.89999998, which would
  accept a probability the reference rejects.
- **The turn policy is part of the contract.** It has overlapping segments,
  dropped (`kept=false`) turns, ends up to 3 frames past the audio, flush-step
  turn scores, and Python half-to-even rounding for pre-roll. These were sent
  to the TURN API design.
- **validate.py** runs the reference for a new family without code changes,
  through per-case `"stages": ["encoder"]`.

## Stage 4 (C++)

- **Stream == run by construction.** Every block of frames runs as one graph
  of at least 4 columns (ggml's NEON tinyBLAS threshold), and the KV rings
  are written before attention with a slot order that depends only on the
  position. A stream is bit-identical to a run for any chunking and thread
  count, on the real model and the toy fixture.
- **ggml_ssm_scan's scalar remainder breaks that.** With a d_state that is
  not a multiple of the SIMD step (the toy's first try used 8), a graph's
  first token rounds differently from the later ones. The real d_state of
  128 is fine. The toy fixture now uses 64.
- **The reference's front end is the main mel error.** Its fp32 DFT GEMM is
  cancellation-bound near the 1e-10 mel floor. With a ggml fp32 DFT the C++
  was 1.25e-3 off on jfk. A host fp64 FFT makes the C++ exact to 6e-8, so
  the remaining fe.mel drift (7.9e-4 max) is the reference's own error.
  jfk's speech drift fell from 3e-5 to 6.7e-6.
- **fp64 RoPE (owner decision)** departs from the fp32 reference by its own
  stream-age drift: 3.9e-5 in speech after 6,000 frames on dots-full. With
  fp32 angles it would be 1.2e-5.
- **validate.py needed no SAVA code.** The TURN role path compares the CLI's
  turn / segment lines with the dumper's turn.json.
- **Streaming speed.** A 1-frame step pays for a 4-column graph: 5.4 ms per
  frame with 1 thread, 3.4 ms with 4, on an M1. Offline is 0.94 ms per frame
  with 1 thread.

## Stage 5 (quants)

- **Q8_0 needed no policy work beyond Stage 3's rules.** `mixer.proj`'s
  1026-wide rows fall back to F16. Stream == run stays bit-identical with
  Q8_0 on M1.
- **BF16 is not free on CPU.** The storage is lossless, but ggml's BF16
  matmul on an M1 (no native bf16) is ~40x slower than F32.
- **The pipecat clip benchmark reproduces Inworld's published SAVA row.**
  F32 AUC 0.97946 vs 0.97946 published (A100 fp32), and accuracy within 2
  clips. Q8_0 loses 1.1e-4 AUC.
- **Q8_0 changes turns only on threshold-margin frames.** On dots-full 6 of
  59 turns move or merge, each at a score within 0.004 of 0.9 or 0.03 of
  0.5.
- **16-bit tiers.**
  - **F16 is effectively lossless.** bf16 source values fit f16 except
    1,065 tiny ones, and ggml keeps F16 x F32 matmuls in fp32 activations. It
    passes the F32 tolerance gate, with 0 Pipecat decision flips. It is
    slower per clip frame than F32 on M1 NEON (no tiled F16 x F32 kernel).
  - **BF16 is lossless storage but lossy compute.** ggml rounds the
    activations to bf16, and the M1 runs it 15-22x slower than F32. Shipped
    F16, not BF16.
  - **Measurement noise.** Speed numbers taken while other worktrees ran
    batch jobs (load 10-36) were off by 2-25x; only idle-machine numbers
    are recorded.
