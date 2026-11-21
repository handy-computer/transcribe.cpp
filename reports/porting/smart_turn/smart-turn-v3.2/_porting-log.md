# smart_turn / smart-turn-v3.2 porting log

Surprises worth feeding back into the docs and tooling.

- **ONNX is the only checkpoint.** No safetensors, no config.json:
  `scripts/intake.py inspect` finds nothing, so the intake was written by
  hand from train.py and the ONNX graph. Preflight Gate A can only warn
  (no reference preprocessor / config to cross-check); Gate B passes.
- **Anonymous, transposed weights.** The torch exporter kept the names of
  biases, norms, convs and Gemm weights (`inner.*`) but stored every Linear
  that became a MatMul as `val_N` in `[in, out]` order. Naming them by the
  bias Add that consumes each MatMul (and, for k_proj, which has no bias, by
  the self_attn_layer_norm feeding it) is unambiguous; loading the result
  strictly into train.py's `SmartTurnV3Model` reproduces onnxruntime to
  5e-7 in p. The map lives in `scripts/lib/smart_turn_onnx.py`, shared by
  the converter and the dumper.
- **Intermediates from onnxruntime.** Adding dump points as graph outputs
  changes which fusions onnxruntime applies: p moves by up to 6e-8. The
  dumper writes the unmodified session's p as `head.prob` and bounds the
  instrumented one at 1e-6.
- **Two front ends in transformers.** WhisperFeatureExtractor silently
  switches to a float32 torch STFT when torch is importable. The reference
  env needs torch (for the twin), so the dumper pins the numpy path, which is
  what upstream inference (no torch in requirements_inference.txt) and
  Pipecat's vendored copy run. Difference: 4e-5 in the mel, <= 4e-7 in p.
- **Massive activations.** whisper-tiny's encoder carries channel 375 (and
  16, 128, 247) to |x| ~150 at a few positions from block 2 on. Two correct
  fp32 implementations (onnxruntime vs the PyTorch twin, and the C++ in
  progress) differ there by up to 2e-2, 25x the Stage-2 recipe; p differs by
  <= 5e-6. Expect Stage 4 to widen enc.block.2/3.out and enc.final.
- **Positional table is not bit-equal to sinusoids().** It has 400 rows:
  train.py sets max_source_positions = 400, so from_pretrained cannot load
  whisper-tiny's 1500-row table and re-initialises it with sinusoids(); it
  is frozen (requires_grad False). It still differs from a CPU
  sinusoids(400, 384) by up to 3.1e-5, growing with position (7.6e-6 below
  row 100): float sin / cos of large arguments on the training device. The
  GGUF stores the exported table.
- **Silence is near the cut-off.** All-zero audio normalises to zeros
  (variance 0, eps 1e-7), the mel is a constant -1.5, and p = 0.4615; the
  int8 ONNX gives 0.987 for the same input.
- **validate.py had no TURN stage.** Added `smart_turn` to the
  single-`encoder`-stage families and to the `--revision` pass-through. The
  role's behavioural artifact (`turn.json`, decision parity) is not compared
  yet (Stage 4).
- **Test-set baseline.** `scripts/turn/run_reference_smart_turn_onnx.py`
  reproduces upstream's published accuracy (fp32 93.71% / int8 92.63%) from
  the HF test set (fp32 exactly; int8 within 0.01 points of accuracy); per-clip p for both ONNX files is kept as the debug oracle.
