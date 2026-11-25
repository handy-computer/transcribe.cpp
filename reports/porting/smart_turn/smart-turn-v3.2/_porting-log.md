# smart_turn / smart-turn-v3.2 porting log

Surprises worth feeding back into the docs and tooling.

- **ONNX is the only checkpoint.** No safetensors, no config.json:
  `scripts/intake.py inspect` finds nothing, so the intake was written by
  hand from train.py and the ONNX graph. Gate A can only warn; Gate B passes.
- **Anonymous, transposed weights.** The exporter stored every Linear that
  became a MatMul as `val_N` in `[in, out]` order. Naming each by the bias
  Add consuming it (k_proj, which has no bias, by the self_attn_layer_norm
  feeding it) is unambiguous; loaded strictly into train.py's
  `SmartTurnV3Model` it reproduces onnxruntime to 5e-7 in p. The map is in
  `scripts/lib/smart_turn_onnx.py`, shared by the converter and the dumper.
- **Intermediates from onnxruntime.** Exposing dump points as graph outputs
  changes which fusions apply: p moves by up to 6e-8. The dumper writes the
  unmodified session's p as `head.prob` and bounds the instrumented one at
  1e-6.
- **Two front ends in transformers.** WhisperFeatureExtractor silently
  switches to a float32 torch STFT when torch is importable. The reference
  env needs torch (for the twin), so the dumper pins the numpy path, which
  upstream inference and Pipecat's vendored copy run (4e-5 in the mel,
  <= 4e-7 in p).
- **Massive activations.** whisper-tiny's encoder carries a few channels to
  |x| ~150 from block 2 on. Two correct fp32 implementations differ there by
  up to 2e-2, 25x the Stage-2 recipe, while p differs by <= 5e-6;
  `enc.block.{2,3}.out` and `enc.final` were widened.
- **The positional table is not `sinusoids()`.** train.py's 400 positions
  made from_pretrained re-initialise whisper-tiny's 1500-row table on the
  training device; it differs from a CPU `sinusoids(400, 384)` by up to
  3.1e-5, so the GGUF stores the exported table.
- **Silence is near the cut-off.** All-zero audio normalises to zeros, the
  mel is a constant -1.5, and p = 0.4615; the int8 ONNX gives 0.987.
- **validate.py had no TURN stage.** Added `smart_turn` to the
  single-`encoder`-stage families and the `--revision` pass-through, and a
  `turn.json` comparison (p and decision) for the role.
- **Test-set baseline.** `scripts/turn/run_reference_smart_turn_onnx.py`
  reproduces upstream's published fp32 accuracy exactly (int8 within 0.01
  points); per-clip p for both ONNX files is the debug oracle.
- **Rig-built GGUFs.** The Ryzen speed rows ran on GGUFs converted on the
  rig; their sha256 differ from the canonical files (F32 7c3d8410..., F16
  c647e9e8..., Q8_0 23ff4cde...) only by the converter's last-bit constants
  (rig: 6ec842d1..., f8f60806..., f7362588...). `ingest_perf.py` keys a cell
  by quant, not hash, so nothing checks this.
- **Engine SHA of the Pipecat runs.** One tool binary built at `8f97ab5b`
  ran all six; scripts-only commits made meanwhile put HEAD (`8415e8e2` /
  `dcb63a1b`) in the F16 and Q8_0 headers. They now say `8f97ab5b` and keep
  the recorded value as `engine_sha_recorded`; `eval_cpp.py --engine-sha`
  names a prebuilt tool's commit after checking the engine sources match.
- **Forward map** was not saved at Stage 4; written from the finished port.
- **Upload (2026-11-25, owner sign-off on tolerances, capability tables and
  HF card):** private `handy-computer/smart-turn-v3.2-gguf` (`74045442`);
  downloaded sha256 equal the canonical files and `validate.py all` passes
  on the downloaded F32.
