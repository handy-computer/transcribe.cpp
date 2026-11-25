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

## Stages 6-8 (publication, profile turn-publication-v1)

- **Speed, two targets, CPU only.** `scripts/turn/bench.py` through the
  Python binding's raw C ABI. `m1`: an Apple M1 standing in for the M4
  reference machine (unreachable), engine `8f97ab5b`, load 1.5-2.9 at each
  cell's start. `ryzen-4750u`: the Ryzen 7 PRO 4750U rig, engine `8415e8e2`
  (scripts-only difference), Tctl below 55 C and load below 3 at each cell's
  start. Load and Tctl per cell are in the bench reports.
- **Rig-built GGUFs.** The Ryzen rows were measured on GGUFs converted on the
  rig. Their sha256 differ from the canonical files (F32 7c3d8410...,
  F16 c647e9e8..., Q8_0 23ff4cde...) only by the converter's last-bit
  constants (rig: F32 6ec842d1..., F16 f8f60806..., Q8_0 f7362588...), as
  checked by the rig agent. ingest_perf.py identifies a
  cell by the file name's quant, not by hash, so nothing checks this; it is
  recorded here instead.
- **Accuracy.** The Pipecat runs were repeated at `8f97ab5b` (the earlier
  rows predated any commit), with `eval_cpp.py` now recording its engine
  commit; scored by `scripts/turn/score.py --ref-column p_fp32`.
- **Validation pin.** `validate.py all --family smart_turn` passes at
  `8f97ab5b` on 2026-11-24.
- **Forward map.** `reports/porting/smart_turn/forward-map.md` was missing
  (Stage 4 artifact); written from the finished port.
- **Engine SHA of the Pipecat runs.** All six runs used one tool binary built
  at `8f97ab5b`. The F16 and Q8_0 headers first recorded HEAD
  (`8415e8e2` / `dcb63a1b`): scripts-only commits made while they ran, with
  the same engine sources. The headers now say `8f97ab5b` and keep the
  recorded value as `engine_sha_recorded`. `eval_cpp.py --engine-sha` now
  names a prebuilt tool's commit, after checking the engine sources match.
- **Results at 8f97ab5b.** F32 93.71% (AUC 0.98252, 31,527/31,527 decisions
  equal to the fp32 ONNX), F16 93.70% (31,525), Q8_0 93.68% (31,478).
- **Upload (2026-11-25, owner sign-off on tolerances, capability tables and
  HF card).** F32, F16, Q8_0 and the rendered README uploaded to the PRIVATE
  repo `handy-computer/smart-turn-v3.2-gguf` (commit `74045442`). Round trip: the downloaded
  files' sha256 equal the local canonical files, and `validate.py all` passes
  on the downloaded F32 (8/8 cases); the real-model smoke passes on the
  downloaded files. Public flip deferred.
