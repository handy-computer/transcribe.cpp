# sava / sava porting log

Surprises worth feeding back into the docs and tooling.

## Reference and conversion

- **No HF config.** The configuration lives in the safetensors metadata
  (`format=sava`, `config=<json>`), and HF's safetensors-metadata API reports
  no dtype, so `scripts/intake.py inspect` finds nothing and the intake was
  written by hand. Gate A warns on dtype and front end (as for Silero); Gate
  B passes.
- **bf16 source, fp32 reference.** Upstream upcasts to fp32 on CPU, so the
  intake declares `float32` (manual) and the F32 GGUF is exact.
- **Hooks see the real run.** `sava.run` is plain PyTorch, so forward hooks
  and instance-method wrappers capture the intermediates of the same call;
  the dumper checks the captured logits reproduce the curves bit for bit.
- **A one-off NumPy model of the GGUF** (frame by frame, reading only the F32
  GGUF and its KVs; not kept in the tree) matched the dumps to 2e-6 in
  probability before any C++ existed, which validated the tensor layout.
- **Run and stream differ in the reference** (up to 3.1e-6): chunked SSD and
  windowed attention offline, the recurrence and a per-frame turn head in a
  Session. Upstream's own golden test allows 1e-5 between them.
- **RoPE in fp32 with absolute positions.** The reference's outputs depend on
  stream age (4e-4 after an hour, 3e-3 after 11 h), though the model is shift
  invariant. `ggml_rope`'s iterated `theta_scale` does not reproduce torch's
  per-index frequencies, so `rope.freqs` ships in the GGUF.
- **Quant-policy name collisions.** parakeet-ultra's `vad.*.weight -> Conv`
  rule would have caught a `vad.` prefix, hence `head.vad.*`; the Python
  mirror lacks policy.cpp's `_norm.weight` suffix rule, hence `norm_*`.
  `ggml_ssm_conv` / `ggml_ssm_scan` need F32 kernels and A, hence new F32
  rules in policy.cpp, gguf_common.py and the sync test.
- **Thresholds as float64 KVs.** 0.9 in fp32 is 0.89999998, which accepts a
  probability the reference rejects.
- **validate.py needed no SAVA code**: per-case `"stages": ["encoder"]` and
  the TURN role's `turn.json` comparison.

## C++

- **Stream == run by construction.** Every block runs as one graph of at
  least 4 columns (ggml's NEON tinyBLAS threshold), and the KV rings are
  written before attention with a slot order that depends only on the
  position: bit-identical for any chunking and thread count.
- **`ggml_ssm_scan`'s scalar remainder breaks that.** With a `d_state` that
  is not a multiple of the SIMD step (the toy's first try used 8), a graph's
  first token rounds differently. The real 128 is fine; the toy uses 64.
- **The reference's front end is the main mel error.** Its fp32 DFT GEMM is
  cancellation-bound near the 1e-10 floor: a ggml fp32 DFT was 1.25e-3 off on
  jfk. A host fp64 FFT is exact to 6e-8 and cut jfk's speech drift from 3e-5
  to 6.7e-6; the remaining `fe.mel` drift is the reference's.
- **fp64 RoPE (owner decision)** departs from the fp32 reference by its own
  stream-age drift: 3.9e-5 in speech after 6,000 frames on `dots-full`
  (fp32 angles: 1.2e-5).
- **Batching.** Shared matmuls over all sessions' columns, per-session ops in
  the shapes of a 1-frame step: bit-identical to unbatched. The batch graph
  lives in the first listed session, keyed by process-unique stream ids (no
  model-level scratch).
- **Two bit-exact speedups:** a contiguous attention query (K.Q takes
  tinyBLAS) and a transposed V ring (drops a per-step copy): a streamed frame
  went from 5.4 to 4.6 ms (F32) and 3.9 to 3.0 ms (Q8_0) at 1 thread on an
  M1. The query change moved outputs by up to 1e-5 against older C++ dumps
  (a different kernel), which invalidated earlier Pipecat rows: re-run the
  accuracy sweep after any graph change.

## Quantization and publication

- **F16 is effectively lossless; BF16 is not.** bf16 source values fit f16
  except 1,065 tiny ones, and ggml keeps F16 x F32 matmul activations in
  fp32 (0 Pipecat flips, passes the F32 gate). ggml rounds BF16 activations
  to bf16, and without native bf16 the M1 runs it 15-22x slower than F32.
  Shipped F16, not BF16.
- **Q8_0 changes turns only at threshold margins** (`dots-full`: 6 of 59, each
  within 0.004 of 0.9 or 0.03 of 0.5). `mixer.proj`'s 1026-wide rows fall
  back to F16.
- **Measurement noise.** Speed taken while other worktrees ran batch jobs
  (load 10-36) was off by 2-25x; only idle-machine numbers are recorded.
- **Rig-built GGUFs.** The Ryzen speed rows ran on GGUFs converted on the rig;
  their sha256 differ from the canonical files (F32 45ca0a80..., F16
  55bd69f2..., Q8_0 b44d5ce8...) only by the converter's last-bit constants
  (rig: 907d6da0..., 4100d5b4..., e0c78a78...). `ingest_perf.py` keys a cell
  by quant, not hash, so nothing checks this.
- **Engine SHA.** A scripts-only commit made during a Pipecat run put the
  wrong HEAD (`8415e8e2`) in the F16 header; corrected to `8f97ab5b`, the
  commit the tool was built at.
- **Metadata audit.** `audit_gguf_metadata.py` flagged the missing
  `general.languages` and tripped its streaming name heuristic: ASR
  assumptions. The audit now exempts silero_vad, sava and smart_turn.
- **No per-clip reference from Inworld** (aggregates only), so
  `scripts/turn/run_reference_sava_torch.py` scores the test set with
  upstream's package; `score.py --ref-jsonl` gates against it.
- **API review (draft 6)** removed turn segments and the SAVA extension: the
  library returns per-frame scores and the clip's `p` / `ended`, and
  Inworld's rule is the caller's (`examples/turn_live` still matches
  `sava.segment` on `dots.wav`). The GGUFs keep `stt.sava.policy.*`, unread.
- **Upload (2026-11-25, owner sign-off on tolerances, capability tables and
  HF card):** private `handy-computer/sava-gguf` (`e64066c9`); downloaded
  sha256 equal the canonical files and `validate.py all` passes on the
  downloaded F32.
- **Card re-rendered and re-uploaded after the slimming pass** (upstream
  links point at the HF revision; reworded validation reference).
