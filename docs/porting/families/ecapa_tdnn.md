# ECAPA-TDNN (language ID)

Status: supported (LANGID role). Ported from `handy-computer/langid.cpp`
(`96af3a5`), where the C++ was first written and validated against
SpeechBrain; that repo is superseded by this family.

ECAPA-TDNN is SpeechBrain's speaker / language embedder (architecture pattern
`encoder-classifier`): SpeechBrain Fbank front end, a TDNN block, three
SERes2Net blocks, multi-layer feature aggregation, attentive statistics
pooling, a 256-d embedding and a two-layer classifier over the VoxLingua107
labels. It is not a transcription model. The family only produces logits; the
LANGID role dispatcher (`src/transcribe-langid.cpp`) owns the crop to the
last `max_audio_ms`, the allowed set, softmax, `allowed_mass`, ranking and
top-k (`docs/langid.md`).

Acceptance: tensor parity on eight FLEURS clips (`validate.py`), and top-1
decision parity with SpeechBrain on FLEURS (15 languages x 200 utterances x
3 / 5 / 10 s / full = 12000 decisions): **12000 / 12000**, max abs logit
difference 6.5e-05 (C++ F32, CPU). Shipped matrix: F32 + F16 + Q8_0;
F16 agrees on 11991 / 12000 (near-ties only), Q8_0 on 11175 / 12000 (the
activation quantization inside ggml's Q8_0 matmul; langid.cpp's Q8_0 measured
the same 93.2%).

## Identity

- Family key: `ecapa_tdnn`
- Upstream architecture: `speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN` + `speechbrain.lobes.models.Xvector.Classifier`
- Hugging Face repo: `speechbrain/lang-id-voxlingua107-ecapa`
- Hugging Face revision: `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`
- License: Apache-2.0
- Variants: `lang-id-voxlingua107-ecapa` (107 labels; legacy codes `iw`, `jw`, `tl`, `no` with aliases `he`, `jv`, `fil`, `nb`)

## References

- Canonical reference: SpeechBrain 1.1.1 `EncoderClassifier` (the only
  implementation; `hyperparams.yaml` instantiates SpeechBrain classes by name),
  torch 2.13.0, CPU, one thread.
- Instrumented reference: `scripts/dump_reference_ecapa_tdnn_speechbrain.py`
  (forward hooks on one `classify_batch`; asserts every hook fires once).
- Cross-check reference: `scripts/lib/ecapa_numpy.py`, a NumPy forward pass
  driven only by the GGUF, which proves the converter's rewrites.

## GGUF contract

Keys: `stt.variant`, `stt.frontend.*` (`window = hamming_periodic`,
`pad_mode = constant`, `log_clamp_min = 1e-10`, `top_db = 80`,
`normalize = sentence_mean`, plus the usual sizes), `stt.ecapa_tdnn.*`
(format version, channels, kernel sizes, dilations, res2net scale, SE /
attention / embedding / classifier widths, `asp_eps`, leaky slope) and
`stt.langid.labels.{codes,names,aliases}`. Integer scalars are uint32.

The converter applies four exact rewrites, each proven by the NumPy
reference: BatchNorm to a `scale` / `shift` pair (TDNNBlock is conv -> ReLU
-> BN, so it cannot fold backward); the three activation-free BNs folded
forward into `fc`, `cls.l1`, `cls.out`; and the ASP and MFA input weights
split per concatenated operand. Tensor names follow the
`tools/transcribe-quantize` rules (`.bias` / `.bn.` F32, `.conv.weight` for
k>1 tap-major kernels F32 / F16, every other `.weight` quantizable,
`frontend.mel_filterbank` F32), so the quant policy has no ECAPA entries.
The full catalogue is in `src/arch/ecapa_tdnn/weights.h`.

The front end is family-local (`src/arch/ecapa_tdnn/mel.cpp`):
`transcribe-mel` has no periodic Hamming window, `10*log10`, top-db floor or
per-bin mean subtraction. The filterbank is SpeechBrain's own, captured by the
converter and stored in the GGUF; it is never rebuilt.

## Commands

Reference dumps and tensor parity:

```bash
uv run scripts/validate.py all --family ecapa_tdnn
```

Conversion (F32 only; quantize afterwards):

```bash
uv run --project scripts/envs/ecapa_tdnn scripts/convert-ecapa_tdnn.py speechbrain/lang-id-voxlingua107-ecapa
build/bin/transcribe-quantize models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \
  models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-Q8_0.gguf --quant Q8_0
```

Accuracy and decision parity (FLEURS from the HF cache):

```bash
uv run --project scripts/envs/ecapa_tdnn scripts/langid/ingest.py fleurs --lang all
uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py --engine speechbrain ... --out reports/langid/ref-speechbrain-untrimmed.jsonl
uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py --engine cpp --library build-shared/src/libtranscribe.dylib ... --out reports/langid/cpp-f32-untrimmed.jsonl
uv run scripts/langid/compare.py reports/langid/ref-speechbrain-untrimmed.jsonl reports/langid/cpp-f32-untrimmed.jsonl
uv run scripts/langid/score.py reports/langid/cpp-f32-untrimmed.jsonl --md reports/langid/cpp-f32-untrimmed.md
```

`compare.py` fails on any disagreement that is not on a reviewed near-tie
list (`--near-ties`), and refuses runs whose labels, recipe, manifest hashes
or checkpoint revision differ.

Benchmarks: `scripts/langid/bench.py` (`tools/transcribe-bench` is ASR-only).

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Language ID | open set | `build/bin/transcribe-cli -m models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf samples/fleurs-ja.wav` | `language: ja` | MUST PASS | PASS |
| Language ID | allowed set | `... --allow en,de samples/fleurs-de.wav` | `language: de`, 2 candidates | MUST PASS | PASS |
| Crop | 30 s window | `transcribe_ecapa_tdnn_real_smoke` (`long-45s.wav`) | `audio_ms == 30000` | MUST PASS | PASS |
| Minimum length | 500 ms | `transcribe_ecapa_tdnn_smoke`, `transcribe_langid_dispatch_unit` | `INPUT_TOO_SHORT` below 500 ms | MUST PASS | PASS |
| Transcribe / translate / timestamps / streaming | n/a | `transcribe_session_init` | `UNSUPPORTED_ROLE` | OUT OF SCOPE — not an ASR model | SKIP — not exposed by runtime |
| Batch (offline) | n/a | `transcribe-cli --batch` | refused: ASR-only | OUT OF SCOPE — no batch entry points for new roles in v1 | ACCEPTED GAP — one clip per call |

## Notes

- Regime for every parity number: F32 GGUF, CPU backend, one thread,
  production C++ front end, macOS arm64. On the port, every stage tensor was
  bit-identical to langid.cpp's (ggml 0.20.2 there, 0.25.3 here), so
  `tests/tolerances/ecapa_tdnn.json` carries over unchanged.
- Hardening from the langid.cpp review, fixed in the role dispatcher and
  tested in `tests/langid_dispatch_unit.cpp`: H1 minimum checked after the
  crop and sub-minimum `max_audio_ms` rejected at init; H2 only NULL means
  "all labels"; H3 non-finite PCM and logits rejected; H6 malformed label
  tables fail the load. H4 (no leak when label metadata is bad) is covered by
  `arch_ecapa_tdnn_bad_labels.gguf` in `transcribe_ecapa_tdnn_smoke`
  (`leaks --atExit`: 0 leaks).
- Performance follow-up (not ported): langid.cpp's CPU weight-repacking
  buffer. On the M4 Max CPU (10 s, Q8_0, median of three interleaved rounds)
  it takes 76-81 ms to about 45-51 ms here and 36-39 ms in langid.cpp;
  without it the two are at parity, as is F16 and Metal (14 ms). ARM only:
  ggml has no Q8_0 repack on x86. `cls.out.weight` (107 rows) cannot be
  repacked, so the candidate filter must check the row count or langid.cpp's
  abandon path disables repacking for the whole file. The performance-core
  thread default did not help here (8 threads ties or beats 12).
- `AUTO` placement is unchanged (GPU first). Metal is 5-10x faster than CPU
  on the M4 Max; CPU stays well under real time.
- Q8_0 conv kernels stay F32 (the quantizer's Conv bucket is F32 in every
  preset), so the Q8_0 file is 26.7 MB against langid.cpp's 24.1 MB, and
  `cls.out.weight` is Q8_0 where langid.cpp kept it F16.
