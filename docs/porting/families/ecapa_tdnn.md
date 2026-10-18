# ECAPA-TDNN (language ID)

Status: supported (LANGID role). Ported from `handy-computer/langid.cpp`
(`96af3a5`).

ECAPA-TDNN is SpeechBrain's speaker / language embedder (architecture pattern
`encoder-classifier`): SpeechBrain Fbank front end, a TDNN block, three
SERes2Net blocks, multi-layer feature aggregation, attentive statistics
pooling, a 256-d embedding and a two-layer classifier over the VoxLingua107
labels. It is not a transcription model. The family only produces logits; the
LANGID role dispatcher (`src/transcribe-langid.cpp`) owns the crop to the
first `transcribe_langid_info::max_audio_ms` (30 s), the allowed set,
softmax, `allowed_mass` and ranking (`docs/langid.md`).

Acceptance: tensor parity on eight FLEURS clips (`validate.py`), and top-1
decision parity with SpeechBrain on FLEURS (15 languages x 200 utterances x
3 / 5 / 10 s / full crops, `full` capped to the first 30 s, C++ F32 on CPU).
Shipped matrix: F32 + F16 + Q8_0. Each GGUF's accuracy and agreement live in
the catalog and are rendered on [the model page](../../models/lang-id-voxlingua107-ecapa.md#accuracy). The
loader widens Q8_0 weights to F16 (`widen_q8_0_weights` in
`src/arch/ecapa_tdnn/model.cpp`).

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

## GGUF contract

Keys: `stt.variant`, `stt.frontend.*` (`window = hamming_periodic`,
`pad_mode = constant`, `log_clamp_min = 1e-10`, `top_db = 80`,
`normalize = sentence_mean`, plus the usual sizes), `stt.ecapa_tdnn.*`
(channels, kernel sizes, dilations, res2net scale, SE / attention /
embedding / classifier widths, `asp_eps`, leaky slope) and
`stt.langid.labels.{codes,names,aliases}`. Integer scalars are uint32.

The converter applies four exact rewrites: BatchNorm to a `scale` / `shift`
pair (TDNNBlock is conv -> ReLU -> BN, so it cannot fold backward); the
three activation-free BNs folded forward into `fc`, `cls.l1`, `cls.out`; and
the ASP and MFA input weights split per concatenated operand. Tensor names
follow the `tools/transcribe-quantize` rules (`.bias` / `.bn.` F32,
`.conv.weight` for k>1 tap-major kernels F32 / F16, every other `.weight`
quantizable, `frontend.mel_filterbank` F32), so the quant policy has no
ECAPA entries.
The full catalogue is in `src/arch/ecapa_tdnn/weights.h`.

The front end is the shared `transcribe-mel` (`src/transcribe-mel.cpp`)
with `window_type = "hamming_periodic"` and `normalize = "sentence_mean"`
(`10*log10`, 80 dB top-db floor, per-bin mean subtraction, no frame drop).
The filterbank is SpeechBrain's own, captured by the converter and stored in
the GGUF; it is never rebuilt.

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
uv run scripts/langid/score.py reports/langid/cpp-f32-untrimmed.jsonl --ref reports/langid/ref-speechbrain-untrimmed.jsonl \
  --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.score.json
uv run scripts/catalog/ingest_accuracy.py --models lang-id-voxlingua107-ecapa
```

The catalog rows follow the `langid-publication-v1` profile
(`catalog/_benchmark_profiles.json`): the open-set mean on the 5 s untrimmed
crop of each shipped GGUF's sweep, with that sweep's agreement.

`score.py --ref` fails on any top-1 disagreement with the reference sweep or
any row the two sweeps do not share, and refuses sweeps whose labels, recipe,
manifest hashes or checkpoint revision differ. With `--json` it writes the
score and, beside it, the agreement. Only F32 is gated: score the F16 and Q8_0
sweeps with `--report-only`, which still writes their agreement.

Benchmarks: `scripts/langid/bench.py --profile` (`tools/transcribe-bench` is
ASR-only) writes bench-driver reports under `reports/perf/`, which
`scripts/catalog/ingest_perf.py` folds into the catalog.

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Language ID | open set | `build/bin/transcribe-cli -m models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf samples/fleurs-ja.wav` | `language: ja` | MUST PASS | PASS |
| Language ID | allowed set | `... --allow en,de samples/fleurs-de.wav` | `language: de`, 2 candidates | MUST PASS | PASS |
| Crop | first 30 s | `transcribe_ecapa_tdnn_smoke` (toy model, 35 s of noise), `transcribe_ecapa_tdnn_real_smoke` (`ru-long.wav`) | whole clip == its first `max_audio_ms` (bit-identical candidates / logits); its last `max_audio_ms` differs | MUST PASS | PASS |
| Minimum length | 500 ms | `transcribe_ecapa_tdnn_smoke`, `transcribe_langid_dispatch_unit` | `INPUT_TOO_SHORT` below 500 ms | MUST PASS | PASS |
| Transcribe / translate / timestamps / streaming | n/a | `transcribe_session_init` | `UNSUPPORTED_ROLE` | OUT OF SCOPE — not an ASR model | SKIP — not exposed by runtime |
| Batch (offline) | n/a | `transcribe-cli --batch` | refused: ASR-only | OUT OF SCOPE — no batch entry points for new roles in v1 | ACCEPTED GAP — one clip per call |
