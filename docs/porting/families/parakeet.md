# Parakeet

Status: all 10 parakeet variants are SUPPORTED with manifest-driven numerical
validation against the NeMo canonical reference. The 8 variants introduced
via the 2026-05-09 intake batch (`ctc-0.6b`, `ctc-1.1b`, `rnnt-0.6b`,
`rnnt-1.1b`, `tdt-1.1b`, `tdt_ctc-110m`, `tdt_ctc-1.1b`, `unified-en-0.6b`)
have completed Stages 1–8: intake, oracle dumps, conversion to F32 GGUF,
C++ bring-up at the reference dtype, the full shipping quant matrix
(F32 + F16 + Q8_0 + Q6_K + Q5_K_M + Q4_K_M), publication-grade performance
benchmarks on Apple M4 Max and AMD Ryzen 7 PRO 4750U (numbers live in
the per-variant model cards under `docs/models/<variant>.md`), the full
LibriSpeech test-clean WER sweep, and ship-ready model cards + HF
YAML/READMEs. Stage 7 family overview:
[`reports/wer/parakeet-family.librispeech-test-clean.summary.md`](../../../reports/wer/parakeet-family.librispeech-test-clean.summary.md).
Three variants (`tdt_ctc-110m`, `ctc-1.1b`, `tdt_ctc-1.1b`) miss the
strict `upstream + 0.01pp` gate by 0.01–0.04pp but are ACCEPTED on
CI-overlap grounds; see the family WER summary for rationale.

## Identity

- Family key: `parakeet`
- Upstream architecture string: `parakeet`
- HF source repo: `nvidia/parakeet-tdt-0.6b-v2` (lead variant for shared
  encoder dims and frontend conventions)
- Variants (intake complete):
  - **TDT (encoder-transducer + duration head)**: `tdt-0.6b-v2`,
    `tdt-0.6b-v3`, `tdt-1.1b`, `tdt_ctc-1.1b`, `tdt_ctc-110m`
  - **RNN-T (encoder-transducer, no duration head)**: `rnnt-1.1b`,
    `rnnt-0.6b`, `unified-en-0.6b`
  - **CTC (encoder-ctc)**: `ctc-1.1b`, `ctc-0.6b`

Per-variant intake JSON: `reports/porting/parakeet/<variant>/intake.json`.

## References

- Canonical reference: **NeMo** (`nvidia/parakeet-tdt-0.6b-v2` and
  `nvidia/parakeet-tdt-0.6b-v3` via `ASRModel.from_pretrained`). NeMo is
  NVIDIA's own implementation and the authoritative source for Parakeet TDT
  weights and inference behavior.
  Script: `scripts/dump_reference_parakeet_nemo.py`.
- Instrumented reference: **NeMo** (same script, using forward hooks to
  capture per-stage intermediates without modifying NeMo internals).

Validation status:

- NeMo reference dumps and C++ CPU dumps pass via
  `uv run scripts/validate.py compare --family parakeet --variant <variant>`
  for both `parakeet-tdt-0.6b-v2` and `parakeet-tdt-0.6b-v3`.
- C++ and NeMo transcripts match exactly on `samples/jfk.wav` for both
  variants.

## Environment

```bash
# NeMo reference environment
uv run --project scripts/envs/parakeet ...
```

Python env: `scripts/envs/parakeet/pyproject.toml`
(nemo_toolkit[asr], torch, soundfile, numpy, sentencepiece).

## Golden Manifest

`tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json`

## Current Commands

Full validation:

```bash
uv run scripts/validate.py all --family parakeet
```

Or step by step:

```bash
uv run scripts/validate.py ref     --family parakeet
uv run scripts/validate.py cpp     --family parakeet
uv run scripts/validate.py compare --family parakeet
```

Conversion:

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```

## Capability Validation

One row per advertised capability per variant. Stage 1 drafts the rows
with `Status: TODO`; Stage 4 fills the observed `Status` after running
each command. Existing `tdt-0.6b-v2`/`v3` rows resolve to PASS. New
variants are TODO until their respective Stage 4 run.

Allowed statuses: `PASS` | `SKIP — not exposed by runtime` |
`ACCEPTED GAP — <reason>`.

| Variant | Capability | Mode | Command | Expected | Status |
|---|---|---|---|---|---|
| tdt-0.6b-v2 | Transcribe | explicit en | `build/bin/transcribe-cli -m models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| tdt-0.6b-v3 | Transcribe | auto | `build/bin/transcribe-cli -m models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F32.gguf samples/jfk.wav` | English transcript | PASS |
| tdt-1.1b | Transcribe | explicit en | `build/bin/transcribe-cli -m models/parakeet-tdt-1.1b/parakeet-tdt-1.1b-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| tdt_ctc-1.1b | Transcribe (TDT head) | explicit en | `build/bin/transcribe-cli -m models/parakeet-tdt_ctc-1.1b/parakeet-tdt_ctc-1.1b-F32.gguf --language en samples/jfk.wav` | English with PnC | PASS |
| tdt_ctc-1.1b | Punctuation/casing | output | same as above | output contains capital letters and `,.?!` | PASS |
| tdt_ctc-110m | Transcribe (TDT head) | explicit en | `build/bin/transcribe-cli -m models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-F32.gguf --language en samples/jfk.wav` | English with PnC | PASS |
| rnnt-1.1b | Transcribe | explicit en | `build/bin/transcribe-cli -m models/parakeet-rnnt-1.1b/parakeet-rnnt-1.1b-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| rnnt-0.6b | Transcribe | explicit en | `build/bin/transcribe-cli -m models/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| unified-en-0.6b | Transcribe (offline) | explicit en | `build/bin/transcribe-cli -m models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-F32.gguf --language en samples/jfk.wav` | English with PnC | PASS |
| unified-en-0.6b | Streaming | streaming | n/a | n/a | `ACCEPTED GAP — streaming infra deferred to v2 port` |
| nemotron-speech-streaming-en-0.6b | Transcribe (offline / cache-aware att_context_size=[70,13], 1.12s chunk) | explicit en | `build/bin/transcribe-cli -m models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-F32.gguf --language en samples/jfk.wav` | English transcript with PnC | TODO |
| nemotron-speech-streaming-en-0.6b | Punctuation/casing | output | same as above | output contains capital letters and `,.?!` | TODO |
| nemotron-speech-streaming-en-0.6b | Streaming (cache reuse across chunks) | streaming | n/a | n/a | `ACCEPTED GAP — v1 port ships batch / one-shot only; streaming session API deferred to follow-up port pass` |
| nemotron-speech-streaming-en-0.6b | Other latency settings ([70,0]/[70,1]/[70,6] = 80/160/560ms chunks) | runtime-selectable att_context_size | n/a | n/a | `ACCEPTED GAP — v1 ships 1.12s chunk only; other settings deferred until streaming lands` |
| ctc-1.1b | Transcribe (CTC head) | explicit en | `build/bin/transcribe-cli -m models/parakeet-ctc-1.1b/parakeet-ctc-1.1b-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| ctc-0.6b | Transcribe (CTC head) | explicit en | `build/bin/transcribe-cli -m models/parakeet-ctc-0.6b/parakeet-ctc-0.6b-F32.gguf --language en samples/jfk.wav` | English transcript | PASS |
| all variants | Word timestamps | only if exposed | `transcribe-cli --timestamps word -m <gguf> <wav>` (any variant) | per-word `t0_ms`/`t1_ms` in JSON output | PASS — derived host-side from emit-frame indices (TDT/RNNT) or per-frame argmax (CTC); same code path as the existing v2/v3 word-timestamp gate, no per-variant differences |

## Open decisions before Stage 3 (convert)

These decisions block converter design for the new variants and should
be resolved before the corresponding port enters porting-3-convert:

1. **TDT_CTC head disposition** (`tdt_ctc-1.1b`, `tdt_ctc-110m`):
   ship the TDT head only at runtime, or expose both? Recommended:
   TDT-only for v1 since the pure CTC variants (`ctc-*`) cover that
   path. Drop unused CTC head weights from the GGUF to save ~vocab*d
   floats per model.
2. **CTC variant intake routes through `.nemo`, not HF config/safetensors**
   (`ctc-1.1b`, `ctc-0.6b`): both repos ship a `.nemo` archive alongside the
   HF-Transformers files. The CTC intakes are pinned to the `.nemo` as
   the authoritative source — same path as v2/v3 and the rest of the
   family. The HF `config.torch_dtype=bfloat16` field is misleading
   metadata (training/optimizer dtype, not storage); the `.nemo`
   state_dict and the parallel HF safetensors are both F32. Trust
   storage (F32) for the reference-dtype GGUF; produce BF16 in Stage 5
   quants.

   Preflight Gate A is `.nemo`-aware (see Tooling section): when
   `intake.reference_framework == "nemo"` AND a `.nemo` is in the repo
   siblings, preflight streams `model_config.yaml` out of the archive
   and uses it as the reference for the frontend check, while skipping
   HF `config.json`'s `torch_dtype` for the dtype check. All 10
   parakeet variants now Gate A at WARN-or-PASS (the WARN is "no
   GGUF yet at Stage A", which is expected pre-convert).
3. **Unified-en streaming**: the model carries shared offline+streaming
   weights. v1 transcribe.cpp port targets OFFLINE only. Streaming is
   the same weights, deferred until streaming infra lands.
4. **n_mels for `.nemo`-only variants**: 80 vs 128 confirmed at
   convert time from `model.cfg.preprocessor.features` inside the
   .nemo archive. Wrong default silently degrades WER without
   changing tensor shapes elsewhere.

## Tooling: NeMo-aware preflight

`scripts/preflight.py` knows how to read `.nemo` archives when the
intake declares NeMo as the reference framework. Behavior:

- `load_reference_state` checks the HF repo siblings for a `*.nemo`
  file when `intake.reference_framework == "nemo"`.
- If present, it streams the archive via `HfFileSystem` (no full
  download — `tarfile.open(mode="r|")` reads forward only), pulls
  `model_config.yaml` out, and translates the `preprocessor` block
  into the same shape `_frontend_from_preprocessor` already
  understands (mapping `features → feature_size`, `preemph →
  preemphasis`, etc.).
- The reference dict carries a `nemo_authoritative=True` flag.
  `check_dtype` then skips HF `config.json`'s `torch_dtype` —
  for NeMo families that field is training metadata, not storage,
  so comparing it against the intake's declared dtype produces
  false-positive FAILs. Storage dtype is verified at Gate B against
  the converted GGUF.

This means the family-wide intake convention (".nemo is canonical")
is now actually honored end-to-end by the tooling. No per-variant
overrides; non-NeMo families are unaffected.

## Gaps

- Manifests record `hf_revision` but not local artifact hashes.
- Default CTest no longer has Parakeet source-tree numerical golden
  payloads; use `validate.py` for numerical comparison.
- Encoder dimensions for the .nemo-only variants (1.1b, 110m, rnnt,
  unified, ctc) are not locked at intake time; they are read from the
  archive during Stage 3 convert. Each intake's `intake_gaps`
  enumerates this.

## Stage 3 conversion notes

Per-variant decisions surfaced during Stage 3 (`porting-3-convert`).

### `parakeet-tdt-1.1b`

- **VARIANT_PROFILES dispatch** — keying by `decoder.vocab_size` was
  ambiguous (`tdt-0.6b-v2` and `tdt-1.1b` both ship a 1024-token SPM).
  Re-keyed by output slug; `expected_vocab_size` is asserted at
  convert time. `general.size_label = "1.1B"`, `general.version = "v1"`
  (the upstream repo carries no version suffix).
- **Encoder `use_bias` resolution from state_dict** — NeMo's
  `model.cfg.encoder` for tdt-1.1b *omits* `use_bias`, while the
  `ConformerEncoder` constructor default is `True`. Trusting the
  YAML (with `.get(..., False)`) silently dropped 462 bias tensors.
  The converter now probes `encoder.layers.0.feed_forward1.linear1.bias`
  in the state_dict and treats that as authoritative; v2/v3 still
  resolve to `False` (they have zero linear/conv biases). The new
  `ENCODER_BLOCK_BIAS_TABLE` walks 11 bias tensors per layer when
  `use_bias=True`.
- **Preflight tokenizer alignment** — RNNT/TDT GGUFs pad the
  tokenizer table by one for the blank/start-state token (lives in
  the predictor embed but not in the upstream SPM). Intakes declare
  the SPM-only vocab. `scripts/preflight.py` now backs the blank out
  of the GGUF count when `tokenizer.ggml.blank_token_id == len-1`,
  so the comparison stays apples-to-apples.
- **Stage 4 follow-ups** — the C++ encoder builder rejects
  `(num_mels=80, subsampling_factor=8)` ("only 8/128 implemented");
  it also does not consume the new `enc.blocks.{i}.*.bias` tensors
  even when `stt.parakeet.encoder.use_bias=true` is read. Both are
  Stage 4 (`porting-4-cpp`) work; the GGUF carries the data Stage 4
  will need.

### `parakeet-rnnt-0.6b`, `parakeet-rnnt-1.1b`, `parakeet-unified-en-0.6b`

- **Predictor / joint hparams resolved at convert time, not from cfg
  alone** — NeMo serializes these YAMLs with empty `prednet={}` /
  `jointnet={}` after instantiation (the constructed module carries
  the values; the cfg dump does not). The new
  `resolve_runtime_hparams()` step in the converter prefers the live
  `model.joint` instance attrs (`joint_hidden`, `activation`,
  `num_extra_outputs`) and falls back to state_dict shapes for the
  predictor (`pred_hidden = embed.shape[1]`, `pred_n_layers` = count
  of `dec_rnn.lstm.weight_ih_l<i>`).
- **TDT durations made optional** — pure RNNT variants have no
  `decoding.durations`. The converter writes `stt.parakeet.tdt.*`
  only when durations are present; an asserted invariant
  (`num_extra_outputs == len(durations)`) fails fast on a malformed
  TDT checkpoint. v2/v3/tdt-1.1b paths unchanged (durations are
  present and match).
- **`general.basename`** — now per-profile (`profile.get("basename",
  "parakeet-tdt")`). RNNT-class variants emit `"parakeet-rnnt"`;
  TDT variants keep `"parakeet-tdt"` via the default. Pure
  descriptive metadata; the loader does not gate on it.
- **unified-en archive load** — NeMo 2.7.x's `ConformerEncoder` does
  not accept the streaming kwarg `att_chunk_context_size` that
  unified-en's YAML carries, so `ASRModel.from_pretrained()` fails.
  The converter now falls back to a direct .nemo archive read
  (model_config.yaml + model_weights.ckpt + SPM tokenizer.model)
  via `_DirectNemoArchive`, which exposes only the surface
  `convert()` consumes. Streaming KVs are not emitted; per the
  family-level decision, v1 transcribe.cpp targets offline only.
- **Stage 4 follow-ups (RNNT)** — the C++ loader's
  `read_parakeet_hparams` requires `stt.parakeet.tdt.durations`,
  so the loader-open smoke fails at "gguf load error" for all 3
  RNNT GGUFs. Stage 4 will need to either (a) make the durations
  KV optional and branch on a head-kind discriminator, or (b)
  introduce `stt.parakeet.head_kind = "rnnt"|"tdt"` and let the
  family handler dispatch. The 1.1b geometry caveat from the TDT
  round (`num_mels=80, subsampling_factor=8`) applies equally to
  rnnt-1.1b.

### `parakeet-ctc-0.6b`, `parakeet-ctc-1.1b`

- **Head-kind discriminator KV introduced** —
  `stt.parakeet.head_kind` is now written by every variant
  (`"tdt"` / `"rnnt"` / `"ctc"`). Stage 4's loader should treat the
  key as optional with default `"tdt"` for backwards compat with
  v2/v3 GGUFs already in circulation.
- **CTC head tensors** — the entire CTC head is a single 1×1 conv:
  `decoder.decoder_layers.0.weight` (`(vocab+1, d_model, 1)`) and
  `decoder.decoder_layers.0.bias` (`(vocab+1,)`). Flattened in the
  GGUF to `head.ctc.weight` / `head.ctc.bias`. No predictor, no
  joint, no LSTM — `convert()` branches on `head_kind == "ctc"` and
  walks `CTC_HEAD_TABLE` instead of the predictor + joint walks.
- **`stt.parakeet.predictor.*` and `stt.parakeet.joint.*` skipped
  for CTC** — there is no predictor or joint to describe. CTC
  GGUFs carry only the encoder hparams + frontend + head_kind.
- **`cfg.decoder.num_classes` vs `vocab_size`** —
  ConvASRDecoder names the SPM vocab `num_classes`; RNNTDecoder
  uses `vocab_size`. `read_hparams` now falls back from the
  former to the latter so a single read works across head kinds.
- **Stage 4 follow-ups (CTC)** — the C++ loader currently always
  reads `stt.parakeet.predictor.hidden` etc., so loader-open fails
  on CTC GGUFs at "predictor.hidden missing". Stage 4 must gate
  predictor/joint/tdt reads on `stt.parakeet.head_kind`. Geometry
  caveat (80 mels, sub=8) applies to both CTC variants.

### `parakeet-tdt_ctc-110m`, `parakeet-tdt_ctc-1.1b`

- **Hybrid → TDT-only at runtime** — per Open-decisions #1, the
  hybrid checkpoints ship as TDT-only and the auxiliary CTC head is
  silently dropped. The pure `ctc-*` variants cover the CTC path,
  so duplicating the head wastes ~vocab×d floats per file.
  Implementation: profile carries `head_kind="tdt"` so the converter
  walks the standard TDT path (predictor + joint + durations); a
  separate `drop_aux_ctc` flag (derived from `"tdt_ctc" in slug`)
  adds `ctc_decoder.*` to the expected-unused prefix list so the
  unconsumed-key check stays meaningful.
- **Direct .nemo load for tdt_ctc-1.1b** — NeMo's `restore_from()`
  extracts the full ~4.4 GB tar to a temp dir before reading, which
  doubles transient disk usage. The 1.1b convert blew the disk budget
  on a system that already had the family in `models/`. New profile
  flag `prefer_direct_load=True` short-circuits NeMo's class
  instantiation when a cached `.nemo` is present and routes through
  `_DirectNemoArchive`. The 110m hybrid keeps the standard NeMo path
  (small enough to extract).
- **Stage 4 follow-ups (tdt_ctc)** — the GGUFs load cleanly through
  the existing TDT KV path; runtime fails at the same `(num_mels=80,
  subsampling_factor=8)` geometry as tdt-1.1b.
- **Local attention on tdt_ctc-1.1b only** — uniquely among the 10
  variants, tdt_ctc-1.1b uses NeMo's `LocalAttRelPositionalEncoding`
  with `att_context_size=[128,128]`. The pos_emb buffer is sized
  `[2W+1, d]` (257 instead of 2T-1) and attention is band-restricted
  to ±128 frames per query. C++ honors this via two new GGUF KVs
  (`stt.parakeet.encoder.att_context_{left,right}`, default `-1` =
  full attention) and a -INF row pad on `matrix_bd` before `rel_shift`,
  which moves out-of-band keys to -INF in the post-softmax scores.
  Math is exact for any T (including T > 2W+1).
