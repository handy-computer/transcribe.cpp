# Sortformer

Status: Stage 4 complete (offline + streaming)

Intake signed off 2026-07-19: capability scope approved as drafted;
acceptance set = AMI (DER + JER vs NeMo reference, plus prob-tensor parity).
Stages 1-4 done. Stage 4 offline checkpoint (tensor parity, 5/5) and streaming
checkpoint (AOSC/FIFO + `_compress_spkcache`, `diar.probs` parity, AMI DER)
both green. **C++ AMI DER 14.59% / JER 19.51%** (16 mtgs, very_high_latency,
forced-alignment RTTMs + dihard3 PP) reproduces the measured reference
14.83% / 19.89% within 0.24 pt (per-meeting median |Δ| ~0.02 pt). Next: Stage 5
quants.

Streaming Sortformer is a **frame-level end-to-end neural speaker diarizer**
(architecture pattern `encoder-diarizer`, new to this repo). It is NOT a
transcription model: it consumes 16 kHz mono audio and emits a `T x 4`
matrix of per-frame per-speaker activity probabilities (one row per 80 ms
frame, up to 4 speakers, columns in speaker arrival order). It runs online
with a stateful Arrival-Order Speaker Cache (AOSC) + FIFO.

Primary reason this family is being ported: it is the named hard
dependency of the Parakeet multitalker speaker-attributed ASR path
(`spk_supervision='diar'`). See
`reports/porting/parakeet/multitalker-parakeet-streaming-0.6b-v1/intake.json`.

## Identity

- Family key: `sortformer`
- Upstream architecture string: `nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel`
- Architecture pattern: `encoder-diarizer` (NEST/FastConformer encoder -> 18L Transformer encoder -> 2 FF -> 4 sigmoid/frame)
- Hugging Face repo: `nvidia/diar_streaming_sortformer_4spk-v2.1`
- Hugging Face revision: `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`
- License: NVIDIA Open Model License (v2.1). Sibling `diar_streaming_sortformer_4spk-v2` is CC-BY-4.0.
- Variants:
  - `streaming-4spk-v2.1` — this port (best meeting DER; multitalker's named dependency).
  - `streaming-4spk-v2` — architecturally identical, CC-BY-4.0. Deferred; add by dropping in weights once v2.1 is validated.

### Diarization scope policy

Sortformer is the repo's first diarization-only family. transcribe.cpp
remains a transcription library; diarization-only models are in scope only
when they (a) feed a transcription pipeline in-repo (here: the named
`spk_supervision='diar'` dependency of parakeet multitalker speaker-
attributed ASR), or (b) reuse an encoder family the repo already maintains
(here: the NEST FastConformer is parakeet's ConformerEncoder, reused
verbatim). A diarizer meeting neither clause (e.g. a pyannote
segmentation+clustering port: new architecture, new dependency surface, no
in-repo ASR consumer) is out of scope. Standalone diarization output is
exposed because it falls out of the multitalker dependency for free, via
the pre-existing transcript-independent `transcribe_speaker_segment` ABI —
no diarizer-specific output surface was added.

## Public API (run extension)

`include/transcribe/sortformer.h` — `TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM`
(`SFST`, RUN slot; registered in `docs/extension-kinds.md`). A run produces
no text; results are read via `transcribe_n_speaker_segments` /
`transcribe_get_speaker_segment` (`TRANSCRIBE_FEATURE_DIARIZATION`).

```c
transcribe_sortformer_stream_ext ext;
transcribe_sortformer_stream_ext_init(&ext);          /* preset = DEFAULT (GGUF cfg) */
ext.preset = TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY;
run_params.family = &ext.ext;                          /* probe accepts_ext_kind first */
```

Preset menu (frames are 80 ms; chunk/rc/fifo/update/spkcache geometry per
`k_presets` in `src/arch/sortformer/stream.cpp`, mirroring the upstream
published operating points):

| Preset | Lookahead | Geometry (chunk/rc/fifo/update/cache) | DER (AMI ihm test, FA + dihard3 PP) | Throughput (M4 CPU) |
| --- | --- | --- | --- | --- |
| `VERY_HIGH_LATENCY` | ~30.4 s | 340/40/40/300/188 | 14.59% (full 16 mtgs; REF 14.83%) | ~25-31x realtime |
| `HIGH_LATENCY` | ~10.0 s | 124/1/124/124/188 | not separately gated | — |
| `LOW_LATENCY` | ~1.04 s | 6/7/188/144/188 | 14.80% (3-mtg subset; REF 14.81%) | ~0.85x realtime (many small windows) |
| `DEFAULT` | GGUF cfg | checkpoint values | — | — |

The menu is discrete (jointly-tuned bundles), not a latency dial; only
these bundles are accuracy-validated. Precedence: GGUF cfg < ext preset <
`TRANSCRIBE_SORTFORMER_STREAM_PRESET` env < per-field env (env layers are
validation hooks; `small` is a diagnostic-only env preset, deliberately
not in the public enum). An out-of-range preset or wrong-kind ext is
rejected pre-clear (`run_validate`), preserving the previous result.
Unit test: `tests/sortformer_stream_ext_unit.cpp`
(`TRANSCRIBE_SORTFORMER_GGUF`-gated).

A future push-audio entry point (`transcribe_stream_begin/feed`, live
diarization) registers a separate STREAM-slot kind taking the same preset
enum; the multitalker interop path (raw T x 4 supervision into parakeet's
layer-0 speaker kernel) is internal C++ and does not round-trip through
this surface.

## References

- Canonical reference: NeMo `SortformerEncLabelModel.from_pretrained(...)` + `sortformer_modules.SortformerModules` (streaming speaker cache). Same NeMo toolkit already pinned for parakeet/canary.
- Instrumented reference: `scripts/envs/sortformer/` (to be created at Stage 2, mirroring `scripts/envs/parakeet`).
- Cross-check references:
  - Sortformer paper: https://arxiv.org/abs/2409.06656
  - Streaming Sortformer (AOSC): https://arxiv.org/abs/2507.18446
  - NEST encoder: https://arxiv.org/abs/2408.13106
  - NeMo eval script: `examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py`

## Commands

Oracle audio (deterministic 2-speaker mix, committed):

```bash
uv run scripts/gen_sortformer_oracle_audio.py
# -> samples/sortformer-2spk-mix.wav + tests/golden/sortformer/sortformer-2spk-mix.rttm
```

Reference dumps (NeMo, via scripts/envs/sortformer):

```bash
BASE=build/validate/sortformer/streaming-4spk-v2.1/sortformer-2spk-mix
uv run --project scripts/envs/sortformer scripts/dump_reference_sortformer_nemo.py encoder \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --audio samples/sortformer-2spk-mix.wav --out $BASE/encoder/ref
uv run --project scripts/envs/sortformer scripts/dump_reference_sortformer_nemo.py diarize \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --audio samples/sortformer-2spk-mix.wav --out $BASE/diarize/ref
```

Conversion:

```bash
scripts/convert-sortformer.py -> models/streaming-4spk-v2.1/streaming-4spk-v2.1-F32.gguf (arch `sortformer`).
```

Validation:

```bash
# Tensor parity (offline enc.* + diar.preds_offline + streaming diar.probs):
uv run scripts/validate.py all --family sortformer --backend cpu     # 6/6 green (single-chunk diar.probs)
VALIDATE_SORTFORMER_PRESET=small uv run scripts/validate.py all --family sortformer  # multi-chunk FIFO+compress diagnostic

# Acceptance (DER + JER, official protocol: forced-alignment RTTMs + dihard3 PP).
uv run scripts/diar/ingest_ami.py --config ihm --split test               # audio + manual RTTMs
uv run scripts/diar/fetch_ami_forced_alignment.py --config ihm --split test  # forced-alignment RTTMs + FA manifest
# Reference baseline (one-time):
uv run --project scripts/envs/sortformer scripts/diar/run_reference_sortformer_nemo.py \
  --manifest samples/diar/ami-ihm-test.manifest.jsonl \
  --model nvidia/diar_streaming_sortformer_4spk-v2.1 --preset very_high_latency \
  --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
  --pred-dir reports/diar/pred/streaming-4spk-v2.1-ami-ihm-test-PPdihard \
  --out reports/diar/streaming-4spk-v2.1-REF.ami-ihm-test.PPdihard.jsonl
# C++ port (dumps diar.probs per meeting, applies the SAME dihard3 PP):
uv run --project scripts/envs/sortformer scripts/diar/run_cpp_sortformer.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --gguf models/streaming-4spk-v2.1/streaming-4spk-v2.1-F32.gguf --preset very_high_latency \
  --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
  --pred-dir reports/diar/pred/streaming-4spk-v2.1-cpp-ami-ihm-test \
  --out reports/diar/streaming-4spk-v2.1-CPP.ami-ihm-test.jsonl --backend cpu
uv run scripts/diar/score_der.py \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --pred-dir reports/diar/pred/streaming-4spk-v2.1-cpp-ami-ihm-test \
  --out reports/diar/streaming-4spk-v2.1-CPP.ami-ihm-test-fa.PPdihard.score.json
# -> REF DER 14.83% / JER 19.89% ; C++ DER 14.59% / JER 19.51% (within 0.24 pt).
```

Benchmarks:

```bash
TODO
```

## Capability Validation

> NOTE (non-standard family): the template's forced `MUST PASS` rows are
> transcription rows (explicit-language / auto transcribe). Sortformer
> emits no transcript, so those rows are `OUT OF SCOPE — not a
> transcription model`. The forced-row spirit is preserved by making the
> **diarization** capability the obligated one: streaming diarization,
> offline (single-chunk) diarization, and the raw activity-tensor output
> are the `MUST PASS` rows for this family. These substitutions require
> user sign-off at intake.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Streaming diarization (<=4 spk) | online, AOSC/FIFO cache | port streaming diarize vs NeMo reference at a fixed preset (e.g. 1.04s latency) | prob-tensor parity vs reference; DER + JER within tolerance on AMI | MUST PASS | PASS — `diar.probs` parity (single-chunk 2.96e-4; multi-chunk `small` preset ~3.9e-4 uniform); AMI DER 14.59% / JER 19.51% vs ref 14.83% / 19.89% |
| Offline diarization | single large chunk | port diarize vs NeMo reference | prob-tensor parity vs reference | MUST PASS | PASS — `diar.preds_offline` 2.96e-4 (validate.py all 6/6) |
| Speaker-activity tensor | `include_tensor_outputs` | T x 4 sigmoid probs, port vs reference on the same audio | max-abs-diff within tolerance, columns arrival-order aligned | MUST PASS | PASS — `diar.probs` [150x4] within tolerance |
| Multitalker interop | feed diar supervision | Sortformer T x 4 drives parakeet multitalker layer-0 speaker kernel | multitalker emits speaker-attributed transcript | OUT OF SCOPE — owned by the multitalker port; unblocked once both land | TODO |
| Transcription (text) | n/a | n/a | model produces no text output | OUT OF SCOPE — not a transcription model | TODO |
| Translation | n/a | n/a | n/a | OUT OF SCOPE — not a transcription model | TODO |
| Timestamps (transcription) | n/a | n/a | diarization segment times are intrinsic, not a transcript-timestamp capability | OUT OF SCOPE — not a transcription model | TODO |
| >4 speakers | n/a | n/a | hard cap at 4 speakers | OUT OF SCOPE — architectural 4-speaker cap | TODO |

## Reference baselines (Stage 2 Oracle)

- **Tensor parity oracle** (`build/validate/sortformer/streaming-4spk-v2.1/sortformer-2spk-mix/`): mel + fastconformer + encoder_proj + transformer + offline preds + streaming `diar.probs` [150×4]. Offline preds == streaming probs on this short clip (single chunk). Tolerances in `tests/tolerances/sortformer.json` (provisional).
- **Measured reference DER/JER** on AMI ihm test (16 meetings, 9.06 h), preset `very_high_latency`, collar 0.0s / overlap included (NVIDIA AMI protocol). The reference pipeline REPRODUCES the published number (15.90 @ very_high_latency AMI-IHM). Full 2x2 decomposition of the two protocol levers:

  | predictions | vs manual RTTMs (diarizers-community/ami) | vs forced-alignment RTTMs (nttcslab-sp) |
  |---|---|---|
  | no post-processing | 29.44% | 15.96% |
  | + dihard3 post-processing | 27.95% | **14.83%** |

  - **Official gate = forced-alignment RTTMs + dihard3 PP: DER 14.83%, JER 19.89%** (missed 5.89%, FA 5.60%, confusion 3.34%). File: `reports/diar/streaming-4spk-v2.1-REF.ami-ihm-test-fa.PPdihard.score.json`. Stage 4 / Stage 7 apply the SAME PP to the C++ probs and score vs the SAME forced-alignment RTTMs, gated against this number.
  - The RTTM source is ~13 pts of the published-vs-manual gap; post-processing another ~1 pt. Confusion stays ~3% throughout, so the diarization core is correct; the manual-RTTM inflation was recall against dense annotations (within-utterance pauses), exactly as NVIDIA's forced-alignment `only_words` labels avoid.
  - Ground truth: nttcslab-sp/diar-forced-alignment (ASRU 2025, card ref [7]), `only_words`, full-corpus-ASR partition — the labels NVIDIA used. PP: NeMo v2 `dihard3-dev` config (no AMI-specific or v2.1 PP is published; dihard3 is the meeting-domain proxy).
  - Diagnostic/context files also kept: `...ami-ihm-test.score.json` (no-PP/manual 29.44%), `...ami-ihm-test-fa.noPP.score.json` (no-PP/FA 15.96%).

## Conversion (Stage 3)

`scripts/convert-sortformer.py` -> `models/streaming-4spk-v2.1/streaming-4spk-v2.1-F32.gguf` (F32, 471 MB, arch string `sortformer`). 971 tensors emitted; 19 skipped (2 `preprocessor.*` recomputed in the C++ frontend, 17 BN `num_batches_tracked` counters). Gate B all-PASS. The GGUF tensor-name contract Stage 4's `src/arch/sortformer/weights.cpp` must implement:

- `enc.pre_encode.*` — dw_striding subsample (same as Parakeet).
- `enc.blocks.{0..16}.*` — 17 Conformer/NEST blocks; identical suffix map to Parakeet's `ENCODER_BLOCK_TABLE` (macaron FF1/FF2, rel-pos self-attn with `pos_bias_u/v`, conv module with `conv.bn.{weight,bias,running_mean,running_var}`). `use_bias=true` (linear biases present); `conv_norm_type=batch_norm`.
- `diar.encoder_proj.{weight,bias}` — Linear 512 -> 192 between the two encoders.
- `tf.blocks.{0..17}.*` — 18 post-LN Transformer blocks: `norm_1` -> attn(`attn.{q,k,v,out}`) -> `norm_2` -> FFN(`ff.{in,out}`), ReLU, `pre_ln=false`.
- `diar.fc1.*`, `diar.spk_head.*` (4x384 -> 4 sigmoid), `diar.single_spk_head.*` (4x192).

Tensor-name mapping decisions: Conformer names mirror `convert-parakeet.py` verbatim (shared NeMo ConformerEncoder). Transformer norms named `norm_1`/`norm_2` and BN kept as `.bn.` so `reference_dtype_for`/`policy.cpp` route them to F32; representatives registered in `scripts/lib/test_quant_policy_sync.py`. GGUF slug is the variant (`streaming-4spk-v2.1`), not the longer repo slug.

## Notes

- Frontend `normalize=NA` (no per-feature normalization); 128 mels, n_fft 512, win 400 / hop 160, hann, dither 1e-5. `preemphasis=0.97` CONFIRMED from the loaded preprocessor at Stage 2 (NeMo default applied, not disabled).
- Reaching the published DER later (optional, not a port gate) needs forced-alignment RTTMs + a tuned post-processing config; the `--collar` / `--skip-overlap` knobs in `scripts/diar/score_der.py` are for diagnostics only.
- Output post-processing (probs -> RTTM segments) is dataset-tuned upstream. Port gate is prob-tensor parity; DER/JER use a fixed/documented post-processing and are compared to the reference model, not to the published numbers.
- Streaming operating-point knobs: `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_update_period`, `spkcache_len` (all in 80 ms frames). Published DER corresponds to specific presets (1.04s vs 30.4s input-buffer latency).
