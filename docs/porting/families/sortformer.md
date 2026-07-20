# Sortformer

Status: research

Intake signed off 2026-07-19: capability scope approved as drafted;
acceptance set = AMI (DER + JER vs NeMo reference, plus prob-tensor parity).

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

## References

- Canonical reference: NeMo `SortformerEncLabelModel.from_pretrained(...)` + `sortformer_modules.SortformerModules` (streaming speaker cache). Same NeMo toolkit already pinned for parakeet/canary.
- Instrumented reference: `scripts/envs/sortformer/` (to be created at Stage 2, mirroring `scripts/envs/parakeet`).
- Cross-check references:
  - Sortformer paper: https://arxiv.org/abs/2409.06656
  - Streaming Sortformer (AOSC): https://arxiv.org/abs/2507.18446
  - NEST encoder: https://arxiv.org/abs/2408.13106
  - NeMo eval script: `examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py`

## Commands

Reference run (NeMo):

```bash
# TODO (Stage 2): pin scripts/envs/sortformer, then:
# uv run --project scripts/envs/sortformer python - <<'PY'
# from nemo.collections.asr.models import SortformerEncLabelModel
# m = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
# m.eval()
# segs, probs = m.diarize(audio=["samples/<multispeaker>.wav"], batch_size=1, include_tensor_outputs=True)
# PY
```

Reference dumps:

```bash
TODO  # Stage 2: dump the T x 4 probability tensor + intermediate encoder activations for parity.
```

Conversion:

```bash
TODO  # Stage 3: scripts/convert-sortformer.py -> GGUF (encoder-diarizer arch).
```

Validation:

```bash
TODO  # Parity: raw T x 4 prob-tensor max-abs-diff vs NeMo reference (arrival-order columns; no Hungarian).
TODO  # Acceptance: DER + JER vs the reference model on the same manifest (proposed set: AMI, freely downloadable).
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
| Streaming diarization (<=4 spk) | online, AOSC/FIFO cache | port streaming diarize vs NeMo reference at a fixed preset (e.g. 1.04s latency) | prob-tensor parity vs reference; DER + JER within tolerance on AMI | MUST PASS | TODO |
| Offline diarization | single large chunk | port diarize vs NeMo reference | prob-tensor parity vs reference | MUST PASS | TODO |
| Speaker-activity tensor | `include_tensor_outputs` | T x 4 sigmoid probs, port vs reference on the same audio | max-abs-diff within tolerance, columns arrival-order aligned | MUST PASS | TODO |
| Multitalker interop | feed diar supervision | Sortformer T x 4 drives parakeet multitalker layer-0 speaker kernel | multitalker emits speaker-attributed transcript | OUT OF SCOPE — owned by the multitalker port; unblocked once both land | TODO |
| Transcription (text) | n/a | n/a | model produces no text output | OUT OF SCOPE — not a transcription model | TODO |
| Translation | n/a | n/a | n/a | OUT OF SCOPE — not a transcription model | TODO |
| Timestamps (transcription) | n/a | n/a | diarization segment times are intrinsic, not a transcript-timestamp capability | OUT OF SCOPE — not a transcription model | TODO |
| >4 speakers | n/a | n/a | hard cap at 4 speakers | OUT OF SCOPE — architectural 4-speaker cap | TODO |

## Notes

- Frontend `normalize=NA` (no per-feature normalization); 128 mels, n_fft 512, win 400 / hop 160, hann, dither 1e-5. `preemphasis` is not set in `model_config.yaml` (NeMo default 0.97 assumed) and MUST be confirmed against the Stage 2 Oracle dump; the sibling streaming-parakeet models disable it.
- Output post-processing (probs -> RTTM segments) is dataset-tuned upstream. Port gate is prob-tensor parity; DER/JER use a fixed/documented post-processing and are compared to the reference model, not to the published numbers.
- Streaming operating-point knobs: `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_update_period`, `spkcache_len` (all in 80 ms frames). Published DER corresponds to specific presets (1.04s vs 30.4s input-buffer latency).
