# Nemotron3 Diarization

Status: ported, Stage 4 complete pending tolerance review (Stage 1 intake 2026-09-24; Stage 2 oracle, Stage 3 BF16 GGUF, Stage 4 C++ 2026-09-25)

Intake signed off 2026-09-24: capability scope approved as drafted, with
live push-audio diarization added as MUST PASS. Acceptance set = AMI Test
MHM forced-alignment (primary) + NOTSOFAR1 Eval MHM (5-8 speaker
coverage); DER + JER vs the measured NeMo reference, plus prob-tensor
parity. Reference dtype BF16. License (`openmdw-1.1`) still to be
interpreted before ship.

Nemotron 3 Diarization is a **frame-level end-to-end neural speaker
diarizer** (architecture pattern `encoder-diarizer`) in the Streaming
Sortformer lineage. It is NOT a transcription model: it consumes 16 kHz
mono audio and emits a `T x 8` matrix of per-frame per-speaker activity
probabilities. There is one row per **10 ms** frame, up to 8 speakers, and
the columns are in speaker arrival order. It runs online with the same
Arrival-Order Speaker Cache (AOSC) + FIFO scheme as `sortformer`.

It is a separate family from `sortformer` (v2.1), not a new variant of it.
The two share the AOSC/FIFO streaming algorithm, but the forward graph is
different:

| | `sortformer` (v2.1) | `nemotron3_diar` |
| --- | --- | --- |
| Encoder | 17L FastConformer (dw_striding x8, rel_pos) + 18L post-LN Transformer (d=192) | 31L pre-LN Transformer (d=512), RoPE, feature stacking x8 |
| Head | 2 FF -> sigmoid at 80 ms | proj 512->192 -> subpixel Conv1d x8 -> 2 FF -> sigmoid at 10 ms |
| Speakers | 4 | 8 |
| Cache silence | running mean | learned embedding |
| Storage dtype | F32 | BF16 |

## Identity

- Family key: `nemotron3_diar`
- Upstream architecture string: `nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel` (NeMo); `Nemotron3DiarizationForAudioFrameClassification` (HF transformers)
- Architecture pattern: `encoder-diarizer`
- Hugging Face repo: `nvidia/Nemotron-3-Diarization`
- Hugging Face revision: `f667ed73aee57d40cc39428eb768b4fd87a0a29e`
- License: `openmdw-1.1` (needs human interpretation before ship)
- Variants:
  - `Nemotron-3-Diarization` — this port. Reference dtype **BF16**: the `.nemo` state_dict is all BF16, and the HF `model.safetensors` F32 is a verified lossless upcast.

## References

- Canonical reference: NeMo Speech `SortformerEncLabelModel.restore_from(Nemotron-3-Diarization.nemo)` + `sortformer_modules.SortformerModules` + `modules.transformer_encoder.TransformerEncoder` (`refs/NVIDIA-NeMo/Speech`, nemo 3.x). The existing `scripts/envs/sortformer` NeMo 2.x pin cannot load this encoder.
- Instrumented reference: `scripts/envs/nemotron3_diar/` (to be created at Stage 2, pinned to NVIDIA-NeMo/Speech).
- Cross-check references:
  - HF transformers `models/nemotron3_diarization/modular_nemotron3_diarization.py` (transformers main, 5.18.0.dev0; strict-load converter from the `.nemo`).
  - Sortformer paper: https://arxiv.org/abs/2409.06656
  - Streaming Sortformer (AOSC): https://arxiv.org/abs/2507.18446
  - NeMo eval script: `examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py`
  - C++ sibling: `src/arch/sortformer/stream.cpp` (AOSC/FIFO bookkeeping)

## Commands

Oracle audio (deterministic 8-speaker mix, committed):

```bash
uv run --project scripts/envs/nemotron3_diar scripts/gen_nemotron3_diar_oracle_audio.py
# -> samples/nemotron3-diar-8spk-mix.wav + tests/golden/nemotron3_diar/nemotron3-diar-8spk-mix.rttm
```

Reference dumps (NeMo Speech, via scripts/envs/nemotron3_diar; bit-deterministic):

```bash
BASE=build/validate/nemotron3_diar/Nemotron-3-Diarization/nemotron3-diar-8spk-mix
REF="--model nvidia/Nemotron-3-Diarization --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e --audio samples/nemotron3-diar-8spk-mix.wav"
D="uv run --project scripts/envs/nemotron3_diar scripts/dump_reference_nemotron3_diar_nemo.py"
$D encoder $REF --out $BASE/encoder/ref                       # first streaming step, per-stage
$D diarize --preset very_high_latency $REF --out $BASE/diarize/ref --dump-compress $BASE/diarize/compress
for P in low_latency very_low_latency ultra_low_latency small; do
  $D diarize --preset $P $REF --out $BASE/diarize-$P/ref --dump-compress $BASE/diarize-$P/compress
done
# HF transformers cross-check (not the gate): hf-offline, hf-stream --mode <low|very_low|ultra_low>_latency
```

Acceptance baselines (DER + JER, forced-alignment RTTMs, collar 0, overlap included,
onset/offset 0.5 with no post-processing filter = the model-card protocol):

```bash
uv run scripts/diar/ingest_ami.py --config ihm --split test
uv run scripts/diar/fetch_ami_forced_alignment.py --config ihm --split test
uv run scripts/diar/ingest_notsofar.py                        # 80 eval meetings, close-talk mix
uv run python -c "import json; open('samples/diar/notsofar-mhm-eval-fa-5plus.manifest.jsonl','w').writelines(l for l in open('samples/diar/notsofar-mhm-eval-fa.manifest.jsonl') if json.loads(l)['num_speakers'] >= 5)"   # 45 meetings
V=Nemotron-3-Diarization
for DS in ami-ihm-test-fa notsofar-mhm-eval-fa; do
  uv run --project scripts/envs/nemotron3_diar scripts/diar/run_reference_nemotron3_diar_nemo.py \
    --manifest samples/diar/$DS.manifest.jsonl --model nvidia/Nemotron-3-Diarization \
    --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e --preset very_high_latency \
    --pred-dir reports/diar/pred/$V-REF-$DS-very_high_latency \
    --out reports/diar/$V-REF.$DS.very_high_latency.jsonl
  uv run scripts/diar/score_der.py --manifest samples/diar/$DS.manifest.jsonl \
    --pred-dir reports/diar/pred/$V-REF-$DS-very_high_latency \
    --out reports/diar/$V-REF.$DS.very_high_latency.score.json
done
```

Conversion:

```bash
uv run --project scripts/envs/nemotron3_diar \
  scripts/convert-nemotron3_diar.py nvidia/Nemotron-3-Diarization \
  --repo-id nvidia/Nemotron-3-Diarization \
  --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e
# -> models/Nemotron-3-Diarization/Nemotron-3-Diarization-BF16.gguf (arch `nemotron3_diar`)
# manifest: reports/convert/Nemotron-3-Diarization-BF16.json
```

Validation:

```bash
# Tensor parity: 2 cases x (encoder step 0 + diar.probs at 5 presets + push-audio at 5 presets)
VALIDATE_CPP_THREADS=8 uv run scripts/validate.py all --family nemotron3_diar --variant Nemotron-3-Diarization --backend cpu
#   VALIDATE_NEMOTRON3_DIAR_PRESETS=low_latency,small  restricts presets
#   VALIDATE_NEMOTRON3_DIAR_STREAM_MS=173              push-audio piece size (0 = skip push-audio stages)

# Push-audio / extension contract (real model)
TRANSCRIBE_NEMOTRON3_DIAR_GGUF=models/Nemotron-3-Diarization/Nemotron-3-Diarization-BF16.gguf \
  build/bin/transcribe_nemotron3_diar_stream_unit

# Batch: segment parity vs serial + frozen golden, and bit-exact diar.probs
uv run scripts/batch_parity.py --model models/Nemotron-3-Diarization/Nemotron-3-Diarization-BF16.gguf \
  --list tests/golden/batch/Nemotron-3-Diarization.list --batch-sizes 2,4,8 --backend cpu --speakers \
  --golden-in tests/golden/batch/Nemotron-3-Diarization.cpu.json
uv run --with numpy scripts/batch_tensor_parity.py --model models/Nemotron-3-Diarization/Nemotron-3-Diarization-BF16.gguf \
  --list tests/golden/batch/Nemotron-3-Diarization.list --batch 8 --backend cpu --dump-name diar.probs

# DER/JER gate: C++ over the acceptance sets, same post-processing as the reference
V=Nemotron-3-Diarization; DS=ami-ihm-test-fa; P=very_high_latency
uv run --project scripts/envs/nemotron3_diar scripts/diar/run_cpp_nemotron3_diar.py \
  --manifest samples/diar/$DS.manifest.jsonl --gguf models/$V/$V-BF16.gguf --preset $P \
  --pred-dir reports/diar/pred/$V-BF16-$DS-$P --ref-pred-dir reports/diar/pred/$V-REF-$DS-$P \
  --out reports/diar/$V-BF16.$DS.$P.jsonl
uv run scripts/diar/score_der.py --manifest samples/diar/$DS.manifest.jsonl \
  --pred-dir reports/diar/pred/$V-BF16-$DS-$P --out reports/diar/$V-BF16.$DS.$P.score.json
```

Benchmarks:

```bash
TODO
```

## Capability Validation

> NOTE (non-standard family, same substitution as `sortformer`): the
> template's forced transcription rows are `OUT OF SCOPE — not a
> transcription model`. The obligated rows are the diarization ones
> instead: streaming diarization, offline diarization, and the raw
> activity tensor. These substitutions were user-signed at intake (2026-09-24).
>
> Streaming covers both paths: whole-file streaming via `transcribe_run`
> + the preset run extension (the v2.1 path), and live push-audio input
> via `transcribe_stream_*` (new for diarizers; user-requested
> 2026-09-24).

Preset geometry (spkcache/fifo/chunk/rc/update, 80 ms frames), from the
model card: very_high_latency 30.4 s = 264/40/340/40/300; low_latency
1.04 s = 264/264/9/4/222; very_low_latency 0.64 s = 264/264/6/2/222;
ultra_low_latency 0.32 s = 264/264/3/1/222.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe (explicit language) | n/a | n/a | model produces no text | OUT OF SCOPE — not a transcription model | SKIP — not exposed by runtime (no text output) |
| Transcribe (auto / no hint) | n/a | n/a | model produces no text | OUT OF SCOPE — not a transcription model | SKIP — not exposed by runtime (no text output) |
| Streaming diarization | online AOSC/FIFO, `low_latency` 1.04 s | port vs NeMo reference at the preset, multi-chunk with compression exercised | 10 ms prob-tensor parity; DER + JER within tolerance of the measured reference on the acceptance set | MUST PASS | PASS — low_latency diar.probs 1.3e-5 vs NeMo on both oracle cases; AMI low_latency DER 9.530% / JER 13.103% vs reference 9.529% / 13.104% (16 meetings; +0.001pp / -0.001pp) |
| Offline diarization | `very_high_latency` 30.4 s | port vs NeMo reference | prob-tensor parity; DER + JER within tolerance of the measured reference on AMI | MUST PASS | PASS — diar.probs 1.1e-5 vs NeMo on both oracle cases; AMI DER 9.212% / JER 12.886% = reference 9.212% / 12.886% (16 meetings) |
| Speaker-activity tensor | `include_tensor_outputs` | T x 8 sigmoid probs at 10 ms, port vs reference on the same audio | max-abs-diff within tolerance, arrival-order columns aligned | MUST PASS | PASS — `diar.probs` [T x 8] at every preset within 1.8e-5 (tolerance 9.9e-5), columns arrival-order aligned; encoder-stage tensors 39/39 |
| 5-8 speakers | any preset | 8-speaker oracle `nemotron3-diar-8spk-mix` (all 8 channels active at every card preset) + NOTSOFAR eval MHM >=5-spk subset (`notsofar-mhm-eval-fa-5plus`, 45 meetings) | channels 5-8 active and prob parity holds; DER vs measured reference | MUST PASS | PASS — 8-speaker oracle: all 8 channels, prob parity 1.1e-5..1.8e-5 at every preset; NOTSOFAR 5-7 spk DER 12.135% / JER 15.858% = reference (45 meetings); full NOTSOFAR 9.853% / 12.809% = reference (80) |
| Low-latency presets | `very_low_latency` 0.64 s, `ultra_low_latency` 0.32 s | port vs NeMo reference at each preset | prob-tensor parity on the oracle clip | MUST PASS | PASS — very_low_latency diar.probs 1.4e-5, ultra_low_latency 8.5e-6 on both oracle cases |
| Output-resolution override | `output_subsampling_factor` > 1 (e.g. 80 ms) | n/a | coarser output grid | OUT OF SCOPE — native 10 ms only; downsampling is post-processing. Back in scope if a consumer needs a coarser public grid | SKIP — not exposed by runtime (native 10 ms only) |
| Multitalker interop | feed diar supervision to parakeet multitalker (8 spk) | Nemotron3 diar T x 8 drives the multitalker speaker kernel | speaker-attributed transcript | OUT OF SCOPE — follow-up port; `multitalker.cpp` hard-binds `sortformer::SortformerEmbedded`, 4 speakers and 80 ms | SKIP — not exposed by runtime (follow-up port) |
| Push-audio live diarization | `transcribe_stream_begin/feed/finalize` + a new STREAM-slot ext kind taking the preset | feed the oracle clip in small audio pieces at each streaming preset | incremental speaker segments during feed; final segments and 10 ms probs match the reference streaming runner within tolerance | MUST PASS | PASS — `transcribe_nemotron3_diar_stream_unit` (rows during feed, open turns extend, finalize == run at LL/VHL on both clips); validate.py stream-<preset> diar.probs bit-identical to the whole-file run at all 5 presets x 2 clips |
| Translation | n/a | n/a | n/a | OUT OF SCOPE — not a transcription model | SKIP — not exposed by runtime |
| Timestamps (transcription) | n/a | n/a | segment times are intrinsic to diarization output | OUT OF SCOPE — not a transcription model | SKIP — not exposed by runtime |
| Batch (offline) | run_batch vs serial | `scripts/batch_parity.py` compares hypothesis text only, so it can't check a text-less model as-is; needs a speaker-segment/tensor comparison (v2.1 resolved this row as a user-approved deferral) | identical speaker segments + CPU tensor parity, or `ACCEPTED GAP — serial fallback` | MUST PASS | PASS — explicit lockstep `run_batch`; `batch_parity.py --speakers` identical at 2/4/8 + golden `tests/golden/batch/Nemotron-3-Diarization.cpu.json`; `batch_tensor_parity.py --dump-name diar.probs` bit-exact (8 recordings). CPU throughput unchanged vs serial (compute-bound; see Notes); acceptance sets at batch 8 (30.4 s): all 96 meetings bit-identical to batch 1 (max|d| 0), DER/JER unchanged (AMI 9.212% / 12.886%, NOTSOFAR 9.853% / 12.809%) |

## Reference baselines (Stage 2 Oracle)

Measured NeMo reference (NeMo Speech `cf724ac337d1`, BF16 .nemo, fp32 on CPU;
forced-alignment RTTMs, collar 0, overlap included, onset/offset 0.5 with no
post-processing filter). These are the gates, not the card numbers.

| Dataset | Preset | Meetings | REF DER | REF JER | Card DER | Spk count correct | C++ DER / JER (Stage 4) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AMI test MHM (`ami-ihm-test-fa`) | very_high_latency | 16 | 9.21% | 12.89% | 9.25% | 14/16 (card SCA 87.50%) | 9.212% / 12.886% (= ref) |
| AMI test MHM | low_latency | 16 | 9.53% | 13.10% | 9.48% | 13/16 (card SCA 81.25%) | 9.530% / 13.103% (ref 9.529% / 13.104%) |
| NOTSOFAR eval MHM, all (`notsofar-mhm-eval-fa`) | very_high_latency | 80 | 9.85% | 12.81% | 6.77% | 78/80 (card 93.75%) | 9.853% / 12.809% (= ref) |
| NOTSOFAR eval MHM, 3-4 spk | very_high_latency | 35 | 6.71% | 7.04% | 5.25% | | |
| NOTSOFAR eval MHM, 5-7 spk (`-5plus`) | very_high_latency | 45 | 12.14% | 15.86% | 7.86% | | 12.135% / 15.858% (= ref) |

AMI reproduces the card (MHM == the ingested `ihm` mix-headset condition).
NOTSOFAR MHM runs 1.3-1.5x the card. NVIDIA does not publish its MHM mixing
recipe; ours is the plain zero-padded sum of the close-talk channels. The
reference setup itself is sound. The meeting set matches the card exactly
(35 + 45 meetings x 2 devices = the card's 70 + 90 recordings), speaker
counting is 78/80 correct, and a one-off check on the single-channel
condition (first `sc_*` device per meeting, no mixing) scores DER 9.85%
over 80 meetings (card SC 11.00%) and 11.55% on the 5-7 speaker subset
(card 13.21%). So the MHM gap comes from the mix: MHM false alarms are 4.47%
vs 2.50% on SC, consistent with headset crosstalk being summed in. The gate
compares reference and C++ on this same MHM audio, so the mix only has to be
fixed.

8-speaker oracle (`samples/nemotron3-diar-8spk-mix.wav`, 92 s, generated by
`scripts/gen_nemotron3_diar_oracle_audio.py`): every card preset finds 8
active channels, and authored speakers A..H map to channels 0..7 in arrival
order; returning speakers keep their channel. uk-long.wav was rejected as
speaker H because the reference merges it into speaker E's channel.

## Notes

### Stage 4 (C++ port) notes

- Implementation: `src/arch/nemotron3_diar/` (model.cpp load / step graph / chunk driver / run / run_batch / push-audio; stream.cpp host AOSC state machine + incremental segments; torch_logf.h). Public API: `include/transcribe/nemotron3_diar.h` — preset enum + `N3DR` (RUN slot) / `N3DS` (STREAM slot) extensions; CLI `--diar-preset`. Typed wrappers in the Python / TypeScript / Rust / Swift bindings. Forward map: `reports/porting/nemotron3_diar/forward-map.md`.
- **CPU compute is fp32.** The loader upcasts every BF16/F16 matmul weight to an exact F32 copy on CPU (+~400 MB RAM; the GGUF stays BF16). ggml-cpu's BF16 matmul rounds activations to BF16, which moved `enc.pre_encode.out` by 2.6e-1 and flipped 43 speaker-cache picks at compression #1 on the oracle; F32 matches NeMo to 1e-5 and is ~7x faster on this CPU. `TRANSCRIBE_NEMOTRON3_DIAR_NATIVE_BF16=1` opts out. GPU backends keep BF16 (Stage 6 measures them).
- **Compression is reproduced bit-for-bit against torch CPU.** Boundary scores in the top-k are routinely ~1e-7 apart or exactly tied, so the host code mirrors ATen's `topk(sorted=False)` (libc++ partial_sort / nth_element on (value, index) pairs), torch's Sleef `logf_u10`, and ATen's 4-accumulator 8-term sum. Gate: `scripts/diar/check_nemotron3_diar_compress.py` replays NeMo on the port's own inputs (370/370 compressions identical: every oracle preset x both clips, a 39-min AMI meeting at 30.4 s, and the 46-min TS3003d at 1.04 s - the one 1.04 s meeting whose run diverged). This matches the reference build used here (macOS arm64, torch 2.14, libc++, NEON). A Linux/x86 reference (libstdc++ tie order, AVX2 8-lane sums) could select differently on exact ties; the DER gate is unaffected, and the replay gate would catch it.
- **Residual divergence is fp32 GEMM noise, not a bug.** ~3e-6 input noise can move a near-tied boundary score across a ~1e-5 gap and flip one pick, after which the run differs (6 of 96 acceptance meetings at 30.4 s; the 0.64 s oracle unforced). NeMo run with different thread counts shows the same 1e-5 noise. Tensor validation therefore injects the reference's picks (forced-picks gate, like reference-mel injection) and gates selection by replay and end to end by DER/JER.
- Input / memory contract (`docs/input-limits.md`, bucket 1): no length cap (`max_audio_ms = 0`), `n_ctx` ignored, per-step compute/memory bounded by spkcache + FIFO + chunk + lookahead; only the stored mel and 10 ms probs grow with length. Input under one hop (160 samples) is an empty OK result on run / push-audio / batch (NeMo: 0 frames). Short clips match NeMo to ~1e-7 (160 / 400 / 8000 samples). Abort is polled per chunk (`TRANSCRIBE_ERR_ABORTED`). All pinned in `nemotron3_diar_stream_unit`.
- Frames: NeMo emits floor(n / 160) mel frames (1,471,999 samples -> 9199), not ceil; the driver consumes exactly those, so run, push-audio finalize and batch all return NeMo's frame count.
- Push-audio: mel frames are computed incrementally from a PCM segment starting 2 hops before the first new frame (normalize=none makes frames independent), and a chunk runs once its full window (chunk + lookahead) is final; finalize runs the tail with NeMo's end-of-audio clamps. Output is bit-identical to `transcribe_run` at the same preset.
- Batch: `run_batch` advances recordings in lockstep (in sync streaming the cache/FIFO geometry depends only on the step index), batching steps with identical geometry. Output is bit-identical to serial; on CPU it is no faster (7.1 s vs 6.5 s for 8 recordings at 30.4 s; 135 s vs 138 s at 1.04 s) because the per-step encode is already compute-bound. GPU throughput is a Stage 6 question.
- Throughput (CPU, M4 Max, 12 threads): 30.4 s preset RTF ~0.02; 1.04 s preset RTF ~0.40 (every 0.72 s chunk re-encodes ~540 cache+FIFO+chunk frames through 31 layers). NeMo's CPU reference: RTF 0.25 at 1.04 s. Stage 6 owns tuning.
- Validation hooks (not compile-gated, following the sortformer precedent so `validate.py` works on a normal build; dump-only ones are inert without `TRANSCRIBE_DUMP_DIR`): see `docs/environment-variables.md` (`TRANSCRIBE_NEMOTRON3_DIAR_*`).

- Intake: `reports/porting/nemotron3_diar/Nemotron-3-Diarization/intake.json` (the `known_risks` list there is the authoritative risk inventory).
- NVIDIA ships `Nemotron-3-Diarization.q8_0.gguf` for NeMo-Speech.cpp with `general.architecture = sortformer`. It is not consumed here. Our GGUF uses a distinct arch string.
- The `.nemo` `sortformer_modules` streaming values (fifo 0, chunk 264, rc 0) are training-time settings. Runtime defaults come from the card's preset table.
- Checkpoint tensors not used by the forward, which the converter drops: `sortformer_modules.hidden_to_spks.*` (legacy head, frozen, never called) and `sortformer_modules.activity_head.*` (training-only aux head).
- `preprocessor.featurizer.{window,fb}` are stored in **BF16**, and NeMo's reference mel uses them. The converter embeds them as `frontend.window` [400] and `frontend.mel_filterbank` [128, 257] (F32 storage, BF16-exact values), so the C++ mel must read them rather than recompute. Recomputing in fp32 diverges from NeMo by the BF16 rounding (window max |diff| 1.9e-3).
- Conversion (Stage 3) decisions:
  - GGUF arch `nemotron3_diar`, KV prefix `stt.nemotron3_diar.*`. The streaming defaults written to the GGUF are the card's very_high_latency preset (264/40/340/40/300), not the `.nemo` training values. The AOSC compression constants (`sil_threshold`, boost rates, etc.) come from the `.nemo` `sortformer_modules`.
  - The fused `attn.w_qkv` stays fused as `enc.blocks.{i}.attn.qkv.weight` [1536, 512], rows `[q | k | v]`, each head-major (NeMo views it as `(T, 3, H, D)`).
  - Names: `norm1/norm2` -> `norm_1/norm_2`, `embed_norm` -> `enc.embed.norm`, `ffn.net.{0,3}` -> `ff.{in,out}`, `first_hidden_to_hidden` -> `diar.fc1`, `single_hidden_to_spks` -> `diar.single_spk_head`, `subpixel_upsample` -> `diar.upsample.conv`, `learnable_sil_emb` -> `diar.sil_emb`.
  - Storage: Linear weights BF16 (bit-exact); biases, LayerNorms, `diar.sil_emb` and frontend buffers F32 (lossless upcast). `diar.upsample.conv.weight` is F16 because the loader has no BF16 conv kernel (voxtral precedent). BF16 -> F16 rounds only its sub-6e-5 weights, max |diff| 2.98e-8.
  - New quant-policy rule: `diar.sil_emb` -> Norm (F32), added to both `reference_dtype_for` and `policy.cpp::classify_tensor`, and pinned in `test_quant_policy_sync.py`.
- Frontend: symmetric Hann window (`periodic=False`), STFT `center=True` with zero (`constant`) padding, preemphasis 0.97, `normalize=NA`, dither 1e-5.
