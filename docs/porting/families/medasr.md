# MedASR

Status: research

## Identity

- Family key: `medasr`
- Upstream architecture string: `LasrCtcForCTC` (model_type `lasr_ctc`; encoder model_type `lasr_encoder` — Conformer + RoPE + macaron FFN + CTC head)
- Hugging Face repo: `google/medasr`
- Hugging Face revision: `ae1e4845b4b07479735d93e1e591e566435b7104` (pinned at intake)
- License: **Health AI Developer Foundations** — `https://developers.google.com/health-ai-developer-foundations/terms`. Gated; HF account must accept terms before download. Verify redistribution clauses before pushing converted GGUFs to `handy-computer/*` on the Hub.
- Variants:
  - `medasr` — 105M params, F32, English-only medical-dictation ASR (pre-trained on LibriHeavy 50k h; fine-tuned on ~5k h of de-identified physician dictations across radiology, internal medicine, and family medicine).

## References

- Canonical reference: **transformers** — `AutoModelForCTC.from_pretrained("google/medasr")` + `AutoProcessor.from_pretrained("google/medasr")` against `transformers @ git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5` (transformers version stamp inside config.json is `5.0.0.dev0`; v5.0.0 is unreleased so the commit pin is load-bearing).
- Instrumented reference: same — hooks added to `LasrCtcForCTC.forward` and `LASRFeatureExtractor.__call__` for tensor dumps at Stage 2.
- Cross-check references:
  - `transformers/models/lasr_ctc/` and `transformers/models/lasr_encoder/` at the pinned commit — authoritative for layer definitions, residual scalars, RoPE, and CTC head.
  - `transformers/models/lasr/feature_extraction_lasr.py` at the pinned commit — authoritative for window type, preemphasis, dither, mel filterbank norm, log compression, and STFT center/padding (none declared in `preprocessor_config.json`).
  - 6-gram KenLM (`lm_6.kenlm`, `lm_6.arpa.xz`) shipped on the HF repo — used by publisher headline WER but OUT OF SCOPE for the first ship (greedy CTC only).

There is no NeMo / ESPnet / FunASR / author-side standalone repo for this checkpoint. The Google Health [github.com/google-health/medasr](https://github.com/google-health/medasr) repo only contains notebooks that wrap the HF Transformers path; it is not a separate reference implementation.

## Golden Manifest

`tests/golden/medasr/medasr.manifest.json` (skeleton; Stage 2 fills in the rest).

## Artifacts

| Artifact | Path |
| --- | --- |
| Intake | `reports/porting/medasr/medasr/intake.json` |
| Preflight Gate A | `reports/porting/medasr/medasr/preflight-gate-A.json` |
| Forward map | `reports/porting/medasr/forward-map.md` (TODO at Stage 3) |
| Tolerances | `tests/tolerances/medasr.json` (TODO at Stage 2) |
| Converter report | `reports/convert/medasr-F32.json` (TODO at Stage 3) |
| Reference dump root | `build/validate/medasr/medasr/` (TODO at Stage 2) |
| Bench reports | `reports/perf/<machine>/medasr-medasr-*.json` (TODO at Stage 6) |
| WER scores | `reports/wer/medasr-{REF,F32,F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M}.test-clean.score.json` (TODO at Stage 7) |
| WER summary | `reports/wer/medasr.test-clean.summary.md` (TODO at Stage 7) |

## Commands

Reference run:

```bash
TODO  # Stage 2: scripts/dump_reference_medasr_hf.py decode --model google/medasr --audio samples/jfk.wav --out build/validate/medasr/medasr/jfk/decode/ref
```

Reference dumps:

```bash
TODO  # Stage 2: scripts/dump_reference_medasr_hf.py dump --model google/medasr --audio samples/jfk.wav --out build/validate/medasr/medasr/jfk/
```

Conversion:

```bash
TODO  # Stage 3: uv run --project scripts/envs/medasr scripts/convert-medasr.py google/medasr
```

Validation:

```bash
uv run scripts/validate.py all --family medasr --variant medasr
```

Benchmarks:

```bash
TODO  # Stage 6: uv run scripts/bench/run.py --models medasr --quants q8_0,q4_k_m --samples jfk --backends metal,cpu --iters 3 --warmup 1 --name medasr-publication
```

## Architecture summary

- Pattern: `encoder-ctc` (non-autoregressive; CTC head over a 512-token SentencePiece vocab; CTC blank is id 0, `<epsilon>`).
- Frontend: `LASRFeatureExtractor` — 128-bin log-mel at 16 kHz, n_fft=512, hop_length=160 (10 ms), win_length=400 (25 ms). Window type, preemphasis, dither, mel filterbank norm (slaney vs htk), padding mode, and log compression are **not** declared in `preprocessor_config.json`; they live in the LASRFeatureExtractor source at the pinned transformers commit and must be pinned by Stage 2 oracle dump.
- Encoder: `lasr_encoder` Conformer variant — 17 layers, d_model=512, 8 MHA heads (num_kv_heads=8 = no GQA), FFN intermediate_size=2048 (SiLU). Conformer convolution module kernel_size=32 with non-standard residual scalars `conv_residual_weights=[2.0, 1.0]`. Macaron FFN residual scalars `feed_forward_residual_weights=[1.5, 0.5]`. **RoPE positional encoding** (rope_theta=10000.0, default scaling) inside the encoder attention — NOT relative-position attention. Subsampling is a **single** stride-2 conv (kernel=5, channels=256) producing a 20 ms output frame stride (50 fps) — not the 4x/8x subsampling typical of FastConformer-style ASR encoders.
- Output head: linear CTC head 512 -> 512. Greedy CTC decode = argmax + collapse repeats + drop blanks. Optional 6-gram KenLM beam decoding is shipped (`lm_6.kenlm`) but OUT OF SCOPE for the first ship.
- Tokenizer: SentencePiece BPE (`spiece.model`, ~241 KB) with HF fast-tokenizer wrapper (`LasrTokenizer`, backend=tokenizers). Model output vocab = 512 entries: id 0 = `<epsilon>` (CTC blank), id 1 = `<s>`, id 2 = `</s>`, id 3 = `<unk>`, ids 4–511 = SentencePiece pieces. Ids 512–612 exist in `tokenizer.json` (`<pad>`, 100 T5-style `<extra_id_N>`) but are **above** the model output dim and cannot be emitted by the CTC head.
- Audio length contract: HF pipeline default uses `chunk_length_s=20, stride_length_s=2` for long-form (chunked offline batching, not real-time streaming). The model itself is not advertised as streaming-capable.

## Capabilities (from intake)

- Languages: 1 — English (`en`). Trained on US English; the model card flags lower accuracy on non-native accents and a male-skewed speaker distribution.
- Language detection: no (monolingual).
- Translation: no.
- Timestamps: none — non-AR CTC has no segment/word timestamp head.
- Streaming: no — the chunked offline batching in the HF pipeline is not real-time streaming.
- VAD: no.
- Diarization: no.

## Upstream benchmarks (from model card)

| Dataset | Greedy CTC WER | + 6-gram KenLM | Notes |
|---------|---------------:|---------------:|-------|
| RAD-DICT (radiologist dictation, internal) | 6.6% | 4.6% | Internal Google dataset; not publicly reproducible. |
| GENERAL-DICT (general/internal medicine, internal) | 9.3% | 6.9% | Internal Google dataset; not publicly reproducible. |
| FM-DICT (family medicine, internal) | 8.1% | 5.8% | Internal Google dataset; not publicly reproducible. |
| Eye Gaze (MIMIC chest X-ray, 998 cases) | 6.6% | 5.2% | Publicly available subset of MIMIC; potentially reproducible. |
| LibriSpeech test-clean | — | — | **Not reported.** Stage 7 still gates against measured Oracle reference baseline on test-clean. |

Acceptance dataset for Stage 7 WER gate: **LibriSpeech test-clean** (default per the porting skill; MedASR supports English so it is a valid target). The publisher does not report a LibriSpeech number; the Stage 7 gate uses the measured Oracle reference baseline rather than a publisher target. Absolute WER on test-clean is expected to be substantially worse than general-purpose ASR (Whisper v3 large scores 12.5% on Eye Gaze while MedASR scores 6.6% — the domain mismatch reverses on out-of-domain LibriSpeech). That gap is not a port defect.

## Known risks

See `reports/porting/medasr/medasr/intake.json::known_risks`. Highlights:

1. **transformers v5.0.0.dev0 reference is pinned to a specific dev commit (65dc261512cbdb1ee72b88ae5b222f2605aad8e5).** v5.0.0 is unreleased; any later commit may move LASR class internals and any earlier commit will not register `lasr_ctc`. Env build is more fragile than typical HF Transformers ports.
2. **LASRFeatureExtractor defaults not declared in preprocessor_config.json.** Window, preemphasis, dither, mel filterbank norm, padding mode, STFT center, and log compression must be pinned from the feature_extraction_lasr source at the pinned commit during Stage 2 oracle dump. This is the largest frontend ambiguity for the port.
3. **RoPE inside a Conformer ASR encoder is unusual.** Existing ASR ports (parakeet, sensevoice, funasr_nano) use relative-position or sinusoidal attention. Stage 4 needs a per-encoder-layer RoPE path; the audio-llm/qwen3_asr rope_theta=10000 default applies.
4. **Non-standard macaron FFN scalars `[1.5, 0.5]` and conv-module scalars `[2.0, 1.0]`.** Existing Conformer ports use unscaled or standard 0.5 macaron residuals. The exact pair must be hard-coded into the converter+C++; an off-by-scale residual is a silent accuracy regression with no shape mismatch.
5. **Single stride-2 subsampling (20 ms frame stride, 50 fps).** Sequence length out of the encoder is 2x longer than peer Conformer ports for the same audio. Bench targets must reflect this; do not assume equal-to-peer.
6. **CTC blank id = 0 (`<epsilon>`), not `vocab_size - 1`.** Existing parakeet/sensevoice convention is blank = last id. Converter must emit the correct blank id KV and the decoder must not assume `blank == vocab_size - 1`.
7. **Tokenizer vocab (613) wider than model output dim (512).** Ids 512–612 exist in tokenizer.json but cannot be emitted by the CTC head. Converter should ship the 512-entry model-head vocab, not the full 613-entry tokenizer vocab.
8. **No publisher LibriSpeech number; medical-domain training.** Stage 7 falls back to the measured Oracle reference baseline. Absolute test-clean WER will likely be worse than general-purpose ASR (per `project_upstream_wer_unreproducible` memory).
9. **6-gram KenLM out of scope.** Greedy CTC only for the first ship.
10. **Gated HF repo (Health AI Developer Foundations terms).** Account must accept terms; redistribution of converted GGUFs to `handy-computer/*` requires confirming the license terms permit it.

## Capability Validation

One row per advertised capability. Each row carries two human/agent
columns that are filled at different stages:

- **`Target`** — the *scope decision*, set at Stage 1 and signed off by
  the user. It declares whether this port is obligated to deliver the
  capability. This is the contract Stage 4 implements against.
- **`Status`** — the *observed outcome*, filled at Stage 4 after running
  the command. Stage 1 leaves it `TODO`.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/medasr/medasr-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/medasr/medasr-F32.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | TODO |
| Translate | n/a | not advertised by upstream (English-only, no translation) | n/a | OUT OF SCOPE — model is monolingual English ASR; bringing it in scope would require an upstream translation head that does not exist | TODO |
| Segment timestamps | n/a | non-AR CTC has no segment-timestamp head | n/a | OUT OF SCOPE — CTC architecture has no segment timestamps; bringing it in scope would require an external forced-aligner not part of the model | TODO |
| Word timestamps | n/a | non-AR CTC has no word-timestamp head | n/a | OUT OF SCOPE — CTC architecture has no word timestamps; bringing it in scope would require ctc_forced_align in the runtime | TODO |
| Streaming | n/a | not advertised by upstream (capabilities.streaming=false) | n/a | OUT OF SCOPE — not advertised; HF pipeline `chunk_length_s` is chunked offline batching, not real-time streaming | TODO |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/medasr/medasr-F32.gguf --samples-dir samples/wer/test-clean --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | TODO |
| Voice activity detection | n/a | not advertised by upstream | n/a | OUT OF SCOPE — not advertised; bringing it in scope would require an external VAD model | TODO |
| Speaker diarization | n/a | not advertised by upstream | n/a | OUT OF SCOPE — not advertised; bringing it in scope would require an external diarization model | TODO |
| KenLM 6-gram beam decode | shipped `lm_6.kenlm` | future: `build/bin/transcribe-cli -m models/medasr/medasr-F32.gguf --lm lm_6.kenlm samples/jfk.wav` | publisher headline WER (e.g. RAD-DICT 6.6% -> 4.6%) | OUT OF SCOPE — first ship is greedy CTC only; bringing it in scope would require runtime KenLM integration | TODO |

## Notes

- This is the first `transformers @ dev` reference in the repo. The reference env (`scripts/envs/medasr/pyproject.toml`) must install transformers from the specific commit in the README, not from a PyPI release. Expect ABI churn vs transformers 4.x.
- `samples/jfk.wav` is the standard English smoke test. The HF repo also ships `test_audio.wav` (a medical-dictation clip) — pull this into `samples/medasr/` for a per-family secondary smoke test that exercises the in-domain audio profile.
- Stage 7's WER target is LibriSpeech test-clean, but the model is optimized for medical dictation. Eye Gaze (MIMIC chest X-ray) would be the more representative complementary check if licensing allows; flag if absolute LibriSpeech WER comes in noticeably worse than general ASR without that being a port defect.
