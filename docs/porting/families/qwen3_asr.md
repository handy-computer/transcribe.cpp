# Qwen3-ASR

Status: accuracy (CPU / Metal / Vulkan all pass 13/13 dumps; e2e
transcription matches reference)

Current progress:

- Intake, family note, converter, reference dumper committed.
- `scripts/convert-qwen3_asr.py` emits a 1.5 GB BF16 accuracy GGUF
  (`Qwen3-ASR-0.6B-BF16.gguf`, 613 tensors including the baked
  librosa mel filterbank + Whisper Hann window, 782M dedup params).
- `scripts/dump_reference_qwen3_asr_author.py decode` drives the
  author `qwen_asr` package via forward hooks; all enc.* + dec.*
  dump points land.
- `src/arch/qwen3_asr/` full inference pipeline: hparam reader,
  weight catalog, encoder graph (3× Conv2d + chunked bidirectional
  attention), LM prefill graph (audio injection via concat,
  Qwen3 28-layer GQA LM with flash attention), step graph, byte-
  level BPE decode, transcript assembly.
- `transcribe-cli` transcribes `samples/jfk.wav` with output
  identical to the reference on CPU, Metal, and Vulkan.
- `scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b`
  passes 13/13 tolerances on all three backends.
- Real-model smoke tests (`tests/qwen3_asr_real_smoke_0_6b.cpp`,
  `tests/qwen3_asr_real_smoke_1_7b.cpp`,
  `tests/qwen3_asr_e2e_smoke.cpp`) pass behind
  `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON` +
  `TRANSCRIBE_QWEN3_ASR_0_6B_GGUF` / `TRANSCRIBE_QWEN3_ASR_1_7B_GGUF` /
  `TRANSCRIBE_QWEN3_ASR_GGUF`.

Performance (M4 Max, jfk.wav = 11 s of audio):

| backend | mel    | encode  | decode  | realtime |
|---------|--------|---------|---------|----------|
| CPU     | 63 ms  | 3576 ms | 2256 ms | 2x       |
| Metal   | 66 ms  | 36 ms   | 202 ms  | 33x      |
| Vulkan  | 43 ms  | 150 ms  | 370 ms  | 20x      |

## Identity

- Family key: `qwen3_asr`
- Upstream architecture string: `qwen3_asr`
- Hugging Face repo: `Qwen/Qwen3-ASR-0.6B` (pinned to
  `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`)
- License: Apache-2.0 (model weights), Apache-2.0 (qwen_asr package)
- Variants: `qwen3-asr-0.6b` (first target), `qwen3-asr-1.7b`,
  `qwen3-forced-aligner-0.6b` (separate aligner sibling, different head
  behavior — tracked outside this port's initial scope)

## Architecture

- Pattern: **audio-llm** (audio encoder + causal LM with audio-token
  injection — no cross-attention, no transducer).
- Audio encoder (`Qwen3ASRAudioEncoder`): 3× Conv2d (stride 2, padding 1,
  `downsample_hidden_size=480`) subsampler → flatten `C*F` → linear to
  `d_model=896` → add sinusoidal position embedding (length 1500) →
  18 layers of bidirectional self-attention with block-diagonal
  `cu_seqlens` masking (window_aftercnn = `n_window_infer // (n_window*2)`
  after-CNN frames, default 800/100 = 8) → `ln_post` → `proj1` → GELU →
  `proj2` → `output_dim=1024`.
- Text LM (`Qwen3ASRThinkerTextModel`): 28-layer Qwen3 causal LM.
  `hidden_size=1024`, `intermediate_size=3072`, GQA with
  `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`.
  RMSNorm (eps 1e-6) with per-head Q-norm and K-norm inside attention,
  SwiGLU MLP, interleaved multimodal RoPE
  (`mrope_section=[24,20,20]`, `mrope_interleaved=true`,
  `rope_theta=1_000_000`), tied word embeddings,
  `max_position_embeddings=65536`, `vocab_size=151936`.
- Fusion: the audio encoder output is scattered into the LM input
  embedding at positions where `input_ids == audio_token_id (151676)`;
  the sequence is framed by `audio_start_token_id (151669)` and
  `audio_end_token_id (151670)` in a chat template.
- Output contract: transcript text only (first port). No segment/word
  timestamps (those come from the sibling forced-aligner variant).
  Language tag is emitted as a Qwen-style "language X" prefix from the
  chat template.

## Frontend

Whisper-style log-mel, driven by `WhisperFeatureExtractor` per
`preprocessor_config.json`:

- `sample_rate=16000`, mono
- `n_mels=128`, `hop_length=160`, `n_fft=400` (win_length=400)
- window: `hann_periodic` (numpy hanning default; verify at bridge)
- `center=True`, `padding_mode="reflect"`
- Mel filterbank: `slaney` norm (Whisper default)
- Log-mel is clamped to `max - 8.0`, then `(x + 4.0) / 4.0` normalized
  (Whisper-style per-utterance normalization)
- `dither=0.0`, no preemphasis
- `chunk_length=30` seconds (`n_samples=480000`, `nb_max_frames=3000`)

The existing `transcribe::MelFrontend` already supports Whisper-style
log-mel for Cohere, so we plan to reuse it with a Qwen3-ASR profile
rather than add a second frontend.

## References

- **Canonical reference**: **author repo** — Alibaba Qwen team
  `qwen_asr` Python package, v0.0.6, source vendored at
  `refs/models/qwen3_asr/Qwen3-ASR/`. Provides
  `Qwen3ASRConfig` + `Qwen3ASRForConditionalGeneration` +
  `Qwen3ASRProcessor` registered against the transformers AutoModel.
  Upstream transformers does not (yet) carry these classes.
- **Instrumented reference**: same author repo, via PyTorch forward
  hooks. Will ship as `scripts/dump_reference_qwen3_asr_author.py`.
- **Cross-check reference**: `refs/mlx/mlx-audio/mlx_audio/stt/models/qwen3_asr/`
  (Apple MLX port) and `refs/models/qwen3_asr/qwen3-asr.cpp/` (an
  independent prior C++/ggml port by `predict-woo`, Q8_0-capable).
  Neither is canonical; they are used only for implementation research
  and sanity checks on encoder / decoder intermediates.

Bridge validation:

- Canonical and instrumented references are the same implementation;
  no bridge needed.
- The two cross-check references will be pinned by commit in the family
  note before validation (see `refs/models/qwen3_asr/qwen3-asr.cpp`
  upstream revision; MLX model code is versioned with the mlx-audio
  repo).

## Environment

- OS: macOS 15.x (darwin arm64, primary dev)
- CPU: Apple Silicon (M-series)
- RAM: 24 GB+ recommended for 0.6B bring-up with BF16 reference dumps
- GPU: Metal (secondary); CPU is the numerical source of truth
- Backend/runtime: ggml CPU (Metal is parity-only until CPU is clean)
- Python: 3.11 (see per-family env)
- Reference package versions: `transformers==4.57.6`, `qwen-asr==0.0.6`,
  `torch>=2.2`, `soundfile`, `librosa`, `numpy`, `safetensors`, `gguf`

## Artifacts

- Family note: `docs/porting/families/qwen3_asr.md` (this file)
- Intake: `reports/porting/qwen3_asr/qwen3-asr-0.6b/intake.json`
- Golden manifests:
  `tests/golden/qwen3_asr/qwen3-asr-0.6b.manifest.json`,
  `tests/golden/qwen3_asr/qwen3-asr-1.7b.manifest.json`
- Tolerances: `tests/tolerances/qwen3_asr.json` (0.6B),
  `tests/tolerances/qwen3_asr-1.7b.json` (1.7B)
- Python env: `scripts/envs/qwen3_asr/pyproject.toml`
- Converter: `scripts/convert-qwen3_asr.py`
- Reference dumper: `scripts/dump_reference_qwen3_asr_author.py`
- C++ arch dir: `src/arch/qwen3_asr/`
- Fixture generator hook: `tests/fixtures/make_gguf_fixtures.py` emits
  `arch_qwen3_asr_minimal.gguf`
- Smokes: `tests/qwen3_asr_smoke.cpp` (default ctest, fixture-backed),
  `tests/qwen3_asr_real_smoke_0_6b.cpp` and
  `tests/qwen3_asr_real_smoke_1_7b.cpp` (env-gated structural tests —
  `TRANSCRIBE_QWEN3_ASR_0_6B_GGUF` / `_1_7B_GGUF`),
  `tests/qwen3_asr_e2e_smoke.cpp` (env-gated public-ABI end-to-end,
  `TRANSCRIBE_QWEN3_ASR_GGUF`)

## Known limitations

Things the first port intentionally does not do; tracked as follow-
up work rather than shipped-and-broken.

- **Language hinting is rejected.** `transcribe_params.language == NULL`
  is the supported mode and triggers the model's built-in auto-detect
  (it prefixes the transcript with `language X`, which we strip before
  returning). Any non-null hint, including a language in the
  capability list, returns `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`. The
  `caps.languages` list documents the model's auto-detect coverage,
  not what callers may hint — rendering caller-supplied hints into the
  chat template is a future change.
- **Streaming** (`stream_transcribe` / chunk rollback) is out of scope
  for this port. Upstream Qwen3-ASR may be architecturally usable in a
  streaming mode, but the transcribe.cpp library does not expose or
  implement a streaming path for this family yet. The golden manifests
  therefore advertise `streaming: false`, and the converter does not
  emit `stt.capability.streaming`, so `transcribe_model_capabilities()
  .supports_streaming` is `false` at runtime until the streaming API
  is wired up and validated.
- **Timestamps** are `TIMESTAMPS_NONE`. The sibling
  `qwen3_forced_aligner` variant produces word-level alignments but
  has its own head contract and ships as a separate family.
- **Ragged-tail audio.** The encoder graph runs all chunks at
  `T_enc_padded = n_chunks * per_chunk_aftercnn` rows, then the host-
  side copy at `src/arch/qwen3_asr/model.cpp` trims the padded rows
  off the last chunk down to `T_enc_real = (n_chunks-1) *
  per_chunk_aftercnn + last_chunk_aftercnn`, matching the reference's
  ragged selection. Everything downstream (LM prefill, dec.* dumps)
  sees the trimmed T_enc. The graph-level `enc.*` dumps still carry
  the padded shape — numerical validation in the golden manifest
  runs on `jfk.wav` (single-chunk, no ragged tail) only. Multi-chunk
  audio is validated transcript-level: `tests/qwen3_asr_e2e_smoke.cpp`
  runs both `jfk.wav` and `dots.wav` (35.3s Steve-Jobs-commencement
  excerpt, multi-chunk ragged tail) through the public ABI and
  asserts edit-distance-zero against checked-in reference transcripts,
  so a regression in the ragged-tail trim path fails the e2e smoke.
- **Chat-template token ids** are resolved at load time by name against
  the tokenizer (Phase 1.6 hard-fails if any piece is missing); a
  future variant that renames roles would fail load rather than
  silently produce a wrong prompt.

## Decisions For Implementation

Recorded here during research; resolved during bring-up.

- **MRoPE vs standard RoPE.** The reference materializes a 3D `cos/sin`
  grid, applies an interleaving permutation per `mrope_section`, and
  then applies a standard rotate-half RoPE. For audio + text with only
  one modality active per position (text generation after the audio
  block), the effective rotation reduces to per-section interleaving on
  a 1D position id. Plan: precompute the interleaved cos/sin table
  host-side and feed it to ggml as a position-indexed rope table;
  fall back to per-step cos/sin lookup if the single-modality reduction
  does not hold for the audio region.
- **Audio encoder attention mask.** Block-diagonal from `cu_seqlens`.
  Plan: build a dense `(T, T)` mask host-side and pass to ggml as an
  additive attention bias, the same pattern Cohere uses for its
  padding-aware encoder mask.
- **Per-head Q/K RMSNorm.** Apply RMSNorm on the last dim (`head_dim`)
  of the reshaped `(B, H, T, D)` tensor. In ggml: reshape → rms_norm →
  mul by per-head weight. No new ops required.
- **Tied embeddings.** Converter omits `output.weight`; loader falls
  back to `token_embd.weight` with `TENSOR_DUPLICATED` (same as
  llama.cpp and the existing Cohere decoder path).
- **Prompt template.** The Qwen3 chat template (in
  `chat_template.json`) carries language and hotword context fields.
  Embed the rendered prompt into the GGUF as a string KV, and at
  inference splice the caller-provided language/context into it. The
  tokenizer merge table and special-token ids are the durable part;
  the template is a separate KV.
- **Streaming.** Out of scope for first port. `stream_transcribe`
  (chunk-id based prefix rollback) is a later follow-up.
- **Reuse Cohere's mel frontend?** Yes — same underlying Whisper
  feature-extractor contract. Add a Qwen3-ASR profile; do not duplicate
  the STFT code.
- **llama.cpp decoder reuse.** The LM side is architecturally a
  standard Qwen3 causal LM. Cross-reference `refs/ggml-org/llama.cpp`
  GQA + RoPE + SwiGLU patterns for graph construction; do not import
  llama.cpp code.

## Commands

Conversion (HF → BF16 accuracy GGUF). Pin the `--revision` so
output hashes are reproducible across machines; the golden manifest
records the canonical commit for each variant.

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0

uv run --project scripts/envs/qwen3_asr \
  scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
```

Reference dumps (drives the author `qwen_asr` package with forward
hooks; pins the same revision as the manifest):

```bash
uv run --project scripts/envs/qwen3_asr \
  scripts/dump_reference_qwen3_asr_author.py decode \
  --model Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --audio samples/jfk.wav \
  --out build/validate/qwen3_asr/qwen3-asr-0.6b/jfk/ref
```

Validation (ref + cpp + compare, all backed by the golden manifest —
no `--gguf` override needed; the manifest slug drives discovery):

```bash
uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-0.6b
uv run scripts/validate.py all --family qwen3_asr --variant qwen3-asr-1.7b
```

Transcribe:

```bash
./build/bin/transcribe-cli -m models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
    samples/jfk.wav
```

Real-model structural + e2e tests (opt-in via CMake flag + env var):

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build -j
TRANSCRIBE_QWEN3_ASR_0_6B_GGUF=$PWD/models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
TRANSCRIBE_QWEN3_ASR_1_7B_GGUF=$PWD/models/Qwen3-ASR-1.7B/Qwen3-ASR-1.7B-BF16.gguf \
TRANSCRIBE_QWEN3_ASR_GGUF=$PWD/models/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-BF16.gguf \
  ctest --test-dir build --output-on-failure
```

Benchmarks:

```bash
# See docs/tools/benchmarking.md for the benchmarking harness.
uv run scripts/bench/run.py --family qwen3_asr
```

## Upstream Benchmarks

Publisher-reported; informational only.

| Dataset | Language | Metric | Score | Source |
|---|---|---|---|---|
| LibriSpeech test-clean (Open ASR Leaderboard) | en | WER | 2.13% | HF model-card evaluation_results → `hf-audio/open-asr-leaderboard` |

Broader multilingual WER/CER figures from the Qwen team's blog or paper
should be added when we encounter them during validation.

## Notes

- The existing third-party `qwen3-asr.cpp` port by `predict-woo` is
  *not* canonical; it is a structural sanity check only. Do not
  copy numerics from it without first verifying against the author
  reference.
- The forced-aligner variant (`Qwen3-ForcedAligner-0.6B`) shares the
  audio encoder and most of the LM, but replaces the generation head
  with a token-level alignment readout. Tracked as a follow-on family
  task, not in the initial bring-up.
- The 1.7B variant differs only in LM width/depth (per intake
  `varying_across_variants`). Once the 0.6B accuracy GGUF and C++ path
  are validated, the 1.7B port should be a configuration change rather
  than architectural work.
