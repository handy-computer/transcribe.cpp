# MOSS-Transcribe-Diarize

Status: research

## Identity

- Family key: `moss`
- Upstream architecture string: `MossTranscribeDiarizeForConditionalGeneration` (`model_type: moss_transcribe_diarize`)
- Hugging Face repo: `OpenMOSS-Team/MOSS-Transcribe-Diarize`
- Hugging Face revision: `d7231bbae2587a4af278735eb765b318c4f64edd`
- License: Apache-2.0
- Variants: `moss-transcribe-diarize` (0.9B; Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B decoder)

Architecture pattern: **audio-llm**. log-mel `input_features` → `WhisperEncoder`
(24 layers, d_model 1024, gelu, LayerNorm) → 4× temporal merge (1024 → 4096) →
`VQAdaptor` MLP (Linear 4096→1024, SiLU, Linear 1024→1024, LayerNorm w/ bias) →
`masked_scatter` into Qwen3-0.6B text embeddings at `<|audio_pad|>` (id 151671)
positions → Qwen3 decoder (28 layers, hidden 1024, silu, RMSNorm, GQA 16/8 heads,
head_dim 128, rope_theta 1e6, tie_word_embeddings) → `lm_head`.

Output is emergent plain text in the canonical format `[start][Sxx]text[end]`
(e.g. `[0.48][S01]Welcome[1.66]`); diarization speaker tags and segment
timestamps are generated text, not special tokens.

## References

- Canonical reference: `author_repo_moss` — OpenMOSS custom `trust_remote_code`
  files in the HF repo (`modeling_moss_transcribe_diarize.py`,
  `configuration_moss_transcribe_diarize.py`,
  `processing_moss_transcribe_diarize.py`) plus the OpenMOSS/MOSS-Transcribe-Diarize
  GitHub inference harness (`build_transcription_messages`, `generate_transcription`,
  audio/video loading, transcript parsing). Not in upstream Transformers.
- Instrumented reference: same custom code (single source of truth; no framework fork).
- Cross-check references: `Qwen3Model` and `WhisperEncoder` upstream Transformers
  implementations that the custom code composes.

## Commands

Reference run:

```bash
TODO  # porting-2-oracle: transformers + trust_remote_code=True generate() over samples/jfk.wav
```

Reference dumps:

```bash
TODO  # porting-2-oracle
```

Conversion:

```bash
TODO  # porting-3-convert
```

Validation:

```bash
TODO  # porting-4-cpp
```

Benchmarks:

```bash
TODO  # porting-6-bench
```

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | auto / no language hint (en) | `build/bin/transcribe-cli -m models/moss-transcribe-diarize/moss-transcribe-diarize-<REFDTYPE>.gguf samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | TODO |
| Transcribe | auto / no language hint (zh) | `build/bin/transcribe-cli -m models/moss-transcribe-diarize/moss-transcribe-diarize-<REFDTYPE>.gguf <zh sample>` | non-empty plausible Chinese transcript | MUST PASS | TODO |
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m ... --language en samples/jfk.wav` | non-empty plausible transcript | OUT OF SCOPE — no first-class language token; language is inferred from audio and steered only by prompt text. Back in scope if the runtime exposes a prompt/language selector. | TODO |
| Speaker diarization | default diarize prompt | `build/bin/transcribe-cli -m ... <multi-speaker sample>` | `[Sxx]` speaker tags present in the raw transcript | OUT OF SCOPE — diarized `[Sxx]` output is produced correctly in the raw transcript (rides the same output format the Segment-timestamps row gates), but there is no runtime API surface to expose diarization as a distinct capability; a dedicated API surface is out of scope for this PR. Back in scope when that API lands. | TODO |
| Segment timestamps | default diarize prompt | `build/bin/transcribe-cli -m ... samples/jfk.wav` | output contains `[start]...[end]` numeric timestamps | MUST PASS | TODO |
| Word timestamps | — | — | word-granularity timestamps | OUT OF SCOPE — model emits segment-level `[start][end]` only; no word alignment. | TODO |
| Long-form audio | >30s multi-chunk | `build/bin/transcribe-cli -m ... <audio >30s>` | full transcript across 30s chunk boundaries | MUST PASS | TODO |
| Promptable / hotwords | custom prompt append | `<prompt override>` | hotword-biased transcript | OUT OF SCOPE — custom prompt/hotword injection is not part of the core transcribe path. Back in scope if a prompt-override flag ships. | TODO |
| Acoustic event annotation | promptable | `<prompt override>` | event annotations in output | OUT OF SCOPE — optional promptable feature, not the core transcribe path. | TODO |
| Translate | — | — | English output on non-English audio | OUT OF SCOPE — transcription-only model; no translation capability. | TODO |
| Language detection | — | — | detected language surfaced | OUT OF SCOPE — no detection branch/token; language is not surfaced by the runtime. | TODO |
| Streaming | — | — | streaming path | OUT OF SCOPE — offline model; `capabilities.streaming: false`. | TODO |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/moss-transcribe-diarize/moss-transcribe-diarize-<REFDTYPE>.gguf --samples-dir samples/wer/<dataset> --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | TODO |

## Notes

- Acceptance dataset (user-signed): **LibriSpeech test-clean** (English) as the
  ref-dtype gate, **plus a Chinese diarization set** (Stage 2 picks the specific
  corpus, e.g. AISHELL-4 / Alimeeting) as an on-target companion. The model
  supports English, so LibriSpeech is valid, but it is off the model's primary
  use case (Chinese/English long-form multi-speaker diarization). All publisher
  benchmarks are Chinese diarization CER/cpCER (AISHELL-4, Alimeeting, Podcast,
  Movies); there is no published English number. Downstream gates use the
  measured Oracle reference baseline, not a publisher score. The Chinese set
  needs Chinese audio in `samples/wer/` and cpCER-style scoring infra at Stage 2.
- WER scoring must de-diarize/de-timestamp hypotheses: strip `[Sxx]` speaker
  tags and `[start]`/`[end]` numeric timestamps from generated text before
  scoring against LibriSpeech references. This is a Stage 2 (`reference.entrypoint`)
  concern.
- The default instruction prompt is Chinese and is used for English audio too;
  there is no language/task token (`has_language_tokens: false`).
- Prompt/audio-span construction is non-trivial: the processor interleaves
  literal second-count digit tokens into the `<|audio_pad|>` span. The
  authoritative cadence is **every 5s** (`processor_config.json`:
  `audio_tokens_per_second=12.5`, `time_marker_every_seconds=5`,
  `enable_time_marker=true`) — the constructor default of 2s is overridden by
  the saved processor config. These are emitted as `stt.moss.*` KV. See
  `known_risks` in the intake.
- See `reports/porting/moss/moss-transcribe-diarize/intake.json` and
  `reports/porting/moss/moss-transcribe-diarize/_research_dump.txt` (model card +
  reference code).

## Stage 3 conversion — tensor-mapping decisions

- **Encoder names reuse the `whisper` family**, decoder names reuse the
  `qwen3_asr` text LM. MOSS's `model.whisper_encoder.*` is a stock HF
  `WhisperEncoder`, so it maps to `enc.conv.{0,1}`, `enc.pos_emb`,
  `enc.final_norm`, `enc.blocks.N.{norm_attn,attn.{q,k,v,out},norm_ffn,ffn.{fc1,fc2}}`
  (q/v/out have bias, k does not). The Qwen3 LM maps to
  `dec.token_embd`, `dec.output_norm`,
  `dec.blocks.N.{norm_attn,norm_ffn,attn.{q,k,v,o,q_norm,k_norm},ffn.{gate,up,down}}`.
  This lets Stage 4 share the whisper encoder graph and a Qwen3 decoder graph.
- **VQAdaptor** (`model.vq_adaptor.layers.{0,2,3}`) maps to `adaptor.fc1`
  (Linear 4096→1024), `adaptor.fc2` (Linear 1024→1024), `adaptor.norm_out`
  (final LayerNorm). `norm_out` is deliberately named so the `"norm_"` rule in
  `reference_dtype_for` keeps the LayerNorm scale at F32.
- **Tied head:** `tie_word_embeddings=true` and `lm_head.weight` is absent from
  the state dict; the loader reuses `dec.token_embd.weight` for the output
  projection. No SKIP list needed.
- **Vocab padded to 151936.** The tokenizer names 151672 tokens but the
  embedding/logit width is `text_config.vocab_size=151936`; the token table is
  padded with `<|unused_N|>` so its length matches the logits. Intake +
  manifest `vocab_size` corrected from 151672 to 151936 accordingly.
- **Conv1d kernels → F16.** Reference dtype is BF16 but the loader has no BF16
  conv kernel, so `enc.conv.{0,1}.weight` are stored F16 per
  `reference_dtype_for`. No `policy.cpp` change was needed — every MOSS tensor
  name hits an existing bucket rule (quant-policy sync passes 5/5).
- **Frontend buffers baked in:** `frontend.mel_filterbank` (slaney) and
  `frontend.window` (periodic Hann) are written as F32 tensors so the C++
  MelFrontend is bit-identical.
- GGUF path is `models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-BF16.gguf`
  (upstream-slug convention, matches `Qwen3-ASR-0.6B/` etc.); `stt.variant` is
  the lowercase `moss-transcribe-diarize`.
