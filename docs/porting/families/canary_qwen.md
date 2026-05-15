# Canary-Qwen

Status: research (Stage 1 intake drafted)

## Identity

- Family key: `canary_qwen`
- Upstream architecture string: NeMo `nemo.collections.speechlm2.models.SALM` (FastConformer encoder + Qwen3-1.7B LLM with LoRA, audio-LLM pattern)
- Hugging Face source repos:
  - `nvidia/canary-qwen-2.5b`
- License:
  - canary-qwen-2.5b: **CC-BY-4.0** (commercial use OK with attribution)
- Variants:
  - `canary-qwen-2.5b` — 2.5B params total (1.7B Qwen3 LM frozen + ~0.8B FastConformer perception module + LoRA adapters), English only, no timestamps, BF16, HF safetensors only.

## References

- Canonical reference: **NeMo** (`nvidia/NeMo`, ≥ v2.5.0) via `nemo.collections.speechlm2.models.SALM.from_pretrained('nvidia/canary-qwen-2.5b')`. SALM is NVIDIA's Speech-Augmented Language Model framework — first-party for canary-qwen and the only place the perception → audio-token-scatter → Qwen3 LM glue lives.
- Instrumented reference: same — NeMo SALM with forward hooks on `model.perception.preprocessor`, `model.perception.encoder`, `model.perception.modality_adapter`, the perception output projection, the audio-embedding scatter, and the Qwen3 LM head for tensor dumps in Stage 2.
- Cross-check references:
  - `nvidia/canary-1b-flash` — the encoder of canary-qwen is initialized FROM canary-1b-flash's FastConformer (`pretrained_asr` field in config.json). The `canary` family's encoder code path and `enc.mel.in` numerics are directly comparable.
  - `Qwen/Qwen3-1.7B` — the frozen LM backbone. Tokenizer (BPE, vocab.json + merges.txt + chat_template.json) is pulled from this repo at convert time. Qwen3 LM idioms (Q/K RMSNorm per head, GQA, 1D RoPE, tied word embeddings) are the same shape as the `qwen3_asr` port already covers.
  - Existing `qwen3_asr` port — audio-LLM injection plumbing (audio embeddings scattered into LM input_embeds at audio-token positions) is the same pattern, modulo a different encoder (Qwen3-ASR uses a Whisper-style chunked encoder; canary-qwen uses FastConformer).

## Architecture summary

- Pattern: `audio-llm` (NeMo SALM = Speech-Augmented Language Model). Audio frames are projected to LM hidden size and SCATTERED into the Qwen3 LM input_embeds at positions where `input_ids == audio_locator_tag id`. There is no cross-attention — the LM runs standard self-attention over the joint (text-prompt + audio + text-response) sequence.
- Frontend: NeMo `AudioToMelSpectrogramPreprocessor` — 128-bin slaney-normalized mel filterbank on a 16 kHz mono waveform, hann_periodic window, 25 ms (400-sample) win_length, 10 ms (160-sample) hop, 512-point FFT, preemph=0.97 applied **before** windowing, per-feature normalization. Identical to the canary and parakeet families.
- Encoder (perception): `nemo.collections.asr.modules.ConformerEncoder` (FastConformer) — 32 layers, d_model=1024, 8 heads, ff_expansion=4 (ffn_dim=4096), conv_kernel=9, batch_norm, depth-wise striding subsampling factor=8 (256-channel subsampling conv), `rel_pos` attention with `untie_biases=true`, `xscaling=false`, pos_emb_max_len=5000. Byte-for-byte the same architecture as canary-1b-flash.
- Modality adapter: `nemo.collections.speechlm2.modules.perception.IdentityConnector` (no parameters; pass-through).
- Perception projection: encoder d_model=1024 → `output_dim=2048` (matching Qwen3-1.7B `hidden_size=2048`). Lives inside `AudioPerceptionModule` (the exact tensor name and whether it is one nn.Linear or Linear + activation + Linear is a Stage 2 confirmation).
- LM: `Qwen/Qwen3-1.7B` — 28 layers, hidden_size=2048, 16 Q heads / 8 KV heads (GQA), head_dim=128, intermediate=6144, rope_theta=1e6, max_position_embeddings=40960, tie_word_embeddings=true. Frozen base + LoRA (r=128, alpha=256, dropout=0.01) applied to `q_proj` and `v_proj` only (28 × 2 = 56 LoRA pairs).
- Audio injection: `<|audioplaceholder|>` is added as a special token to the Qwen3 tokenizer at SALM training time. At inference, the LM input_ids contain one audio_placeholder per encoder output frame (12.5 frames/s after factor-8 subsampling); the perception output is scattered into the corresponding rows of input_embeds before the LM forward.
- Generation contract: chat-format prompt `[{"role":"user","content":"Transcribe the following: <|audioplaceholder|>","audio":[...]}]`. Greedy decoding with `max_new_tokens=128` is the model card example.
- Output head: tied to `embed_tokens` (no separate `lm_head` tensor — TENSOR_DUPLICATED fallback at load).
- Audio length contract: native ≤ 40 s; max_total_tokens=1024 (prompt + audio + response). With 12.5 frames/s, 40 s → 500 audio frames, leaving 524 tokens for prompt + response. Long-form / chunking: out of scope for v1.

## Capabilities (from intake)

| Variant | Languages | Translation | Timestamps | Streaming | VAD | Diarization | Notes |
|---|---|---|---|---|---|---|---|
| canary-qwen-2.5b | en | no | none | no | no | no | English-only despite a multilingual encoder; PnC supported; dual-mode (ASR + LLM-only) — LLM mode is out of scope for transcribe-cli. |

## Upstream benchmarks (from model card)

Open ASR Leaderboard, NeMo v2.5.0, greedy decoding, whisper-normalizer v0.1.12.

| Dataset | WER (%) |
|---|---:|
| LibriSpeech test-clean | 1.60 |
| LibriSpeech test-other | 3.10 |
| AMI | 10.18 |
| GigaSpeech | 9.41 |
| Earnings22 | 10.42 |
| SPGISpeech | 1.90 |
| TEDLIUM v3 | 2.72 |
| VoxPopuli (en) | 5.66 |
| Open ASR Leaderboard mean | **5.63** |

**Acceptance dataset for Stage 7 WER gate: LibriSpeech test-clean** (English; default per the porting skill — model is English-only and reports a published 1.60% on this dataset). Stage 7 will reproduce the upstream-reported WER within ±0.01 absolute at the reference dtype (BF16) as the ship gate.

## Environment

Python env: `scripts/envs/canary_qwen/pyproject.toml` *(to be created at Stage 2)*. Expected pins: `nemo_toolkit[asr]>=2.5.0` (SALM lives under `nemo.collections.speechlm2`, added in 2.5.0), `torch`, `soundfile`, `numpy`, `transformers` (for the Qwen3 tokenizer at convert time), plus `gguf` for the converter.

## Artifacts

| Artifact | Path |
| --- | --- |
| Intakes | `reports/porting/canary_qwen/<variant>/intake.json` |
| Preflight Gate A | `reports/porting/canary_qwen/<variant>/preflight-gate-A.json` |
| Forward map | `reports/porting/canary_qwen/forward-map.md` *(Stage 4)* |
| Tolerances | `tests/tolerances/canary_qwen.json` *(Stage 2 provisional → Stage 4 final)* |
| Converter reports | `reports/convert/<variant>-BF16.json` |
| Reference dump root | `build/validate/canary_qwen/<variant>/` |
| WER reports | `reports/wer/<variant>-<preset>.librispeech-test-clean.{jsonl,score.json}` |
| WER summaries | `reports/wer/<variant>.librispeech-test-clean.summary.md` |
| User-facing model cards | `docs/models/<variant>.md` |
| HF README specs | `scripts/hf_cards/<variant>.yaml` |
| HF README rendered | `models/<variant>/README.md` |

## Commands

Reference dump (per variant — perception encoder + audio-token scatter + LM decode):

```bash
TODO  # Stage 2 will produce scripts/dump_reference_canary_qwen_nemo.py modeled on dump_reference_canary_nemo.py + dump_reference_qwen3_asr.py
```

Conversion:

```bash
TODO  # Stage 3 will produce scripts/convert-canary-qwen.py — must merge LoRA A/B into Qwen3 q_proj/v_proj at convert time, pull tokenizer from Qwen/Qwen3-1.7B, add the audio_placeholder special token
```

Quantization (produces F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M):

```bash
uv run scripts/quantize-all.py models/<variant>/<variant>-BF16.gguf
```

Validation:

```bash
uv run scripts/validate.py all --family canary_qwen --variant <variant>
```

WER:

```bash
uv run scripts/wer/run.py \
  --model models/<variant>/<variant>-BF16.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/<variant>-BF16.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/<variant>-BF16.librispeech-test-clean.jsonl
```

Benchmarks (deferred until after the public HF release):

```bash
uv run scripts/bench/run.py \
  --models <variant> \
  --quants q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name <variant>-publication
```

## Known risks

See per-variant `reports/porting/canary_qwen/<variant>/intake.json::known_risks`. Family-wide highlights:

1. **Audio-LLM injection (scatter into LM input_embeds).** Same pattern as `qwen3_asr` / `qwen2_audio`; mismatch in the scatter index or the projection dim silently shifts the LM context with no shape error.
2. **`<|audioplaceholder|>` is an added special token, not in upstream Qwen3-1.7B vocab.** Its concrete ID is decided when SALM training adds it; Stage 2 must pin the exact ID from the SALM checkpoint's added_tokens.
3. **LoRA on q_proj/v_proj only (r=128, alpha=256, all 28 LM layers).** Converter must merge LoRA A/B into the base Qwen3 q_proj/v_proj at convert time so the GGUF and the loader stay LoRA-unaware. Inspect tensor names at Stage 3 to decide whether the safetensors stores merged weights or separate base + LoRA pairs.
4. **Qwen3 LM idioms.** Q/K RMSNorm per head (norm along head_dim), GQA (16 Q / 8 KV heads, head_dim=128), standard 1D RoPE (rope_theta=1e6, max_pos=40960), tied word embeddings (omit lm_head; TENSOR_DUPLICATED fallback). Reuse the `qwen3_asr` ops; do NOT carry over the interleaved multimodal RoPE used there — canary-qwen uses standard 1D RoPE because audio is injected as embeddings into a 1D position grid.
5. **FastConformer encoder is byte-for-byte canary-1b-flash.** The pretrained_asr field declares it. Reuse the existing canary/parakeet rel_pos relative-shift code path. Encoder weight dtype here is BF16 (vs F32 in canary-1b-flash safetensors) — tolerances retuned at Stage 4.
6. **Frontend dither override.** config declares dither=1e-5 (NeMo training default); inference must use 0.0 to keep Stage 2 oracle dumps deterministic and Stage 4 numerical comparisons reproducible.
7. **Distribution: HF safetensors only.** No `.nemo` archive, no tokenizer, no preprocessor on HF (unlike canary-1b-flash). Tokenizer + chat template must be pulled from `Qwen/Qwen3-1.7B`. The frontend cfg lives inside the SALM config.json under `perception.preprocessor`.
8. **English only.** The encoder was multilingual in the canary-1b-flash starting point but SALM was trained English-only. Multilingual transcription is not advertised; the runtime should accept only `--language en` (or no language hint, since the Qwen3 LM has no source-language conditioning anyway).
9. **LLM mode (text-only QA / summarization on transcript) is out of scope.** transcribe-cli targets ASR only. The dual-mode capability is documented for completeness but the C++ runtime ships the ASR path only.
10. **Greedy decoding default.** The model card example uses `max_new_tokens=128` greedy; this is what produces the upstream 5.63% mean WER. First port uses greedy.

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with `Status: TODO`; Stage 4 fills in the observed `Status` after running each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream but the public CLI/API does not surface a way to verify it.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally not exercised here; reason names what would unblock the row.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/canary-qwen-2.5b/canary-qwen-2.5b-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/canary-qwen-2.5b/canary-qwen-2.5b-BF16.gguf samples/jfk.wav` | non-empty plausible transcript on the no-hint path (model has no source-language conditioning, so behavior should be identical to the en path) | TODO |
| Punctuation & capitalization | always-on (no toggle exposed by SALM prompt) | `build/bin/transcribe-cli -m models/canary-qwen-2.5b/canary-qwen-2.5b-BF16.gguf --language en samples/jfk.wav` | transcript carries punctuation and mixed-case | TODO |
| Translate | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Segment timestamps | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Word timestamps | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Streaming | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Voice activity detection | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Speaker diarization | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Language detection | n/a | not advertised by upstream (English only) | n/a | SKIP — not advertised |
| Dual-mode LLM (text-only QA on transcript) | non-ASR mode | `model.llm.disable_adapter()` + text-only generate | reasoning / summarization output | SKIP — not exposed by runtime (transcribe-cli is ASR only; LLM mode is out of scope for v1) |

## Notes

- **Similarity to existing ports.** Canary-Qwen sits at the intersection of two existing families:
  - The **encoder half** is byte-for-byte canary-1b-flash (FastConformer 32-layer, d_model=1024, n_heads=8, ff_expansion=4, conv_kernel=9, dw_striding factor=8, rel_pos with untie_biases=true). The `pretrained_asr` field in config.json declares this initialization. Reuse the existing canary / parakeet encoder code path.
  - The **decoder half** is the audio-LLM injection pattern used by `qwen3_asr` (audio frames scattered into LM input_embeds at audio-token positions), but with the LM being Qwen3-1.7B with LoRA on q_proj/v_proj — vs Qwen3-ASR's bespoke `Qwen3ASRForConditionalGeneration` class. Standard 1D RoPE applies (no interleaved multimodal RoPE needed here, unlike qwen3_asr).
  - This is the **first NeMo SALM port** in the repo; canary-qwen is also the **first audio-LLM port whose encoder is FastConformer** (qwen3_asr/cohere/qwen2_audio all have Whisper-style chunked encoders).
- **Why a new family vs adding a variant under `canary`.** Reference class differs (`SALM` vs `EncDecMultiTaskModel`), architecture pattern differs (`audio-llm` vs `encoder-decoder`), tokenizer differs (Qwen3 BPE vs concatenated SP), prompt protocol differs (Qwen chat template + audio scatter vs positional task-token slots). Sharing the `canary` family key would force the C++ loader to branch on `model_type` for everything except the encoder. A separate family keeps the bring-up surgical.
- **Stage 2 dumper plan.** Modeled on `dump_reference_canary_nemo.py` (encoder side) + `dump_reference_qwen3_asr.py` (audio-token scatter + LM decode side). Dump tensors at: `mel.in`, `enc.subsampling.{0..2}.{conv,bn}.out`, `enc.blocks.{0..31}.{ffn1,attn,conv,ffn2,norm_out}`, `perception.proj.out` (the d_model=1024 → output_dim=2048 projection), `lm.input_embeds.scattered`, `lm.blocks.{0..27}.{attn,ffn}`, `lm.out_norm`, and `lm.logits` (greedy-decoded transcript + step-N logits for autoregressive gate).
- **Stage 3 converter plan.** New `scripts/convert-canary-qwen.py` driven by `scripts/envs/canary_qwen/`. Steps: (a) load SALM via NeMo, (b) extract perception encoder + projection weights, (c) extract Qwen3 LM weights post-LoRA-merge, (d) pull Qwen3 tokenizer from `Qwen/Qwen3-1.7B`, (e) add the audio_placeholder special token at the ID the SALM tokenizer assigned, (f) emit BF16 GGUF with the canary-family encoder tensor names + qwen3-family LM tensor names, (g) emit `stt.canary_qwen.audio_locator_id` KV.
- **Stage 4 C++ plan.** New `src/arch/canary_qwen/`. Encoder code path can be a thin subclass of (or include from) `src/arch/canary/` (FastConformer 32-layer, rel_pos, dw_striding factor=8). LM code path can be a thin subclass of (or include from) `src/arch/qwen3_asr/` (Qwen3 LM with Q/K RMSNorm, GQA, 1D RoPE, tied embeddings) — but skip the qwen3_asr-specific interleaved multimodal RoPE.
- **English-only smoke test.** Use `samples/jfk.wav` for the Stage 4 transcribe smoke test (same as canary). No multilingual samples needed.
