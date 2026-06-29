# Granite Speech

Status: research

## Identity

- Family key: `granite`
- Upstream architecture strings: `granite_speech`, `granite_speech_plus`
- Hugging Face repos:
  - `ibm-granite/granite-4.0-1b-speech` (pinned `bd87ab862416353633ea431fe49b1614003623c5`)
  - `ibm-granite/granite-speech-4.1-2b` (pinned `8f4bb5f31ae98971bd218169f00065a041d20058`)
  - `ibm-granite/granite-speech-4.1-2b-plus` (pinned `edd3bf54fbb06d8e263aa0c1939321d67b073f86`)
- License: Apache-2.0 (all variants)
- Variants:
  - `granite-4.0-1b-speech`: audio-llm, AR; supports en/fr/de/es/pt/ja transcription and bidirectional translation.
  - `granite-speech-4.1-2b`: audio-llm, AR; same supported languages; improved punctuation/casing and keyword biasing over 4.0-1b.
  - `granite-speech-4.1-2b-plus`: audio-llm, AR; en/fr/de/es/pt only (no ja); adds speaker diarization tags (`[Speaker N]:`) and word-level timestamps (`[T:N]` centisecond format) in the text stream. Encoder concatenates mid-layer (idx 3) and final-layer hidden states (`cat_hidden_layers=[3]`).

Sibling family: `granite_nar` (the non-autoregressive `ibm-granite/granite-speech-4.1-2b-nar` editor model). Same underlying Conformer encoder dims and Granite-4.0-1b-base LLM weights, but the bidirectional-editor decoding pipeline is structurally distinct enough to warrant its own family. See `docs/porting/families/granite_nar.md`.

Note: IBM publishes an official LLM-only GGUF at `ibm-granite/granite-4.0-1b-speech-GGUF`. It ships the inner `granite-4.0-1b-base` LLM weights as `architecture=granite` but does not include the Conformer audio encoder or the Q-Former projector, so it is not reusable as a transcribe.cpp speech model. We will produce a fused encoder+projector+LLM GGUF (`granite_speech` architecture) ourselves.

## References

- Canonical reference: `transformers` mainline for the AR variants (`GraniteSpeechForConditionalGeneration`, `GraniteSpeechPlusForConditionalGeneration`).
- Instrumented reference: same as canonical. To be configured under `scripts/envs/granite/` at Stage 2.
- Cross-check references:
  - `ibm-granite/granite-4.0-1b-speech-GGUF` (LLM-only; useful for tensor-name and quant-ladder conventions on the Granite LLM half).
  - Open ASR Leaderboard reproduction harness on HuggingFace (`hf-audio/open-asr-leaderboard`). IBM's published WER numbers come from this.

Transformers version pinning:

- `granite_speech` (4.0-1b, 4.1-2b): `transformers>=4.52.1`. Model cards pin 4.54.0 / 4.57.6 respectively.
- `granite_speech_plus` (4.1-2b-plus): requires `transformers>=5.8` (config pins `5.6.0.dev0`); the PyPI release may lag, so the env may need a pinned commit SHA off HEAD.

The reference env (`scripts/envs/granite/pyproject.toml`) must cover both `granite_speech` and `granite_speech_plus`. Stage 2 picks a transformers SHA new enough to register `granite_speech_plus`.

## Architecture

The AR variants share a near-identical skeleton:

- **Audio encoder**: Conformer with block-attention (model_type `granite_speech_encoder` / `granite_speech_plus_encoder`).
  - 16 layers, `hidden_dim=1024`, 8 heads at `dim_head=128`, depthwise `conv_kernel_size=15`, `conv_expansion_factor=2`, `feedforward_mult=4`, `input_dim=160` (= 80 logmels with a 2-frame stack at input), `output_dim=348`, `max_pos_emb=512` (Shaw relative-position embedding), `context_size=200` (block-attention boundary).
  - Macaroni FFN halves (0.5 residual scaling on both half-FFNs), GLU-gated convolution module, self-conditioned CTC bypass from a middle layer.
  - `4.1-2b-plus` adds `cat_hidden_layers=[3]`: encoder hidden state at layer 3 is concatenated along the feature dim with the final-layer output; projector input becomes `2*1024=2048`.
- **Projector**: BLIP-2 Q-Former (`model_type=blip_2_qformer`). 2 layers, `hidden_size=1024`, `intermediate_size=4096`, 16 heads, absolute position embedding, `cross_attention_frequency=1`, `layer_norm_eps=1e-12`. `encoder_hidden_size=1024` for base, `2048` for `-plus`. Trainable queries cross-attend to encoder output windows of `window_size=15`. The projector vocab_size=30522 is vestigial from BLIP-2 and unused.
  - `downsample_rate=5`. Effective audio-token rate into the LM is ~10 Hz.
- **Text LM**: `granite-4.0-1b-base` (`model_type=granite`). 40 layers, `hidden_size=2048`, `intermediate_size=4096`, 16 query heads / 4 KV heads (GQA), `vocab_size=100353`, `max_position_embeddings=4096`, `rope_theta=10000` (standard RoPE, no scaling), `rms_norm_eps=1e-05`, SiLU activation. Granite scalar multipliers `embedding_multiplier=12`, `logits_scaling=8`, `attention_multiplier=0.0078125 (=1/128)`, `residual_multiplier=0.22` apply on every forward pass; missing any one of them silently degrades accuracy.
  - `tie_word_embeddings`: false for 4.0-1b and 4.1-2b, **true** for 4.1-2b-plus and 4.1-2b-nar.
- **Fusion**: the projector output is scattered into the LM input embedding at positions where `input_ids == audio_token_index (100352)`.

## Frontend

All AR variants use `GraniteSpeechFeatureExtractor`:

- `sample_rate=16000`, mono.
- `n_mels=80`, `hop_length=160` (10 ms), `win_length=400` (25 ms), `n_fft=512` (window zero-padded to 512 before FFT). The asymmetry between `win_length=400` and `n_fft=512` is a classic mismatch trap.
- Window/center/padding/mel-norm not declared in `preprocessor_config.json`; values follow `torchaudio.transforms.MelSpectrogram` defaults: Hann periodic window, `center=True`, reflect padding, htk-style mel scale with no filter normalization. Stage 2 (oracle) must confirm these by running the reference processor end-to-end.
- After the projector's `downsample_rate=5`, the effective audio-token rate into the LM is ~10 Hz for the AR variants.

The encoder consumes `80 mels x 2-frame stack = input_dim=160`. The 2-frame stack is implicit in the encoder definition; the converter must preserve it on the path from mel to encoder input.

The `granite-speech-4.1-2b-plus` repo is missing `preprocessor_config.json`. The processor is registered as `granite_speech_plus` in HEAD transformers and presumably reuses `GraniteSpeechFeatureExtractor` defaults; the oracle must confirm.

## Commands

Reference WER run (per-variant prompt picked automatically from `--model`):

```bash
uv run --project scripts/envs/granite \
  scripts/wer/run_reference_granite_transformers.py \
    --model ibm-granite/granite-4.0-1b-speech \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/granite-4.0-1b-speech-REF.test-clean.jsonl
```

Reference tensor dumps (per-variant):

```bash
uv run --project scripts/envs/granite \
  scripts/dump_reference_granite_transformers.py encoder \
    --model ibm-granite/granite-4.0-1b-speech \
    --audio samples/jfk.wav \
    --out build/validate/granite/granite-4.0-1b-speech/jfk/encoder/ref

uv run --project scripts/envs/granite \
  scripts/dump_reference_granite_transformers.py decode \
    --model ibm-granite/granite-4.0-1b-speech \
    --audio samples/jfk.wav \
    --out build/validate/granite/granite-4.0-1b-speech/jfk/decode/ref
```

Conversion (preserves source dtypes; one HF repo per variant):

```bash
uv run --project scripts/envs/granite \
  scripts/convert-granite.py ibm-granite/granite-4.0-1b-speech \
  --repo-id ibm-granite/granite-4.0-1b-speech
```

Validation (ref-dtype, CPU strict; `--variant` selects per-variant manifest):

```bash
uv run scripts/validate.py all \
  --family granite --variant granite-4.0-1b-speech
```

Or step by step:

```bash
uv run scripts/validate.py ref     --family granite --variant granite-4.0-1b-speech
uv run scripts/validate.py cpp     --family granite --variant granite-4.0-1b-speech
uv run scripts/validate.py compare --family granite --variant granite-4.0-1b-speech
```

Benchmarks (per-variant):

```bash
uv run scripts/perf/bench.py \
  --model models/granite-4.0-1b-speech/granite-4.0-1b-speech-Q8_0.gguf \
  --backend metal \
  --machine "$(uname -srm | tr ' ' _)"
```

Full WER sweep (Stage 7):

```bash
for PRESET in BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  uv run scripts/wer/run.py \
    --model models/granite-4.0-1b-speech/granite-4.0-1b-speech-${PRESET}.gguf \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/granite-4.0-1b-speech-${PRESET}.test-clean.jsonl
  uv run scripts/wer/score.py reports/wer/granite-4.0-1b-speech-${PRESET}.test-clean.jsonl
done
```

## Capability Validation

One row per advertised capability per variant. Stage 1 drafts the rows with `Status: TODO`; Stage 4 fills in observed statuses after running each command.

Allowed statuses: `PASS`, `SKIP - not exposed by runtime`, `ACCEPTED GAP - <reason>`.

| Capability | Variant | Mode | Command / test | Expected observable | Status |
|------------|---------|------|----------------|---------------------|--------|
| Transcribe | granite-4.0-1b-speech | explicit language hint | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS (matches reference exact) |
| Transcribe | granite-4.0-1b-speech | auto / no language hint | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | PASS (same transcript; --language is a no-op for granite which has one chat prompt) |
| Translate (X->En) | granite-4.0-1b-speech | non-English source audio | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf --translate --target-language en samples/german.wav` | non-empty English transcript on non-English audio | PASS ("at the beach, the bathing suit, ...") |
| Translate (En->X) | granite-4.0-1b-speech | English source audio + target language hint | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-BF16.gguf --translate --target-language de samples/jfk.wav` | non-empty non-English transcript on English audio | PASS ("und so meine amerikanischen freunde: fragen sie nicht ...") |
| Keyword biasing | granite-4.0-1b-speech | hotword prompt | n/a | hotword preserved verbatim in transcript | SKIP - not exposed by runtime (no --prompt / hotword flag) |
| Transcribe | granite-speech-4.1-2b | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript with punctuation/casing | PASS (matches reference exact) |
| Transcribe | granite-speech-4.1-2b | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf samples/jfk.wav` | non-empty plausible transcript | PASS |
| Translate (X->En) | granite-speech-4.1-2b | non-English source audio | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf --translate --target-language en samples/german.wav` | non-empty English transcript on non-English audio | PASS |
| Translate (En->X) | granite-speech-4.1-2b | English source audio + target language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-BF16.gguf --translate --target-language de samples/jfk.wav` | non-empty non-English transcript on English audio | PASS ("und so, meine amerikaner, fragen sie nicht ...") |
| Keyword biasing | granite-speech-4.1-2b | hotword prompt | n/a | hotword preserved verbatim with position-aware decoding | SKIP - not exposed by runtime |
| Transcribe | granite-speech-4.1-2b-plus | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS (granite-4 system-role template; transcript matches reference within one comma) |
| Transcribe | granite-speech-4.1-2b-plus | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-BF16.gguf samples/jfk.wav` | non-empty plausible transcript | PASS |
| Translate | granite-speech-4.1-2b-plus | n/a | n/a | runtime rejects `--translate` (`supports_translate=false`) | SKIP - not exposed by runtime (upstream model card lists -plus as ASR-only; the base granite-speech-4.1-2b is the AST variant. The fused LLM can be prompt-coerced to translate — the smoke produced "Ich bitte meine Mitbürger, ..." — but the converter advertises `stt.capability.translate=false`, so the dispatcher rejects the task.) |
| Word timestamps | granite-speech-4.1-2b-plus | structured per-word timestamps | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-BF16.gguf --timestamps word samples/jfk.wav` | clean transcript + per-word t0/t1, parsed from the model's `[T:N]` end-of-word centisecond markers | PASS (model emits `[T:N]` markers, e.g. "_ [T:30] and [T:57] so [T:95] my [T:125] fellow [T:160] americans ..."; the runtime drops the `_` placeholders, unwraps the mod-1000 / 10 s rollover, and returns structured words. Verified against IBM's transformers reference with the model-card timestamps prompt; the earlier `[SS:N]` reading was an artifact of an incorrect prompt string.) |
| Speaker diarization | granite-speech-4.1-2b-plus | multi-speaker audio | n/a | `[Speaker N]:` tags preceding speaker turns in the text | SKIP - not exposed by runtime (user-deferred to later) |
| Incremental decoding | granite-speech-4.1-2b-plus | prefix_text continuation | n/a | transcript continues from supplied prefix | SKIP - not exposed by runtime |

## Notes

- The IBM-published `ibm-granite/granite-4.0-1b-speech-GGUF` strips the speech encoder; our port produces fused-stack GGUFs (encoder + projector + LM in one file).
- Stage 4 WER sanity on LibriSpeech test-clean (BF16 ref-dtype):

  | Variant | REF full 2620 | REF 512 | C++ 512 | Δ (C++ - REF on 512) |
  |---------|---------------|---------|---------|----------------------|
  | granite-4.0-1b-speech      | 1.42% | 1.24% | 1.26% | +0.02 pp |
  | granite-speech-4.1-2b      | 1.35% | 1.61% | 1.35% | -0.26 pp |
  | granite-speech-4.1-2b-plus | 1.48% | 1.26% | 1.29% | +0.03 pp |

  All three pass (C++ within bootstrap CI of REF on the same 512 subset).
  - **1b**: 95% CIs overlap identically with REF; the 0.02 pp gap is 5 utterances where a single token's argmax flips a character on BF16 noise (e.g. "chiaroscuroists" -> "chiaroscurosts").
  - **2b**: looks "better than REF" entirely because of one utterance (1995-1836-0004) where the HF reference enters an "and missus vanderpool" repetition loop (reproduced on both MPS and CPU PyTorch) and never recovers; C++ also loops but breaks out one token sooner on BF16 noise. Excluding that utt, C++ and REF score identically at 1.085%.
  - **-plus**: requires `add_generation_prompt=True` (assistant role marker appended) on the prompt. With False (the original Stage-2 dumper default) both REF and C++ produce 25-27 empty hypotheses on short test-clean clips, blowing up to a ~26% WER. With True the model opens the assistant turn explicitly and the empties disappear (1.29% on C++ 512). The Stage-2 dumper, the WER reference runner, and the C++ prompt builder were all updated to use True; the family-doc Capability Validation rows reflect that prompt. The reference dumps in `build/validate/granite/granite-speech-4.1-2b-plus/jfk/ref/` were regenerated so the 15/15 tensor checks remain valid.
- Upstream LibriSpeech test-clean WER targets (porting-7-wer gate, ref-dtype C++ must score <= REF + 0.01): the published 4.0-1b number is 1.42 (matches our REF full exactly); 4.1-2b is 1.33 (we measure REF full 1.35 — within bootstrap CI); 4.1-2b-plus published is 1.44 (we measure REF full 1.48 — within bootstrap CI). The `granite_nar` sibling is tracked separately.
