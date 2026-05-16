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

Reference run:

```bash
TODO
```

Reference dumps:

```bash
TODO
```

Conversion:

```bash
TODO
```

Validation:

```bash
TODO
```

Benchmarks:

```bash
TODO
```

## Capability Validation

One row per advertised capability per variant. Stage 1 drafts the rows with `Status: TODO`; Stage 4 fills in observed statuses after running each command.

Allowed statuses: `PASS`, `SKIP - not exposed by runtime`, `ACCEPTED GAP - <reason>`.

| Capability | Variant | Mode | Command / test | Expected observable | Status |
|------------|---------|------|----------------|---------------------|--------|
| Transcribe | granite-4.0-1b-speech | explicit language hint | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-bf16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | TODO |
| Transcribe | granite-4.0-1b-speech | auto / no language hint | `build/bin/transcribe-cli -m models/granite-4.0-1b-speech/granite-4.0-1b-speech-bf16.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | TODO |
| Translate (X->En) | granite-4.0-1b-speech | non-English source audio | `<actual supported command>` | non-empty English transcript on non-English audio | TODO |
| Translate (En->X) | granite-4.0-1b-speech | English source audio + target language hint | `<actual supported command>` | non-empty non-English transcript on English audio | TODO |
| Keyword biasing | granite-4.0-1b-speech | hotword prompt | `<actual supported command>` | hotword preserved verbatim in transcript | TODO |
| Transcribe | granite-speech-4.1-2b | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-bf16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript with punctuation/casing | TODO |
| Transcribe | granite-speech-4.1-2b | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b/granite-speech-4.1-2b-bf16.gguf samples/jfk.wav` | non-empty plausible transcript | TODO |
| Translate (X->En) | granite-speech-4.1-2b | non-English source audio | `<actual supported command>` | non-empty English transcript on non-English audio | TODO |
| Translate (En->X) | granite-speech-4.1-2b | English source audio + target language hint | `<actual supported command>` | non-empty non-English transcript on English audio | TODO |
| Keyword biasing | granite-speech-4.1-2b | hotword prompt | `<actual supported command>` | hotword preserved verbatim with position-aware decoding | TODO |
| Transcribe | granite-speech-4.1-2b-plus | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-bf16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | TODO |
| Transcribe | granite-speech-4.1-2b-plus | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus-bf16.gguf samples/jfk.wav` | non-empty plausible transcript | TODO |
| Word timestamps | granite-speech-4.1-2b-plus | text-protocol output | `<actual supported command>` | `[T:N]` centisecond markers interleaved with the transcript text | TODO |
| Speaker diarization | granite-speech-4.1-2b-plus | multi-speaker audio | `<actual supported command>` | `[Speaker N]:` tags preceding speaker turns in the text | TODO |
| Incremental decoding | granite-speech-4.1-2b-plus | prefix_text continuation | `<actual supported command>` | transcript continues from supplied prefix | TODO |

## Notes

- The IBM-published `ibm-granite/granite-4.0-1b-speech-GGUF` strips the speech encoder; our port will produce fused-stack GGUFs.
- Upstream LibriSpeech test-clean WER targets for the porting-7-wer gate (ref-dtype C++ must score <= target + 0.01): `1.42 / 1.33 / 1.44` for `4.0-1b / 4.1-2b / 4.1-2b-plus` respectively. The `granite_nar` sibling is tracked separately.
