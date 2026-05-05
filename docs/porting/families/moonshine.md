# Moonshine

Status: research

## Identity

- Family key: `moonshine`
- Upstream architecture string: `moonshine` (`MoonshineForConditionalGeneration`)
- Hugging Face repo: `UsefulSensors/moonshine-tiny` (first port; per-variant
  intakes pin each variant's repo + revision under
  `reports/porting/moonshine/<variant>/intake.json`)
- Hugging Face revision: `390624ed33d594443aa4aa221f5b9f283b545b5a`
- License: MIT (see model card)
- Variants:
  - `moonshine-tiny` (27M params, 6 enc / 6 dec layers, hidden 288, F32 reference dtype, English-only).
  - `moonshine-base` (61M params; deferred — first port targets `tiny`).

## References

- Canonical reference: **transformers** — `transformers.models.moonshine.{MoonshineForConditionalGeneration, MoonshineConfig}` paired with `AutoProcessor` (which composes `Wav2Vec2FeatureExtractor` + `PreTrainedTokenizerFast`). This is the class set the UsefulSensors/moonshine-tiny model card documents; no `trust_remote_code` required. First-party in transformers since 4.48.
- Instrumented reference: **transformers** (same class set, with forward hooks for tensor dumps in Stage 2).
- Cross-check references:
  - `refs/mlx/mlx-audio/mlx_audio/stt/models/moonshine/` — clean MLX Python implementation. Useful for reading the conv frontend, partial-RoPE attention, and SwiGLU decoder MLP without HF abstraction layers.

## Commands

Reference run:

```bash
TODO  # uv run scripts/validate.py ref --family moonshine --variant moonshine-tiny
```

Reference dumps:

```bash
TODO  # uv run --project scripts/envs/moonshine \
      #   scripts/dump_reference_moonshine_transformers.py decode \
      #   --model UsefulSensors/moonshine-tiny \
      #   --audio samples/jfk.wav \
      #   --out build/validate/moonshine/moonshine-tiny/jfk/ref \
      #   --torch-threads 1
```

Conversion:

```bash
TODO  # uv run --project scripts/envs/moonshine scripts/convert-moonshine.py UsefulSensors/moonshine-tiny
```

Validation:

```bash
TODO  # uv run scripts/validate.py all --family moonshine --variant moonshine-tiny
```

Benchmarks:

```bash
TODO  # uv run scripts/wer.py --family moonshine --variant moonshine-tiny --dataset librispeech-test-clean
```

## Architecture summary

- Pattern: `encoder-decoder` (audio encoder + autoregressive text decoder with cross-attention).
- Frontend: **raw 16 kHz waveform** — no STFT, no mel. A learned 3-layer Conv1d stack subsamples the audio: Conv1 (kernel 127, stride 64, no bias) → tanh → GroupNorm(num_groups=1) → Conv2 (kernel 7, stride 3, GELU) → Conv3 (kernel 3, stride 2, GELU). Total temporal stride 384.
- Encoder: 6 transformer blocks, hidden 288, 8 heads (= 8 KV heads, no GQA on tiny), GELU FFN (intermediate 1152), pre-norm, partial RoPE (`partial_rotary_factor=0.9`) on self-attn, final LN.
- Decoder: token embedding (vocab 32768), 6 transformer blocks with self-attn (causal, partial RoPE) + cross-attn (no RoPE) + SwiGLU FFN (`fc1: hidden→2*intermediate`, split-gate-silu, `fc2: intermediate→hidden`), final LN, logits head tied to token embedding.
- Tokenizer: byte-level BPE, vocab 32768. Special tokens: bos=1, eos=2, pad=2, decoder_start=1. No language or task tokens.
- Generation contract: decoder seeded with `decoder_start_token_id=1`; greedy or temperature sampling until eos=2 or `max_length=194`.

## Capabilities (from intake)

- Languages: English only (`en`).
- Language detection: no.
- Translation: no.
- Timestamps: none (model card does not advertise; no timestamp tokens in the vocab).
- Streaming: no.
- VAD / diarization: no.

## Upstream benchmarks (from model card)

- LibriSpeech test-clean (en): 4.55 WER %
- LibriSpeech test-other (en): 11.68 WER %
- TED-LIUM (en): 5.69 WER %
- Common Voice mean (en): 12.65 WER %
- GigaSpeech (en): 14.21 WER %
- AMI (en): 22.84 WER %
- Earnings22 (en): 20.73 WER %
- SPGISpeech (en): 7.43 WER %
- VoxPopuli (en): 14.11 WER %

Acceptance dataset for Stage 7 WER gate: **LibriSpeech test-clean** (upstream-reported 4.55 %).

## Known risks

See `reports/porting/moonshine/moonshine-tiny/intake.json::known_risks`. Highlights:

1. Frontend is a learned conv stack on raw waveform (no mel, no STFT). New to our pipeline; Gate A's `frontend_config` cross-check sees `feature_size=1` from the Wav2Vec2FeatureExtractor — that is the conv input channel, not a mel bin count.
2. Variable-length audio input (no pad-or-trim contract). `max_position_embeddings=194` constrains the decoder, not the encoder.
3. Partial RoPE (`partial_rotary_factor=0.9`) — same approach as cohere; reuse patterns from `src/arch/cohere/`.
4. Two distinct MLP shapes in one model (encoder GELU vs decoder SwiGLU).
5. Tied word embeddings — converter must not duplicate `decoder.embed_tokens`.
6. Cross-attn KV cache is precomputed once from the encoder; self-attn cache grows per step. Distinguish in cache management.
7. `pad_head_dim_to_multiple_of=8` is set in HF config; verify whether transformers actually pads V to head_dim=40 for tiny (head_dim is 36 nominal). MLX implementation does not pad. Confirm at oracle stage.

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it. The row
  stays here so readers see the gap honestly.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/moonshine-tiny/moonshine-tiny-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS — exact JFK quote: "And so my fellow Americans ask not what your country can do for you, ask what you can do for your country." |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/moonshine-tiny/moonshine-tiny-F32.gguf samples/jfk.wav` | non-empty plausible transcript (English-only model — auto path is the same as explicit `en`) | PASS — identical output to explicit `--language en` (no language token in this model) |

## Notes

- Moonshine is English-only and emits transcript-only output (no language tokens, no task tokens, no timestamp tokens). Translate / language-detect / timestamps rows are intentionally absent from the capability table because the model does not advertise them.
- First port targets greedy argmax single-utterance transcription. Temperature sampling, beam search, and any post-hoc timestamp extraction are out of scope.
- **Stage 4 subset WER vs HF reference**: identical. C++ 3.25% vs HF reference 3.25% on the same `samples/wer/test-clean.512.manifest.jsonl` (512 utterances). All 512 hypotheses are byte-equal between the two — no per-utterance diffs. Substitution/deletion/insertion counts match exactly (269/41/43). HF reference runner is the new `wer` subcommand on `scripts/dump_reference_moonshine_transformers.py` (greedy, num_beams=1, max_new_tokens=192 — same regime as the decode dumper). Reports: `reports/wer/moonshine-tiny-F32.test-clean-512.score.json` (C++) and `reports/wer/moonshine-tiny-REF.test-clean-512.score.json` (HF).
