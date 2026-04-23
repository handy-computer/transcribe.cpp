# Whisper

Status: research

## Identity

- Family key: `whisper`
- Upstream architecture string: `whisper` (`WhisperForConditionalGeneration`)
- Hugging Face repo: `openai/whisper-tiny`
- Hugging Face revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
- License: Apache-2.0 (see HF repo)
- Variants: `whisper-tiny` (first port). Sibling HF repos (`whisper-tiny.en`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3`, etc.) share architecture and will be added as separate variants.

## References

- Canonical reference: **transformers** — `transformers.models.whisper.{WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor}`. This is the class set the openai/whisper-tiny model card documents; no `trust_remote_code` required.
- Instrumented reference: **transformers** (same class set, with forward hooks for tensor dumps in Stage 2).
- Cross-check references:
  - `refs/ggml-org/whisper.cpp` — ggml-native C++ implementation. Primary cross-check for encoder graph shape, mel frontend, and numerical behavior under ggml.
  - `refs/mlx/mlx-audio/mlx_audio/stt/models/whisper/` — MLX Python implementation (also carries the DTW word-timestamp path and decoding loop).
  - `openai/whisper` PyPI package — original OpenAI implementation; useful for mel constants and decoding strategy.

Bridge validation (to be completed in Stage 2):

- Transcript: TODO
- Token ids: TODO
- Frontend: TODO
- Encoder: TODO
- Decoder/logits: TODO
- Known drift: TODO

## Environment

- OS: TODO
- CPU: TODO
- RAM: TODO
- GPU: TODO
- Backend/runtime: TODO
- Python: TODO
- Reference package versions: TODO (transformers, torch, soundfile, numpy)

## Artifacts

- Intake: `reports/porting/whisper/whisper-tiny/intake.json`
- Preflight Gate A: `reports/porting/whisper/whisper-tiny/preflight-gate-A.json`
- Reference run report: TODO
- Golden manifest: `tests/golden/whisper/whisper-tiny.manifest.json` (skeleton; filled in Stage 2)
- Reference dumps: TODO (`dumps/ref/whisper/whisper-tiny/...`)
- C++ dumps: TODO (`dumps/cpp/whisper/whisper-tiny/...`)
- Validation report: TODO
- Benchmark report: TODO
- Converter report: TODO

## Commands

Reference run:

```bash
uv run --project scripts/envs/whisper scripts/validate.py ref \
  --family whisper --variant whisper-tiny
```

Reference dumps:

```bash
uv run --project scripts/envs/whisper scripts/dump_reference_whisper.py \
  --variant whisper-tiny
```

Conversion:

```bash
uv run --project scripts/envs/whisper scripts/convert-whisper.py openai/whisper-tiny
```

Validation:

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny
```

Benchmarks:

```bash
uv run scripts/wer.py --family whisper --variant whisper-tiny --dataset librispeech-test-clean
```

## Architecture summary

- Pattern: `encoder-decoder` (audio encoder + autoregressive text decoder with cross-attention).
- Frontend: 80-bin log-mel spectrogram, 16 kHz, fft=400, hop=160, Hann periodic window, Slaney filterbank norm, center=True / reflect padding, `log10 → clamp(max-8) → (+4)/4` dynamic-range compression. Input pad-or-trimmed to 30 s (3000 mel frames).
- Encoder: 2× Conv1d (first stride=1, second stride=2) → sinusoidal positional embedding (max_source_positions=1500) → 4 encoder blocks (d_model=384, 6 heads, GELU, pre-norm, standard MHSA + FFN) → final LN.
- Decoder: learned positional embedding (max_target_positions=448) + token embedding, 4 decoder blocks (self-attn + cross-attn to encoder) with GELU FFN, final LN, logits head tied to token embedding (vocab 51865).
- Tokenizer: GPT-2 style BPE with bytes-to-unicode mapping; 50258 base + 1607 added tokens (99 language tokens, 2 task tokens, auxiliary controls, 1501 timestamp tokens).
- Generation contract: forced prefix `<|startoftranscript|> <|lang|> <|task|> <|no_timestamps|>`, suppress_tokens list applied at decode time, optional DTW-over-cross-attention for word timestamps.

## Capabilities (from intake)

- Languages: 99 (see intake.json)
- Language detection: yes
- Translation: yes (target = English)
- Timestamps: segment, word
- Streaming: no (not first-class; chunked 30s windows only)
- VAD / diarization: no

## Upstream benchmarks (from model card)

- LibriSpeech test-clean (en): 7.54 WER %
- LibriSpeech test-other (en): 17.15 WER %
- Common Voice 11.0 (multilingual aggregate): 141.0 WER % (as reported)

Acceptance dataset for Stage 6 WER gate: **LibriSpeech test-clean** (upstream-reported 7.54 %).

## Known risks

See `reports/porting/whisper/whisper-tiny/intake.json::known_risks`. Highlights:

1. Whisper-specific log-mel compression (not a standard stat normalization).
2. 30-second input contract (pad-or-trim; long audio requires chunked decoding).
3. Forced decoder prefix tokens are part of the contract, not optional.
4. Tokenizer base vocab (50258) ≠ model output dim (51865); added tokens include 1501 timestamp ids.
5. Suppress_tokens list must be applied at decode.

## Notes

- Whisper is the closest existing ggml reference (`refs/ggml-org/whisper.cpp`). Prefer re-using shapes and mel code conventions rather than reinventing.
- First port targets single-chunk (≤30 s) transcription, single language, no timestamps. Word timestamps, translation, and chunked long-form decoding are follow-ups.
