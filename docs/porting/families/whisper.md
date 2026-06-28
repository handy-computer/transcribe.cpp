# Whisper

Status: research

## Identity

- Family key: `whisper`
- Upstream architecture string: `whisper` (`WhisperForConditionalGeneration`)
- Hugging Face repo: `openai/whisper-tiny` (first port; per-variant intakes
  pin each variant's repo + revision under
  `reports/porting/whisper/<variant>/intake.json`)
- License: Apache-2.0 (see each HF repo)
- Variants ported (12):
  - Multilingual `tiny`, `base`, `small`, `medium`, `large`, `large-v2`
    (F32 reference dtype; 80 mel bins; 99 languages).
  - English-only `tiny.en`, `base.en`, `small.en`, `medium.en` (F32; 80
    mel bins; English-only — no language token, no `<|translate|>` task
    token in the decoder prefix).
  - `large-v3` and `large-v3-turbo` (F16 reference dtype; 128 mel bins;
    100 languages — `large-v3` adds Cantonese/`yue`). `large-v3-turbo` is
    the v3 encoder paired with a 4-layer decoder.

## References

- Canonical reference: **transformers** — `transformers.models.whisper.{WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor}`. This is the class set the openai/whisper-tiny model card documents; no `trust_remote_code` required.
- Instrumented reference: **transformers** (same class set, with forward hooks for tensor dumps in Stage 2).
- Cross-check references:
  - `refs/ggml-org/whisper.cpp` — ggml-native C++ implementation. Primary cross-check for encoder graph shape, mel frontend, and numerical behavior under ggml.
  - `refs/mlx/mlx-audio/mlx_audio/stt/models/whisper/` — MLX Python implementation (also carries the DTW word-timestamp path and decoding loop).
  - `openai/whisper` PyPI package — original OpenAI implementation; useful for mel constants and decoding strategy.

## Commands

Reference run:

```bash
uv run scripts/validate.py ref --family whisper --variant whisper-tiny
```

Reference dumps:

```bash
uv run --project scripts/envs/whisper \
  scripts/dump_reference_whisper_transformers.py decode \
  --model openai/whisper-tiny \
  --audio samples/jfk.wav \
  --out build/validate/whisper/whisper-tiny/jfk/ref \
  --torch-threads 1
```

Conversion:

```bash
uv run --project scripts/envs/whisper scripts/convert-whisper.py openai/whisper-tiny
```

Validation:

```bash
uv run scripts/validate.py all --family whisper --variant whisper-tiny
```

Manual long-form / no-speech parity check:

```bash
cmake --build build --target transcribe-cli

printf '%s\n' \
  samples/noise.wav \
  samples/jobs-silence.wav \
  samples/whole-earth.wav \
  samples/love-loss.wav \
  samples/death.wav \
  samples/dots-full.wav \
| build/bin/transcribe-cli -q \
    --backend cpu \
    --threads 1 \
    -m models/whisper-tiny/whisper-tiny-F32.gguf \
    --language en \
    --timestamps segment \
    --batch /dev/stdin \
    --batch-jsonl
```

Expected first-order behavior for the manual check:

- `samples/noise.wav` and `samples/jobs-silence.wav` produce empty text and no segments.
- The longer Jobs speech excerpts produce segment-timestamp JSONL with coverage close to the input duration.
- With `whisper-tiny-F32`, decoded text for these samples should match the pinned Transformers reference exactly when the reference is run with the same generation knobs.

A scripted long-form parity regression (Stage 3 — `condition_on_prev_tokens` and prev-context stitching) is available:

```bash
uv run --project scripts/envs/whisper \
  scripts/whisper_long_form_parity.py \
    --model openai/whisper-tiny \
    --gguf models/whisper-tiny/whisper-tiny-F32.gguf \
    --audio samples/whole-earth.wav
```

It runs HF transformers and `transcribe-cli` on the same clip with `return_timestamps=True` and `condition_on_prev_tokens=True`, then asserts WER ≤ 0.05 between the two transcripts. As of Stage 3 ship, `whole-earth.wav` (~84 s, 3 chunks) and `love-loss.wav` (~197 s, 7 chunks) both match HF at WER 0.0000 with identical segment counts, exercising the prev-context window assembly and the `skip_ending_double_timestamps` rule. Pass `--no-condition` to test the no-prev-tokens path.

For Hugging Face long-form comparisons, build processor inputs with `truncation=False`. Without that flag, `WhisperProcessor` pads/trims to the normal 30-second window and the reference only evaluates the first chunk. The reference invocation used for this manual check is:

```python
inputs = processor(
    pcm,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
    truncation=False,
)
generated = model.generate(
    **inputs,
    language="en",
    task="transcribe",
    return_timestamps=True,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    do_sample=False,
    num_beams=1,
)
text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
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

Acceptance dataset for Stage 7 WER gate: **LibriSpeech test-clean** (upstream-reported 7.54 %).

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

## Variant quirks

- **`.en` variants and bos_token_id.** The HF `whisper-*.en` checkpoints
  ship a `generation_config.json` with `bos_token_id=50257`, but the
  literal `<|endoftext|>` token in their `tokenizer.json` is at id
  **50256** (one less than the multilingual location). The converter
  trusts the literal token text, so the GGUF correctly carries
  `tokenizer.ggml.bos_token_id=50256`. Whisper decoding does not use
  `bos`; it uses `<|startoftranscript|>` and the task tokens, which are
  unaffected. Preflight Gate B `tokenizer_alignment` reports the
  intake↔reference mismatch as a `fail` for `.en` variants — accepted as
  a known HF metadata inconsistency, not a runtime defect.
- **`.en` variants and capabilities.** `whisper-*.en` are English-only;
  their tokenizer has no language tokens and no `<|translate|>` task
  token. The converter writes `stt.capability.lang_detect=False` and
  `stt.capability.translate=False` for these, derived from the
  `len(languages) == 1` check in `read_hparams`.
- **`whisper-large-v3` family mel filterbank.** `large-v3` and
  `large-v3-turbo` use 128 mel bins (not 80) and ship a
  `preprocessor_config.json` without an embedded `mel_filters` array —
  WhisperFeatureExtractor recomputes them at load time. The converter
  reproduces this via `transformers.audio_utils.mel_filter_bank` and
  bakes the result into the GGUF as `frontend.mel_filterbank`.
- **`whisper-large-v3-turbo` decoder depth.** Turbo is the v3 encoder
  paired with a 4-layer decoder (vs. 32 for full v3). The loader reads
  decoder layer count dynamically from `stt.whisper.decoder.n_layers`,
  no special-case needed.
- **F16 vs F32 reference dtype.** `large-v3` and `large-v3-turbo` ship
  `float16` safetensors; everything else (`tiny` … `large-v2`) ships
  `float32`. The converter detects the source dtype from the safetensors
  header and emits `<variant>-F32.gguf` or `<variant>-F16.gguf`
  accordingly.

## Family-specific implementation notes

These items are real C++ work for Stage 4 / 5 but do NOT flow through intake → convert → validate (the GGUF remains the canonical numerical reference):

- **C++ mel frontend (deferred).** Stage 4 bringup runs the encoder against the reference mel injected via `TRANSCRIBE_MEL_FROM_REF=<ref dir>`. `validate.py cpp` sets this env var automatically for the whisper family. The C++ mel frontend (slaney filterbank + Hann periodic window + whisper_logmel dynamic-range compression) is a follow-up. Tolerance file currently records `enc.mel.in` as zero-drift because we read the reference dump; the entry will need widening when the C++ frontend lands.
- **whisper.cpp `.bin` loader compatibility (deferred).** The plan calls for accepting upstream whisper.cpp `.bin` files alongside our GGUF. Not implemented in Stage 4; tracked for a later stage.
- **KV-cached decoder (shipped).** The decoder runs through a KV-cached path: cross K/V are precomputed once from the encoder output into the cross cache; self K/V are written per-step into the self cache. The same graph builder handles both the prompt pass (`n_tokens > 1, n_past = 0`) and per-step generation (`n_tokens = 1, n_past = current`). Self- and cross-caches are F16 by default (flip with `--kv-type f32` for tighter parity). F16 cache introduces ~8× wider per-block `max_abs` drift than F32, concentrated in a few outlier elements; transcripts and 300-sample LibriSpeech WER are unchanged within CI.
- **Beam search / temperature fallback (deferred).** Greedy argmax with `suppress_tokens` + first-step `begin_suppress_tokens` is enough to match the reference dumps' transcripts byte-for-byte under `do_sample=False, num_beams=1`. Beam search and temperature fallback (whisper.cpp's `temperature_inc` strategy) are follow-ups for the production decode path.
