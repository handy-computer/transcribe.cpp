# Whistle

Status: research

## Identity

- Family key: `whistle`
- Upstream architecture string: `WhistleForSpeechRecognition` (`model_type: whistle`, `weights_version: 2.0.0`)
- Hugging Face repo: `Cactus-Compute/whistle`
- Hugging Face revision: `b358ddadd89b7a713b5aa131f23032d3cca1b251`
- License: Apache-2.0 (`LICENSE` in the HF repo)
- Variants: `whistle` (~55M params; 16.9 MB deployed at 2/4-bit CQ; English, German, French, Spanish, Italian, Dutch, Polish)

Architecture pattern: **encoder-decoder**. 16 kHz audio → 80-bin log-mel
(n_fft 512, win 400, hop 160; Slaney-normalized filterbank shipped in the
`.cact`) → 3-stage depthwise/pointwise Conv2D stem (128 ch, 8× time
downsampling, 1280 → 512 projection, 80 ms frames) → 8 encoder blocks
(d_model 512; GQA self-attention 8q/2kv heads, qk 48 / v 64, q/k
ZCRMSNorm, gate_proj, sigmoid `attn_gate`; depthwise conv module k=9 with
pw1 512→1024 / pw2; two HadamardMLPs; 4-lane mHC hyper-connections) →
8 Needle-3 decoder blocks (self-attention with causal q/k/v taps of width 3,
RoPE θ=1e5, gated cross-attention to the encoder with sigmoid `cross_gate`,
HadamardMLP, 4-lane mHC) plus Engram hashed n-gram embeddings at decoder
sites 3 and 7 → tied 8199-way head (8192 SentencePiece text pieces + 7
`<|lang|>` tokens at 8192–8198). Hard cap of 30 s per pass and 320 decoder
tokens.

The decoder is laddered: any depth ≥ 2 is deployable upstream via
`--audio-depth`. This port targets the full 8-layer depth.

## References

- Canonical reference: `author_repo_cactus_needle_engine`, the closed-source
  prebuilt Needle engine (`Cactus-Compute/needle3@2ae11323dc000f5e70c49f7403efa6af12ba9e67`
  platform binaries / `cactus-needle` 3.2.0 wheel) running
  `whistle.cact`. It is a **black-box oracle**: it exposes the transcript,
  detected language, word times (`needle_transcribe`) and the encoder
  output per 80 ms frame (`needle_embed`), and no intermediate tensors.
  - **There is no first-party source for the audio path** (mel details,
    stem, encoder block, cross-attention, `pe_gate`, decode loop). The
    user accepted the engine-only oracle on 2026-10-09 over pausing for
    Cactus to publish code or adopting a third-party port, and with it
    explicitly waived the CLAUDE.md "do not guess" rule for those
    components. Every reconstruction is confirmed against engine outputs.
- Instrumented reference: none. Stage 2 dumps are limited to what the
  engine emits (encoder embedding, transcript, language, word times).
- Partial first-party source: `refs/cactus/needle` (`git@github.com:cactus-compute/needle.git@ef3cf754`),
  `needle/model/architecture.py` (ZCRMSNorm, tapped GQA attention,
  HadamardMLP, mHC + Sinkhorn, Engram, RoPE, depth ladder) and
  `needle/model/export.py` / `quantize.py` (`.cact` layout, CQ
  dequantization, SentencePiece reference tokenizer). Authoritative for
  the shared decoder blocks and the weight format.
- Cross-check references (non-authoritative):
  - `refs/cactus/needle` branch `origin/experiments@646ebdc` `src/model.py`:
    April 2026 speech encoder-decoder prototype (stem, encoder block,
    gated cross-attention).
  - `mrfakename/whistle-ONNX@14fc92cb`: a third-party PyTorch
    reimplementation on dequantized `.cact` weights. Not adopted as the
    reference.

### Weights

The reference-dtype GGUF comes from the deployed `whistle.cact`, not from
`checkpoints/whistle.safetensors`. The safetensors file holds F32 QAT
masters (`run.stage=qapt`, 2/4-bit weights, int8 KV and activations), and
the shipped, benchmarked model is the CQ-quantized `.cact` (user decision,
2026-10-09). CQ matmul weights (145 × 2-bit, 7 × 4-bit, group 128) are
dequantized to F32. FP16 tensors stay F16 and FP32 tensors (mel
filterbank, Hadamard permutations) stay F32. Reference dtype label: `F32`.

## Commands

The engine is loaded via ctypes from `scripts/lib/whistle_engine.py`, not
through the `cactus-needle` wrapper. That avoids telemetry and keeps writes
out of `~/.cache/cactus-needle`. On first use the pinned wheel
(`Cactus-Compute/needle3@2ae11323`, `cactus_needle-3.2.0`) is extracted
to `../refs/cactus/needle3-engine/3.2.0/wheel/`. `WHISTLE_ENGINE_LIB` and
`WHISTLE_CACT` override the library and weights paths. The converter is
created at Stage 3.

Acceptance manifest (≤30 s subset). The engine rejects longer audio, and
`samples/wer/` is gitignored, so regenerate the subset after
`scripts/wer/ingest.py librispeech`:

```bash
uv run --no-project python -c "
import json, wave
with open('samples/wer/librispeech-test-clean.manifest.jsonl') as f, \
     open('samples/wer/librispeech-test-clean-le30s.manifest.jsonl', 'w') as g:
    for l in f:
        w = wave.open(json.loads(l)['audio'])
        if w.getnframes() / w.getframerate() <= 30.0: g.write(l)
"   # 2611 of 2620 rows
```

Reference run (engine oracle):

```bash
uv run --project scripts/envs/whistle \
  scripts/wer/run_reference_whistle_engine.py \
    --manifest samples/wer/librispeech-test-clean-le30s.manifest.jsonl \
    --out reports/wer/whistle-REF.librispeech-test-clean-le30s.jsonl
uv run scripts/wer/score.py reports/wer/whistle-REF.librispeech-test-clean-le30s.jsonl
```

Reference dumps (engine-observable outputs only; all manifest cases):

```bash
uv run scripts/validate.py ref --family whistle --variant whistle
# per case: uv run --project scripts/envs/whistle scripts/dump_reference_whistle_engine.py \
#   {encoder|decode} --audio samples/jfk.wav --language en \
#   --out build/validate/whistle/whistle/jfk/ref
```

Conversion:

```bash
uv run --project scripts/envs/whistle \
  scripts/convert-whistle.py --repo-id Cactus-Compute/whistle \
    --revision b358ddadd89b7a713b5aa131f23032d3cca1b251
```

Validation:

```bash
uv run scripts/validate.py all --family whistle --variant whistle
```

Benchmarks:

```bash
uv run scripts/bench/run.py \
  --models whistle \
  --quants f32,q8_0,q4_k_m \
  --samples jfk \
  --backends cpu,metal \
  --iters 3 --warmup 1 \
  --name whistle-publication
```

## Capability Validation

One row per advertised capability. `Target` is the Stage 1 scope decision
(user-signed for every row that is not forced); `Status` is filled at
Stage 4.

- `MUST PASS`: in scope. Stage 4 must resolve the row to `PASS`.
- `OUT OF SCOPE — <reason>`: deferred. Stage 4 may resolve it to SKIP or
  ACCEPTED GAP. The reason names what would bring it back in scope.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript matching the engine oracle | MUST PASS | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | TODO |
| Language detection | auto, non-English | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/fleurs-de.wav` (also `fleurs-fr`, `fleurs-es`) | `detected-language: de` / `fr` / `es` line matching the engine's `language` | MUST PASS | TODO |
| No-speech handling | silence / steady noise | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/noise.wav` | empty transcript and no detected language, as the engine returns | MUST PASS | TODO |
| Translate | n/a | n/a | n/a | OUT OF SCOPE — not advertised upstream (transcription only) | TODO |
| Segment timestamps | n/a | n/a | n/a | OUT OF SCOPE — upstream emits word-level times only | TODO |
| Word timestamps | `--timestamps word` | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --timestamps word samples/jfk.wav` | per-word start/end within ±1 frame (80 ms) of the engine's `words` | MUST PASS | TODO |
| Keyword biasing | `--vocabulary` | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --vocabulary "<terms>" samples/product-names.wav` | biased terms survive as the engine's `keywords=` run does | OUT OF SCOPE — engine bias weights/algorithm undocumented and decode-time (not prompt-based); back in scope once characterized | TODO |
| Encoder embedding | n/a | n/a | per-80 ms encoder rows | OUT OF SCOPE — transcribe.cpp exposes no embedding API; used internally only as the Stage 2 oracle checkpoint | TODO |
| Decoder depth ladder | n/a | n/a | N-layer rung (N ≥ 2) transcript | OUT OF SCOPE — port ships the full 8-layer depth; back in scope if the runtime gains a depth knob or ladder rungs ship as variants | TODO |
| Streaming | n/a | n/a | n/a | OUT OF SCOPE — model is an offline 30 s window (`capabilities.streaming: false`); the engine's live mode is a host-side re-decode/agreement policy | TODO |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/whistle/whistle-F32.gguf --samples-dir samples/wer/librispeech-test-clean --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | TODO |

## Notes

- **Scope sign-off (2026-10-09).** The user signed off two MUST PASS rows
  whose upstream mechanism is closed:
  - Word timestamps: DTW over cross-attention, with layers and heads
    undocumented.
  - No-speech handling: mechanism undocumented.

  Stage 2 must characterize both from engine outputs. If Stage 4 can't
  reproduce them, that is a user-signed blocker, not a downgrade.
  Keyword biasing and streaming are OUT OF SCOPE.

- Acceptance dataset: LibriSpeech test-clean restricted to ≤30 s
  (`librispeech-test-clean-le30s`, 2611 of 2620 rows; user decision
  2026-10-09). The publisher reports 4.31% on the full split. That number
  is context only; gates use the measured engine Oracle baseline on the
  subset.
- **Oracle WER baseline (Stage 2).**
  - Score: **4.30%** (95% CI 4.06–4.56%) on `librispeech-test-clean-le30s`.
  - Run: 2611 utterances, 0 errors, all detected `en`.
  - Settings: forced `en` per row, engine default search, Whisper
    `EnglishTextNormalizer`.
  - Breakdown: Sub/Del/Ins 1758/206/288.
  - Forced `en` and auto-detect produce byte-identical hypotheses on all
    2611 rows, so the language prompt does not change the decode here.
  - The header must carry `"language": "en"`. `score.py` routes the
    normalizer by header language, and a placeholder value silently falls
    back to `BasicTextNormalizer` (an earlier run scored 4.76% that way).
  - Files: `reports/wer/whistle-REF.librispeech-test-clean-le30s.{jsonl,score.json}`.
  - Publisher full-split figure is 4.31% (−0.01 pp delta; within noise).
    Context only. This measured run is the Stage 4 / Stage 7 gate.
  - **Stage 7 does not pick up the subset automatically.** Its LibriSpeech
    branch hard-codes `librispeech-test-clean.manifest.jsonl`. Override
    `$MANIFEST` with `samples/wer/librispeech-test-clean-le30s.manifest.jsonl`
    for this family.
- **Oracle observables (Stage 2).**
  - `needle_embed` returns `ceil(seconds / 0.08)` rows × 512, e.g. 138 for
    the 11.0 s jfk. It is dumped as `enc.final`. Whether it is taken
    before or after `encoder/final_norm` is undocumented.
  - `needle_transcribe` returns text, language, word times and the token
    count. There are no token IDs, logits or mel.
  - Output is deterministic across repeat calls.
  - Audio over 30 s errors with `audio limit is 30 s`.
  - The C API, the dylib exports and the CLI offer no beam-width or greedy
    knob. The only decode knobs are `--depth` (ladder) and `--threads`.
  - `noise.wav` gives empty text and empty language.
  - `jobs-silence.wav` (5.3 s) gives "Thank you." with language `en`.
- **Beam vs greedy.** The engine defaults to 5-beam search, and
  transcribe.cpp decodes greedily. Stage 2 found no way to run the engine
  greedy, so the REF baseline is beam search. Stage 4 either implements the
  engine's search or records the greedy-vs-beam WER delta as an accepted
  gap. Exact `transcript_compare` will likely fail for greedy decoding;
  Stage 4 chooses the compare mode.
- **Numerics.** The engine runs int8 KV and 8-bit activations (QAT
  targets), so the F32 ggml port cannot be bit-exact against the oracle.
  Parity is judged on the encoder embedding, transcripts and WER. The
  Stage 2 `enc.final` tolerance is a provisional magnitude budget that the
  port will not meet; Stage 4 replaces it with a measured, justified one.
- **Frontend unknowns.** Window type, log flavor/clamp, normalization,
  STFT centering/padding and inference dither are engine-internal. They
  are fixed at Stage 4 bring-up by matching `needle_embed` (`enc.final`). The filterbank itself
  is copied verbatim from the `.cact`.
- **Novel ops.**
  - HadamardMLP: Kronecker 16×16 ⊗ 32×32 Walsh-Hadamard with fixed
    permutations and a rank-8 correction.
  - mHC: 4 lanes, Sinkhorn-normalized lane mixing.
  - Engram: host-side n-gram hashing plus a dilated 4-tap conv.
  - Causal q/k/v taps: the KV state also carries 2 previous projections.
- Long-form audio over 30 s needs a chunking policy. The engine's own
  long-audio path is its streaming API.
- **Acceptance set exceeds the 30 s cap.** 9 of 2620 LibriSpeech
  test-clean utterances run past 30 s (max 34.95 s). The engine baseline
  and the C++ port must handle them the same way, or the WER gate
  compares different inputs.
- **Reference-run hygiene (Stage 2).**
  - Set `NEEDLE_TELEMETRY=0`, because the `cactus-needle` wrapper sends
    usage pings and re-fetches `config.json` on every load.
  - Point `NEEDLE_WHISTLE_WEIGHTS` at the HF-cache `whistle.cact` and
    `NEEDLE3_LIB_PATH` at an engine under `refs/`, so nothing is written
    to `~/.cache/cactus-needle`.
