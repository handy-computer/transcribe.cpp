# Canary

Status: shipped (Stage 7 complete; bench numbers pending)

## Identity

- Family key: `canary`
- Upstream architecture string: `canary` (NeMo `EncDecMultiTaskModel`; encoder is `FastConformerEncoder`, decoder is `TransformerDecoder`)
- Hugging Face source repos:
  - `nvidia/canary-1b-v2`
  - `nvidia/canary-1b-flash`
  - `nvidia/canary-180m-flash`
  - `nvidia/canary-1b` (original; **CC-BY-NC-4.0**, distinct from rest of family)
- License:
  - canary-1b-v2 / canary-1b-flash / canary-180m-flash: **CC-BY-4.0** (commercial use OK with attribution)
  - canary-1b: **CC-BY-NC-4.0** (non-commercial only) — converter must surface this in `general.license`.
- Variants:
  - `canary-1b-v2` — 978M params, 32-layer FastConformer encoder + 8-layer Transformer decoder, 25 European languages, word + segment timestamps, F32, .nemo only.
  - `canary-1b-flash` — 883M params, 32 + 4, en/de/es/fr, word + segment timestamps (experimental), F32, .nemo + an encoder-only HF Transformers shim (`config.json`, `preprocessor_config.json`, `model.safetensors`).
  - `canary-180m-flash` — 182M params, 17 + 4, en/de/es/fr, word + segment timestamps (experimental), F32, .nemo only.
  - `canary-1b` — 1B params, 24 + 24, en/de/es/fr, no timestamps, F32, .nemo only, **CC-BY-NC-4.0**.

## References

- Canonical reference: **NeMo** (`nvidia/NeMo`) via `nemo.collections.asr.models.EncDecMultiTaskModel.from_pretrained(<hf_repo>)`. NeMo is NVIDIA's first-party framework and the only way to run the full multitask AED — every variant's HF repo ships a `.nemo` tar archive consumed by this class.
- Instrumented reference: same — NeMo with forward hooks added on the FastConformer encoder, the cross-attention layer, the Transformer decoder, and the LM head for tensor dumps in Stage 2.
- Cross-check references:
  - `nvidia/canary-1b-flash` HF Transformers `FastConformerModel` shim — exposes the encoder only (`nemo_decoder_type='none'`); useful for the mel-spectrogram contract and encoder shape but cannot generate transcripts.
  - Parakeet port (`scripts/convert-parakeet.py`, `src/arch/parakeet/`) — the FastConformer encoder, relative-position attention with rel_pos shift, and the depth-wise striding subsampling are the same shapes Parakeet already implements.

## Environment

Python env: `scripts/envs/canary/pyproject.toml` *(to be created at Stage 2)*  — expected pins: `nemo_toolkit[asr]`, `torch`, `soundfile`, `numpy`, `sentencepiece`, plus `gguf` for the converter. NeMo version pin is variant-sensitive: canary-1b's model card explicitly references `r1.23.0`, while the v2 / flash variants need a newer NeMo (≥2.x or main pre-2.5).

```bash
# Reference run (per variant, once the env exists)
uv run --project scripts/envs/canary \
  scripts/dump_reference_canary_nemo.py decode \
    --model nvidia/<variant> \
    --audio samples/jfk.wav \
    --source-lang en --target-lang en \
    --task asr \
    --out build/validate/canary/<variant>/jfk/decode/ref
```

## Golden Manifests

- `tests/golden/canary/canary-1b-v2.manifest.json`
- `tests/golden/canary/canary-1b-flash.manifest.json`
- `tests/golden/canary/canary-180m-flash.manifest.json`
- `tests/golden/canary/canary-1b.manifest.json`

All four are Stage 1 skeletons (identity-only, `_skeleton: true`); Stage 2 (`porting-2-oracle`) fills in the reference entrypoint, frontend, tokenizer summary, capabilities, and per-family case set.

## Artifacts

| Artifact | Path |
| --- | --- |
| Intakes | `reports/porting/canary/<variant>/intake.json` |
| Preflight Gate A | `reports/porting/canary/<variant>/preflight-gate-A.json` |
| Forward map | `reports/porting/canary/forward-map.md` |
| Tolerances | `tests/tolerances/canary.json` |
| Converter reports | `reports/convert/<variant>-F32.json` |
| Reference dump root | `build/validate/canary/<variant>/` |
| WER reports | `reports/wer/<variant>-<preset>.librispeech-test-clean.{jsonl,score.json}` |
| WER summaries | `reports/wer/<variant>.librispeech-test-clean.summary.md` |
| User-facing model cards | `docs/models/<variant>.md` |
| HF README specs | `scripts/hf_cards/<variant>.yaml` |
| HF README rendered | `models/<variant>/README.md` |

## Commands

Reference dump (per variant — encoder + decode):

```bash
uv run --project scripts/envs/canary \
  scripts/dump_reference_canary_nemo.py encoder \
    --model nvidia/<variant> \
    --audio samples/jfk.wav \
    --out build/validate/canary/<variant>/jfk/encoder/ref

uv run --project scripts/envs/canary \
  scripts/dump_reference_canary_nemo.py decode \
    --model nvidia/<variant> \
    --audio samples/jfk.wav \
    --out build/validate/canary/<variant>/jfk/decode/ref
```

Conversion:

```bash
uv run --project scripts/envs/canary \
  scripts/convert-canary.py nvidia/<variant> --repo-id nvidia/<variant>
```

Quantization (produces F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M):

```bash
uv run scripts/quantize-all.py models/<variant>/<variant>-F32.gguf
```

Validation:

```bash
uv run scripts/validate.py all --family canary --variant <variant>
```

WER:

```bash
uv run scripts/wer/run.py \
  --model models/<variant>/<variant>-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/<variant>-F32.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/<variant>-F32.librispeech-test-clean.jsonl
```

Benchmarks (deferred until after the public HF release — easier to
download a single GGUF on each target machine than to copy 25 GB of
mixed quants):

```bash
uv run scripts/bench/run.py \
  --models <variant> \
  --quants q8_0,q4_k_m \
  --samples jfk \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name <variant>-publication
```

## Architecture summary

- Pattern: `encoder-decoder` (multitask AED — Attention-based Encoder-Decoder; both ASR and AST in a single model, switched by a task-token prompt).
- Frontend: NeMo `AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures` — 128-bin slaney-normalized mel filterbank on a 16 kHz mono waveform, hann_periodic window, 25 ms (400-sample) win_length, 10 ms (160-sample) hop, 512-point FFT, preemph=0.97 applied **before** windowing, per-feature normalization (mean/std across time, per mel band). Same family as Parakeet.
- Encoder: `FastConformerEncoder` with depth-wise striding subsampling (factor 8, 256-channel subsampling conv), relative-position attention (`rel_pos`), conv_kernel_size=9, SiLU activation, pre-norm. Layer count varies (17 / 24 / 32 across variants).
- Decoder: `TransformerDecoder` (cross-attends to encoder output). Layer count varies (4 / 8 / 24).
- Generation contract: a multitask prompt is fed into the decoder *before BOS*. Slot order is positional, not lookup-by-name:
  - canary-1b: `[<source_lang>, <target_lang>, <taskname>, <pnc>]` (4 slots, `taskname='asr'|'s2t_translation'`).
  - canary-1b-v2 / 1b-flash / 180m-flash: `[<source_lang>, <target_lang>, <task>, <pnc>, <toggle_timestamps>]` (5 slots, `task='asr'|'ast'`).
- Output head: LM head over the concatenated SP vocabulary. Decoding is beam search by default for the original canary-1b (beam=5, length_penalty=1.0) and greedy by default for the flash variants (beam=1).
- Tokenizer: concatenated SentencePiece — one SP model per language concatenated into a single vocabulary. canary-1b-v2 is 16,384 pieces; flash/180m-flash/1b vocab sizes are not stated on model cards (Stage 2 fills from .nemo).
- Audio length contract: native ≤40 s direct inference. <1 s is symmetrically zero-padded to 1 s. >40 s is handled by an external chunked inference script with 1 s overlap (canary-1b-v2 chunk len defaults to 40 s; canary-1b-flash 10 s; canary-180m-flash 10 s). **Long-form / streaming is out of scope for the v1 port.**

## Capabilities (from intake)

| Variant | Languages | Translation | Timestamps | Notes |
|---|---|---|---|---|
| canary-1b-v2 | 25 (BCP-47: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk) | yes (EN↔X for 24 langs, both directions) | word + segment | timestamps come from a side `_timestamps_asr_model` CTC aligner shipped inside the .nemo archive; AED itself is text-only |
| canary-1b-flash | 4 (en, de, es, fr) | yes (EN↔de, EN↔es, EN↔fr) | word + segment (experimental) | greedy beam=1 default; F1=95.5% on LS test-clean at 200 ms collar |
| canary-180m-flash | 4 (en, de, es, fr) | yes (EN↔de, EN↔es, EN↔fr) | word + segment (experimental) | greedy beam=1 default; F1=93.5% on LS test-clean |
| canary-1b | 4 (en, de, es, fr) | yes (EN↔de, EN↔es, EN↔fr) | none | beam=5 default; older 4-slot prompt without `<toggle_timestamps>` |

None of the variants advertise language detection (auto-detect), streaming, VAD, or speaker diarization.

## Upstream benchmarks (from model cards)

Selected highlights — per-variant intake.json carries the full table.

| Dataset | canary-1b-v2 | canary-1b-flash | canary-180m-flash | canary-1b |
|---|---|---|---|---|
| LibriSpeech test-clean (WER %) | 2.18 | 1.48 | 1.87 | 1.48 |
| LibriSpeech test-other (WER %) | 3.56 | 2.87 | 3.83 | 2.93 |
| Open ASR Leaderboard mean (WER %) | 7.15 | 6.35 | — | 6.5 (HF ASR Leaderboard, older) |
| FLEURS En→De (BLEU) | 29.4 (En→X mean over 24 langs) | 32.27 | 28.18 | 32.15 |
| FLEURS De→En (BLEU) | 29.08 (X→En mean over 24 langs) | 35.50 | 32.08 | 33.98 |

**Acceptance dataset for Stage 7 WER gate: LibriSpeech test-clean** (English; default per the porting skill — every variant supports English and reports a published score on this dataset). Stage 7 will reproduce the upstream-reported WER within ±0.01 absolute at the reference dtype as the ship gate.

## Known risks

See per-variant `reports/porting/canary/<variant>/intake.json::known_risks`. Family-wide highlights:

1. **Multitask prompt format is positional and variant-sensitive.** The original canary-1b uses 4 slots (`source_lang, target_lang, taskname, pnc`); canary-1b-v2 / 1b-flash / 180m-flash use 5 slots (`source_lang, target_lang, task, pnc, toggle_timestamps`) with `task` not `taskname`. Mismatched prompts silently swap task semantics with no shape error.
2. **Concatenated SentencePiece tokenizers.** One SP model per language concatenated into the unified vocab; piece IDs depend on concatenation order. Converter must preserve order.
3. **FastConformer encoder with rel_pos attention + factor-8 depth-wise striding subsampling.** Same shape as Parakeet — the existing parakeet rel-shift code path is reusable.
4. **NeMo FilterbankFeatures applies preemph=0.97 BEFORE windowing/STFT.** Standard NeMo ASR frontend trap. Per-feature normalization (per mel band, across time) — not per-utterance global stats.
5. **Cross-attention from transformer decoder to encoder output.** Padding mask on the cross-attention keys must propagate from encoder input lengths after the factor-8 subsampling.
6. **Decoding default differs by variant.** canary-1b: beam=5, length_penalty=1.0. flash variants: beam=1 greedy. Mismatch with upstream's measured-config beam will move WER.
7. **Audio length contract.** Native ≤40 s; <1 s is zero-padded to 1 s; long-form needs an external chunker. Streaming is not native — out of scope for v1.
8. **Timestamps are not native to the AED.** canary-1b-v2 / 1b-flash / 180m-flash advertise word + segment timestamps but produce them via a side CTC aligner inside the .nemo archive, not from the AED itself. Out of scope for v1.
9. **License divergence.** canary-1b is CC-BY-NC-4.0; the rest are CC-BY-4.0. Converter must surface license correctly in `general.license` so non-commercial restrictions on canary-1b ride along with the GGUF.
10. **.nemo archive distribution.** No HF `config.json` / `tokenizer_config.json` / `generation_config.json` (except canary-1b-flash's encoder-only HF Transformers shim). Converter must mirror the parakeet pattern of unpacking the .nemo before extraction.

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

The table covers all four variants as a single matrix — the same row maps
to each variant's `models/<variant>/<variant>-F32.gguf` artifact. Variant-
specific differences (timestamp support, language coverage, license) are
flagged in the row notes.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS — verified on all four variants 2026-05-07 (matches reference text under normalized compare) |
| Transcribe | explicit language hint (de) | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language de samples/german.wav` | non-empty plausible German transcript | PASS — verified on canary-180m-flash, canary-1b-flash, canary-1b-v2 against `samples/german.wav` 2026-05-07 (canary-1b excluded — does not advertise de) |
| Transcribe | explicit language hint (es, fr) | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language <es\|fr> samples/<es\|fr>.wav` | non-empty plausible transcript in the requested language | ACCEPTED GAP — no es/fr samples shipped (same path as de) |
| Transcribe | wide multilingual (canary-1b-v2 only) | `build/bin/transcribe-cli -m models/canary-1b-v2/canary-1b-v2-F32.gguf --language <ru\|uk\|pl\|...> samples/<lang>.wav` | non-empty plausible transcript in the requested language | ACCEPTED GAP — 25-lang sample set not shipped; runtime accepts each language id (verified by v2 loading 25 language tokens from `general.languages`), unblocked once samples land |
| Transcribe | auto / no language hint | n/a — Canary requires explicit source_lang in the multitask prompt | n/a | SKIP — not advertised (no language-detection capability on any Canary variant) |
| Translate | EN→de | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language en --translate --target-language de samples/jfk.wav` | non-empty plausible German translation of JFK | PASS — verified on all four variants 2026-05-07 (`--target-language` wired through `params->target_language`; canary-1 uses explicit `<\|translate\|>` task token, canary2 infers from src!=tgt) |
| Translate | EN↔es / EN↔fr | same as above with `--target-language es\|fr` | non-empty plausible translation in the requested language | ACCEPTED GAP — no es/fr reference samples shipped; runtime path identical to the de case |
| Translate | X→EN for 24 langs (canary-1b-v2 only) | `build/bin/transcribe-cli -m models/canary-1b-v2/canary-1b-v2-F32.gguf --language <X> --translate --target-language en samples/<X>.wav` | non-empty plausible English translation | ACCEPTED GAP — multilingual sample set not shipped; runtime path identical to the EN→de case |
| Segment timestamps | only canary-1b-v2 / 1b-flash / 180m-flash | n/a — not exposed by runtime in v1 | n/a | SKIP — not exposed by runtime (timestamps decoder path / `_timestamps_asr_model` aligner not ported in v1; intake known_risks notes timestamps are explicitly experimental upstream) |
| Word timestamps | only canary-1b-v2 / 1b-flash / 180m-flash | same | n/a | SKIP — not exposed by runtime (same reason) |
| Streaming | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Voice activity detection | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Speaker diarization | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Punctuation & capitalization toggle | `<pnc>` slot in multitask prompt | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language en --no-pnc samples/jfk.wav` | lowercase de-punctuated transcript when `--no-pnc` is set; `--pnc` (default) keeps capitalization and punctuation | PASS — verified on canary-180m-flash, canary-1b-flash, canary-1b 2026-05-07 (`transcribe_canary_params::pnc`). canary-1b-v2 ignores the nopnc prompt slot upstream — verified against NeMo `model.transcribe(..., pnc='no')` returning identical PNC-on output, so this is a model-side limitation rather than a runtime bug |

## Notes

- This is the first encoder-decoder port in the repo. Existing ports — Parakeet (encoder-transducer), SenseVoice (encoder-CTC), Whisper (encoder-decoder), Cohere/Qwen3-ASR (audio-LLM) — share infrastructure pieces but Canary's *multitask prompt protocol* with positional task tokens is unique to this family. Whisper's `<|startoftranscript|>` style is the closest analogue but Whisper uses runtime token detection, not a fixed prompt slot order.
- Canary's FastConformer encoder shares the rel_pos relative-shift implementation that Parakeet already needs. The C++ encoder code can probably be re-used with parameter-only changes (n_layers, ffn_dim, etc.).
- Use `samples/jfk.wav` for the English smoke test. German/Spanish/French sample clips and the wider 25-language multilingual set need to be sourced for canary-1b-v2 — pull from FLEURS or Common Voice during Stage 2 dumper setup.
- canary-1b is CC-BY-NC-4.0. Do not bake "Apache-2.0" or default license metadata into its converter output; surface the non-commercial restriction in `general.license`.
- canary-1b-v2 ships timestamps via a separate CTC aligner inside the .nemo archive (`_timestamps_asr_model`). The model card says these can be deleted to save memory; Stage 2 should decide whether to keep them in the converted GGUF or strip (default: strip — timestamps out of scope for v1).
- **Stage 3 tensor-name decisions** (`scripts/convert-canary.py`):
  - Encoder block names mirror `src/arch/parakeet/weights.cpp` verbatim (FastConformer module class is shared); the converter is a pass-through with no transposes.
  - Decoder layer names use `dec.layer.{i}.{norm1, self_attn.{q,k,v,o}, norm2, cross_attn.{q,k,v,o}, norm3, ffn.{up,down}}` rather than NeMo's `first_sub_layer`/`second_sub_layer`/`third_sub_layer`. The latter is fine for source comprehension but unfriendly to a reader of the future Stage 4 `src/arch/canary/` code.
  - `encoder_decoder_proj` is emitted only when present in state_dict (180m-flash has it: 512→1024). canary-1b / canary-1b-flash / canary-1b-v2 all have matching encoder/decoder dim and the projection is absent; the C++ loader reads `stt.canary.decoder.encoder_decoder_proj` (bool) to know whether to expect the tensor.
  - `transf_encoder.num_layers` is 0 for every shipping variant — the converter asserts and rejects nonzero rather than silently emitting an incomplete GGUF.
  - Two tokenizer flavors live under one converter: `CanaryTokenizer` (aggregate of spl_tokens + per-language SP, used by canary-1b/1b-flash/180m-flash) and `CanaryBPETokenizer` (single 16K SP with all specials baked in, used by canary-1b-v2). The flat `tokenizer.ggml.tokens` array is uniform; routing arrays under `stt.canary.tokenizer.{lang_codes,lang_offsets,lang_sizes}` are `["all"]/[0]/[vocab_size]` for the BPE flavor.
  - SP `score` is emitted as 0.0 for the aggregate flavor (NeMo's offset bookkeeping is fragile) and from `SP.GetScore` for the BPE flavor. The AED runtime never consumes scores — only a future text-encoding path would, and that can ship per-language SP protos as separate KV blobs if needed.
  - Special-token IDs: canary-1b runtime is `bos=3 eos=2 pad=1`; canary-1b-flash / canary-180m-flash / canary-1b-v2 runtime is `bos=4 eos=3 pad=2`. The canary-1b-flash intake originally captured the sub-SP defaults (1/2/0) — corrected at Stage 3 to match the multitask runtime IDs (4/3/2).
- **Stage 4 frontend parity**: canary uses the production `transcribe::MelFrontend` directly — there is no env-var ref-mel injection path on this family. Frontend parity is exercised inside the standard `validate.py compare` flow via the `enc.mel.in` tensor. Observed drift on `samples/jfk.wav` (180m-flash, F32, CPU, 2026-05-07): `max_abs=5.19, mean_abs=1.6e-3`. Identical mechanism as parakeet — fp64 STFT in C++ vs fp32 STFT in NeMo, plus a single-frame pad-zeroing difference at the trailing mel frame (`t=T_mel-1`, where NeMo masks beyond the valid signal length and the C++ side computes the partial frame). Both effects are documented in the family tolerance file's `_comment` block; the absolute drift is below the parakeet-family tolerance (`max_abs=10.0`) and the downstream encoder dampens the edge perturbation by the final per-block LayerNorm.
- **Stage 4 missing-bias bug (FIXED)**: the first WER pass on canary-180m-flash returned 19.96% on test-clean.512 (vs upstream 1.87%) with a long tail of "what what what…" loop hallucinations. Root cause was NOT numerical drift: the `CanaryBlock` weight catalog and the `to_view(BlockView)` projection were copied from parakeet (bias-free FFN/attn linears) without re-reading the GGUF tensor list. NeMo's `ConformerEncoder` always emits biases on every Linear (FF1/FF2 linear1/linear2, attention Q/K/V/out — the converter exports them all under `enc.blocks.*.ff*.linear*.bias` and `enc.blocks.*.attn.linear_*.bias`), but the loader silently dropped them and the `use_bias=False` KV the converter wrote happened to make the gap look "intentional." Across 17 layers the missing biases compounded into mean drift that reached `1.5e1` at intermediate blocks (vs `1.4e-1` post-fix) and the decoder cascade hit `15.8` mean drift at the final FFN (vs `1.2` post-fix), big enough to flip the top-1 logit on harder utterances. After loading the biases, drift falls to fp32-level on every gate tensor and the LibriSpeech transcripts return to upstream-quality output. The lesson: do not propagate parakeet's bias-free assumption to a new conformer family without verifying tensor-by-tensor against the GGUF — `stt.canary.encoder.use_bias` was a misread of an unrelated config field; truth lives in the tensor catalog.
- **Stage 4 mid-generation tensor (DEFERRED)**: per Stage 4 spec, autoregressive families should dump a `dec.logits_raw.gen<N>` (N >= 8) tensor on both the C++ and reference sides to gate step-graph correctness past the prompt pass. Neither side does this yet for canary; correctness past the first generated token is currently inferred from matching transcripts. Add to both `dump_reference_canary_nemo.py` (capture decoder logits at step 8) and the C++ `model.cpp` step loop in a follow-up.

## Release artifacts

| Variant | License | HF target | User card | HF spec |
|---|---|---|---|---|
| canary-180m-flash | CC-BY-4.0 | `handy-computer/canary-180m-flash-gguf` | [docs/models/canary-180m-flash.md](../../models/canary-180m-flash.md) | [scripts/hf_cards/canary-180m-flash.yaml](../../../scripts/hf_cards/canary-180m-flash.yaml) |
| canary-1b-flash | CC-BY-4.0 | `handy-computer/canary-1b-flash-gguf` | [docs/models/canary-1b-flash.md](../../models/canary-1b-flash.md) | [scripts/hf_cards/canary-1b-flash.yaml](../../../scripts/hf_cards/canary-1b-flash.yaml) |
| canary-1b-v2 | CC-BY-4.0 | `handy-computer/canary-1b-v2-gguf` | [docs/models/canary-1b-v2.md](../../models/canary-1b-v2.md) | [scripts/hf_cards/canary-1b-v2.yaml](../../../scripts/hf_cards/canary-1b-v2.yaml) |
| canary-1b | **CC-BY-NC-4.0** | `handy-computer/canary-1b-gguf` | [docs/models/canary-1b.md](../../models/canary-1b.md) *(pending — WER sweep in flight)* | [scripts/hf_cards/canary-1b.yaml](../../../scripts/hf_cards/canary-1b.yaml) *(pending)* |

Note that the converter writes the per-variant license string into
`general.license` and a permalink into `general.license.link`. canary-1b
is the only family member under a non-commercial license; downstream
tooling that reads the GGUF KV can rely on this distinction.

## Release WER (LibriSpeech test-clean, full 2620 utt)

F32 ref-dtype gate enforced (|Δ| ≤ 1pp); quants reported, not gated.
Per-variant tables under `reports/wer/<variant>.librispeech-test-clean.summary.md`.

| Variant | F32 | F16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Upstream | Δ vs upstream |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canary-180m-flash | 1.94% | 1.94% | 1.93% | 1.93% | 1.90% | 1.93% | 1.87% | +0.07pp |
| canary-1b-flash   | 1.62% | 1.62% | 1.62% | 1.65% | 1.64% | 1.59% | 1.48% | +0.14pp |
| canary-1b-v2      | 1.92% | 1.92% | 1.91% | 1.94% | 1.93% | 1.91% | 2.18% | **−0.26pp** |
| canary-1b         | 1.55% | 1.55% | 1.55% | 1.57% | TBD   | TBD   | 1.48% | +0.07pp |

Same-wav NeMo reference comparison (where measured): canary-180m-flash
F32 = 1.94% port vs 1.93% NeMo on the identical 2620 wavs — one
substitution out of ~27k reference words.

## Release status

- Stage 1–5 complete for all four variants.
- Stage 6 (bench): deferred. Will run after the public HF release —
  one GGUF download per machine is cheaper than copying the matrix
  around.
- Stage 7 (WER): complete for canary-180m-flash, canary-1b-flash,
  canary-1b-v2; canary-1b in flight (Q5_K_M + Q4_K_M still running).
- Stage 8: HF YAMLs + READMEs + user-facing cards drafted for the three
  Stage-7-complete variants. canary-1b assets follow once the WER
  sweep finishes. Upload is a manual `hf upload` step per variant.
