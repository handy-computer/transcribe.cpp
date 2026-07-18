# GigaAM

Status: ported. Stage 4 numerical validation green at ref dtype (F32),
Stage 5 quant matrix shipped, Stage 7 FLEURS ru WER measured. All four
ASR variants (`gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt`, `gigaam-v3-ctc`,
`gigaam-v3-e2e-ctc`) ready for HF upload. `v3_ssl` (encoder-only HuBERT
pretraining checkpoint) intentionally out of scope.

## Identity

- Family key: `gigaam`
- Upstream architecture string: `gigaam` (HF `model_type`); paper name "GigaAM" (Sber/ai-sage, InterSpeech 2025, arXiv:2506.01192)
- Hugging Face repo: `ai-sage/GigaAM-v3` (five sibling branches upstream; four ported)
- License: MIT
- Variants in scope (one HF branch each):

  | Variant | HF branch | Head | num_classes | Vocab style |
  |---|---|---|---|---|
  | `gigaam-v3-e2e-rnnt` | `main` (`v3_e2e_rnnt`) | RNN-T | 1025 | SentencePiece, 1024 pieces, punctuation + Cyrillic casing in vocab |
  | `gigaam-v3-e2e-ctc` | `e2e_ctc` | CTC | 257 | SentencePiece, 256 pieces, punctuation + Cyrillic casing in vocab |
  | `gigaam-v3-rnnt` | `rnnt` | RNN-T | 34 | charwise (inline in config.json), 33 entries: space + а-я (no ё), lowercased no-punct |
  | `gigaam-v3-ctc` | `ctc` | CTC | 34 | charwise (inline in config.json), same 33-entry list as `gigaam-v3-rnnt` |

  Encoder dims are identical across the four branches (16-layer
  Conformer, d=768, 16 heads, rotary PE, conv1d subsampling factor 4).
  Stage 2 dumps showed encoder weights diverge across variants after
  per-head fine-tuning (different `enc.out` magnitudes per variant), so
  each variant ships its own GGUF rather than a shared encoder.

  Upstream also publishes `v3_ssl`: an encoder-only HuBERT-CTC
  pretraining checkpoint with no head, tokenizer, or decoding path.
  transcribe.cpp is an inference-only ASR runtime with no
  encoder-output emission CLI, so `v3_ssl` has no user-observable
  capability here and is intentionally **out of scope** for this port.

Canonical bench samples: `ru` (model is Russian-only; English `jfk`/`dots` would decode to garbage).

## References

- Canonical reference: **author_repo_gigaam**. The [`gigaam`](https://github.com/salute-developers/GigaAM) PyPI package (MIT). Loader entrypoint: `gigaam.load_model("v3_e2e_rnnt")` (or `"v3_rnnt"`, `"v3_ctc"`, `"v3_e2e_ctc"`) returns a `GigaAMASR` with `.transcribe(wav_file)` and `.transcribe_longform(wav_file)`.
- Instrumented reference: **author_repo_gigaam** (same package). Use `torch` forward hooks on `ConformerEncoder`, `RNNTHead`, `CTCHead`, and `FeatureExtractor` to capture per-stage intermediates without modifying the upstream classes. The HF mirror at `ai-sage/GigaAM-v3` ships the same class definitions inside `modeling_gigaam.py` and can be loaded via `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`: same code path, second source of truth.
- Cross-check references: HuggingFace `transformers` AutoModel path (against the modeling_gigaam.py shim). Note: NeMo is NOT a reference for this family; GigaAM uses its own Conformer + RNN-T implementations wired through omegaconf/Hydra.

## Commands

Per-family Python env: `scripts/envs/gigaam/`. Replace `<variant>` with
one of `gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt`, `gigaam-v3-ctc`,
`gigaam-v3-e2e-ctc`, and `<variant-key>` with the matching
`v3_e2e_rnnt` / `v3_rnnt` / `v3_ctc` / `v3_e2e_ctc`.

Reference run (sanity-check upstream weights on a Russian sample):

```bash
uv run --project scripts/envs/gigaam python -c \
  "import gigaam; m = gigaam.load_model('v3_e2e_rnnt', fp16_encoder=False, device='cpu'); print(m.transcribe('samples/ru.wav'))"
```

Reference dumps (per-stage tensor sidecars + transcript.json, used by
`scripts/validate.py`):

```bash
uv run --project scripts/envs/gigaam scripts/dump_reference_gigaam_author.py decode \
  --variant <variant-key> \
  --audio samples/ru.wav \
  --out build/validate/gigaam/<variant>/ru/ref
```

Conversion (writes the reference-dtype F32 GGUF):

```bash
uv run --project scripts/envs/gigaam scripts/convert-gigaam.py ai-sage/GigaAM-v3 \
  --repo-id <variant> --variant-key <variant-key>
```

Validation (tensor-by-tensor + transcript exact compare):

```bash
uv run scripts/validate.py all --family gigaam --variant <variant>
```

Quantize (Stage 5 derived presets F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M):

```bash
uv run scripts/quantize-all.py models/<variant>/<variant>-F32.gguf
```

WER (FLEURS ru, acceptance dataset for the family; see Notes below):

```bash
# 1) one-time corpus ingest (skip if samples/wer/fleurs-ru.manifest.jsonl exists)
uv run scripts/wer/ingest.py fleurs --lang ru

# 2) measure upstream baseline on FLEURS ru (gigaam author package)
uv run --project scripts/envs/gigaam scripts/wer/run_reference_gigaam_author.py \
  --variant <variant-key> \
  --manifest samples/wer/fleurs-ru.manifest.jsonl \
  --out reports/wer/<variant>-REF.fleurs-ru.jsonl

# 3) measure C++ port on the same manifest, ref dtype and every quant
for PRESET in F32 F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  uv run scripts/wer/run.py \
    --model models/<variant>/<variant>-${PRESET}.gguf \
    --manifest samples/wer/fleurs-ru.manifest.jsonl \
    --language ru \
    --out reports/wer/<variant>-${PRESET}.fleurs-ru.jsonl
  uv run scripts/wer/score.py reports/wer/<variant>-${PRESET}.fleurs-ru.jsonl --language ru
done
```

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS`: command ran and the observable matched.
- `SKIP (not exposed by runtime)`: capability is advertised upstream
  but the public CLI/API does not surface a way to verify it. The row
  stays here so readers see the gap honestly.
- `ACCEPTED GAP (<reason>)`: capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

Do not invent observables the runtime cannot actually produce (e.g.
do not write "detected language equals X" if the CLI does not print
the detected language); those rows resolve to SKIP, not invented
checks.

Each row covers all four ported variants (`gigaam-v3-e2e-rnnt`,
`gigaam-v3-rnnt`, `gigaam-v3-ctc`, `gigaam-v3-e2e-ctc`); commands below
use `e2e-rnnt` as the representative case, and Stage 4 verified the
observable on every variant against `samples/ru.wav`.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe (Russian) | explicit language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf --language ru samples/ru.wav` | non-empty plausible Russian transcript | PASS. All 4 variants emit a fluent Russian transcript ("Важно различать глаголы и дополнения." for e2e variants; "важно различать глаголы и дополнения" for the non-e2e variants). validate.py Transcript: ok (exact) on every variant. |
| Transcribe (Russian) | auto / no language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-F32.gguf samples/ru.wav` | non-empty plausible Russian transcript (monolingual model: auto path equals explicit-`ru`) | PASS. Auto-path output bit-identical to the explicit `--language ru` path on all 4 variants. |
| Punctuation + casing | inherent in e2e_rnnt / e2e_ctc vocab | same transcribe commands above | output contains capital letters and `.` for e2e variants; lowercased / no-punct for non-e2e | PASS. E2e variants emit "Важно ... дополнения." with capital "В" and trailing period; non-e2e variants emit lowercased no-punct as expected from their 33-entry charwise vocab. |
| Translate | only if exposed | `build/bin/transcribe-cli -m <gguf> --translate --target-language en samples/ru.wav` | runtime should reject (model is monolingual) | SKIP (not exposed by upstream). The CLI surfaces `--translate`, but the gigaam runtime returns `run: unsupported task` because the family's `capabilities.translate=false`. No translate head exists upstream. |
| Language detection | only if exposed | `build/bin/transcribe-cli -m <gguf> samples/ru.wav` and grep CLI output for a "detected language" line | a detected-language string in CLI output | SKIP (not exposed by runtime). The transcribe.cpp CLI does not print a detected-language line for any family, and the gigaam runtime sets `n_languages=0` (monolingual). The auto path is wired to the explicit-ru path, but no observable is surfaced to verify "detection". |
| Word timestamps | only if exposed | inspect `transcribe-cli` output for a non-zero `words:` count | per-word `t0`/`t1` lines in CLI output | ACCEPTED GAP (word-timestamp derivation lives in the upstream `gigaam` package; HF `modeling_gigaam.py` does not implement it). transcribe.cpp emits per-token timestamps (40 ms granularity = encoder frame stride) but does not aggregate them to words. CLI shows `words: 0`. |
| Segment timestamps (>25 s longform) | only if exposed | n/a (upstream uses PyAnnote VAD segmentation) | per-segment `t0`/`t1` from a >25 s sample | ACCEPTED GAP (PyAnnote VAD dependency). transcribe.cpp supports short-form (≤25 s) at every gigaam variant; long-form chunking is not ported. Stage 4 sample (`ru.wav`, 4.5 s) is well within the short-form path. |

## Notes

- **Acceptance dataset (Stage 7): FLEURS ru** (`fleurs:ru` → `ru_ru`
  per `scripts/wer/ingest.py:96`). transcribe.cpp already supports
  FLEURS ru ingest + scoring. **Upstream does NOT publish a FLEURS ru
  WER**, so Stage 7 will measure the upstream baseline ourselves by
  running `gigaam.load_model('v3_e2e_rnnt').transcribe()` over
  the ingested FLEURS ru manifest, then apply the standard
  `C++ ≤ upstream + 0.01pp` gate against that measured baseline.
  Text-norm both sides (e2e_rnnt emits cased+punctuated text; FLEURS
  references are cased; align casing/punctuation handling in the
  WER scorer).
- **Frontend gotchas to lock at Stage 4.** `mel_scale=htk`, `mel_norm=null`,
  `center=false`, log-not-log10 scaling via `SpecScaler`
  (`log(clamp(x, 1e-9, 1e9))`). Any default-aware port will get at least
  one of these wrong silently.
- **Rotary positional embeddings inside Conformer attention** are
  unusual (rel_pos is the parakeet/nemo norm). The `[cos; sin]` stacked
  PE tensor and `rtt_half` rotation in modeling_gigaam.py must be
  reproduced element-for-element.
- **Vocab differs across branches.** The lowercased Russian-only
  (ctc/rnnt) and the punctuation+casing (e2e_ctc/e2e_rnnt) branches
  split their num_classes budget very differently. `tokenizer.model`
  SHA-256 must be per-variant; do not assume reuse across branches.
- **Long-form (>25 s) parity is out of scope.** Upstream
  `transcribe_longform` chains a PyAnnote VAD pipeline + per-segment
  forward. transcribe.cpp will support short-form (≤25 s) at every
  variant.
- **Charwise vs SentencePiece tokenizer paths.** The `rnnt` and `ctc`
  branches use an inline 33-entry character vocab in `config.json`;
  the e2e branches use a SentencePiece `tokenizer.model` file. The
  converter and the C++ loader both need to handle both paths. The
  charwise vocab for `rnnt` and `ctc` is expected to match SHA-256
  (confirmed at Stage 2).
- **Reference-WER measurement (Stage 7) is family-wide.** Because
  upstream does not publish FLEURS ru WER for any of the five
  branches, Stage 7 measures the upstream baseline for each ASR
  variant ourselves via the `gigaam` package, then applies the
  C++ ≤ upstream + 0.01pp gate per variant. Text-norm for the lowercased
  `rnnt`/`ctc` variants must strip punctuation, lowercase, and fold
  ё→е on the FLEURS reference; the e2e variants can either compare
  cased+punctuated or strip both sides.

### Stage 3 converter decisions

Captured here so Stage 4 (`src/arch/gigaam/`) does not have to rederive
them from the converter source.

- **`conv.batch_norm` is LayerNorm at runtime.** The state_dict key is
  named `conv.batch_norm.{weight,bias}` for legacy NeMo-naming reasons,
  but the module is `torch.nn.LayerNorm` (no `running_mean` /
  `running_var` / `num_batches_tracked`). The converter renames to
  `enc.blocks.{i}.conv.ln.{weight,bias}` so the loader expectation is
  unambiguous.
- **LSTM bias collapse.** PyTorch's nn.LSTM stores `bias_ih_l{i}` and
  `bias_hh_l{i}` as two separate vectors both added to the gate
  pre-activation. The converter sums them into a single
  `pred.lstm.{i}.bias`. Gate order is PyTorch native (i, f, g, o).
- **Joint output sequential.** GigaAM's `head.joint.joint_net` is
  `Sequential(ReLU, Linear)`: index 1 is the output linear. Parakeet's
  TDT uses `Sequential(Linear, ReLU, Linear)` (output at index 2). The
  converter flattens to `joint.out.{weight,bias}` for both shapes; the
  activation is recorded in `stt.gigaam.joint.activation`.
- **No relative-position artifacts in attention.** Rotary PE eliminates
  the `self_attn.linear_pos`, `pos_bias_u`, `pos_bias_v` tensors that
  parakeet carries. The encoder block table has 34 tensors per layer
  (vs parakeet's 41 with biases).
- **Frontend buffers baked into the GGUF.** The converter ships
  `frontend.mel_filterbank` (transposed to `[n_mels=64, n_freq_bins=161]`)
  and `frontend.window` (length 320 Hann periodic) directly from the
  model's `preprocessor.featurizer.0.*` state_dict so the C++
  MelFrontend uses bit-identical buffers. `stt.frontend.mel_norm = "htk"`
  encodes the HTK mel-scale + no slaney area normalization combination
  for completeness, but the baked tensors are authoritative.
- **Encoder weights do NOT appear identical across variants** (observed
  during Stage 2 dumps: `enc.out` magnitudes differ per variant).
  The intake's hypothesis of shared encoder weights was wrong;
  the converter emits a full GGUF per variant, not a shared encoder
  GGUF with per-variant head shards.

---

# GigaAM Multilingual

Status: **Stage 1 intake done for the 220M `ctc` variant** (2026-07-18);
Gate A green, capability scope user-signed. Separate HF repo
`ai-sage/GigaAM-Multilingual` (arXiv:2607.10371). Conformer + charwise CTC,
same core math as the gigaam-v3 base branches but a 70-char multilingual
vocab and (for `large_ctc`) a bigger 24-layer/d=1024 encoder.

## Identity

- Hugging Face repo: `ai-sage/GigaAM-Multilingual`
- License: MIT
- Languages: ru, en, kk (Kazakh), ky (Kyrgyz), uz (Uzbek). "Best-in-class"
  on ru/kk/ky/uz; **English is moderate quality** (FLEURS en 12.2%, CV en 26.0%).
- Reference: same `author_repo_gigaam` code path; README documents
  `transformers.AutoModel.from_pretrained("ai-sage/GigaAM-Multilingual", revision="ctc", trust_remote_code=True).transcribe(wav)`.
  Recommended env: `transformers==5.*`, `torch==2.10.*`.
- Variants (one HF branch each):

  | Variant | HF branch | Params | Encoder | Head | num_classes | Vocab | Status |
  |---|---|---|---|---|---|---|---|
  | `gigaam-multilingual-ctc` | `ctc` (== `main`) | 220M | 16-layer, d=768 | CTC | 71 | charwise 70-char multilingual (inline) | **intake (this run)** |
  | `gigaam-multilingual-large-ctc` | `large_ctc` | 600M | 24-layer, d=1024 | CTC | 71 | same 70-char vocab | **not yet intaken** |
  | `ssl` / `large_ssl` | `ssl` / `large_ssl` | 220M / 600M | encoder-only HuBERT | — | — | — | **out of scope** (no head/decoding; same rationale as `v3_ssl`) |

  70-char vocab (blank id = 70): space, apostrophe, Latin a-z (26),
  Cyrillic а-я + ё (33), and 9 Turkic/Kazakh letters (і ғ қ ң ү ұ һ ә ө).
  Lowercased output, no digits, no punctuation beyond apostrophe, no casing.
  No language conditioning — a single CTC head over one shared vocab, so
  a `--language` hint is a no-op.

Canonical bench samples (Stage 2 chooses): a Russian sample plus at least
one Kazakh/Kyrgyz/Uzbek sample to exercise the Turkic vocab range.

## Capability Validation (Stage 1 draft — `gigaam-multilingual-ctc`)

Stage 1 drafts these rows with `Status: TODO`; Stage 4 fills `Status`.
`Target` is the Stage-1 scope contract (`MUST PASS` / `OUT OF SCOPE — <reason>`).

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe (multilingual) | auto / no language hint | `build/bin/transcribe-cli -m models/gigaam-multilingual-ctc/gigaam-multilingual-ctc-F32.gguf <sample>.wav` | non-empty plausible transcript in the sample's language (ru/en/kk/ky/uz) | MUST PASS | TODO |
| Transcribe (multilingual) | explicit `--language <lang>` | `build/bin/transcribe-cli -m <gguf> --language kk <sample>.wav` | output identical to the auto path (model has no language conditioning; flag is a no-op) | MUST PASS | TODO |
| Lowercased no-punct output | inherent charwise vocab | same transcribe commands | output is lowercased, no punctuation except apostrophe, no digits | MUST PASS | TODO |
| Offline batch | batch decode | `scripts/validate.py` / batch harness over N samples | batch output matches per-sample output | MUST PASS | TODO |
| Translate | only if exposed | `build/bin/transcribe-cli -m <gguf> --translate --target-language en <sample>.wav` | runtime rejects (no translate head upstream) | OUT OF SCOPE — no translate head exists upstream; would need a translation-capable checkpoint | TODO |
| Language detection | only if exposed | grep CLI output for a "detected language" line | none (no LID head, no language tokens) | OUT OF SCOPE — model has no language-ID head or language tokens; CLI surfaces no detected-language observable | TODO |
| Word timestamps | only if exposed | inspect `transcribe-cli` for non-zero `words:` count | per-word `t0`/`t1` | OUT OF SCOPE — reference `modeling_gigaam.py` now implements `frames_to_words`, but transcribe.cpp emits per-token (40 ms) timestamps only and does not aggregate to words; would need runtime word-aggregation | TODO |
| Segment timestamps (>30 s longform) | only if exposed | >30 s sample | per-segment `t0`/`t1` | OUT OF SCOPE — PyAnnote VAD dependency; short-form (≤30 s) only, same as gigaam-v3 | TODO |
| Streaming | n/a | n/a | n/a | OUT OF SCOPE — not advertised (`streaming: false`) | TODO |

## Notes (Multilingual)

- **Acceptance dataset: FLEURS, starting ru + kk** (user-signed 2026-07-18),
  not LibriSpeech. Gate on ru + kk first to get a clean signal on the
  implementation, then expand to ky/uz (and en for context) once we are
  confident in the C++ port. The model supports English but English is its
  weakest capability, and the whole value is the ru/kk/ky/uz set.
  Per-language text normalization (lowercase, strip punctuation,
  numerals→words, Uzbek apostrophes, ё handling) must match the model's
  lowercased charwise output. Upstream publishes FLEURS numbers under its
  own normalization/exclusions; Stage 7 measures the reference baseline
  ourselves per language and gates C++ against the measured baseline.
- **`large_ctc` is a distinct architecture** (24-layer, d=1024, feat_in=1024)
  and needs its own intake/oracle/convert/tolerances run — no reuse of the
  220M ctc tolerances.
- **`modeling_gigaam.py` differs from v3** (newer regenerated build, +word
  timestamps/longform), but the frontend/encoder/CTC math classes are the
  same. Stage 2 dumps confirm frontend/encoder bit-exactness against v3.
