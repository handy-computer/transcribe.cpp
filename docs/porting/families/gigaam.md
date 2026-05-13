# GigaAM

Status: research

## Identity

- Family key: `gigaam`
- Upstream architecture string: `gigaam` (HF `model_type`); paper name "GigaAM" (Sber/ai-sage, InterSpeech 2025, arXiv:2506.01192)
- Hugging Face repo: `ai-sage/GigaAM-v3` (five sibling branches, one per variant)
- License: MIT
- Variants (all five in scope; one HF branch each; **Stage 1 intake complete for all five**):

  | Variant | HF branch | Head | num_classes | Vocab style | Stage 1 intake |
  |---|---|---|---|---|---|
  | `gigaam-v3-e2e-rnnt` | `main` (`v3_e2e_rnnt`) | RNN-T | 1025 | SentencePiece, 1024 pieces, punctuation + Cyrillic casing in vocab | DONE |
  | `gigaam-v3-e2e-ctc` | `e2e_ctc` | CTC | 257 | SentencePiece, 256 pieces, punctuation + Cyrillic casing in vocab | DONE |
  | `gigaam-v3-rnnt` | `rnnt` | RNN-T | 34 | charwise (inline in config.json), 33 entries: space + а-я (no ё), lowercased no-punct | DONE |
  | `gigaam-v3-ctc` | `ctc` | CTC | 34 | charwise (inline in config.json), same 33-entry list as `gigaam-v3-rnnt` | DONE |
  | `gigaam-v3-ssl` | `ssl` | none (encoder-only HuBERT-CTC pretrain) | n/a | n/a (no decoder, no tokenizer) | DONE |

  Encoder dims are identical across all five branches (16-layer
  Conformer, d=768, 16 heads, rotary PE, conv1d subsampling factor 4).
  Encoder weights are expected — but not yet confirmed — to be shared
  across the four ASR variants after SSL pretraining + per-head
  fine-tuning. The SSL checkpoint is the common ancestor pre-finetune.
  Stage 3 cross-compares encoder tensors across branches to decide
  whether the same GGUF weights can back multiple variants or whether
  each ships its own.

## References

- Canonical reference: **author_repo_gigaam** — the [`gigaam`](https://github.com/salute-developers/GigaAM) PyPI package (MIT). Loader entrypoint: `gigaam.load_model("v3_e2e_rnnt")` (or `"v3_rnnt"`, `"v3_ctc"`, `"v3_e2e_ctc"`, `"v3_ssl"`) → returns a `GigaAMASR` with `.transcribe(wav_file)` and `.transcribe_longform(wav_file)`.
- Instrumented reference: **author_repo_gigaam** (same package). Use `torch` forward hooks on `ConformerEncoder`, `RNNTHead`, `CTCHead`, and `FeatureExtractor` to capture per-stage intermediates without modifying the upstream classes. The HF mirror at `ai-sage/GigaAM-v3` ships the same class definitions inside `modeling_gigaam.py` and can be loaded via `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` — same code path, second source of truth.
- Cross-check references: HuggingFace `transformers` AutoModel path (against the modeling_gigaam.py shim). Note: NeMo is NOT a reference for this family; GigaAM uses its own Conformer + RNN-T implementations wired through omegaconf/Hydra.

## Commands

> Stage 2/3/4 will fill these. Skeletons below assume the per-family
> env at `scripts/envs/gigaam/` (TODO) and the dump/convert script
> names follow the family convention (e.g. `dump_reference_gigaam.py`,
> `convert-gigaam.py`).

Reference run:

```bash
TODO  # uv run --project scripts/envs/gigaam python -c "import gigaam; \
      #   m = gigaam.load_model('v3_e2e_rnnt'); print(m.transcribe('<ru-sample>.wav'))"
      # (use a Russian-language sample; samples/jfk.wav is English and will
      #  produce a phonetic-Russian guess, not a meaningful validation.)
```

Reference dumps:

```bash
TODO  # uv run --project scripts/envs/gigaam scripts/dump_reference_gigaam.py \
      #   --variant gigaam-v3-e2e-rnnt \
      #   --manifest tests/golden/gigaam/gigaam-v3-e2e-rnnt.manifest.json
```

Conversion:

```bash
TODO  # uv run scripts/convert-gigaam.py \
      #   --in $TRANSCRIBE_MODELS_DIR/gigaam-v3-e2e-rnnt \
      #   --out models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-f32.gguf
```

Validation:

```bash
TODO  # uv run scripts/validate.py all --family gigaam --variant gigaam-v3-e2e-rnnt
```

Benchmarks (Stage 7 acceptance — FLEURS ru):

```bash
TODO  # 1) one-time corpus ingest
      # uv run scripts/wer/ingest.py fleurs --lang ru
      #
      # 2) measure upstream baseline on FLEURS ru (gigaam package)
      # uv run --project scripts/envs/gigaam scripts/wer/run_reference_gigaam_author.py \
      #   --variant v3_e2e_rnnt \
      #   --manifest samples/wer/fleurs-ru.manifest.jsonl
      #
      # 3) measure C++ port on the same manifest
      # uv run scripts/wer/run.py \
      #   --model models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-f32.gguf \
      #   --manifest samples/wer/fleurs-ru.manifest.jsonl
```

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

Do not invent observables the runtime cannot actually produce (e.g.
do not write "detected language equals X" if the CLI does not print
the detected language) — those rows resolve to SKIP, not invented
checks.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe (Russian) | explicit language hint | `build/bin/transcribe-cli -m models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-f32.gguf --language ru <ru-sample>.wav` | non-empty plausible Russian transcript with punctuation+casing | TODO |
| Transcribe (Russian) | auto / no language hint | `build/bin/transcribe-cli -m models/gigaam-v3-e2e-rnnt/gigaam-v3-e2e-rnnt-f32.gguf <ru-sample>.wav` | non-empty plausible Russian transcript (monolingual model: auto path equals explicit-`ru`) | TODO |
| Punctuation + casing | inherent in e2e_rnnt vocab | same transcribe commands above | output contains capital letters and standard punctuation tokens (`.,?!`) | TODO |
| Translate | only if exposed | n/a — model is monolingual Russian | — | TODO (expected: SKIP — not exposed by upstream) |
| Language detection | only if exposed | n/a — model is monolingual Russian | — | TODO (expected: SKIP — not exposed by upstream) |
| Word timestamps | only if exposed | n/a — HF `modeling_gigaam.py` does not derive per-word timings (`gigaam` package exposes `word_timestamps=True` but transcribe.cpp will not reproduce that derivation at Stage 4) | — | TODO (expected: ACCEPTED GAP — word-timestamp derivation lives in the gigaam package, not in modeling_gigaam.py) |
| Segment timestamps (>25 s longform) | only if exposed | n/a — upstream uses PyAnnote VAD segmentation, out of scope for transcribe.cpp short-form parity | — | TODO (expected: ACCEPTED GAP — PyAnnote dependency) |

## Notes

- **Acceptance dataset (Stage 7): FLEURS ru** (`fleurs:ru` → `ru_ru`
  per `scripts/wer/ingest.py:96`). transcribe.cpp already supports
  FLEURS ru ingest + scoring. **Upstream does NOT publish a FLEURS ru
  WER** — Stage 7 will measure the upstream baseline ourselves by
  running `gigaam.load_model('v3_e2e_rnnt').transcribe()` over
  the ingested FLEURS ru manifest, then apply the standard
  `C++ ≤ upstream + 0.01pp` gate against that measured baseline.
  Text-norm both sides (e2e_rnnt emits cased+punctuated text; FLEURS
  references are cased — align casing/punctuation handling in the
  WER scorer).
- **Frontend gotchas to lock at Stage 4.** `mel_scale=htk`, `mel_norm=null`,
  `center=false`, log-not-log10 scaling via `SpecScaler`
  (`log(clamp(x, 1e-9, 1e9))`). Any default-aware port will get at least
  one of these wrong silently.
- **Rotary positional embeddings inside Conformer attention** are
  unusual (rel_pos is the parakeet/nemo norm). The `[cos; sin]` stacked
  PE tensor and `rtt_half` rotation in modeling_gigaam.py must be
  reproduced element-for-element.
- **Vocab differs across branches.** The same num_classes (1025)
  budget is split very differently between the lowercased
  Russian-only (ssl/ctc/rnnt) and the punctuation+casing
  (e2e_ctc/e2e_rnnt) branches. `tokenizer.model` SHA-256 must be
  per-variant — do not assume reuse across branches.
- **Long-form (>25 s) parity is out of scope.** Upstream
  `transcribe_longform` chains a PyAnnote VAD pipeline + per-segment
  forward. transcribe.cpp will support short-form (≤25 s) at every
  variant.
- **`gigaam-v3-ssl` is encoder-only.** No decoding, no tokenizer, no
  transcribe(). Stage 7 (WER) does not apply to this variant — the
  acceptance gate is Stage 4 encoder-tensor numerical parity against
  `gigaam.load_model('v3_ssl').encoder(features)`. transcribe.cpp has
  no encoder-output emission CLI today, so the SSL variant is loadable
  but has no user-observable runtime capability until that path is
  added. Open design question, not blocking the other four variants.
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
