# GigaAM

Status: ported. Stage 4 numerical validation green at ref dtype (F32),
Stage 5 quant matrix shipped, Stage 7 FLEURS ru WER measured. All four
ASR variants (`gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt`, `gigaam-v3-ctc`,
`gigaam-v3-e2e-ctc`) ready for HF upload. `v3_ssl` (encoder-only HuBERT
pretraining checkpoint) intentionally out of scope.

`gigaam-multilingual-large-ctc` (ai-sage/GigaAM-Multilingual, branch
`large_ctc`) converts and validates at F32 on the existing runtime with no
C++ change; WER, benchmarks, and publication are not done. See
[GigaAM Multilingual Large CTC](#gigaam-multilingual-large-ctc).

## Identity

- Family key: `gigaam`
- Upstream architecture string: `gigaam` (HF `model_type`); paper name "GigaAM" (Sber/ai-sage, InterSpeech 2025, arXiv:2506.01192)
- Hugging Face repo: `ai-sage/GigaAM-v3` (five sibling branches upstream; four ported)
- Also converted, not yet published: `gigaam-multilingual-large-ctc` from `ai-sage/GigaAM-Multilingual`, see [GigaAM Multilingual Large CTC](#gigaam-multilingual-large-ctc)
- License: MIT
- v3 variants in scope (one HF branch each):

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

Canonical bench samples for the v3 variants: `ru` (model is Russian-only; English `jfk`/`dots` would decode to garbage).

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
  variant. (Superseded: audio past 25 s is now encoded in overlapping
  windows natively; see `docs/models/gigaam.md`.)
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

## GigaAM Multilingual Large CTC

Variant `gigaam-multilingual-large-ctc` is the `large_ctc` branch of
[`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual),
pinned at commit `3905cd51c3ed4e88c8edf33f3302969ba480a327` (the repo's
`main` branch is the smaller `ctc` model). It is the v3 encoder at 24
layers, d_model=1024, 16 heads, with a CTC head over 71 classes: a
70-entry charwise vocab (space, apostrophe, a-z, Russian а-я and ё, and
і ғ қ ң ү ұ һ ә ө) plus blank at id 70. Languages: ru, en, kk, ky, uz.
Offline only; there is no language input or language-detection output.

| Item | Path |
|---|---|
| Intake | `reports/porting/gigaam/gigaam-multilingual-large-ctc/intake.json` |
| Golden manifest | `tests/golden/gigaam/gigaam-multilingual-large-ctc.manifest.json` |
| Tolerances | `tests/tolerances/gigaam-multilingual-large-ctc.json` |
| Variant sources and snapshot loader | `scripts/lib/gigaam_hf.py` (tests: `scripts/lib/test_gigaam_hf.py`) |

### Commands

Conversion (writes `models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-F32.gguf`,
2341378624 bytes, 824 tensors):

```bash
uv run --project scripts/envs/gigaam scripts/convert-gigaam.py ai-sage/GigaAM-Multilingual \
  --repo-id gigaam-multilingual-large-ctc
```

The converter downloads the snapshot (2.3 GB) at the pinned commit into
`$TRANSCRIBE_MODELS_DIR/<slug>@<commit>/` or the Hugging Face cache, both
of which keep each commit in its own directory; with `HF_HUB_OFFLINE=1`
it uses that local copy without network access, and fails when the
pinned commit was never downloaded. Unlike converters that take
`lib.hf_source.resolve_model_dir`, it accepts no other repo id and no local directory, so `general.repo_url`
and the reference dumps' `hf_revision` name the source the weights came
from.

The repo and commit of every GigaAM variant are in `VARIANTS` of
`scripts/lib/gigaam_hf.py`, which the converter, the reference dumper,
and the WER runner read. `scripts/lib/test_gigaam_hf.py` fails when a
golden manifest names another source, and the dumper refuses the
`--revision` that `validate.py` passes from the manifest when it differs
from the pin. Change the pin and the manifest together.

Reference dumps and validation use the same pin:

```bash
uv run scripts/preflight.py --family gigaam --variant gigaam-multilingual-large-ctc --gate A
uv run scripts/preflight.py --family gigaam --variant gigaam-multilingual-large-ctc --gate B
uv run scripts/validate.py all --family gigaam --variant gigaam-multilingual-large-ctc
uv run scripts/quantize-all.py models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-F32.gguf
TRANSCRIBE_GIGAAM_GGUF=$PWD/models/gigaam-multilingual-large-ctc/gigaam-multilingual-large-ctc-F16.gguf \
  ctest --test-dir build -R gigaam
```

### Reference

The reference is the family's `gigaam` package pin (6e4b027c), as for v3.
That commit has no `load_model("multilingual_large_ctc")`, so
`scripts/lib/gigaam_hf.py` instantiates the package's `FeatureExtractor`,
`ConformerEncoder`, `CTCHead`, and `CTCGreedyDecoding` from the snapshot's
`config.json`, whose `_target_` entries name the same classes in
`modeling_gigaam`, and loads `pytorch_model.bin` with its `model.` key
prefix removed. The upstream commits from 6e4b027c to 7447938 change the
encoder only for padded batch inputs. The repo's own `modeling_gigaam.py`
on the same weights produces bit-identical `frontend.mel.out`, `enc.out`,
and `ctc.log_probs` on `samples/ru.wav`.

The v3 config lists `mel_scale: htk` and `mel_norm: null`; the
Multilingual config omits both. The package's `FeatureExtractor` passes
neither to torchaudio and defaults `center` to true, so the converter
reads the values the module sets (sample rate, FFT, window and hop
lengths, mel count, `center`, mel scale and norm, frequency range) from
it, `htk` with no norm for every variant, and refuses a config value that
disagrees. The rest of `stt.frontend.*` (periodic Hann window, no dither,
pre-emphasis, or normalization) is fixed by `FeatureExtractor`'s code.

### Capability Validation

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe (ru, en, kk, ky, uz) | auto / no language hint | `build/bin/transcribe-cli -m <gguf> <wav>` | transcript equal to the reference's | PASS on `samples/ru.wav`, `samples/fleurs-ru.wav`, `samples/fleurs-en.wav`, and two FLEURS validation utterances each for kk, ky, uz (not in the repo): exact match for F32, F16, Q8_0 on CPU and Metal and Q6_K, Q5_K_M on Metal; Q4_K_M on Metal differs in one letter on one kk utterance. |
| Transcribe | explicit language hint | same with `--language kk` (or any of the five) | same transcript as without the hint | PASS. The hint has no effect on CTC decoding; a language outside `general.languages`, such as `--language de`, fails with `run: unsupported language`. |
| Translate, language detection, streaming | n/a | n/a | n/a | SKIP (not exposed upstream or by the runtime), as for v3. |
| Word timestamps | n/a | n/a | n/a | ACCEPTED GAP, as for v3: token timestamps only. |

Numerical validation at F32 on CPU, one thread, against the ru golden
case is within the per-variant tolerances, with an exact transcript; the
observed drift is in the `_comment` of the tolerance file. The other four
languages have no golden case: their numerical parity rests on the shared
graph and on the transcript checks above. The reference WER baseline, the
per-language WER sweep, and benchmarks have not been run;
`scripts/wer/run_reference_gigaam_author.py --variant multilingual_large_ctc`
runs the reference side.

`TRANSCRIBE_GIGAAM_GGUF=<gguf> ctest --test-dir build -R gigaam` passes
with the F32, F16, and Q8_0 GGUFs of this variant.
