# Conversion

`scripts/convert-<family>.py` converts an upstream checkpoint into a
GGUF file that `transcribe.cpp`'s loader can ingest. **Conversion is
Python. Quantization is C++.** These are separate tools with separate
responsibilities; see [`quantization.md`](quantization.md).

## Scope

What conversion does:

- Read the upstream format (NeMo `.nemo` archive / HuggingFace safetensors).
- Apply any required layout transforms.
- Rename upstream tensor names to canonical GGUF names.
- Embed the tokenizer (SentencePiece / BPE) as GGUF KV.
- Emit architecture metadata (hparams, frontend config, variant string,
  language list, capability flags).
- Write tensors in the source/reference dtype for that family
  (currently Parakeet: **F32**; Cohere: **BF16**).

What conversion does **not** do:

- Produce a lossy quantized GGUF (Q8_0, Q4_K_M, etc.). That is
  `transcribe-quantize`'s job. Converters do not expose `--quant`.
- Decide per-tensor quant policy. Bucket classification used to live
  here; post-unification it lives in the shared C++ policy table.
- Call `gguf.quants.quantize()` for any block-quantized type.

## Current families

| Family   | Script                          | Env                          | Source format                |
|----------|---------------------------------|------------------------------|------------------------------|
| parakeet | `scripts/convert-parakeet.py`   | `scripts/envs/parakeet/`     | NeMo `.nemo` archive (via `ASRModel.from_pretrained`) |
| cohere   | `scripts/convert-cohere.py`     | `scripts/envs/cohere/`       | HuggingFace safetensors (bf16) |

Each converter is a single-file script with inline documentation of its
tensor catalog, hparam map, and layout transforms. No base class. See
[`../porting/3-conversion.md`](../porting/3-conversion.md) for why
single-file is the chosen shape.

## CLI

```bash
# Parakeet: pass an HF repo id (or a local .nemo path). NeMo resolves
# and downloads the checkpoint via ASRModel.from_pretrained.
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2

# Cohere: pass an HF repo id. huggingface_hub.snapshot_download pulls
# the checkpoint into $TRANSCRIBE_MODELS_DIR/<slug>/ (or the HF cache
# if unset), then the script converts it.
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py CohereLabs/cohere-transcribe-03-2026
```

Both converters also accept a local checkpoint path for offline /
custom-checkpoint use. Pass `--repo-id` in that case so the output slug
can be derived:

```bash
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py <model-dir> --repo-id CohereLabs/cohere-transcribe-03-2026
```

The output dtype is family-specific and matches the reference/source
dtype. Use [`quantization.md`](quantization.md) for any derived F16 or
block-quantized GGUF.

## Output naming and layout

Match `llama.cpp`'s filename convention: `<slug>-<QUANT>.gguf` with the
quant preset uppercase. The `<slug>` is the HF repo name (everything after
the last `/` in the repo id) and **is also the directory name** — one
directory per HF repo, so multiple variants of a family coexist cleanly.

```text
models/
├── parakeet-tdt-0.6b-v2/                        # nvidia/parakeet-tdt-0.6b-v2
│   ├── parakeet-tdt-0.6b-v2-F32.gguf
│   ├── parakeet-tdt-0.6b-v2-F16.gguf
│   └── parakeet-tdt-0.6b-v2-Q4_K_M.gguf
└── cohere-transcribe-03-2026/                   # CohereLabs/cohere-transcribe-03-2026
    ├── cohere-transcribe-03-2026-BF16.gguf
    └── cohere-transcribe-03-2026-Q5_K_M.gguf
```

This matches the raw-checkpoint layout under `$TRANSCRIBE_MODELS_DIR`
(also keyed by HF repo name), so `models/<slug>/` on the converted side
mirrors `$TRANSCRIBE_MODELS_DIR/<slug>/` on the source side.

The `--repo-id` flag on each converter builds both the directory and
filename for you via `slug_from_repo_id()` + `gguf_name()` in
`scripts/lib/gguf_common.py`. Quantized siblings live next to the
accuracy GGUF under the same `<slug>/` directory.

Tools that discover a GGUF by `--family` (`scripts/validate.py`,
`scripts/bench/run.py`) scan `models/*/` and filter by `stem.startswith(family)`.

## First-GGUF rule

The **first** GGUF for a new family must match the reference dump's
dtype exactly (recorded in the sidecar `model_dtype`). If the reference
loaded BF16, the first GGUF is BF16. If F32, then F32. A dtype mismatch
makes the C++ inference path operate at a different precision than the
reference and causes tolerances to absorb a hidden gap instead of
genuine drift.

See [`../porting/3-conversion.md`](../porting/3-conversion.md) for the
full accuracy-GGUF-first policy, tied-weight handling, and multi-
component bucket rules.

## Adding a new family

1. Create `scripts/envs/<family>/pyproject.toml` with the upstream
   loader's dep stack. NeMo and Transformers do not co-install
   cleanly — each family gets its own env.
2. Copy `convert-parakeet.py` or `convert-cohere.py` as a starting
   point. Both are deliberately readable top-to-bottom.
3. Write the hparam map, tensor catalog, and layout transforms inline.
4. Import shared helpers from `scripts/lib/` (GGUF KV helpers,
   fp encoding, manifest writing). **Do not** import a per-family
   base class — there isn't one, and there shouldn't be one until we
   have 5+ families of the same shape.
5. Update the C++ loader (`src/arch/<family>/weights.cpp`) to accept
   the full quant allowlist — see `transcribe::weights::kQuantLinearTypes`
   in [`quantization.md`](quantization.md).

## Shared library

`scripts/lib/` holds code that every converter uses but that doesn't
justify a class hierarchy:

- `gguf_common.py` — KV writer helpers, tensor name canonicalization,
  fp32/f16/bf16 `encode_for_gguf()`.
- `quant_policy.py` — preset name registry (names only, no math;
  quantization math lives in C++).

Import with a two-line `sys.path.insert` at the top of each converter.
This matches `llama.cpp`'s `gguf-py` pattern: a local importable module,
not an installable package.

## Converter manifest

Each conversion should write enough provenance to reconstruct the run.
Target path:

```text
reports/convert/<family>/<variant>-<QUANT>.json
```

Contents: source repo + revision, source file sha256, converter
revision, output sha256, tensor count, skipped / tied / fused tensors,
quant preset.

See [`../porting/3-conversion.md`](../porting/3-conversion.md#converter-manifest).
