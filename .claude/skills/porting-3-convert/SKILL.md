---
name: porting-3-convert
description: Converts a model family's upstream checkpoint into a reference-dtype GGUF that transcribe.cpp's loader can ingest. Produces ONLY the reference-dtype artifact (F32 / F16 / BF16 per the intake); the full quant matrix is a Stage 5 (porting-5-quants) concern. Use this after porting-2-oracle has emitted the contract tensor dumps. Input: intake.json, per-family converter, and the upstream checkpoint. Output: models/<slug>/<slug>-<REFDTYPE>.gguf, reports/convert/<variant>-<refdtype>.json, and Preflight Gate B green.
---

# porting-3-convert

Third stage of the porting pipeline. Runs the family's converter to emit
the reference-dtype GGUF, records provenance, and clears Preflight Gate B.
The full quant matrix belongs to Stage 5.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` exists and is schema-valid.
- `build/validate/<family>/<variant>/dump_coverage.json` exists (output of `porting-2-oracle`).
- The per-family converter exists at `scripts/convert-<family>.py`. If not, this skill creates a skeleton from the closest existing converter.
- The per-family reference env exists at `scripts/envs/<family>/pyproject.toml`.
- `$TRANSCRIBE_MODELS_DIR` is set.

## Workflow

```
Convert progress:
- [ ] Step 1: Confirm or author the per-family converter
- [ ] Step 2: Identify reference dtype from intake
- [ ] Step 3: Run the converter (reference dtype only)
- [ ] Step 4: Write the converter manifest
- [ ] Step 5: Run structural check (Preflight Gate B)
- [ ] Step 6: Sign-off review
```

### Step 1: Per-family converter

Check for `scripts/convert-<family>.py`.

**If present**: reuse. For sibling variants (e.g. `-v3` when `-v2` is ported) the converter usually handles the new variant via `--repo-id` alone; confirm by reading the converter's source for any hard-coded variant checks.

**If absent**: mirror the closest existing converter
(`convert-cohere.py`, `convert-parakeet.py`, or `convert-qwen3_asr.py`).
Preserve source dtypes, emit the loader KV used by
`src/arch/<family>/weights.cpp`, and surface only unresolved tensor-name
or sharding decisions to the user.

### Step 2: Identify reference dtype (read intake)

```bash
uv run python -c "import json; \
  print(json.load(open('reports/porting/<family>/<variant>/intake.json'))['dtype']['expected'])"
```

Expect one of `float32`, `float16`, `bfloat16`. If `null` or `unresolved`, the intake is incomplete — return to `porting-1-intake`.

Map to GGUF preset suffix:
- `float32` → `F32`
- `float16` → `F16`
- `bfloat16` → `BF16`

### Step 3: Run the converter (execute)

```bash
uv run --project scripts/envs/<family> \
  scripts/convert-<family>.py <hf_repo_or_local_path> \
  --repo-id <hf_repo>
```

The output path follows the llama.cpp convention:

```
models/<repo-last-segment>/<repo-last-segment>-<REFDTYPE>.gguf
```

Example for parakeet v2 (reference dtype F32):

```
models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf
```

**Ask-point**: if the converter needed any tensor-name mapping decisions the skill could not infer (NeMo's dual-LSTM bias collapse, tied-embedding handling, fused BatchNorm orientation, etc.), surface them to the user. Record each decision in the family doc's Notes section.

### Step 4: Converter manifest (execute)

Write `reports/convert/<variant>-<refdtype>.json`:

```python
# uv run python -c '...'
import json, hashlib, pathlib, subprocess
family = "<family>"
variant = "<variant>"
refdtype = "<REFDTYPE>"   # F32 / F16 / BF16
gguf = pathlib.Path(f"models/{variant}/{variant}-{refdtype}.gguf")
sha = hashlib.sha256(gguf.read_bytes()).hexdigest()
head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
manifest = {
    "family": family,
    "variant": variant,
    "refdtype": refdtype,
    "gguf_path": str(gguf),
    "gguf_bytes": gguf.stat().st_size,
    "gguf_sha256": sha,
    "source_hf_repo": json.load(open(f"reports/porting/{family}/{variant}/intake.json"))["hf_repo"],
    "source_hf_revision": json.load(open(f"reports/porting/{family}/{variant}/intake.json"))["hf_revision"],
    "converter_script": f"scripts/convert-{family}.py",
    "converter_commit": head,
}
out = pathlib.Path(f"reports/convert/{variant}-{refdtype}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
```

The manifest lets future runs prove provenance without re-reading the GGUF.

### Step 5: Structural checks (execute)

Two gates. Preflight Gate B cross-checks intake ↔ GGUF KV ↔ reference:

```bash
uv run scripts/preflight.py --family <family> --variant <variant> --gate B
```

Loader-open smoke: the C++ loader must open the new GGUF:

```bash
build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf samples/jfk.wav >/dev/null
```

For a brand-new family where `src/arch/<family>/` doesn't exist yet, the loader returns `TRANSCRIBE_ERR_UNSUPPORTED_ARCH`. That is acceptable at Stage 3 — note it in sign-off; Stage 4 (`porting-4-cpp`) brings up the arch. For an established family the smoke must exit 0. A per-family real-model smoke (`tests/<family>_real_smoke.cpp`) is a Stage 4 artifact, not a Stage 3 gate.

### Step 6: Sign-off

Report:
- GGUF path and size.
- Reference dtype used.
- Converter manifest path.
- Preflight Gate B result.
- Any tensor-name mapping decisions captured in the family doc.

**Do not commit.**

## Postconditions

- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists.
- `reports/convert/<variant>-<REFDTYPE>.json` exists and records the SHA + source revision.
- Preflight Gate B is green.
- The full quant matrix is NOT generated here — that is Stage 5 (`porting-5-quants`).

## Pointers (read, not execute)

- `docs/porting/3-conversion.md` — converter patterns and tensor-name guidance
- `docs/tools/conversion.md` — CLI contract and filename convention
- Existing converters as template references:
  - `scripts/convert-parakeet.py` (NeMo `.nemo`)
  - `scripts/convert-cohere.py` (Transformers)
  - `scripts/convert-qwen3_asr.py` (author-repo)
- `scripts/lib/gguf_common.py` — shared KV-write helpers (execute only indirectly, via the converter)
- `src/arch/<family>/weights.cpp` for any already-ported family — the loader's authoritative tensor-name and shape expectations
