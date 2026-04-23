---
name: porting-1-intake
description: Produces the machine-readable intake packet (reports/porting/<family>/<variant>/intake.json) that locks down identity and human-judgment decisions at the start of a new model-family port. Use this as the FIRST step when porting a new speech model to transcribe.cpp, before any converter or C++ work. Input is a Hugging Face repo URL or org/name plus a family key and variant name; output is a schema-valid intake.json plus a draft family doc, with Preflight Gate A green.
---

# porting-1-intake

First stage of the porting pipeline. Runs the mechanical intake bootstrapper, walks the user through the human-judgment ask-points, schema-validates the result, and runs Preflight Gate A. On green, the port is ready for `porting-2-refdump`.

## Preconditions

- User provided: a Hugging Face URL or `org/name`, a stable `family` key (e.g. `parakeet`, `whisper`), a `variant` name (e.g. `parakeet-tdt-0.6b-v2`, `whisper-base`).
- Working directory is the transcribe.cpp repo root.
- `uv` is installed.
- `$TRANSCRIBE_MODELS_DIR` is set (see the parent CLAUDE.md).

## Workflow

Copy and track this checklist:

```
Intake progress:
- [ ] Step 1: Run scripts/intake.py inspect (mechanical draft)
- [ ] Step 2: Fill human-judgment fields via research. If they cannot be found then ask.
- [ ] Step 3: Schema-validate the intake.json
- [ ] Step 4: Run Preflight Gate A
- [ ] Step 5: Pre-fill family doc draft from intake
- [ ] Step 6: Seed golden manifest skeleton
- [ ] Step 7: Sign-off review
```

### Step 1: Mechanical draft (execute)

```bash
uv run scripts/intake.py inspect \
  --repo <org/name> \
  --family <family> \
  --variant <variant> \
  --out reports/porting/<family>/<variant>/intake.json
```

This writes a draft with mechanical fields populated (hf_revision, config, dtype distribution, frontend, tokenizer summary, capabilities) and null-valued human-judgment fields.

**If the family is already ported** (there is a prior `reports/porting/<family>/<other-variant>/intake.json`): read the prior intake and carry its `reference_framework`, `reference_rationale`, `architecture_pattern`, and `known_risks` forward as DEFAULTS for the ask-points. Do not silently copy them — confirm each with the user.

### Step 2: Potential Ask-points

Try and research and fill out the following questions/ask-points on your own. Fill the corresponding fields in the intake JSON. See `docs/porting/1a-intake.md` for field semantics.

1. **Reference framework choice.**: The reference implementation should always be the source of truth. If there is not one, this is a red flag and must immediately be told to the human.

2. **Architecture pattern.** One of `encoder-transducer`, `encoder-decoder`, `audio-llm`, `encoder-ctc`. The script's `config.architecture_candidates` is a heuristic starting point. If it doesn't fit, propose a new pattern and have the user accept it based on your research of the architecture.

3. **Acceptance dataset.** Default: LibriSpeech test-clean (upstream-reported WER must be captured in `upstream_benchmarks`). For non-English-only models LibriSpeech is NOT a valid target — ask the user which dataset the publisher reports on (FLEURS `<lang>`, Common Voice `<lang>`, etc.) and record that as the acceptance dataset. This choice drives the Stage 6 WER hard gate.

4. **Known risks.** Anything the agent cannot infer from HF metadata: novel positional encoding, custom attention masks, multimodal fusion quirks, streaming, long-sequence degradation, per-layer dtype differences, upstream doing one precision but our quantize path assuming another. Free-form list; one risk per entry. Only flag these if they are CRITICAL blockers for the next stages. If not, DO NOT ask the human.

If `scripts/intake.py` reported non-empty `intake_gaps`, surface them here too. Gaps affecting dtype, frontend, tokenizer IDs, architecture pattern, or reference framework must be resolved or explicitly accepted by the user.

### Step 3: Schema validation (execute)

```bash
uv run python -c "import json, jsonschema; \
  s=json.load(open('docs/porting/families/_intake-schema.json')); \
  d=json.load(open('reports/porting/<family>/<variant>/intake.json')); \
  jsonschema.validate(d, s); print('OK')"
```

If validation fails, fix the intake in place and re-run. Do not proceed with a schema-invalid intake.

### Step 4: Preflight Gate A (execute)

```bash
uv run scripts/preflight.py --family <family> --variant <variant> --gate A
```

Gate A cross-checks declared intake values against the reference framework's config/preprocessor/tokenizer files. If it fails, the intake and the reference disagree on something load-bearing — fix the intake or the declaration and re-run.

### Step 5: Family doc draft (execute)

If `docs/porting/families/<family>.md` does not exist yet, create it by copying `_template.md` and filling the Identity, References, and Artifacts sections from the intake. Leave the Commands section with real `uv run` invocations for this family.

```bash
[ -f docs/porting/families/<family>.md ] || \
  cp docs/porting/families/_template.md docs/porting/families/<family>.md
```

### Step 6: Golden manifest skeleton (execute)

Seed `tests/golden/<family>/<variant>.manifest.json` with the identity fields derived from intake. This is a **skeleton** — `porting-2-refdump` completes it with `reference.entrypoint`, frontend, tokenizer_summary, capabilities, and the contract case set. Writing the skeleton now lets later stages reference `manifest.source_model` and `manifest.variant` without having to re-derive them.

```python
# uv run python -c '...'
import json, pathlib
intake = json.loads(pathlib.Path("reports/porting/<family>/<variant>/intake.json").read_text())
skel = {
    "schema": "transcribe-golden-manifest-v1",
    "family": intake["family"],
    "variant": intake["variants"][0]["name"],
    "source_model": {"hf_repo": intake["hf_repo"], "hf_revision": intake["hf_revision"]},
    "expected_dtype": intake["dtype"]["expected"],
    "dtype_source": intake["dtype"]["source"],
    "_skeleton": True,  # porting-2-refdump removes this when filling the rest
}
out = pathlib.Path(f"tests/golden/{intake['family']}/{intake['variants'][0]['name']}.manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(skel, indent=2))
```

### Step 7: Sign-off

Report to the user:
- Path to the intake.json.
- Path to the family doc draft.
- Preflight Gate A result.
- Any unresolved `intake_gaps`.
- Whether the acceptance dataset is LibriSpeech or an alternate (name it).

**Do not commit.** The user authors the commit manually after reviewing.

## Postconditions

- `reports/porting/<family>/<variant>/intake.json` exists and validates against `docs/porting/families/_intake-schema.json`.
- All four human-judgment fields (`reference_framework`, `reference_rationale`, `architecture_pattern`, `known_risks`) are filled.
- Upstream acceptance target is captured in `upstream_benchmarks` — LibriSpeech test-clean by default, or an explicit alternate for non-English models.
- `docs/porting/families/<family>.md` exists as a draft.
- `tests/golden/<family>/<variant>.manifest.json` exists as a skeleton (identity fields only, `_skeleton: true`).
- Preflight Gate A is green.

## Pointers (read, not execute)

- `docs/porting/1a-intake.md` — full field semantics for intake.json
- `docs/porting/families/_intake-schema.json` — authoritative schema
- `docs/porting/families/_template.md` — family doc starting shape
- `docs/porting/0-porting.md` — architecture pattern reference
- Existing intakes: `reports/porting/parakeet/*/intake.json`, `reports/porting/cohere/*/intake.json`, `reports/porting/qwen3_asr/*/intake.json` — use as shape references for a new family
