---
name: porting-1-intake
description: First stage for a new speech-model port. Produces reports/porting/<family>/<variant>/intake.json, drafts the family capability table, seeds the golden manifest skeleton, and clears Preflight Gate A.
---

# porting-1-intake

First stage of the porting pipeline. Creates the intake packet, fills the
human-judgment fields, drafts the family doc's capability table, and
clears Preflight Gate A.

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

**If the family is already ported** (there is a prior `reports/porting/<family>/<other-variant>/intake.json`): read the prior intake and carry its `reference_framework`, `reference_rationale`, `architecture_pattern`, and `known_risks` forward as defaults.

### Step 2: Human-judgment fields

Fill these fields from research. Ask only when the answer cannot be found
or a decision needs user sign-off. See `docs/porting/1a-intake.md` for
field semantics.

1. **Reference framework choice.**: The reference implementation should always be the source of truth. If there is not one, this is a red flag and must immediately be told to the human.

2. **Architecture pattern.** One of `encoder-transducer`, `encoder-decoder`, `audio-llm`, `encoder-ctc`. The script's `config.architecture_candidates` is a heuristic starting point. If it doesn't fit, propose a new pattern and have the user accept it based on your research of the architecture.

3. **Acceptance dataset.** Default: LibriSpeech test-clean. Capture any
   publisher-reported score in `upstream_benchmarks` when available for
   context, but downstream gates use the measured Oracle reference
   baseline, not the publisher score. For models not supporting English
   LibriSpeech is NOT a valid target. This must be flagged and become a
   hard stop for the time being.

4. **Known risks.** List critical risks that affect later stages
   (frontend/tokenizer ambiguity, custom attention masks, streaming,
   long-sequence behavior, dtype quirks, etc.). One risk per entry. Do
   not ask the user about routine or low-impact observations.

5. **Capability validation table.** Draft one row per advertised
   capability in `docs/porting/families/<family>.md`, fill `Target`, and
   leave `Status: TODO` for Stage 4. `Target` is one of:
   - `MUST PASS` — in scope; Stage 4 is obligated to make the row resolve to `PASS`.
   - `OUT OF SCOPE — <reason>` — explicitly deferred for this port; Stage 4 may resolve it to SKIP or ACCEPTED GAP. The reason names what would bring it back in scope.

   Forced `MUST PASS`: explicit-language transcription, auto/no-hint
   transcription, offline batch, and streaming when
   `capabilities.streaming: true`. All other targets require user
   sign-off. Stage 4 fills `Status` and may not downgrade a `MUST PASS`
   row without the user re-signing. Capabilities the runtime cannot
   observe still get `OUT OF SCOPE` rows so the gap stays visible.

If `scripts/intake.py` reported `intake_gaps`, resolve or explicitly
accept any gap affecting dtype, frontend, tokenizer IDs, architecture
pattern, or reference framework.

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

Seed `tests/golden/<family>/<variant>.manifest.json` with the identity fields derived from intake. This is a **skeleton** — `porting-2-oracle` completes it with `reference.entrypoint`, frontend, tokenizer_summary, capabilities, and the per-family case set the dumper actually runs (typically `samples/jfk.wav` plus any extra audio the family wants). Writing the skeleton now lets later stages reference `manifest.source_model` and `manifest.variant` without having to re-derive them.

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
    "_skeleton": True,  # porting-2-oracle removes this when filling the rest
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
- **The Capability Validation table, row by row, with your proposed
  `Target` for each.** Ask the user to confirm or amend every `Target`
  that is not a forced `MUST PASS` (the transcribe rows, streaming, and
  batch are forced). This per-row in/out-of-scope confirmation is the
  contract Stage 4 implements against — a `MUST PASS` row Stage 4 cannot
  satisfy is a blocker, an `OUT OF SCOPE` row may resolve to SKIP /
  ACCEPTED GAP. Do not treat this as a row-count summary; the user must
  sign off the scope of each capability here.

**Do not commit.** The user authors the commit manually after reviewing.

## Postconditions

- `reports/porting/<family>/<variant>/intake.json` exists and validates against `docs/porting/families/_intake-schema.json`.
- All four human-judgment fields (`reference_framework`, `reference_rationale`, `architecture_pattern`, `known_risks`) are filled.
- Acceptance dataset is captured in `upstream_benchmarks` — LibriSpeech
  test-clean by default, or an explicit alternate for non-English models.
  Any publisher-reported score is context only; downstream gates use the
  measured Oracle reference baseline.
- `docs/porting/families/<family>.md` exists as a draft and includes a `## Capability Validation` table with one row per advertised capability; every row has its `Target` filled (`MUST PASS` or `OUT OF SCOPE — <reason>`, user-signed for non-forced rows) and `Status: TODO`. Rows for capabilities the runtime cannot observe are marked `OUT OF SCOPE` so they may resolve to SKIP at Stage 4.
- `tests/golden/<family>/<variant>.manifest.json` exists as a skeleton (identity fields only, `_skeleton: true`).
- Preflight Gate A is green.

## Pointers (read, not execute)

- `docs/porting/1a-intake.md` — full field semantics for intake.json
- `docs/porting/families/_intake-schema.json` — authoritative schema
- `docs/porting/families/_template.md` — family doc starting shape
- `docs/porting/0-porting.md` — architecture pattern reference
- Existing intakes: `reports/porting/parakeet/*/intake.json`, `reports/porting/cohere/*/intake.json`, `reports/porting/qwen3_asr/*/intake.json` — use as shape references for a new family
