# Numerical Validation

Numerical validation is the tensor-by-tensor gate used during bring-up
and after backend or performance changes. It is also the primary
development methodology: you write reference dumps first, then build
the C++ implementation stage by stage until each tensor matches.

## Flow

```text
1. Study architecture, decide dump points
2. Write reference dump script, generate reference tensors
3. Implement C++ stage by stage, comparing against reference at each stage
4. Set tolerances from observed data
5. Gate: all tensors within tolerance, transcript correct
```

## Dump Principles

**Dump at normalization boundaries.** LayerNorm, BatchNorm, and RMSNorm
often reset activation scale and reduce accumulated numerical noise.
These are good gate points: a correct implementation should usually
produce stable agreement there, while many real bugs still produce
obvious divergence.

**Distinguish gate tensors from debug tensors.** Gate tensors are the
validation contract — they must be dumped on both sides (symmetric)
and must have entries in the tolerance file. Debug tensors are
additional C++-only dumps useful during development (e.g. conformer
sub-stage outputs like `enc.block.0.ff1`). Debug tensors show as
MISSING in the comparison but don't cause validation failure as long
as they're not in the tolerance file.

**Gate tensors must be symmetric.** Every gate tensor dumped on one
side must exist on the other. A MISSING gate tensor (one that has a
tolerance entry) is a real failure.

**Use first/mid/last layer outputs as the default gate.** Dump block 0,
one middle block, and the last block for each repeated component. This
catches load/layout bugs, accumulation drift, and final-stage behavior
without making every layer a permanent tolerance entry. Dump all layers
temporarily when you need to bisect a divergence to a specific layer.

**Dump pre-softmax logits as the numerical gate.** Post-softmax or
log-probability tensors can contain `-inf` when softmax underflows or
masked positions are carried through, and the affected positions may
differ between implementations. Pre-softmax logits avoid this and are
the meaningful comparison. See
[`4a-numerical-troubleshooting.md`](4a-numerical-troubleshooting.md#log-softmax-infinities).

**Dump embedding lookups.** These verify tokenizer alignment and weight
loading. They should be near-exact (within floating-point round-off)
and are a cheap early gate.

## Deciding What to Dump

Before writing any code, study the model architecture and identify the
stage boundaries. Ask:

- What is the frontend? (mel spectrogram, raw waveform features, etc.)
- What transforms happen before the first block? (conv subsampling,
  linear projection, positional encoding addition)
- What is a "block"? (conformer block, transformer layer, LSTM step)
- How many blocks are there in each component?
- Is there an encoder-decoder bridge? (projection, cross-attention K/V)
- What does the decoder consume? (token embeddings, positional encoding)
- What is the output head? (linear + softmax, CTC, joint network)

Map these to dump points. The table below shows common patterns — not
every row applies to every architecture:

| Stage              | Name pattern           | Notes                           |
|--------------------|------------------------|---------------------------------|
| Frontend output    | `enc.mel.in`           | Family-specific frontend        |
| First transform    | `enc.pre_encode.out`   | Conv subsampling, linear proj   |
| Positional enc     | `enc.pos_emb`          | Sinusoidal, learned, RoPE, etc  |
| Encoder layers     | `enc.block.<N>.out`    | First/mid/last by default       |
| Encoder final      | `enc.final`            | Post-final-norm, key gate point |
| Enc-dec bridge     | `enc_dec_proj.out`     | If present                      |
| Decoder embeddings | `dec.token_emb`        | Near-exact gate                 |
|                    | `dec.pos_emb`          | Near-exact gate                 |
| Decoder input norm | `dec.embed_norm`       | After embedding + normalization |
| Decoder layers     | `dec.block.<N>.out`    | First/mid/last by default       |
| Decoder final      | `dec.out_before_head`  | Post-final-norm, key gate point |
| Pre-softmax logits | `dec.logits_raw`       | Numerical gate                  |
| Post-softmax       | `dec.logits`           | Informational, expect inf diffs |

CTC models have no decoder rows. RNNT/TDT models have LSTM states
instead of transformer blocks. Whisper-style models have a full
cross-attention transformer decoder. Audio-LLM models (token
injection) need an additional injection/fusion validation stage; see
[`4a-numerical-troubleshooting.md`](4a-numerical-troubleshooting.md#audio-token-injection).
The implementer chooses which rows apply and adds family-specific names
as needed, documented in the tolerance file.

When the decoder is architecturally a standard LLM (GQA, RoPE,
SwiGLU, RMSNorm), the dump points and expected behavior will be
similar to `llama.cpp` models. The per-layer outputs, final norm,
and pre-softmax logits follow the same patterns.

## Bring-Up: Reference First, Then C++

### Step 1: Study the architecture

Read the reference implementation end to end. Understand the forward
pass, the preprocessing pipeline, and any special tokens or prompt
construction. Identify the dump points above. Write down what shapes
you expect at each stage.

### Step 2: Write the reference dump script

Write `scripts/dump_reference_<family>_<ref>.py` with subcommands for
each stage (`mel`, `encoder`, `decode`). Create
`scripts/envs/<family>/pyproject.toml` with the reference's
dependencies.

Always use the model author's canonical implementation as the
reference. If the model was released through HuggingFace Transformers,
use Transformers. If through NeMo, use NeMo. Don't switch frameworks
mid-bringup.

The dump script has two jobs:
1. Generate the reference tensors you'll build toward.
2. Serve as executable documentation of the model's forward pass.

Run it on `samples/jfk.wav`. Verify shapes match your expectations.
Verify the transcript is correct. These reference dumps are now fixed —
you won't regenerate them during C++ development unless you find a bug
in the dump script itself.

### Step 3: Implement and validate stage by stage

Build the C++ implementation incrementally, using the reference dumps
as your test oracle at each stage:

**Frontend.** Implement the mel/feature extractor. Dump it. Compare
against the reference mel. Fix until it matches. Common issues: wrong
window function, missing preemphasis, dither mismatch, normalization
differences, and off-by-one STFT frame alignment. Don't proceed until
this is clean — every downstream tensor depends on it.

**Encoder.** Implement the graph builder. Start with just the
pre-encode stage (conv subsampling or equivalent), dump, compare. Then
add blocks one at a time or all at once — the per-layer dumps will
tell you where divergence starts if something is wrong. Work until
`enc.final` matches.

**Encoder-decoder bridge.** Implement the projection or cross-attention
K/V computation. Dump, compare.

**Decoder.** Same pattern. Start with embedding lookups (should be
near-exact — if not, you have a tokenizer or weight loading bug). Add
layers, dump per-layer outputs. Work until `dec.out_before_head`
matches. Then add the head and compare `dec.logits_raw`.

At each stage, the comparison is:

```bash
TRANSCRIBE_DUMP_DIR=build/validate/<family>/jfk/cpp \
  build/bin/transcribe-cli -m <model.gguf> --backend cpu samples/jfk.wav

uv run scripts/compare_tensors.py \
  build/validate/<family>/jfk/cpp \
  build/validate/<family>/jfk/ref
```

Run without tolerances during development to see raw numbers. You're
looking for tensors that are close (within dtype noise) vs tensors
that are completely wrong (off by orders of magnitude or wrong shape).

### Step 4: Classify tensors and set tolerances

Once all stages pass informally, classify each tensor into one of
three roles:

**Gate** — post-normalization tensors where drift is small and stable.
A real bug (wrong weights, wrong op, wrong layer order) would produce
divergence far beyond the observed baseline. Examples: `enc.final`,
`dec.out_before_head`, `dec.token_emb`, `dec.logits_raw`.

**Informational** — pre-normalization or amplification stages where
the dominant divergence is from known precision differences, not
implementation bugs. Reported for visibility but set wide tolerances.
Examples: `enc.pre_encode.out`, `enc.block.0.out`.

**Debug** — C++-only tensors useful during development but not part
of the validation contract. These are dumped by the C++ graph builder
but not by the reference script, and are not listed in the tolerance
file. They show as MISSING in the comparison output but don't cause
failure. Examples: conformer sub-stage outputs like
`enc.block.0.{ff1,attn,conv,ff2}`.

Both gate and informational tensors go in the tolerance file and
must be symmetric (present on both sides). Debug tensors do not.

Set tolerances from observed data:
- Gate tensors: ~10x the observed CPU max_abs.
- Informational tensors: ~3x the observed max_abs.
- Don't guess numbers in advance — run the comparison and read them.

Write the tolerance file to `tests/tolerances/<family>.json`. Include
a `_comment` documenting the reference framework, the dtype, known
divergence causes, and the attenuation profile you observed.

### Step 5: Verify

Once the bring-up is complete, use `validate.py` for end-to-end
validation:

```bash
# Full validation: ref dumps + C++ dumps + comparison
uv run scripts/validate.py all --family <family>

# Or step by step
uv run scripts/validate.py ref     --family <family>
uv run scripts/validate.py cpp     --family <family>
uv run scripts/validate.py compare --family <family>

# With a different backend
uv run scripts/validate.py cpp --family <family> --backend metal
```

During incremental development, you can still run the pieces manually:

```bash
TRANSCRIBE_DUMP_DIR=build/validate/<family>/<variant>/jfk/cpp \
  build/bin/transcribe-cli --backend cpu -m <model.gguf> samples/jfk.wav

uv run scripts/compare_tensors.py \
  build/validate/<family>/<variant>/jfk/cpp \
  build/validate/<family>/<variant>/jfk/ref \
  --tolerances tests/tolerances/<family>.json
```

All comparisons should exit 0 with every matched tensor within
tolerance. If the reference side writes `transcript.json`, `validate.py`
also requires the C++ transcript text to match the reference text
character-for-character. CPU validation is the default gate. If an
accelerator backend needs a different tolerance policy, document the
manual `compare_tensors.py --tolerances <backend-file>` command in the
family note; `validate.py compare` currently auto-selects
`tests/tolerances/<family>.json`.

`validate.py` assumes an accuracy GGUF already exists. It can auto-pick a
GGUF under `models/<family>/` or accept `--gguf`, but conversion belongs
to the converter path rather than validation.

## Troubleshooting

This document defines what to dump, how to compare it, and how
tolerances become the validation contract. Failure diagnosis lives in
[`4a-numerical-troubleshooting.md`](4a-numerical-troubleshooting.md),
including common mel/STFT off-by-one errors, dtype and accumulation
drift, layout bugs, attention mask or position mismatches, token
injection errors, and log-softmax infinities.

## Per-Family Notes

Each family's tolerance file (`tests/tolerances/<family>.json`) is the
authoritative source for that family's expected divergence profile,
reference framework, and known causes. Don't duplicate profiles here —
they change as implementations improve.

The current tolerance parser supports a flat mapping from tensor name to
`max_abs` and `mean_abs`, plus `_comment` for documentation:

```json
{
  "_comment": ["why the tolerances are what they are"],
  "enc.final": {"max_abs": 0.5, "mean_abs": 0.01}
}
```

Tensor names present in the tolerance file are the validation contract.
If a contract tensor is missing on either side, comparison fails.
C++-only debug tensors may appear as MISSING in comparison output without
failing as long as they are not in the tolerance file.

## Failure Handling

A validation failure should answer:

- Which tensor first diverged?
- Was the shape wrong? (weight loading or graph construction bug)
- Was the magnitude completely off? (wrong op, wrong layer order)
- Was it close but over tolerance? (precision issue, dtype mismatch)
- Did the transcript change? (functional regression)
- Was this backend-specific? (compare CPU vs accelerator)
