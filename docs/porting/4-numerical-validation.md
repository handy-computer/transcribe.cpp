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
reset the scale of activations and attenuate accumulated error. These
are the natural gate points — where a correct implementation should
produce tight agreement and a bug would produce obvious divergence.

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

**Dump pre-softmax logits as the numerical gate.** Post-softmax
log-probabilities produce `-inf` for near-zero-probability tokens, and
the exact set of `-inf` positions differs between implementations.
Pre-softmax logits avoid this and are the meaningful comparison. See
[Known Patterns: log-softmax inf](#log-softmax-inf).

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
injection) need an additional injection/fusion validation stage —
see [Known Patterns: Multimodal fusion](#multimodal-fusion). The
implementer chooses which rows apply and adds family-specific names
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
differences. Don't proceed until this is clean — every downstream
tensor depends on it.

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
accelerator backend needs different tolerance policy, use a separate
tolerance file for that backend and document the command in the family
note.

`validate.py` assumes an accuracy GGUF already exists. It can auto-pick a
GGUF under `models/<family>/` or accept `--gguf`, but conversion belongs
to the converter path rather than validation.

## Known Patterns

### log-softmax inf

Any model outputting log-probabilities over a vocabulary will produce
`-inf` for tokens where `softmax(x)` rounds to zero. Different
implementations round differently, so the exact set of `-inf`
positions differs, creating infinite element-wise diffs.

**Affected:** every seq2seq decoder with log-softmax output, CTC
models with log-prob output.

**Fix:** dump `dec.logits_raw` (pre-softmax) as the numerical gate.
Set `dec.logits` tolerance to `"inf"` for both max_abs and mean_abs.

### Baked frontend tensors

Converters may bake filterbanks, windows, or other frontend buffers
from the original checkpoint. These must match the reference
implementation, not necessarily the original checkpoint. If the
reference computes its own window (e.g. `torch.hann_window`), the
converter should bake that same window, not the one from the
checkpoint's preprocessor state dict.

Verify during converter development by comparing the baked tensor
against what the reference implementation uses.

### Dtype matching

**Use the reference's natural dtype.** Let the reference implementation
load in its default dtype; don't coerce weights to match the GGUF dtype
unless the production C++ path is intentionally doing that conversion
and you are validating that specific export precision.

A bf16 GGUF compared against an f32 reference will produce non-zero
diffs — that is expected, not a bug. Characterise them, set tolerances
from observed data, and document the cause in the tolerance file's
`_comment`. If mean_abs is low and output-adjacent gate tensors are
stable, the drift is just precision noise and the validation is sound.

**Only add precision-matching reference modes after you prove you need
them.** If the default reference already produces the correct transcript
and all gate tensors are within tolerance, stop there. A
precision-simulating path (e.g. round-tripping weights through f16
before comparing) is an exception that belongs in the dump script with
a comment explaining why: "GGUF intentionally stores X as f16, so this
validation target checks exported-model numerics." It should not be the
default.

**The first accuracy GGUF must use the same dtype as the reference.**
The reference drives the numerics; the converter follows. If the
reference loaded bf16 (because that is the model's natural dtype), the
first GGUF must be bf16. If the reference loaded f32, the first GGUF
must be f32. A dtype mismatch between the reference and the GGUF
introduces a precision gap that comparison tolerances end up absorbing
silently — hiding real bugs behind noise. See the Conversion doc for
the required dtype ordering.

### Preprocessing mismatches

Reference implementations may apply dither, preemphasis, or other
preprocessing that the C++ frontend must match. Check the feature
extractor source code, not just model config defaults. Common issues:

- Dither applied at inference time in one implementation but not the
  other.
- Preemphasis coefficient missing or hardcoded differently.
- Window function computed differently (periodic vs symmetric Hann).
- Normalization using biased vs unbiased variance.

### Multimodal fusion / token injection

Audio-LLM models (Qwen3-ASR, Qwen2-Audio) inject audio encoder
output into the text token sequence by replacing placeholder tokens.
The correctness of this injection is critical and easy to get wrong:
off-by-one in audio length calculation, wrong placeholder token
count, wrong mask positions.

Dump the token embedding sequence **before and after injection** as
separate tensors. The pre-injection tensor should show standard
token embeddings. The post-injection tensor should show audio
features at exactly the placeholder positions and token embeddings
everywhere else. If these don't match, every downstream decoder
layer will diverge and the per-layer diffs won't help diagnose the
root cause.

### Padding boundary artifacts

The last frame of conv-subsampled features often shows large diffs
because the exact boundary handling differs between implementations
(ggml im2col vs PyTorch native conv). This is typically one frame
out of hundreds and does not affect transcription quality. Set
tolerances to accommodate it rather than trying to match it exactly.

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
