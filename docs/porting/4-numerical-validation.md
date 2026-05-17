# Numerical Validation

Numerical validation is the tensor-by-tensor gate used during bring-up
and after backend or performance changes. The procedural workflow —
generating the oracle, converting, implementing C++ stage by stage,
finalizing tolerances — lives in the porting skills under
`.claude/skills/` (`porting-2-oracle`, `porting-3-convert`,
`porting-4-cpp`). This doc covers the conceptual contract: *what to
dump*, *why those points*, and *how to read a failure*.

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

**Preserve C++ intermediates before scheduler allocation.** If a C++
graph builder stores intermediate tensor pointers for a post-compute
dump pass, each stored tensor must be marked with
`transcribe::debug::mark_tensor_for_dump()` while the graph is being
built. The ggml scheduler can otherwise reuse intermediate buffers, so
the dumper may read a later tensor's data through an earlier tensor
handle. This usually shows up as unrelated dump files being byte-exact
equal or as huge early-stage tolerances that disappear when final
outputs are checked. Marking is a no-op unless `TRANSCRIBE_DUMP_DIR` is
enabled, so normal inference keeps live-range packing unchanged.

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

## Tensor Roles and Tolerances

Each tensor falls into one of three roles:

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

Tolerance derivation:
- Provisional tolerances are sized from per-tensor magnitude in
  `porting-2-oracle` (`1e-4 × p99_abs` for max, `1e-5 × rms` for mean,
  with a `1e-6` floor).
- Finalization happens in `porting-4-cpp` against observed C++ drift.
- Don't guess numbers in advance — run the comparison and read them.

Tolerance files live at `tests/tolerances/<family>.json` and include
a `_comment` documenting the reference framework, the dtype, known
divergence causes, and the attenuation profile observed.

## Troubleshooting

Failure diagnosis lives in
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
