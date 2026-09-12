# Granite Speech 5.0 CTC (TurboCTC)

Status: bring-up

## Identity

- Family key: `granite5_ctc`
- Upstream architecture string: `granite_speech5_ctc` (`GraniteSpeech5ForCTC`); encoder `granite_speech5_encoder` (`GraniteSpeech5Encoder`)
- Hugging Face repo: `ibm-granite/granite-speech-5.0-470m-turboctc` (pinned `18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0`)
- Hugging Face revision: `18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0`
- License: Apache-2.0
- Variants:
  - `granite-speech-5.0-470m-turboctc`: 473 M params, English-only, encoder + CTC head, non-autoregressive greedy decode.

Kept out of the existing `granite` family: that family is `audio-llm` (Conformer
encoder plus a BLIP-2 Q-Former projector plus a Granite-4.0 LLM decoder), while
this model is a bare Conformer encoder with a tied CTC head and no LLM at all.
Mainline transformers agrees with that split -- `modular_granite_speech5.py`
derives from Parakeet and Moonshine-Streaming, not from `granite_speech`. The
`5` in the key marks the Granite Speech 5.0 generation so a future Granite 5.0
autoregressive model can take its own key without a rename.

## References

- Canonical reference: mainline `transformers >= 5.16.0` (`transformers.models.granite_speech5`). No `trust_remote_code`: `config.json` carries no `auto_map`, so `AutoModelForCTC.from_pretrained` resolves to the in-tree classes. Pin `5.17.0` (current on PyPI) in `scripts/envs/granite5_ctc/` at Stage 2.
- Instrumented reference: same as canonical.
- Cross-check references:
  - `src/arch/granite/encoder.cpp` + `src/granite_conformer/shaw_attn.h`: the block-local Shaw relative-position attention is the same scheme (different `context_size`).
  - `src/conformer/conformer.{h,cpp}` (`build_conformer_block`, `ConvNormType::BatchNorm`): the macaron conformer block used by `parakeet`.
  - `transformers.models.parakeet`: `GraniteSpeech5EncoderBlock` and `GraniteSpeech5ForCTC` inherit from `ParakeetEncoderBlock` / `ParakeetForCTC`; `GraniteSpeech5EncoderAttention` inherits from `MoonshineStreamingEncoderAttention`.
  - The in-repo `granite_encoder.py` / `modeling_ctc_conformer.py` / `processing_ctc_conformer.py` files are a **stale alternate packaging** (`model_type: ctc_conformer`) using the granite-speech-4.x tensor names (`to_q`/`to_kv`, `ff1`, `depth_conv`, `batch_norm`). They cannot load the shipped `model.safetensors`. Read them as prose documentation of the training-time model; do not treat them as the reference.

## Architecture

Pattern: `encoder-ctc`. Conformer encoder -> tied CTC head -> greedy collapse. No
decoder, no projector, no LLM, no autoregressive loop.

- **Encoder** (`GraniteSpeech5Encoder`): `num_hidden_layers=16`, `hidden_size=1024`,
  `num_attention_heads=8`, `head_dim=128` (`num_key_value_heads=8`, so no GQA),
  `intermediate_size=4096`, `hidden_act=silu`, `conv_kernel_size=7`,
  `conv_expansion_factor=2` (conv inner dim 2048), `context_size=128`,
  `max_position_embeddings=512`, `attention_bias=true`.
  - `input_linear`: `Linear(320 -> 1024)`, bias. 320 = 80 mels x 2 (deltas) x 2 (frame stacking).
  - Block: macaron conformer. `x += 0.5*ff1(norm_ff1(x))`; `x += attn(norm_attn(x))`;
    `x += conv(norm_conv(x))`; `x += 0.5*ff2(norm_ff2(x))`; `x = norm_out(x)`.
    All five norms are LayerNorm; the conv module's internal norm is **BatchNorm1d**.
  - FFN: `linear1(1024->4096) -> SiLU -> linear2(4096->1024)`, both **with bias**
    (this is what `attention_bias=true` actually controls here).
  - Attention: `q_proj`/`k_proj`/`v_proj` bias-free, `o_proj` **with bias**.
    Block-local over `context_size=128` frames; the sequence is right-padded to a
    whole number of blocks and the pad columns are `-inf`-masked. Shaw relative
    positions via a per-layer `rel_pos_emb` table of `[2*512+1, 128] = [1025, 128]`,
    indexed by `clamp(i - j, ±128) + 512`; bias `= q @ (rel * head_dim^-0.5)^T`,
    added to the scaled `q@k^T`.
  - Conv module: `pointwise_lin1(1024 -> 4096)` -> GLU over the channel dim ->
    (mask pad frames to 0) -> depthwise `Conv1d(2048, 2048, k=7, groups=2048, bias=False, padding=3)`
    -> BatchNorm1d(2048) -> SiLU -> `pointwise_lin2(2048 -> 1024)`. Pointwise stages
    are `nn.Linear` in `(B, T, C)`, not `Conv1d`.
  - **Subsampling blocks** (`subsample_layers=[0, 1]`): blocks 0 and 1 halve time.
    ff1 and attention run at the input rate; then the conv module's depthwise conv
    uses `stride=2`, the residual is mean-pooled over non-overlapping pairs
    (`unfold(1,2,2).mean(-1)`, dropping a trailing odd frame), and the conv output is
    **trimmed** to the pooled length: `h = pooled + conv_out[:, :pooled.shape[1]]`.
    ff2 and `norm_out` then run at the halved rate. The padding mask is halved with
    AND semantics after each subsampling block.
  - **Self-conditioned CTC**: after block index 7 (`num_hidden_layers // 2`, 0-based),
    `h += out_mid(softmax(out(h)))`, where `out: Linear(1024 -> 16384)` and
    `out_mid: Linear(16384 -> 1024)`. This runs at **inference**, not only training.
- **CTC head**: `ctc_head: Linear(1024 -> 16384)`, **tied** to `encoder.out`
  (`tie_word_embeddings=true`, `_tied_weights_keys` maps `ctc_head.{weight,bias} -> encoder.out.{weight,bias}`).
  The checkpoint contains no `ctc_head.*` tensors.
- **Frame rates**: mel 100 Hz -> frame stacking x2 -> 50 Hz at block 0 -> 25 Hz at block 1
  -> **12.5 Hz** for blocks 2..15 and the CTC output. Total downsample 8x. One
  128-frame attention block therefore spans 2.56 s, 5.12 s, then 10.24 s of audio
  depending on the layer.
- **Decode**: argmax over 16384 logits per frame -> collapse repeats -> drop blank
  (id 0) -> tokenizer-decode the survivors. The model id space is 1:1 with the
  tokenizer, so there is no offset and no special-token block to strip.

## Frontend

`GraniteSpeech5FeatureExtractor`. New to this repo; shares nothing with the
granite-4.x extractor except the normalization formula.

- `sample_rate=16000`, mono, `n_mels=80`, `hop_length=160`, `win_length=400`, `n_fft=512`.
  **`win_length != n_fft`**: the 400-sample Hann window is zero-padded to 512 inside
  the STFT. The intake schema's `frontend` object has no `win_length` field, so this
  value only lives here and in the GGUF KVs.
- `torchaudio.transforms.MelSpectrogram` at library defaults: `power=2.0`, Hann
  **periodic**, `center=True`, `pad_mode=reflect`, `f_min=0`, `f_max=8000`,
  `mel_scale=htk`, `norm=None` (**unnormalized** triangular filters, not Slaney).
  None of these are declared in the checkpoint; Stage 2 must confirm them against the
  reference extractor.
- Log: `clamp_min(1e-10)` then **log10** (base 10, not natural log).
- Normalization (`per_utterance`): floor at `global_max - 8.0` where `global_max` is
  the maximum over **both** time and mel bins of the whole utterance, then the fixed
  affine `x/4 + 1`. This is bit-for-bit `src/transcribe-mel.cpp`'s existing
  `per_utterance` mode (`(x + 4) / 4`), which granite-4.x already requires. It is a
  dynamic-range floor, **not** CMVN.
- Deltas: `torchaudio.functional.compute_deltas(win_length=3)` concatenated along the
  channel axis, 80 -> 160.
- Frame stacking: transpose, then reshape to pack 2 consecutive frames, 160 -> 320.
- **Framing rule**: `mel_frames = num_samples // hop_length`;
  `num_frames = 2 * ceil(mel_frames / 2)`; the **waveform** is right-padded with zeros
  to `(num_frames - 1) * hop_length + 1` samples; the melspec is sliced to
  `[..., :num_frames]`. With `center=True` torchaudio emits `mel_frames + 1` frames, so
  for even `mel_frames` this drops exactly one trailing frame (matching the existing
  `per_utterance` `n_out = n_frames - 1`), and for odd `mel_frames` it **keeps** the
  trailing center-pad frame and pads the waveform to fill the stacking pair. granite-4.x
  does the opposite (drops the odd frame), so its framing code must not be reused.
  `global_max` is taken **after** the slice.
- Attention mask marks `ceil(mel_frames / 2)` encoder frames valid, which **equals** the
  emitted frame count, so the trailing partial group is valid, not masked. Verified at
  Stage 2: `valid=550/550` on jfk (even) and `valid=1767/1767` on dots (odd). The mask
  does real work only across a batch of uneven lengths.

## Tokenizer

GPT-2-style byte-level BPE, 16384 entries, no normalizer, `ByteLevel` pre-tokenizer
(`add_prefix_space=false`) and `ByteLevel` decoder (`add_prefix_space=true`). The only
added token is `<|blank|>` at id 0, which is simultaneously the CTC blank and
`pad_token_id`. `tokenizer_config.json` declares `tokenizer_class=ParakeetTokenizer`,
but the artifact is byte-level BPE; spaces come from the byte-level round-trip, not
from SentencePiece detokenization. `vocab_sha256`:
`3a2c2ef57ee23c9ecbe72f220b9c1e838783df49a159082c39f9ed91c32e7fde`.

## Dtype

`bfloat16` (`config.dtype`), confirmed by the safetensors header: 502 BF16, 32 F32,
16 I64. The F32 tensors are the 16 BatchNorm `running_mean`/`running_var` pairs; the
I64 are `num_batches_tracked` (inference-irrelevant, drop at convert). Note that
transformers sets `_keep_in_fp32_modules_strict = ["conv.norm"]`, so the reference
upcasts the **entire** BatchNorm module (including the BF16-stored affine
`weight`/`bias`) to F32 at load, so the reference forward is not pure BF16 at the conv
norm.

Stage 3 revised the Stage 1 note that said to fold BatchNorm at convert time: the
converter emits `conv.bn.{weight,bias,running_mean,running_var}` verbatim and the
**loader** folds them into an affine scale+shift, which is what every other Conformer
family here does (`conv_bn_fused_scale` / `conv_bn_fused_bias` in
`src/conformer/conformer.h`) and what `policy.cpp` assumes when it pins the running
stats to F32 "for the at-load BN-fusion math". Folding at convert would also cost
Stage 4 the ability to diff the BN parameters against the reference dumps.

## Oracle (Stage 2)

Oracle dtype is **F32**, not the BF16 the weights ship as. The BF16 to F32 upcast is
lossless, so these are the shipped weights unchanged, and F32 activations over BF16
weights is exactly the transcribe.cpp compute regime. Measured on jfk: a BF16 forward
gives an identical transcript and identical token ids, but tensors differ from F32 by up
to 2-3 % of `p99_abs` (worst: `enc.block.1.post_ff2`, max |diff| = 0.44), roughly 250x
the `1e-4 x p99_abs` tolerance budget. A BF16 oracle would have forced blind tolerance
widening at Stage 4. Do not re-dump at bf16.

Dump cases and the frame cascade they cover:

| Case | Duration | mel_frames | stacked | block 0 out | block 1 out |
|------|----------|-----------|---------|-------------|-------------|
| `jfk` | 11.00 s | 1100 (even) | 550 | 275 (even in, no drop) | 137 (odd in, drop) |
| `dots` | 35.33 s | 3533 (odd) | 1767 | 883 (odd in, drop) | 441 (odd in, drop) |

Together they cover both residual-pooling parities at both subsampling blocks, which one
case cannot. The pooled residual is `unfold(1,2,2).mean(-1)`, which silently drops a
trailing odd frame, and the conv output is then trimmed to that length; an implementation
that rounds the other way passes jfk block 0 and fails everywhere else.

Hook identities verified bit-exactly at Stage 2 (max |diff| = 0.0 on jfk). These are the
Stage 4 sub-step contract:

- `encoder.out(enc.block.7.out)` == `enc.ctc.mid_logits`
- `encoder.out_mid(softmax(enc.ctc.mid_logits))` == `enc.ctc.mid_injection`
- `ctc_head(enc.out)` == `enc.ctc_logits`
- `ctc_head.weight/bias` share torch storage with `encoder.out` (identical `data_ptr`),
  confirming `tie_word_embeddings=true`. The converter emits that `[16384, 1024]` matrix
  once and points both consumers at it.

`enc.block.7.out` is captured **before** the mid-injection, so block 8's input equals
`enc.block.7.out + enc.ctc.mid_injection`.

**Reference WER baseline (the Stage 4 / Stage 7 gate):** 1.33 % on LibriSpeech
test-clean, 95 % CI [1.20 %, 1.47 %], 2620 utterances, 0 errors (527 sub / 87 del /
92 ins). Scored with `EnglishTextNormalizer`. The publisher reports no LibriSpeech
number, so there is nothing to compare it against; the gate is this measured run, not a
published score. Artifacts:
`reports/wer/granite-speech-5.0-470m-turboctc-REF.test-clean.{jsonl,score.json}`.

## Conversion (Stage 3)

Converter: `scripts/convert-granite5_ctc.py`. Reference dtype **BF16** (the shipped
storage dtype), emitted as
`models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf`
(947,824,480 bytes, 520 tensors: 341 F32, 131 BF16, 48 F16). Provenance, SHA-256, and the source
revision are recorded in
`reports/convert/granite-speech-5.0-470m-turboctc-BF16.json`.

GGUF architecture string: `granite_speech5_ctc` (the upstream `model_type`), matching
the `granite_speech` / `granite_speech_nar` convention where the arch string names the
upstream architecture rather than our family key.

Per-tensor storage dtype comes from `gguf_common.reference_dtype_for`, the Python
mirror of `tools/transcribe-quantize/policy.cpp::classify_tensor`: norms, biases,
BatchNorm running stats and the frontend buffers are pinned F32; the three conv kernels
per block downcast to F16 (the loader has no BF16 conv kernel); everything else stays
BF16. This family needed **no new rule** in either copy. Representative names are
registered in `scripts/lib/test_quant_policy_sync.py` so the two copies cannot drift
apart here unnoticed.

### Tensor map

| HF checkpoint | GGUF | Storage |
|---|---|---|
| `encoder.input_linear.{weight,bias}` | `enc.input_linear.{weight,bias}` | BF16 / F32 |
| `encoder.out.{weight,bias}` | `enc.ctc_proj.{weight,bias}` | BF16 / F32 |
| `encoder.out_mid.{weight,bias}` | `enc.ctc_bypass.{weight,bias}` | BF16 / F32 |
| `encoder.layers.{i}.norm_feed_forward1.*` | `enc.blocks.{i}.norm_ff1.*` | F32 |
| `encoder.layers.{i}.feed_forward1.linear{1,2}.*` | `enc.blocks.{i}.ff1.linear{1,2}.*` | BF16 / F32 |
| `encoder.layers.{i}.norm_self_att.*` | `enc.blocks.{i}.norm_attn.*` | F32 |
| `encoder.layers.{i}.self_attn.q_proj.weight` | `enc.blocks.{i}.attn.q.weight` | BF16 |
| `encoder.layers.{i}.self_attn.{k,v}_proj.weight` | `enc.blocks.{i}.attn.kv.weight` (fused) | BF16 |
| `encoder.layers.{i}.self_attn.o_proj.{weight,bias}` | `enc.blocks.{i}.attn.out.{weight,bias}` | BF16 / F32 |
| `encoder.layers.{i}.self_attn.rel_pos_emb.weight` | `enc.blocks.{i}.attn.rel_pos_emb.weight` | BF16 |
| `encoder.layers.{i}.norm_conv.*` | `enc.blocks.{i}.norm_conv.*` | F32 |
| `encoder.layers.{i}.conv.pointwise_lin{1,2}.*` | `enc.blocks.{i}.conv.pointwise{1,2}.*` | F16 / F32 |
| `encoder.layers.{i}.conv.depthwise_conv.weight` | `enc.blocks.{i}.conv.depthwise.weight` | F16 |
| `encoder.layers.{i}.conv.norm.{weight,bias}` | `enc.blocks.{i}.conv.bn.{weight,bias}` | F32 |
| `encoder.layers.{i}.conv.norm.running_{mean,var}` | `enc.blocks.{i}.conv.bn.running_{mean,var}` | F32 |
| `encoder.layers.{i}.norm_feed_forward2.*` | `enc.blocks.{i}.norm_ff2.*` | F32 |
| `encoder.layers.{i}.feed_forward2.linear{1,2}.*` | `enc.blocks.{i}.ff2.linear{1,2}.*` | BF16 / F32 |
| `encoder.layers.{i}.norm_out.*` | `enc.blocks.{i}.norm_out.*` | F32 |
| `encoder.layers.{i}.conv.norm.num_batches_tracked` | *(dropped)* | — |
| *(baked from torchaudio)* | `frontend.mel_filterbank`, `frontend.window` | F32 |

### Mapping decisions

1. **`k_proj` and `v_proj` are concatenated into `attn.kv`** (`[hidden, 2*inner]`,
   K rows first, V second). `src/granite_conformer/shaw_attn.h::ShawAttnWeights` takes a
   fused `attn_kv_w` because granite 4.x's checkpoint stores a fused `to_kv`; granite 5.0
   splits it. Fusing at convert time makes the shared Shaw helper reusable verbatim,
   which is the largest single Stage 4 lever in this port. The transform is reversible
   and was verified exact: rows `[0, inner)` equal `k_proj`, rows `[inner, 2*inner)`
   equal `v_proj`, max |diff| 0.0 on blocks 0 / 9 / 15.
2. **Block tensor names follow granite 4.x** (`convert-granite.py`) — `attn.q` /
   `attn.kv` / `attn.out` / `attn.rel_pos_emb`, `conv.pointwise{1,2}` /
   `conv.depthwise` / `conv.bn.*` — because Stage 4 forks that encoder and the shared
   Shaw helper. Two names instead follow this checkpoint's own vocabulary, where
   granite 4.x diverges from it: `ff{1,2}.linear{1,2}` (granite 4.x says `ff*.up` /
   `.down`) and `norm_out` (granite 4.x says `norm_post`). Both also match
   `src/conformer/conformer.h`'s `BlockView` field names, and `norm_out` avoids
   colliding conceptually with `encoder.out`, which is a different tensor entirely.
3. **`conv.pointwise_lin{1,2}` are `nn.Linear` here, not `Conv1d`** — 2-D
   `[out, in]`, no trailing kernel axis, unlike NeMo's conformer. They are the same
   1x1 operator, so they keep the `conv.*` names, but their 2-D layout routes them to
   the Linear quant bucket. The loader correspondingly uses `GET_LIN`; Stage 4 must
   not expect granite 4.x's 3-D `[1, in, out]` layout for these two slots.
4. **`num_batches_tracked` is dropped** (16 I64 scalars, training-only counters).
5. **The frontend buffers are baked from torchaudio, not librosa.** The reference
   front-end *is* `torchaudio.transforms.MelSpectrogram`, so
   `torchaudio.functional.melscale_fbanks(..., norm=None, mel_scale="htk")` and
   `torch.hann_window(400, periodic=True)` are the exact buffers it uses internally.
   Verified bit-exact (max |diff| 0.0) against a live `MelSpectrogram`'s
   `mel_scale.fb` and `spectrogram.window`. The sibling granite converters use
   librosa for the same buffers; prefer torchaudio wherever the reference is
   torchaudio.

The converter cross-checks the stacked encoder width three independent ways and
refuses to run if they disagree: mainline `processor_config.json[feature_size]`, the
derived `n_mels x (1 + deltas) x stack_factor`, and the actual shape of
`encoder.input_linear.weight`. This matters because the repo ships two front-end
config files (mainline `processor_config.json` and the stale `trust_remote_code`
`preprocessor_config.json`) and only the stale one spells `stack_factor` / `deltas`.

### Gate B

`preflight --gate B` is PASS on `dtype_consistency`, `tokenizer_alignment`, and
`capabilities`, with two expected WARNs and no FAIL:

- `frontend_config` — the same `declared.normalization=per_utterance` vs
  `reference.normalization=none` artifact carried over from Gate A. The GGUF and the
  intake agree (`per_utterance`); the "reference" side is preflight defaulting an
  absent `normalize` key to `none`.
- `architecture_sanity` — `granite_speech5_ctc` does not contain the family key
  `granite5_ctc`. Soft by design: preflight warns rather than fails when a family key
  differs from the upstream architecture string. `granite_nar` carries the identical
  warn (`granite_speech_nar` vs `granite_nar`).

Loader-open smoke returns `unsupported architecture`, which is the expected Stage 3
outcome for a family whose `src/arch/granite5_ctc/` does not exist yet — the GGUF
header and KV block parsed, and dispatch is what failed. Stage 4 brings up the arch.

## Commands

Reference run (Stage 2):

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/wer/run_reference_granite5_ctc_transformers.py \
    --model ibm-granite/granite-speech-5.0-470m-turboctc \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/granite-speech-5.0-470m-turboctc-REF.test-clean.jsonl
```

Reference dumps (Stage 2):

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/dump_reference_granite5_ctc_transformers.py encoder \
    --model ibm-granite/granite-speech-5.0-470m-turboctc \
    --audio samples/jfk.wav \
    --out build/validate/granite5_ctc/granite-speech-5.0-470m-turboctc/jfk/encoder/ref
```

Conversion (Stage 3, preserves source bf16):

```bash
uv run --project scripts/envs/granite5_ctc \
  scripts/convert-granite5_ctc.py ibm-granite/granite-speech-5.0-470m-turboctc \
  --repo-id ibm-granite/granite-speech-5.0-470m-turboctc
```

Validation (Stage 4):

```bash
uv run scripts/validate.py all --family granite5_ctc --variant granite-speech-5.0-470m-turboctc
```

Benchmarks (Stage 6):

```bash
uv run scripts/perf/bench.py \
  --model models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  --backend metal \
  --machine "$(uname -srm | tr ' ' _)"
```

Full WER sweep (Stage 7):

```bash
for PRESET in BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  uv run scripts/wer/run.py \
    --model models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-${PRESET}.gguf \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/granite-speech-5.0-470m-turboctc-${PRESET}.test-clean.jsonl
  uv run scripts/wer/score.py reports/wer/granite-speech-5.0-470m-turboctc-${PRESET}.test-clean.jsonl
done
```

## C++ implementation (Stage 4)

`src/arch/granite5_ctc/` — `weights`, `encoder`, `decoder`, `model`,
`capabilities`, plus the family header `granite5_ctc.h`. GGUF architecture
string `granite_speech5_ctc`. Forward map:
`reports/porting/granite5_ctc/forward-map.md`.

Reused rather than reimplemented: `src/granite_conformer/shaw_attn.h`
verbatim (the converter's fused `attn.kv` is what makes that possible),
`conformer::macaron_ff_residual`, `conformer::fused_batch_norm`,
`conformer::conv_2d_dw_direct_f32`, and `transcribe::MelFrontend`.

### Three things that were not reusable

1. **Frame-count override on the frontend.** The reference emits
   `2*ceil(mel_frames/2)` mel frames, which equals `n_frames-1` when
   `mel_frames` is even and `n_frames` when it is odd — the per_utterance
   mode's own rule (always drop the trailing centre-pad frame) is right in
   the first case and wrong in the second. Since the count is a
   per-utterance property it cannot live in `MelConfig`, so
   `MelFrontend::compute` grew a trailing `out_frames` argument (0 = each
   mode's existing behaviour; no other family passes it). The
   per-utterance max is taken over exactly the emitted frames, matching
   `mel[..., :num_frames].amax()`.
2. **Three attention geometries per forward.** `T` changes at blocks 0 and
   1, so the block-local pad mask and zero-pad tile differ per stage.
   `EncoderBuild` carries one `AttnStage` per distinct length (3 here) and
   a `block -> stage` index. granite 4.x builds exactly one.
3. **In-block subsampling.** `pooled = mean of frame pairs` (built from two
   strided views plus an add, so no pooling kernel is needed and every
   backend works), then `x = pooled + conv_out[:len(pooled)]`. The stride-2
   conv emits `ceil(T/2)` and the pooled residual `floor(T/2)`, so the conv
   output is trimmed down — never the reverse.
### Batch path

`run_batch()` is a real parallel path: per-utterance frontends run on a
shared thread budget, then ONE encoder dispatch over the utterance batch on
the activation's `ne[2]`. Same-length batches take the mask-free graph and
are bit-identical to single-shot. Uneven batches pad to `T_max` and apply
the reference's three masks — zeroed pad frames after `input_linear`,
masked pad key columns in every block-local attention, and zeroed pad
frames before each depthwise conv. The third one is load-bearing: without
it an utterance's own padding reaches its real frames through the kernel-7
receptive field. Per-utterance valid lengths halve with `floor` at each
subsampling block, matching `downsample_attention_mask`'s AND-over-pairs
on a prefix mask.

The block-local pad mask uses a large finite negative (`-1e30`) rather than
`-INF`, mirroring the reference's `masked_fill(..., finfo.min)`. It only
matters in a batch, where a short utterance can leave a whole 128-frame
block with no real keys: `-INF` there makes softmax produce NaN, which then
survives the multiplicative conv mask (`0 * NaN = NaN`) and poisons real
frames. Wherever `-INF` was well defined the two agree exactly.

Varied-length batches are NOT bit-exact against single-shot (~1e-2 on the
logits, text still byte-identical). Localized: the divergence is a single
frame — the one whose depthwise-conv window straddles the end of the
sequence — and it then spreads through block-local attention. Masked pad
slots and the depthwise op's own zero padding are algebraically identical
(verified by a host-side replay of that frame: masked and truncated inputs
give a bit-identical conv output), so this is ggml choosing a different
kernel for `OW = 82` vs `OW = 261`, not a masking error. The magnitudes
agree: an actual mask failure would leak O(10) values, and a host replay
with masking disabled differs by 11.3 versus the 4.9e-04 observed. The
repo-standard tensor-parity gate (`--wav ... --batch 4`, same length) is
bit-exact.

### Input / memory contract

Unbounded, and the family says so rather than pretending otherwise. Block-
local attention costs `O(T * context_size)`, not `O(T^2)`, and there is no
decoder context to exhaust, so memory is linear in audio length. Nothing is
rejected up front and nothing is truncated; `n_ctx` reaches no graph and is
ignored (recorded on the session only so `transcribe_session_get_limits`
can report it). The only floor is the frontend's: a clip must survive both
subsampling blocks with at least one frame left, which is reported as
`TRANSCRIBE_ERR_INVALID_ARG` with the clip duration in the message.
Allocation failures return `TRANSCRIBE_ERR_OOM`, never a silently shrunk
context.

### Reference-dtype WER gate (Stage 4)

Full LibriSpeech test-clean, 2620 utterances, BF16 GGUF on Metal.

| Run | WER | 95% CI | Sub / Del / Ins | Errors |
|---|---|---|---|---|
| Oracle reference (Stage 2, transformers F32) | 1.33% | [1.20%, 1.47%] | 527 / 87 / 92 | 0 |
| C++ BF16, batch 1 | 1.33% | [1.20%, 1.47%] | 527 / 87 / 92 | 0 |
| C++ BF16, batch 8 | 1.33% | [1.20%, 1.47%] | 527 / 87 / 92 | 0 |

Gate (C++ batch-1 <= Oracle + 0.01pp): **PASS at 0.00pp**.

Two stronger statements than the gate requires:

- **2619 of 2620 C++ hypotheses are byte-identical to the reference.** The one
  that differs is `6930-81414-0001`, `kaffir` vs `kaafir` — a rare proper noun
  where a borderline CTC argmax flipped under BF16 rounding. Both spellings are
  wrong against the transcript, so the error counts are unchanged.
- **Batch 8 is byte-identical to batch 1 on all 2620 utterances.** test-clean has
  mixed lengths, so this run exercises the padded-batch masking end to end; the
  ~1e-2 logit differences that path introduces never flip an argmax.

Artifacts: `reports/wer/granite-speech-5.0-470m-turboctc-BF16.test-clean.{b1,b8}.{jsonl,score.json}`.

### Encoder performance work

Two changes landed after the first Stage 4 pass, both gated at the full
Stage 4 bar. M4, `transcribe-bench`, jfk (11 s), Metal, F32 GGUF:

| build | encode | wall | vs parakeet-tdt-0.6b-v2 F32 |
|---|---|---|---|
| first Stage 4 pass | 114.9 ms | 120.6 ms | encode 6% slower |
| + direct depthwise | 101.4 ms | 107.2 ms | encode 6% faster |
| + skew positional bias | 96.0 ms | 102.2 ms | **encode 11% faster, wall 18% faster** |

(parakeet-tdt-0.6b-v2 F32 on the same host: encode 108.2 ms, wall 125.3 ms.)
The same 16% encode saving shows up at BF16 (109.6 -> 91.8 ms) and on the
35 s case (306.5 -> 256.7 ms). CPU barely moves (930 -> 914 ms); see the
BF16 threading finding below for why.

Where the time went before the work, measured by ablation on Metal: block-
local attention 34%, conv module 30%, the two macaron FFNs 30%, mid-CTC
self-conditioning 3.7%. The FFNs were already at roughly M4 F32 peak
(~2.7 TFLOPS) while attention and conv ran at ~500-680 GFLOPS on 31% of
the arithmetic, so both changes target throughput, not arithmetic.

1. **Depthwise conv: direct, not im2col.** `conformer::conv_2d_dw_direct_f32`
   replaced `conv_2d_dw_f32`. im2col inflates the activation by
   `conv_kernel` to feed a matmul whose K is that same `conv_kernel`, which
   is nearly all memory traffic; the direct op skips the fold. The
   batch-size-stability requirement that ruled out `conv_1d_dw_f32` still
   holds, because the direct op is taken at every batch size.

   This is NOT bitwise equal to the im2col form, and it pushed
   `enc.ctc_logits` on the dots case from 6.150e-02 to 1.513e-01, past its
   old budget. Localized before it was absorbed: against an F32-tier GGUF
   and the F32 oracle the direct build matches at the same 1e-5-relative
   level as im2col on both cases and is marginally *better* on this very
   tensor (dots 8.106e-05 vs 8.202e-05); `mean_abs` improved too. Only the
   `max_abs` order statistic over 441x16384 values moved. The tolerance
   entry was re-finalized to 0.227 with that evidence recorded in
   `tests/tolerances/granite5_ctc.json`.

2. **Shaw positional bias: skew, not a ctx*ctx lookup.** `shaw_block_attn`
   grew an optional `pos_rows` argument. The old path materializes a
   `[head_dim, ctx, ctx]` table per layer (8 MB at ctx=128) and feeds a
   batched GEMM with N = n_heads = 8. The new path gathers only the
   `2*ctx - 1` distinct relative offsets (130 KB), does one fat GEMM
   against q, and rotates the result into key columns with the existing
   `conformer::rel_shift`. About 2x the multiply-adds at a far better GEMM
   shape, and it lands already in `kq`'s axis order so two permute+cont
   round trips disappear.

   Valid because every entry of `compute_attention_dists` is a function of
   `(query - key)` alone, clamped or not. The identity was checked
   host-side in numpy before any C++ was written. granite 4.x still passes
   `pos_rows = nullptr` and takes the old branch bit-unchanged.

Gates re-run on both changes: `validate.py all` exit 0 (35/35 both cases),
batch text parity vs the frozen golden at 2/4/8, batch tensor parity
bit-exact (max_abs 0.0), 59/59 ctest, full test-clean 1.33% at batch 1 and
batch 8 with the identical 527/87/92 split. The skew build is byte-identical
to the F32 reference on all 2620 utterances (the first Stage 4 pass differed
on one, `6930-81414-0001`).

Still on the table, not done: the conv module's elementwise chain is
21.3 ms for 14.7 GFLOP (~690 GFLOPS) after change 1, with roughly eight
passes (LN, GLU+sigmoid, fused BN, SiLU, two transposes, masks) between
pointwise matmuls that should run at FFN speed.

### Known perf finding (for Stage 6)

The CPU BF16 matmul path does not thread: encode is 8.9 s at 1 thread and
9.4 s at 8 on an M4 for 11 s of audio. The same graph on an F32 GGUF scales
6.9 s -> 2.4 s, so the graph parallelizes and the BF16 kernel is the
bottleneck. Metal is unaffected (247 ms, RTF ~44x), which is what the WER
gates ran on. Not a Stage 4 blocker; recorded here so Stage 6 does not
rediscover it.

## Quant matrix (Stage 5)

`uv run scripts/quantize-all.py models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf`

| preset | size | notes |
|--------|------|-------|
| BF16 | 947.8 MB | reference tier, the Stage 4 validated artifact |
| F16 | 948.3 MB | |
| Q8_0 | 505.6 MB | |
| Q6_K | 391.8 MB | |
| Q5_K_M | 335.7 MB | |
| Q4_K_M | 279.1 MB | smallest shippable |

All six load and emit a plausible `text:` on `samples/jfk.wav`. Published to
the private `handy-computer/granite-speech-5.0-470m-turboctc-gguf`.

### Tentative per-quant WER (Stage 5; Stage 7 is authoritative)

Full LibriSpeech test-clean (2620 utterances), Modal L4 CUDA, batch 1, 0
errors in every cell. Oracle reference (Stage 2, transformers F32) is 1.33%.

| preset | size | WER | word errors | vs BF16 | hyps differing from BF16 | RTF |
|--------|------|-----|-------------|---------|--------------------------|-----|
| BF16 | 947.8 MB | 1.33% | 707 (528 S / 87 D / 92 I) | — | — | 163.9x |
| F16 | 948.3 MB | 1.33% | 705 | -2 | 2 / 2620 | 92.2x |
| Q8_0 | 505.6 MB | 1.33% | 706 | -1 | 15 / 2620 | 100.3x |
| Q6_K | 391.8 MB | 1.33% | 707 | +0 | 32 / 2620 | 106.4x |
| Q5_K_M | 335.7 MB | 1.34% | 708 | +1 | 56 / 2620 | 102.9x |
| Q4_K_M | 279.1 MB | 1.34% | 711 | +4 | 121 / 2620 | 103.3x |

Every tier sits inside the BF16 95% CI of [1.20, 1.47]. The whole matrix
spans 4 word errors out of ~53k words, so the ordering between tiers is not
statistically meaningful here — only the magnitude is: this model quantizes
essentially for free down to Q4_K_M.

That is also the answer to the risk the pointwise policy change introduced.
Those 32 tensors are 201 MB and had never been block-quantized before; at
Q4_K_M they now carry Q4_K weights, and the cost is +4 errors. The
hypothesis-divergence column shows the change is real (121 utterances decode
differently at Q4_K_M) without being harmful.

The RTF column is `rtf_wall` from the sweep and is not a perf measurement:
BF16 reads 163.9x while every other tier clusters near 100x, which no weight
format explains on its own (BF16 is the *largest* file and the fastest cell).
It is contaminated by container warm-up and download time. Stage 6 measures
throughput properly from the stage counters; ignore these numbers.

### Quant policy change: pointwise convs are Linear, not ConvPw

The first cut came out at Q4_K_M 403.6 MB, of which **201 MB was F16**.
`policy.cpp::classify_tensor` routed the 32
`enc.blocks.N.conv.pointwise{1,2}.weight` tensors to `Bucket::ConvPw`, which
every preset pins at F16. The comment justifying that pin said the cost was
"~2 MB across all of them" — true when parakeet was the only conformer in
tree, and off by two orders of magnitude for the granite conformer
(d_model 1024, conv expansion 4096).

They are not convs here. Upstream they are `nn.Linear`
(`conv.pointwise_lin{1,2}`, see the Tensor map), the converter stores them
2-D, and `encoder.cpp` runs them through plain `ggml_mul_mat`
(`x = ggml_mul_mat(ctx, b.conv_pointwise1_w, x)`). Two changes:

- `classify_tensor` now takes the tensor's `ne0`. A pointwise stored 2-D
  `[in, out]` is `Bucket::Linear`; only the PyTorch `Conv1d` layout
  `[1, in, out]` (`ne0 == 1`) stays `ConvPw`/F16, because no block quant can
  encode a 1-element row. The name-only overload keeps the conservative
  `ConvPw` answer.
- `weights.cpp` loads them with `GET_LIN` instead of `GET_CONV`, so the
  derived presets' block-quantized types pass the allowlist. The reference
  GGUF still stores them F16, which the linear allowlist already covers, so
  the BF16 artifact is unchanged and no re-conversion was needed.

Numerically a no-op on the reference tier: `validate.py all` stayed 35/35
with `enc.ctc_logits` at max_abs 1.513e-01, digit-identical to the pre-change
run. Sizes moved Q4_K_M 403.6 -> 279.1 MB (-31%), Q8_0 600.0 -> 505.6 MB.

`scripts/lib/test_quant_policy_sync.py` is unaffected: `reference_dtype_for`
has no shape argument and keeps the name-based Conv answer for both layouts,
which stays correct because F16 is a legal storage dtype for a Linear too.
The split only matters to the quantizer, which does see the shape.

**Not applied to granite 4.x.** Those variants store pointwise in the
`[1, in, out]` layout, so they would need a converter change, which changes
their reference BF16 GGUFs and forces re-convert + re-validate + re-quantize
+ republish across four already-public repos, for ~13% (202 MB off a ~1.6 GB
Q4_K_M, since the 2B LLM decoder dominates the file). Deferred past Stage 8
by user decision. The policy rule is shape-driven, so granite 4.x picks the
win up automatically once its converter emits 2-D.

## Benchmarks (Stage 6)

Reproduction (apple-m4, `build/`):

```
uv run scripts/bench/run.py --models granite-speech-5.0-470m-turboctc \
  --quants q8_0,q4_k_m --samples jfk,dots --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 --name granite-speech-5.0-470m-turboctc-publication
```

Publication scope, iters 3 / warmup 1. Vulkan is filtered out on this
machine (no `build-vulkan/`), so the driver's fallthrough reaches the
Metal-only binary and reports `model load: backend error` per cell; it
writes no report and does not affect sign-off.

| backend | quant | sample | encode ms | wall ms | RTF |
|---------|-------|--------|-----------|---------|-----|
| Metal | q8_0 | jfk (11.0 s) | 93.8 | 99.9 | 110.1x |
| Metal | q8_0 | dots (35.3 s) | 251.9 | 271.5 | 130.2x |
| Metal | q4_k_m | jfk | 96.7 | 103.3 | 106.5x |
| Metal | q4_k_m | dots | 259.6 | 279.1 | 126.6x |
| CPU | q8_0 | jfk | 381.8 | 387.5 | 28.4x |
| CPU | q8_0 | dots | 1169.4 | 1186.6 | 29.8x |
| CPU | q4_k_m | jfk | 417.6 | 423.2 | 26.0x |
| CPU | q4_k_m | dots | 1254.9 | 1272.2 | 27.8x |

**Q8_0 is faster than Q4_K_M on both backends**, by 3% on Metal and 7-9% on
CPU, despite being 1.8x the file size. K-quant unpacking costs more than the
bandwidth it saves at this model size. Q8_0 is the better default
recommendation for this family on both accuracy (Stage 5: 1.33% vs 1.34%) and
speed; Q4_K_M earns its place only where the 279 MB footprint matters.

### Long audio (widening, non-gating)

`--samples dots-full` (305.9 s), same scope:

| backend | quant | encode ms | RTF |
|---------|-------|-----------|-----|
| Metal | q8_0 | 2172.8 | 131.1x |
| Metal | q4_k_m | 2227.2 | 128.3x |
| CPU | q8_0 | 8154.5 | 36.9x |
| CPU | q4_k_m | 9585.9 | 31.5x |

RTF is flat from 35 s to 306 s on Metal (130.2x -> 131.1x) and *improves* on
CPU (29.8x -> 36.9x). That is the block-local Shaw attention paying off: cost
is linear in T, so there is no quadratic term to erode RTF as audio grows.
This is the regime where the family beats a global-attention encoder, and the
publication samples (11 s / 35 s) understate it.

### Batch throughput (Step 4b, non-gating)

`build/bin/transcribe-batch-bench -m <F16> samples/jfk.wav --batch-sizes
1,2,4,8,16,32 --iters 3`, Metal, report at
`reports/perf/apple-m4/granite5-ctc-batch_granite-speech-5.0-470m-turboctc_batch_metal.json`:

| n_batch | 1 | 2 | 4 | 8 | 16 | 32 |
|---------|---|---|---|---|----|----|
| per-utt ms | 97.3 | 94.2 | 94.4 | 96.4 | 96.7 | 97.7 |

Essentially flat: best case is 3% at batch 2-4, and it regresses back to
batch-1 latency by 32. A single 11 s clip (441 encoder frames x 1024) already
saturates the GPU, so there is no idle width for batching to fill. This does
NOT downgrade the `Batch (offline)` capability row, which is PASS on its own
terms (a real `run_batch()` ships and is byte-identical to serial); it just
means the feature buys throughput only on CPU or for shorter clips.

## Capability Validation

`Target` is the Stage 1 scope decision (user-signed). `Status` is the Stage 4
observed outcome and is `TODO` until then. A `MUST PASS` row may not be
downgraded without the user re-signing.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | PASS — `and so my fellow americans ask not what your country can do for you ask what you can do for your country`, character-identical to the reference transcript |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf samples/jfk.wav` | same transcript as the hinted run (the model has no language conditioning, so `--language` is inert) | MUST PASS | PASS — byte-identical to the hinted run, as expected (`--language` reaches no branch) |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-BF16.gguf --list tests/golden/batch/granite-speech-5.0-470m-turboctc.list --batch-sizes 2,4,8 --backend cpu --golden-in tests/golden/batch/granite-speech-5.0-470m-turboctc.cpu.json`<br>`uv run scripts/batch_tensor_parity.py --model <same> --wav samples/jfk.wav --batch 4 --backend cpu --dump-name dec.ctc_logits` | byte-identical hypotheses + CPU tensor parity | MUST PASS | PASS — text byte-equal vs serial at 2/4/8 over 24 mixed-length utterances (golden `tests/golden/batch/granite-speech-5.0-470m-turboctc.cpu.json`, frozen over the 24 utterances in `...turboctc.list`; `--samples-dir samples/wer/test-clean` globs all 2620 and does NOT reproduce the golden); same-length CPU tensor parity bit-exact (max_abs=0.0) at batch=4; full test-clean batch-8 WER equals batch-1 |
| Language detection | auto-detect | not exercised | model is English-only; no detection branch and no language tokens in the 16384-entry vocab | OUT OF SCOPE — monolingual English model; would return in scope only if IBM ships a multilingual Granite 5.0 CTC variant | SKIP — not exposed by runtime |
| Translate | `--target-language` | not exercised | no translation head, no prompt surface, no non-English training data | OUT OF SCOPE — English-only ASR; would return in scope only with an upstream AST-capable variant | SKIP — not exposed by runtime |
| Segment timestamps | segment granularity | `build/bin/transcribe-cli -m <BF16> --timestamps segment samples/jfk.wav` | `TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS` | OUT OF SCOPE — no segmentation policy is defined for this family; would return in scope once a split rule (silence or max-duration) is chosen | SKIP — not exposed by runtime. `max_timestamp_kind` is `TRANSCRIBE_TIMESTAMPS_NONE`, so the request is rejected. One segment is still built internally to carry the text; it spans the clip bounds and makes no timing claim |
| Word timestamps | word granularity | `build/bin/transcribe-cli -m <BF16> --timestamps word samples/jfk.wav` | `TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS` | OUT OF SCOPE — **re-signed by CJ 2026-09-12**, down from MUST PASS at intake. An implementation existed and passed the structural check, but its word END times were wrong (see below) and there is no reference alignment to validate against, so it was stripped rather than shipped. Returns in scope with a real word-end rule plus something to check it against | SKIP — not exposed by runtime; the word-building code was removed from `decoder.cpp` and `max_timestamp_kind` left at `TRANSCRIBE_TIMESTAMPS_NONE` |
| Streaming | `--stream-chunk-ms` | not exercised | `capabilities.streaming: false`; and the `per_utterance` log-mel floor depends on an utterance-global maximum, so chunked decoding changes the feature values | OUT OF SCOPE — non-streaming model with a non-causal frontend; would return in scope only with a fixed-max ("global") frontend variant plus an upstream streaming recipe | SKIP — not exposed by runtime |
| Speaker diarization | multi-speaker | not exercised | single-speaker CTC output, no speaker head | OUT OF SCOPE — no diarizer in the architecture | SKIP — not exposed by runtime |

## Known Limitations

Drawn from the intake capability flags and what the port actually surfaced.
Nothing here is speculative; each item is either a declared upstream
capability the model does not have, or a measured result from Stages 4-7.

**English only.** Monolingual. The 16384-entry byte-level BPE vocab carries
no language tokens and the graph has no language-conditioning branch, so
`--language` reaches nothing and is inert. There is no language-detection
path to expose. A multilingual Granite 5.0 CTC variant would change this;
none is published.

**No translation.** No AST head, no prompt surface, no non-English training
data.

**No streaming.** `capabilities.streaming` is false upstream, and the
frontend makes it more than a missing feature: the `per_utterance` log-mel
floor is computed from an utterance-global maximum, so chunked decoding
changes the feature values themselves. Streaming needs a fixed-max
("global") frontend variant plus an upstream streaming recipe, not just a
chunking loop.

**No diarization.** Single-speaker CTC output, no speaker head.

**No timestamps of any granularity.** `max_timestamp_kind` is
`TRANSCRIBE_TIMESTAMPS_NONE`, so `--timestamps segment|word|token` all return
`TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS`. One segment is still constructed
internally to carry the transcript text, but it spans the clip bounds and
makes no timing claim, and token `t0_ms`/`t1_ms` are left at 0.

This is a **withdrawal**, not an omission. Word timestamps were implemented,
signed off as MUST PASS at intake, and passed the Stage 4 structural check
(present, monotonic, in bounds). They were stripped at Stage 8 because the
structural check was too weak to catch the actual defect: CTC is peaky, a
token occupies exactly the one frame where it wins, and `close_word` took a
word's end from its last token, so the end time was always `emitting frame +
one frame`. Nearly every word reported exactly 80 ms of duration regardless
of how long it took to say — `americans`, `impossible` and `backwards` all
measured one frame, and only genuinely multi-token words differed (`dots`
0.24 s, `10` 0.16 s). The onsets were sound; the durations were not.

Two things made shipping it the wrong call. Upstream neither advertises nor
emits timings, so there is no reference alignment to validate against, and a
wrong duration is worse than no duration for any consumer that trusts it.
Reinstating needs a real word-end rule (the next token's emitting frame, or
the last frame before the next word's onset) and a ground truth to check it
against. The same defect is worth checking for in parakeet's CTC variants,
which ship timestamps today.

**Batch (offline): PASS.** The family ships an explicit `run_batch()`
parallel fast path, and it is WER-neutral in the strongest sense available:
batch 8 is byte-identical to batch 1 on all 2620 test-clean utterances, and
CPU tensor parity at batch 4 is bit-exact (max_abs 0.0). This is not a
serial fallback. What Stage 6 measured is that the *throughput* win is small
on GPU: per-utterance latency on Metal goes 97.3 ms at batch 1 to a best of
94.2 ms at batch 2-4 and back to 97.7 ms at batch 32, because a single 11 s
clip already saturates the device. Batching is worth more on CPU and for
short clips.

**Streaming row is omitted from this section's posture list** because the
model does not stream natively; see the Capability Validation table, where
it is `SKIP - not exposed by runtime` against an `OUT OF SCOPE` target.

**Input length is unbounded, with one floor.** Block-local attention is
`O(T * context_size)`, so memory is linear in audio length; nothing is
rejected or truncated and `n_ctx` is ignored by design. The one hard floor
is that a clip must survive both subsampling blocks with at least one frame
remaining, returned as `TRANSCRIBE_ERR_INVALID_ARG`.

**CPU BF16 matmul does not thread.** Encode is 8.9 s at 1 thread and 9.4 s
at 8 on an M4 for 11 s of audio; the same graph on an F32 GGUF scales 6.9 s
to 2.4 s. The graph parallelizes; the BF16 kernel is the bottleneck. Metal
is unaffected. Users on CPU should prefer a quantized tier over BF16, which
is also the faster choice on every measurement in Stage 6.

## Notes

- Single-variant family at intake time.
- A sibling `ibm-granite/granite-speech-5.0-470m-turboctc-nc` exists (4.85 % vs 5.00 %
  aggregate Open ASR WER, SentencePiece instead of byte-level BPE). It is
  **noncommercially licensed** and out of scope, but this family key may later need
  to carry two tokenizer types.
- Publisher benchmarks are bar-chart PNGs only. The single numeric public figure is
  5.00 % aggregate WER on the Open ASR leaderboard (blog post, results as of
  2026-08-25) plus >12,600 RTFx on an H200 with batched inference. There is **no**
  published LibriSpeech test-clean number, so the `porting-7-wer` gate anchors on the
  measured Oracle reference baseline from `porting-2-oracle`, not on a publisher score.
- Preflight Gate A: `dtype_consistency` PASS, `tokenizer_alignment` PASS,
  `frontend_config` WARN (`declared.normalization=per_utterance` vs
  `reference.normalization=none`) is a preflight artifact, since
  `preprocessor_config.json` has no `normalize` key and the normalization is applied
  in extractor code. `capabilities` WARN is the standard "no GGUF yet" skip.
- Highest-risk parts of this port, in order (all three came out clean at Stage 4,
  gated on `enc.block.{0,1}.post_conv` on both dump cases): the in-block stride-2
  subsampling with a pooled+trimmed residual; the exact frontend framing rule in the
  odd-`mel_frames` case; and the F32 BatchNorm island inside an otherwise BF16 graph.
- Stage 1 called `tokenizer_class=ParakeetTokenizer` in `tokenizer_config.json`
  "misleading". It is not: `ParakeetTokenizer._decode` is where the CTC run-length
  collapse lives (`itertools.groupby`, then drop `pad_token_id`). `generate()` returns
  the raw per-frame argmax, so decoding with a plain byte-level BPE tokenizer would
  emit every repeated frame. The C++ reproduces the groupby-then-drop-blank order,
  which is what keeps `[A, blank, A]` as two tokens and `[A, A]` as one.
- **Word timestamps were withdrawn at Stage 8** (re-signed OUT OF SCOPE by CJ,
  2026-09-12; MUST PASS at intake). The intake reasoning was sound as far as it went:
  one encoder frame is exactly 80 ms (`8 * 160 / 16000`), the greedy collapse yields
  the emitting frame per surviving token, and parakeet's CTC convention gives
  `t0 = 80 ms * step_at_emit`. Word-boundary detection did not carry over —
  `src/arch/parakeet/model.cpp` splits on the SentencePiece marker `▁` (`E2 96 81`)
  while this tokenizer is byte-level BPE, where a word opens on a raw piece beginning
  with `Ġ` (`C4 A0`) and the utterance's first word carries no marker — so Stage 4
  reimplemented it. What the intake got wrong was `t1 = t0 + 80 ms`: that is not a
  word end, it is the emitting frame plus one frame, so every single-token word
  reported 80 ms of duration no matter its true length. The Stage 4 observable
  (present, monotonic, in-bounds) could not catch that, because a uniformly-80 ms
  answer satisfies all three. Removed from `decoder.cpp` and `capabilities.cpp`
  rather than shipped; see Known Limitations for the reinstatement bar.

  The lesson generalizes: a structural timestamp check passes on output that is
  structurally perfect and semantically meaningless. Any future timestamp capability
  in this repo needs an observable that would fail on constant-width words.
