# Forward map - crisperwhisper

Reference: `crisperwhisper` 2.0.2 (PyPI, MIT), transformers backend @
`TransformersEngine` / `CrisperWhisperModel._transcribe_v2`
Closest in-tree analog: `src/arch/whisper/` + the shared `src/whisper_graph/`

**The graph is stock Whisper and is not re-derived here.** `src/whisper_graph/`
is shared verbatim with the `whisper` family: the encoder, all four decoder
graph families, the KV cache, and the tensor catalog. Every row in the Encoder
and Decoder tables below is "unchanged from whisper", and their gate tensors are
the same names `scripts/dump_reference_crisperwhisper_author.py` already emits.

What this port actually builds is the **decode harness above the graph**, which
shares nothing with whisper: the mode-tag prompt, prompt-artifact stripping, the
`<ctx>` continuation long-form, and Viterbi word timing over supervised
cross-attention. Those are the rows that carry real work.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Load + resample to 16 kHz mono | `audio.load_audio` | `[n_samples]` | — | host-side, existing CLI/wav path | whisper |
| Log-mel, 80 bins, 10 ms hop | `TransformersEngine.extract_features_with_mel` → HF `WhisperFeatureExtractor` | `[80, 3000]` | `enc.mel.in` | `transcribe::MelFrontend`, `normalize="per_utterance"` (the `whisper_logmel` tag) | whisper (identical; filterbank + Hann baked into the GGUF) |
| Keep the 10 ms mel for word timing | same call returns `mel` numpy alongside the tensor | `[80, 3000]` | — | retain the pre-encoder mel buffer on the session; it is the blank-energy source, NOT a second frontend | **new** — whisper discards it |

Note: the mel runs at 10 ms while encoder frames are 20 ms. The aligner
resamples energy 2:1 (`word_timing._resample_1d`), it does not re-run the mel.

## Encoder

Unchanged from whisper. `whisper_graph::build_encoder_graph`, called with
CrisperWhisper's `HParams`. Listed for gate-tensor traceability only.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Conv stem k=3 s=1 + GELU | `WhisperEncoder.conv1` | `[3000, d]` | `enc.conv1.out` | `conformer::conv_1d_f32` + `ggml_gelu_erf` | whisper |
| Conv stem k=3 s=2 + GELU | `WhisperEncoder.conv2` | `[1500, d]` | `enc.conv2.out` | same, stride 2 | whisper |
| + learned pos emb | `WhisperEncoder.embed_positions` | `[d, 1500]` | `enc.pos_emb`, `enc.embed.out` | `ggml_add` of a GGUF-baked F32 table | whisper |
| N pre-LN blocks | `WhisperEncoderLayer` | `[d, 1500]` | `enc.block.{0,3,6,8,11}.out` | `whisper_graph` `build_block`; q/v/out have bias, k does not | whisper |
| Final LN | `WhisperEncoder.layer_norm` | `[d, 1500]` | `enc.final` | `conformer::layer_norm` | whisper |

## Decoder

Unchanged from whisper. `whisper_graph::build_cross_kv_graph` +
`build_decoder_graph_kv` + `build_step_graph`.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token + pos embed | `WhisperDecoder.embed_tokens/_positions` | `[d, S]` | `dec.token_emb`, `dec.pos_emb`, `dec.embed_sum` | `ggml_get_rows` + `ggml_add`; **pure GGUF read/add, tolerance pinned at exact 0.0** | whisper |
| N pre-LN blocks (self + cross + FFN) | `WhisperDecoderLayer` | `[d, S]` | `dec.block.{0,3,6,8,11}.out` | `whisper_graph` cached/step MHA; cross-K/V precomputed per chunk | whisper |
| Final LN | `WhisperDecoder.layer_norm` | `[d, S]` | `dec.out_before_head` | `conformer::layer_norm` | whisper |
| Tied head → logits | `proj_out` (tied to `embed_tokens`) | `[vocab, S]` | `dec.logits_raw`, `dec.logits` | `ggml_mul_mat` on `dec.token_embd.weight`, no bias | whisper |
| Mid-generation step | greedy loop step 20 | `[vocab]` | `dec.logits_raw.gen20` | static step graph, `n_past > 0` | whisper (satisfies the Stage-4 `gen<N>`, N≥8 rule) |

## Generation / KV Path

This is where the port diverges. Every row here is new code in
`src/arch/crisperwhisper/model.cpp`.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Build mode-tag prompt | `prompt.PromptBuilder._build` | `[n_prompt]` i32 | `dec.token_ids` | tokenize `"[verbatim_1]…[verbatim_5]"` then append the Whisper prefix. Tags come **before** the prefix, no `<\|startofprev\|>` wrapper. Ids from `stt.crisperwhisper.mode.*_token_ids` | **none** — whisper prepends `<\|startofprev\|>` + prompt AFTER sot |
| Whisper prefix | `TransformersEngine.get_decoder_prefix` | 4 tokens | — | `[sot, <\|lang\|>, <\|transcribe\|>, <\|notimestamps\|>]`; `<\|notimestamps\|>` is **always** present | whisper (same tokens, different position relative to the prompt) |
| Optional `<ctx>` context | `prompt.PromptBuilder._build` | variable | — | `" <ctx> " + last N confirmed words + " <ectx>"` appended to the tag block, **before** any hotwords (training order) | **none** |
| Greedy decode + suppression | `_run_generate` | `[vocab]`/step | `dec.logits_raw.gen20` | existing whisper host loop minus every timestamp rule; apply `suppress_tokens` each step, `begin_suppress_tokens` on step 0 | whisper (strip the timestamp-rule block) |
| Stop | `max_new_tokens=256` or EOT | — | — | EOT = 50257; cap from `max_new_tokens`, and prompt + `<ctx>` eat into the 448 budget first (intake risk 7) | whisper |
| Strip prompt artifacts | `prompt.strip_prompt_artifacts` + `decode_tokens(skip_special=True)` | text | `transcript.json` | drop ids in `stt.crisperwhisper.prompt_artifact_token_ids`; **keep** `event_token_ids` (they are verbatim output) | **none** — whisper strips by tokenizer special flag; here all 31 added tokens are `special: false` |
| Word grouping | `word_timing.group_tokens_into_words` | `[W][tok idx]` | — | flush on special / space (id 220) / leading-space piece | **none** |
| Token logp from attention | `word_timing.token_logp_from_attention` | `[T, F]` | — | `attn**5.0`, row-normalize, log | **none** |
| Blank logp from mel energy | `word_timing.blank_logp_from_mel_energy` | `[F]` | — | mean over mels → p10/p90 normalize → resample 2:1 → `(1-e)**3.0` → log → `-3.0` | **none** |
| Word Viterbi with virtual blanks | `word_timing.viterbi_align_words_with_blanks` | `[W](start,end)` | `word_timestamps.json` | logsumexp-collapse each word's token rows, then DP over `2W+1` states; frame = 20 ms | **none** |
| Gap split + monotonize | `split_interword_gaps`, `monotonize_words` | `[W]` | `word_timestamps.json` | gaps ≤ 0.1 s split at midpoint; clamp starts forward at seams | **none** |
| Long-form windows | `longform/base.make_chunks` | `[n_chunks][≤30 s]` | — | 30 s window, 26 s stride, 4 s overlap | **none** — whisper's timestamp-token long-form does not transfer |
| Long-form stitch | `longform/continuation.continuation_transcribe_with_word_timestamps` | text | — | per chunk: decode with `<ctx>` of last 12 confirmed words; drop a trailing word only when its start ≥ stride, capped at 2; lift timings by `i*stride` | **none** |

**Ordering constraint discovered in the reference:** `continuation_transcribe`
delegates to the `_with_word_timestamps` variant whenever
`timestamp_aware_drop=True` (the default), so **long-form depends on word
timing**. Word timing must land before long-form, not after.

## Cross-attention capture (the one shared-component change)

The Viterbi needs, per generated token, the head-averaged post-softmax
cross-attention over encoder frames for the 10 `(layer, head)` pairs in
`stt.crisperwhisper.word_timing.alignment_heads`.

| Concern | Decision |
|---|---|
| Reference | `_run_generate_with_attention` + `_stack_step_attention`: sum over selected `(layer, head)`, divide by count, take the **last** query row per step |
| ggml | `whisper_graph` decoder builders take an optional `capture_cross_layers` list; when a layer is listed the builder forces the manual attention path for that layer and marks the softmax output as a graph output |
| Whisper impact | passes an empty list; graph is bit-identical (verified: 42/42 tensors, transcripts exact) |
| Why not flash | `ggml_flash_attn_ext` never materializes the probability matrix. Capture layers fall back to `mul_mat` + `soft_max_ext`, which is also the path the reference's `attn_implementation="eager"` uses |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Explicit language | `language="en"` forces `<\|lang\|>` | `params.language` → lang token | Transcribe / explicit language hint |
| Auto language | **not reachable** via `transcribe()`; `language` defaults to `"en"` | whisper's lang-detect prefill on the shared prefill graph | Transcribe / auto, no language hint |
| Verbatim mode | `mode="verbatim"` (package default) | `--mode verbatim`; default per `stt.crisperwhisper.mode.default` | Verbatim mode |
| Intended mode | `mode="intended"` | `--mode intended` | Intended mode |
| Word timestamps | `word_timestamps=True` | `--timestamps word` | Word timestamps |
| Long-form | `> 30 s` → continuation | automatic above 30 s | Longform (>30 s) |
| Segment timestamps | never emitted (`<\|notimestamps\|>` forced) | cap `max_timestamp_kind` at WORD | Segment timestamps — OUT OF SCOPE |
| Translate | token present, no reference path | `stt.capability.translate=false` | Translate — OUT OF SCOPE |
| Batch (offline) | no batched transcribe on this backend | `run_batch()` over the shared batched step graph | Batch (offline) |

## Deviations From Closest Analog

- **Prompt position.** Whisper puts a conditioning prompt after
  `<|startofprev|>` and before `<|startoftranscript|>`. CrisperWhisper puts
  mode tags before the whole Whisper prefix with no wrapper token. Reusing
  whisper's prompt builder would silently produce a different sequence.
- **No timestamp tokens at all.** Every timestamp rule in whisper's host loop
  (`max_initial_timestamp`, monotonicity, paired-emission, the
  `no_timestamps` toggle) is dead code here and must not be ported.
- **Added tokens are not `special`.** All 31 carry `special: false`, so
  `skip_special_tokens` does not strip them. Stripping is by explicit id list,
  and the 15 vocal-event tokens must survive into the output.
- **Word timing is Viterbi, not DTW.** Whisper's median-filtered DTW is not
  used; the blank state comes from mel energy, which is why the frontend row
  above retains the 10 ms mel.
- **Long-form is prompt continuation, not timestamp stitching.**

## Variant Notes

- `crisperwhisper-2.0-small`: family baseline (d=768, 12+12, 80 mels, vocab
  51896). The only variant implemented at this stage.
- `crisperwhisper-2.0-medium` / `-large`: pure size changes (24+24 / 32+32).
  Same 80-mel, 51896-vocab, same token ids. Expected to need no code change.
- `crisperwhisper-2.0-turbo`: **not a size variant.** 128 mels, vocab 51897,
  100 languages, 4 decoder layers against a 32-layer encoder, and every special
  token id above 50357 shifted by +1. Nothing may hardcode a token id; all ids
  come from GGUF KV, which the Stage-3 converter already guarantees. Its
  `alignment_heads` differ and cannot be reused from the other three.
