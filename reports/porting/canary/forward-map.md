# Forward map - canary

Reference: `nemo.collections.asr.models.EncDecMultiTaskModel` @ NeMo v2.7.2
Closest in-tree analog: `src/arch/cohere/` (encoder-decoder; FastConformer + Transformer decoder)

A compact map from the reference forward pass to the C++ port. Family-level
because the four canary variants share architecture; per-variant differences
go in "Variant Notes" only when they affect graph shape, control flow,
capabilities, or validation coverage.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Mel / log-mel filterbank | `model.preprocessor` (`AudioToMelSpectrogramPreprocessor` -> `FilterbankFeatures`) | `[T_mel, 128]` | `enc.mel.in` | `transcribe::MelFrontend` (n_fft=512, win=400, hop=160, hann_periodic, preemph=0.97, per-feature CMVN) | parakeet/cohere mel frontend |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Pre-encode subsampling (factor 8) | `model.encoder.pre_encode` (ConformerEncoder dw_striding) | `[T_enc, d_enc]` | `enc.pre_encode.out` | `transcribe::conformer::build_pre_encode` over `PreEncodeView` | parakeet pre-encode |
| Relative positional embedding | `model.encoder.pos_enc` (`RelPositionalEncoding`) | `[2*T_enc-1, d_enc]` | `enc.pos_emb` | host-built sin/cos table, uploaded as graph input | parakeet/cohere `pos_emb_in` |
| FastConformer block (FF1 -> rel-pos MHSA -> ConvModule (BN) -> FF2 -> LN) | `model.encoder.layers[i]` | `[T_enc, d_enc]` | `enc.block.{0, n/2, n-1}.out` | `transcribe::conformer::build_conformer_block` over `BlockView` (bias-FREE on linears, like parakeet) | parakeet block |
| Final encoder LN | implicit in `layers[-1]` (norm_out) | `[T_enc, d_enc]` | `enc.final` (when no proj; alias for last block out) | `conf::named` on the final tensor | parakeet final |
| Encoder->decoder projection (180m-flash only) | `model.encoder_decoder_proj` (Linear, d_enc=512 -> d_dec=1024) | `[T_enc, d_dec]` | `enc.final` | `ggml_mul_mat(proj.W, x) + proj.b` after the last block | cohere `enc_dec_proj` |

## Decoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token embedding | `model.transf_decoder._decoder.embedding.token_embedding` | `[seq_len, d_dec]` | (implicit, before first norm) | `ggml_get_rows(dec.embed.token.weight, ids)` | cohere `dec.token_emb` |
| Sinusoidal positional embedding (frozen) | `model.transf_decoder._decoder.embedding.position_embedding` | `[seq_len, d_dec]` | (implicit) | `ggml_get_rows(dec.embed.pos_enc, pos_ids)` | cohere `dec.pos_emb` |
| Embedding LayerNorm | `transf_decoder._decoder.embedding.layer_norm` | `[seq_len, d_dec]` | (implicit) | `LN(tok_emb + pos_emb, w=norm.weight, b=norm.bias)` | cohere `dec.embed_norm` |
| Pre-LN self-attention sublayer | `transf_decoder._decoder.layers[i].first_sub_layer` (LN -> MHSA -> residual) | `[seq_len, d_dec]` | `dec.layer.{i}.self_attn.out` | `x + mha(LN1(x), causal_mask, q,k,v,o)` (KV-cached) | cohere self_attn (split out) |
| Pre-LN cross-attention sublayer | `transf_decoder._decoder.layers[i].second_sub_layer` (LN -> XATTN -> residual) | `[seq_len, d_dec]` | `dec.layer.{i}.cross_attn.out` | `x + mha(LN2(x), enc_states, q,k,v,o)` (cross-KV cache) | cohere cross_attn (split out) |
| Pre-LN FFN sublayer | `transf_decoder._decoder.layers[i].third_sub_layer` (LN -> Linear -> ReLU -> Linear -> residual) | `[seq_len, d_dec]` | `dec.layer.{i}.ffn.out` | `x + ffn_out_b + W_out * relu(W_in * LN3(x) + ffn_in_b)` | cohere ffn (split out) |
| Final decoder LN | `transf_decoder._decoder.layer_norm` (produces `dec.norm.{w,b}` GGUF tensor) | `[seq_len, d_dec]` | (implicit) | `LN(x, dec.norm.{w,b})` | cohere `dec_final` |
| LM head | `transf_decoder.log_softmax.dense` (untied; explicit `dec.head.{weight,bias}`) | `[seq_len, vocab]` | `dec.lm_head.logits.0` | `ggml_mul_mat(dec.head.weight, x) + dec.head.bias` | cohere head (BUT canary is UNTIED) |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Multitask prompt assembly (canary2 = 5 slots: `<source_lang>`, `<target_lang>`, `<task>`, `<pnc>`, `<toggle_timestamps>`; canary = 4 slots, no `<toggle_timestamps>`) | `nemo.collections.common.prompts.canary*` | `[prompt_len]` i32 | `dec.prompt_ids` | host-side build via `stt.canary.special.*_id` lookups + per-language `stt.canary.tokenizer.lang_codes` | qwen3_asr prompt build (closest) |
| Prompt pass (n_past=0, n_tokens=prompt_len; populates self-attn KV cache) | greedy/beam decode prologue | `[vocab, prompt_len]` | `dec.lm_head.logits.0` (last-position logits) | `build_decoder_graph_kv(n_tokens=prompt_len, n_past=0)` | cohere prompt pass |
| Cross-attention KV pre-compute | computed once per utterance from `enc.final` | `[d_dec, T_enc]` per layer (k,v) | (no gate tensor — verified via `dec.layer.*.cross_attn.out`) | `build_cross_kv_graph` | cohere cross-KV pre-compute |
| Step pass (n_tokens=1, n_past=k) | autoregressive step loop | `[vocab, 1]` | (terminate on EOS=`<\|endoftext\|>` or `<\|nospeech\|>`) | `build_decoder_graph_kv(n_tokens=1, n_past=k)` | cohere step pass |
| Mid-generation gate tensor | step graph after >=8 completed iterations | `[vocab, 1]` | `dec.logits_raw.gen8` | TODO — emit on both sides per Stage 4 mid-gen requirement; mirror cohere's pattern when added | cohere `dec.logits_raw` |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en/de/es/fr) with explicit lang | `task=asr`, `source_lang=target_lang=<l>` | `transcribe-cli --language <l>` produces non-empty plausible transcript | TODO row in `docs/porting/families/canary.md` |
| Translate en->{de,es,fr} | `task=s2t_translation`, `source_lang=en, target_lang=<l>` | `transcribe-cli --task translate --target-lang <l>` (1b is en-only ASR; 1b-flash/1b-v2/180m-flash translate) | TODO row |
| Translate {de,es,fr}->en | same, reversed | same | TODO row |
| Word/segment timestamps | `toggle_timestamps=yes` (1b-flash, 180m-flash, 1b-v2 only — 1b does not advertise) | `--timestamps {word,segment}` -> SKIP/ACCEPTED-GAP for v1; runtime exposure TBD | TODO row (likely SKIP) |
| Punctuation/capitalization toggle | `pnc=yes/no` | not user-facing in CLI v1 | likely SKIP — runtime always passes `pnc=yes` |
| Streaming | not advertised on any variant | n/a | n/a |
| Voice activity detection | not advertised | n/a | n/a |

## Deviations From Closest Analog

- **LM head is untied**: cohere ties the head to the token embedding; canary ships explicit `dec.head.weight` and `dec.head.bias` and the converter loads them as separate tensors. The C++ head must `mul_mat(head.weight, x) + head.bias`, not `mul_mat(token_emb, x) + head.bias`.
- **GGUF tensor naming for the decoder**: canary uses `dec.layer.{i}.{self_attn,cross_attn,ffn}.{q,k,v,o,up,down}.{weight,bias}` and `dec.layer.{i}.norm{1,2,3}.{weight,bias}`, not cohere's `dec.blocks.{i}.{self_attn,cross_attn,ff}.*` / `norm_{self,cross,ff}`. The decoder weight loader must read the canary names verbatim.
- **Encoder linears are bias-free** (`stt.canary.encoder.use_bias = False` on every variant): canary's encoder is structurally parakeet's, not cohere's. The shared `transcribe::conformer::BlockView` slots for FFN/attention biases stay null.
- **KV namespace**: `stt.canary.encoder.*` / `stt.canary.decoder.*` / `stt.canary.special.*_id` / `stt.canary.tokenizer.{lang_codes,lang_offsets,lang_sizes,prompt_format}` — distinct from `stt.cohere.*`.
- **Multitask prompt is positional**: 4 slots (canary) or 5 slots (canary2). The prompt format is published via `stt.canary.tokenizer.prompt_format` and the per-task token IDs via `stt.canary.special.*_id`. Mismatched prompt format silently swaps task semantics with no shape error — same risk class as qwen3_asr.

## Variant Notes

- `canary-180m-flash`: 17/4 enc/dec layers, enc_d_model=512, dec_d_model=1024 -> `stt.canary.decoder.encoder_decoder_proj=True` and a separate `enc.proj.{weight,bias}` linear runs after the last block. prompt=canary2, vocab=5248. Unique among the four — only variant with the projection.
- `canary-1b-flash`: 32/4 enc/dec layers, enc_d_model=dec_d_model=1024, no proj. prompt=canary2, vocab=5248. Same code path as 180m minus the projection branch.
- `canary-1b-v2`: 32/8 enc/dec layers, enc_d_model=dec_d_model=1024, no proj. prompt=canary2, vocab=16384.
- `canary-1b`: 24/24 enc/dec layers, enc_d_model=dec_d_model=1024, no proj. prompt=`canary` (4 slots, no toggle_timestamps), vocab=4128. The only variant on the legacy 4-slot prompt format.
