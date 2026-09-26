# Forward map - nemotron3_diar

Reference: NeMo Speech `SortformerEncLabelModel` (nemo.collections.asr.models.sortformer_diar_models) + `modules/transformer_encoder.py::TransformerEncoder` + `modules/sortformer_modules.py::SortformerModules` @ `cf724ac337d1` (`refs/NVIDIA-NeMo/Speech`)
Closest in-tree analog: src/arch/sortformer/ (AOSC/FIFO streaming driver, host state machine, diar head, speaker-segment surface). The encoder has no in-tree analog; RoPE/GELU pre-LN attention patterns follow ggml-reference-map.

`encoder-diarizer`: no tokenizer, no decoder, no text. Product = `T x 8`
per-frame speaker-activity probabilities at **10 ms** (high_resolution=True,
output_subsampling_factor=1), arrival-order columns, max 8 speakers. The
inference product path is ALWAYS the sync streaming loop
(`streaming_mode=True`, `async_streaming=False`, `forward_streaming`); there
is no separate offline forward (the "offline" capability = the
very_high_latency preset).

Dims (GGUF `stt.nemotron3_diar.*`): encoder 31L pre-LN d=512 h=8 (head_dim 64)
dff=2048 GELU(exact), RoPE NEOX theta 1e4 rotary_fraction 1, no qkv bias,
out_proj bias, LN eps 1e-5; feature stacking x8 (1024->512, no bias);
encoder_proj 512->192; subpixel Conv1d 192->1536 k3 pad1 -> [T,8,192] ->
[8T,192]; head relu -> fc1 192->192 -> relu -> single_spk_head 192->8 ->
sigmoid. AOSC cache holds 512-dim pre_encode embeddings at 80 ms.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Log-mel | `AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures` (`process_signal`; streaming_mode => no peak normalization) | [128, T_mel], T_mel = floor(n/160) (9200 for 92 s, 9133 for 91.337 s) | `enc.mel.in` | `transcribe::MelFrontend`: preemph 0.97, n_fft 512, win 400, hop 160, constant center pad, normalize none, log(x+2^-24), dither 0; **window + filterbank from GGUF** (`frontend.window` / `frontend.mel_filterbank`, BF16-exact); the driver keeps the first floor(n/160) frames | sortformer load() MelConfig + cohere GGUF-buffer injection |

## Encoder (per streaming step, over the `[spkcache | fifo | chunk]` concat)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Feature stacking (pre_encode) | `subsampling.py::FeatureStacking.forward` (via `_call_pre_encode`): (B,C,T)->(B,T,C), zero-pad T to x8, reshape (T/8, 8*128) frame-major, `proj` Linear no bias | [T_diar, 512] | `enc.pre_encode.out` | mel window [T_win,128] (time-major) -> pad -> view [1024, T/8] (8 consecutive 128-dim frames per row, frame-major) -> mul_mat(proj) | new (reshape + linear) |
| Concat cache | `forward_streaming_step`: `concat_embs([spkcache, fifo, chunk_pre_encode_embs])` | [S+F+T_diar, 512] | (host) | host concat, as sortformer | sortformer `run_diar_streaming_core` |
| embed_norm | `TransformerEncoder.forward_internal` (rope branch: no xscale, dropout no-op) -> `embed_norm` LN | [T_cat, 512] | `enc.embed_norm.out` | `ggml_norm` + mul + add | sortformer `layer_norm` |
| Pre-LN block x31 | `TransformerBlock.forward`: `x += attn(norm1(x))`; `x += ffn(norm2(x))` | [T_cat, 512] | `enc.layers.{0,1,15,30}.out` | per block below | cohere/qwen3 pre-LN blocks (structure only) |
| final_norm | `final_norm` LN | [T_cat, 512] | `enc.final_norm.out` | layer_norm | - |
| encoder_proj | `frontend_encoder`: transpose + `sortformer_modules.encoder_proj` 512->192 | [T_cat, 192] | `diar.encoder_proj.out` | linear | sortformer |

### Attention (`MultiHeadAttention.forward`, rope branch)
```
qkv = w_qkv(x).view(T, 3, H, D)          # rows [q | k | v], head-major; no bias
q,k = rope(q,k)                          # positions 0..T_cat-1 (cache_len=0: t_q == t_k); NEOX rotate_half
out = softmax(q k^T / sqrt(D) + padmask) v   # flex attention, default scale; sync batch=1 => no padding
out_proj(out)                             # bias
```
ggml (as built): one mul_mat for qkv, `ggml_view_4d` q/k/v head slices, `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX, n_dims=64, freq_base=1e4)`,
manual F32 attention `mul_mat(k, q)` -> `soft_max_ext(scale=1/8)` -> `mul_mat(v^T, kq)` (no flash; K/V stay F32). Positions restart at 0 every step (RoPE over the concat,
not absolute stream time). FFN: `net.0` Linear -> `nn.GELU()` (exact erf) -> `net.3` Linear => `ggml_gelu_erf`.

## Decoder

No decoder. The head is the upsampler + sigmoid stack (`forward_infer`).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| (transformer_encoder) | `transformer_encoder` is None for this model (no post-LN stack) | - | - | skipped | differs from sortformer |
| Subpixel conv | `SortformerModules.upsample_hidden`: Conv1d(192, 1536, k=3, pad=1) over time | [T_cat, 1536] | `diar.subpixel_conv.out` | `ggml_im2col` (F32, pad 1) + mul_mat with the kernel viewed [576, 1536] (channels in ne0); F16 kernel upcast to F32 on CPU | conformer conv helpers |
| Pixel shuffle | reshape (T, 8, 192) -> (8T, 192): channel block j of frame t = output frame 8t+j | [8*T_cat, 192] | `diar.upsample.out` | ne [1536, T] -> reshape [192, 8T] (contiguous, no permute) | new |
| Speaker logits | `forward_speaker_logits`: relu -> fc1 -> relu -> single_hidden_to_spks | [8*T_cat, 8] | `diar.logits` | linear/relu | sortformer head |
| Sigmoid | `forward_infer`: `sigmoid(logits) * output_mask` (mask all-ones in sync) | [8*T_cat, 8] | (inside `diar.probs`) | `ggml_sigmoid` | sortformer |

## Streaming path (`forward_streaming` / `forward_streaming_step`, sync)

Per chunk from `streaming_feat_loader` (lc=0 for every card preset; rc per preset):
1. pre_encode(chunk mel window incl. rc) -> chunk_embs [T_diar, 512] (T_diar = ceil(win/8)).
2. concat [spkcache | fifo | chunk_embs] -> encoder -> encoder_proj -> upsample -> head -> **hi-res preds [8*T_cat, 8]**.
3. `downsample_preds(hi, 8)`: avg-pool x8 (ceil_mode, count_include_pad=False) -> preds80 [T_cat, 8] (drives the cache).
4. `streaming_update(state, chunk_embs, preds80, lc, rc)` (host): identical to sortformer's sync update EXCEPT
   - `use_learnable_sil_emb=True`: NO `_get_silence_profile` update; `_compress_spkcache` fills disabled slots
     with `learnable_sil_emb` (GGUF `diar.sil_emb`) instead of the running mean;
   - `spkcache_preds` before first compression = `preds80[:S] ++ pop_preds` (equivalent to sortformer's seed-on-first-compress).
5. chunk output = `hi[(S+F+lc)*8 : (S+F+lc+C)*8]` (10 ms), appended to total_preds.
6. End: `total_preds[:T_mel]` (output_subsampling_factor 1) -> T_mel rows.

Presets (80 ms frames; spkcache/fifo/chunk/rc/update): very_high_latency 264/40/340/40/300;
low_latency 264/264/9/4/222; very_low_latency 264/264/6/2/222; ultra_low_latency 264/264/3/1/222;
small (validation only) 24/10/20/2/20. `spkcache_sil_frames_per_spk=1`, compress constants from GGUF KVs.

Push-audio (`transcribe_stream_*`): the chunk loop is causal given the right context, and the per-chunk mel
equals the whole-utterance mel on shared frames (Stage 2), so a feed can run every chunk whose window
(chunk + rc mel frames) is fully available, and finalize runs the tail with the utterance-final geometry
(`min(...)` clamps in `streaming_feat_loader`). Final output length = NeMo's T_mel.

### Implementation + validation status (Stage 4)

All rows above are implemented in `src/arch/nemotron3_diar/` (model.cpp: load, step graph, chunk driver,
run / run_batch / push-audio; stream.cpp: host AOSC state machine + segment tracker; torch_logf.h) and gated:
every Stage-2 tensor passes on both oracle cases (92 s and the non-aligned 91.337 s), diar.probs at every
preset within 1.8e-5 of NeMo, push-audio bit-identical to the whole-file run, batched bit-identical to serial.

Parity-critical details found during bring-up (all verified against the reference):
- CPU compute must be fp32: ggml-cpu BF16 matmuls round activations to BF16, which flips cache-compression
  picks. The loader upcasts BF16/F16 matmul weights to exact F32 on CPU (default; `..._NATIVE_BF16=1` opts out).
- The compression top-k boundary is routinely separated by ~1e-7 (and exact ties are common), so the host
  compression reproduces torch CPU arithmetic bit-for-bit: `torch.topk(sorted=False)` = ATen's
  partial_sort / nth_element on (value, index) pairs; `torch.log` = Sleef logf_u10 (torch_logf.h); the
  8-speaker sum = ATen's 4-accumulator order. With that, C++ compression == torch compression on identical
  inputs (replay gate: 370/370 compressions). The remaining source of divergence is fp32 GEMM noise (~1e-5) moving a boundary score across a
  ~5e-6 gap on long meetings (6 of 96 acceptance meetings at very_high_latency); DER/JER are unchanged
  to 0.001pp (see the family doc).
- NeMo frames: floor(n / 160) valid mel frames (not ceil); the chunk driver consumes exactly those.

## Generation / KV Path

No autoregressive generation. N/A.

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Offline diarization | `diarize()` at very_high_latency | `transcribe_run` + preset run ext -> speaker segments; `diar.probs` dump | Offline diarization |
| Streaming diarization | `diarize()` at low_latency | same, preset LOW_LATENCY | Streaming diarization |
| Low-latency presets | very_low / ultra_low | preset ext | Low-latency presets |
| Push-audio | reference streaming at the same preset | `transcribe_stream_begin/feed/finalize` + STREAM-slot preset ext; incremental speaker segments | Push-audio live diarization |
| Speaker-activity tensor | preds [T,8] | `diar.probs` dump | Speaker-activity tensor |
| Batch | n/a | `run_batch` | Batch (offline) |
| Text / translate / timestamps | none | not exposed | OUT OF SCOPE rows |

## Deviations From Closest Analog

- Encoder is a 31L pre-LN RoPE Transformer over the concat (sortformer: FastConformer + 18L post-LN transformer). No transformer_encoder stage, no xscale, no rel-pos table.
- pre_encode is non-overlapping feature stacking (no conv subsampling, no left context needed).
- Output is 10 ms (x8 subpixel upsampler); the cache/FIFO machinery still runs on 80 ms avg-pooled probs.
- Learned silence embedding replaces the running-mean silence profile.
- 8 speakers, 4 card presets (different geometry from v2.1).
- BF16 weights (sortformer F32); mel window/fb are GGUF buffers.

## Variant Notes

- `Nemotron-3-Diarization`: family baseline (this port).
