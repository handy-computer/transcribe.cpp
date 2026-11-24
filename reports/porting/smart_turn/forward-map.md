# Forward map - smart_turn

Reference: pipecat-ai/smart-turn (`inference.py`, `train.py` `SmartTurnV3Model`) and its
exported fp32 graph `smart-turn-v3.2-gpu.onnx` from pipecat-ai/smart-turn-v3 @ f766f81d
(onnxruntime 1.23.2, CPU); front end transformers 4.48.2 `WhisperFeatureExtractor(chunk_length=8)`.
Closest in-tree analog: src/arch/whisper/ (the encoder is a Whisper-tiny-shaped encoder:
4 layers, d_model 384, 6 heads, 80 mels, 400 positions).

One clip per call: the last 8 s (128,000 samples), left zero padded, give 800 mel frames and
400 encoder positions. No decoder, no KV cache, no streaming. All gates live in
`tests/tolerances/smart_turn.json`; `head.prob` with decision parity (p > 0.5) is the ship gate.

Written at Stage 8 from the finished port (the Stage 4 map was not saved); every row is
implemented and validated, none is open.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| crop to the last 8 s, left zero pad | `inference.py` `truncate_audio_to_last_n_seconds`, feature extractor `padding="max_length"` | [128000] | - | host: `transcribe-turn.cpp` crops, `prepare_window` pads | - |
| zero mean, unit variance over the whole window (padding included) | `WhisperFeatureExtractor` `do_normalize=True` (`zero_mean_unit_var_norm`, eps 1e-7) | [128000] | `fe.window` | host fp64 sums, `stt.smart_turn.waveform_normalize(_eps)` | - |
| Whisper log-mel (Hann 400, hop 160, 80 Slaney mels, log10 clamp, (x + 4) / 4) | `WhisperFeatureExtractor._np_extract_fbank_features` | [80, 800] | `enc.mel.in` | `whisper::install_mel_from_buffers` (GGUF filterbank and window), `MelFrontend::compute` | whisper |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| conv1 k3 + GELU, conv2 k3 s2 + GELU | `WhisperEncoder.conv1/conv2` | [384, 800], [384, 400] | `enc.conv1.out`, `enc.conv2.out` | `whisper::build_encoder_graph` (im2col conv) | whisper encoder |
| + sinusoidal positions | `embed_positions` | [384, 400] | `enc.embed.out` | GGUF `enc.pos_emb`, `ggml_add` | whisper encoder |
| 4 pre-LN blocks: MHA (6 heads, k without bias) + GELU FFN 1536 | `WhisperEncoderLayer` | [384, 400] | `enc.block.{0,1,2,3}.out` | whisper encoder block; manual attention by default (`flash::apply_env_overrides` can switch it) | whisper encoder |
| final LayerNorm | `WhisperEncoder.layer_norm` | [384, 400] | `enc.final` | `ggml_norm` + affine | whisper encoder |

## Decoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| attention pooling: softmax over time of Linear(384,256)-Tanh-Linear(256,1) | `SmartTurnV3Model.pool_attention` | [400] | `head.pool_weights` | `mul_mat` + bias, `tanh`, `soft_max` over T | - |
| pooled sum over time | `torch.sum(weights * h, dim=1)` | [384] | `head.pooled` | `mul_mat(cont(transpose(enc)), weights)` | - |
| classifier Linear(384,256)-LayerNorm-GELU-Linear(256,64)-GELU-Linear(64,1) | `SmartTurnV3Model.classifier` (Dropout identity) | [1] | `head.logit` | `mul_mat` + bias, `ggml_norm` (eps from GGUF), `gelu_erf` | - |
| sigmoid | `torch.sigmoid` | [1] | `head.prob` | `ggml_sigmoid` | - |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| none: one forward pass per clip | - | - | - | - | - |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| turn score of a clip | `predict_endpoint`: p, `prediction = p > 0.5` | `transcribe_turn_run`: `p`, `ended = p >= threshold` (default 0.5 from `stt.smart_turn.threshold`) | TURN run: PASS (8 golden cases, Pipecat test set decision parity) |
| streaming | none | `transcribe_turn_stream_feed` returns NOT_IMPLEMENTED (`frame_samples` 0) | omitted (does not stream) |
| language | none (23 training languages, no conditioning) | no language control | - |

## Deviations From Closest Analog

- No decoder and no token path: the Whisper encoder output feeds the pooling head directly.
- The window is fixed (800 mel frames, 400 positions) whatever the clip length; Whisper pads to
  30 s instead.
- Waveform normalisation (zero mean, unit variance over the padded window) runs before the mel;
  Whisper ASR does not normalise.
- AUTO resolves to CPU (the validated path); a named backend is honoured.

## Variant Notes

- `smart-turn-v3.2`: the family baseline. Quantized files keep the conv stem, positional table,
  front-end buffers, norms, biases and the one-row head projections (`head.pool.2.weight`,
  `head.cls.6.weight`) in F32 (`tools/transcribe-quantize/policy.cpp`).
