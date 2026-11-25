# Smart Turn v3.2 (end of turn)

Status: bring-up (TURN role). Stages 1-3 done: intake, oracle, F32 GGUF.

Smart Turn answers "has the speaker finished their turn?" for one clip of
16 kHz audio (architecture pattern `encoder-classifier`). It scores the last
8 s: the whisper-tiny encoder, attention pooling over its 400 positions and a
small MLP give one probability. It does not stream and does not read text.
Usage of the role: [`docs/turn.md`](../../turn.md).

## Identity

- Family key: `smart_turn`; variant `smart-turn-v3.2`; GGUF
  `general.architecture = smart_turn`
- Upstream module: `SmartTurnV3Model` in smart-turn's `train.py`
  (github.com/pipecat-ai/smart-turn @ `4e12f2b`), exported to ONNX
- Hugging Face repo: `pipecat-ai/smart-turn-v3`
- Hugging Face revision: `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`
- Source file: `smart-turn-v3.2-gpu.onnx` (fp32, sha256
  `ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be`).
  `smart-turn-v3.2-cpu.onnx` (sha256 `2bb02631...`) is upstream's static
  int8 quantization of the same model; it is not converted, only scored for
  comparison.
- License: BSD-2-Clause (model and code). Golden clips from
  `pipecat-ai/smart-turn-data-v3.2-test` are CC-BY-4.0 (`samples/README.md`).
- Parameters: 8,000,386 (F32 GGUF 32.1 MB)

## References

- Canonical: onnxruntime 1.23.2 on the fp32 ONNX, run as smart-turn's
  `inference.py` `predict_endpoint` does (session options
  `ORT_SEQUENTIAL`, inter-op 1, `ORT_ENABLE_ALL`; CPU, 1 thread). Front end:
  transformers 4.48.2 `WhisperFeatureExtractor(chunk_length=8)` on its
  numpy path. Upstream's decision is `p > 0.5`.
- Instrumented: `scripts/dump_reference_smart_turn_onnx.py`. The same ONNX
  graph with the dump points added as outputs. Exposing them stops a few
  onnxruntime fusions, so its p is within 6e-8 of the canonical session (the
  dumper bounds it at 1e-6 and writes the canonical p as `head.prob`).
- Cross-check: a PyTorch twin, `SmartTurnV3Model` copied from `train.py`
  (`scripts/lib/smart_turn_onnx.py`) with the ONNX weights mapped in, torch
  2.9.0, sdpa attention. The dumper writes its tensors to `<out>/torch/` and
  the per-tensor differences to `<out>/bridge.json`.

Versions are pinned in `scripts/envs/smart_turn/pyproject.toml`: onnx 1.19.1,
onnxruntime 1.23.2, transformers 4.48.2, torch 2.9.0, numpy 2.3.4 (the
versions smart-turn's `requirements.txt` exports with).

### Front end

`inference.py` (and Pipecat's `LocalSmartTurnAnalyzerV3`):

```
window = audio[-128000:]  or  [zeros(128000 - n) | audio]   left padded
x      = (window - mean) / sqrt(var + 1e-7)      whole window, padding included, float32
mel    = Whisper log-mel of x: reflect pad 200, periodic Hann 400, hop 160,
         |X|^2, slaney 80-bin filterbank, log10(max(., 1e-10)),
         drop the last frame, max(., max - 8), (. + 4) / 4      -> [80, 800]
```

transformers uses a float32 torch STFT when torch is importable and a
float64 numpy STFT otherwise. `inference.py`'s requirements have no torch and
Pipecat ships a numpy copy (`_whisper_features.py`), so the reference pins the
numpy path. On the golden clips: numpy vs torch path 4.0e-5 max in the mel,
4.2e-7 in p; numpy path vs Pipecat's copy 1.2e-7 in the mel, 1.8e-6 in p. The
filterbank is transformers' `mel_filter_bank(slaney)` (the extractor is built
fresh, not from whisper-tiny's `preprocessor_config.json`).

### Forward pass

```
h  = GELU(conv1(mel)) ; GELU(conv2(h)) ; + pos_emb[400]     whisper-tiny stem
h  = 4 x pre-LN block (6 heads x 64, k without bias, FFN 1536, erf GELU)
h  = LayerNorm(h)                                            [400, 384]
s  = Linear(256,1)(tanh(Linear(384,256)(h)))                 [400]
w  = softmax_t(s) ; pooled = sum_t w[t] h[t]                 [384]
z  = Linear(64,1)(GELU(Linear(256,64)(GELU(LayerNorm(Linear(384,256)(pooled))))))
p  = sigmoid(z)
```

All LayerNorm eps are 1e-5. The export computes attention as torch SDPA does:
q and k are each scaled by 64^-0.25 before the product.

### Bridge validation

The ONNX stores 26 of the Linear weights as anonymous MatMul initializers
(`val_17` ...) in `[in, out]` order. `map_onnx_weights` names each by the
bias `Add` that consumes the MatMul's output, or, for the bias-free k
projection, by the `self_attn_layer_norm` feeding it, and transposes it. The
map is checked by loading it strictly into the rebuilt `SmartTurnV3Model`:

| Input | twin vs onnxruntime, p |
|---|---|
| 5 random waveforms (0.5 to 12.5 s) | <= 5.4e-7 (sdpa and eager attention alike) |
| 8 golden clips | <= 5.4e-6 (silence), others <= 3.0e-6 |

Per tensor (max over the golden clips, `bridge.json`): `enc.conv2.out`
1.6e-5, `enc.block.1.out` 6.6e-5, `enc.block.2.out` 6.2e-3,
`enc.block.3.out` 2.0e-2, `enc.final` 6.8e-3, `head.pooled` 5.1e-5,
`head.logit` 2.2e-5. The large encoder numbers are Whisper's massive
activations: from block 2, channel 375 (then 16, 128, 247) reaches |x| 80 to
153 at a few positions, and fp32 reduction order shows there (about 1e-4
relative). The final LayerNorm and the softmax pooling contract it again.

The GGUF closes the loop: its tensors, loaded back into the twin, with a mel
computed from its own `frontend.mel_filterbank` / `frontend.window`
(3.6e-6 from transformers' mel), reproduce onnxruntime's p to <= 5.4e-6 on
the golden clips.

## Tensor map

| ONNX | torch (`SmartTurnV3Model`) | GGUF | numpy shape |
|---|---|---|---|
| `inner.encoder.conv{1,2}.{weight,bias}` | `encoder.conv{1,2}.*` | `enc.conv.{0,1}.{weight,bias}` | `[384, 80, 3]`, `[384, 384, 3]`, `[384]` |
| `inner.encoder.embed_positions.weight` | `encoder.embed_positions.weight` | `enc.pos_emb.weight` | `[400, 384]` |
| `inner.encoder.layers.i.self_attn_layer_norm.*` | same | `enc.blocks.i.norm_attn.*` | `[384]` |
| MatMul `val_{17,92,165,238}` (T) + `...q_proj.bias` | `...self_attn.q_proj.*` | `enc.blocks.i.attn.q.*` | `[384, 384]` |
| MatMul `val_{25,100,173,246}` (T), no bias | `...self_attn.k_proj.weight` | `enc.blocks.i.attn.k.weight` | `[384, 384]` |
| MatMul `val_{32,107,180,253}` (T) + `...v_proj.bias` | `...self_attn.v_proj.*` | `enc.blocks.i.attn.v.*` | `[384, 384]` |
| MatMul `val_{75,148,221,294}` (T) + `...out_proj.bias` | `...self_attn.out_proj.*` | `enc.blocks.i.attn.out.*` | `[384, 384]` |
| `inner.encoder.layers.i.final_layer_norm.*` | same | `enc.blocks.i.norm_ffn.*` | `[384]` |
| MatMul `val_{79,152,225,298}` (T) + `...fc1.bias` | `...fc1.*` | `enc.blocks.i.ffn.fc1.*` | `[1536, 384]` |
| MatMul `val_{88,161,234,307}` (T) + `...fc2.bias` | `...fc2.*` | `enc.blocks.i.ffn.fc2.*` | `[384, 1536]` |
| `inner.encoder.layer_norm.*` | `encoder.layer_norm.*` | `enc.final_norm.*` | `[384]` |
| MatMul `val_311` (T) + `inner.pool_attention.0.bias` | `pool_attention.0.*` | `head.pool.0.*` | `[256, 384]` |
| MatMul `val_313` (T) + `inner.pool_attention.2.bias` | `pool_attention.2.*` | `head.pool.2.*` | `[1, 256]` |
| Gemm `inner.classifier.0.*` | `classifier.0.*` | `head.cls.0.*` | `[256, 384]` |
| LayerNormalization `inner.classifier.1.*` | `classifier.1.*` | `head.cls.1.*` | `[256]` |
| Gemm `inner.classifier.4.*` | `classifier.4.*` | `head.cls.4.*` | `[64, 256]` |
| Gemm `inner.classifier.6.*` | `classifier.6.*` | `head.cls.6.*` | `[1, 64]` |
| (WhisperFeatureExtractor) | | `frontend.mel_filterbank`, `frontend.window` | `[80, 201]`, `[400]` |

(T) = transposed by the converter. Every tensor is F32; ggml `ne` is the
reverse of the numpy shape. The positional table is the exported one: it is
within 3.1e-5 of transformers' `sinusoids(400, 384)` but not equal, so the
runtime must load it.

## GGUF contract

Keys: `stt.variant`, `stt.smart_turn.source_sha256`,
`stt.capability.{lang_detect, translate, timestamps, streaming}` (all false),
`stt.smart_turn.encoder.{n_layers 4, d_model 384, n_heads 6, ffn_dim 1536,
num_mel_bins 80, max_source_positions 400, activation "gelu"}`,
`stt.smart_turn.max_audio_samples` 128000, `stt.smart_turn.threshold` 0.5,
`stt.smart_turn.waveform_normalize` true,
`stt.smart_turn.waveform_normalize_eps` 1e-7, and convert-whisper.py's
`stt.frontend.*` (`type mel`, 80 mels, 16 kHz, `n_fft` / `win_length` 400,
hop 160, `hann_periodic`, `normalize whisper_logmel`, dither 0, pre-emphasis
0, f 0 to 8000, `pad_mode reflect`, `center` true, `mel_norm slaney`,
`chunk_length` 8, `n_samples` 128000, `nb_max_frames` 800). `general.*`
carries the 23 languages, BSD-2-Clause and the source URL.

## Oracle

Golden cases (`tests/golden/smart_turn/smart-turn-v3.2.manifest.json`):

| case | length | window | p (fp32) | p (int8 ONNX) | label |
|---|---:|---|---:|---:|---|
| `jfk` | 11.0 s | crop | 0.948620 | 0.805015 | |
| `ja` | 7.2 s | pad | 0.990261 | 0.989737 | |
| `noise` | 15.0 s | crop | 0.977672 | 0.957368 | |
| `silence` | 8.0 s, zeros | crop | 0.461506 | 0.987037 | |
| `turn-complete-en` | 3.6 s | pad | 0.990633 | 0.989329 | complete |
| `turn-incomplete-en` | 6.1 s | pad | 0.005583 | 0.006176 | incomplete |
| `turn-incomplete-en-long` | 12.0 s | crop | 0.006419 | 0.006422 | incomplete |
| `turn-uncertain-en` | 5.5 s | pad | 0.274234 | 0.271964 | incomplete |

The `turn-*` clips are human recordings from the v3.2 test set (ids in
`samples/README.md`). `silence` (variance 0, constant features) sits near
the cut-off.

Dumps (`build/validate/smart_turn/smart-turn-v3.2/<case>/ref/`, numpy
shapes, batch dropped, time-major like the whisper family's):
`fe.window [128000]` (normalised window), `enc.mel.in [800, 80]`,
`enc.conv1.out [800, 384]`, `enc.conv2.out [400, 384]`,
`enc.pos_emb [400, 384]`, `enc.embed.out`, `enc.block.{0..3}.out`,
`enc.final [400, 384]`, `head.pool_scores [400]`, `head.pool_weights [400]`,
`head.pooled [384]`, `head.logit [1]`, `head.prob [1]`; plus `turn.json`
(p, the decision, p of the int8 ONNX), `bridge.json` and `torch/`.
`enc.pos_emb` and `head.pool_scores` are debug-only (no tolerance entry).

Test-set baseline (`scripts/turn/run_reference_smart_turn_onnx.py`,
decision `p > 0.5`, rows kept in `reports/turn/`):

| ONNX | shard 0 (3,153 clips) | all 31,527 clips | published |
|---|---:|---:|---:|
| fp32 (`-gpu`) | 93.69% | 93.71% (FPR 3.51, FNR 2.78) | 93.71% (3.51, 2.78) |
| int8 (`-cpu`) | 92.48% | 92.62% (FPR 4.76, FNR 2.63) | 92.63% (4.73, 2.64) |

The fp32 numbers reproduce upstream's report exactly, which pins the front
end and the decision rule. The int8 file lands 0.01 points lower, with FPR 0.03 points
higher (onnxruntime's int8 kernels on arm64 here; upstream's host is not
recorded). int8 and fp32 agree on
96.12% of all decisions (mean |dp| 0.042, max 0.95). FPR / FNR are percent of
all clips, as upstream reports them.

## Commands

```bash
# Reference dumps (all manifest cases)
uv run scripts/validate.py ref --family smart_turn
# one clip
uv run --project scripts/envs/smart_turn scripts/dump_reference_smart_turn_onnx.py encoder \
  --audio samples/jfk.wav --out build/validate/smart_turn/smart-turn-v3.2/jfk/ref

# Test-set baseline (fp32 and int8 ONNX)
uv run --project scripts/envs/smart_turn scripts/turn/run_reference_smart_turn_onnx.py \
  --out reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl

# Conversion (downloads the pinned ONNX; models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf)
uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py
uv run scripts/preflight.py --family smart_turn --variant smart-turn-v3.2 --gate B

# Validation (Stage 4)
uv run scripts/validate.py all --family smart_turn --variant smart-turn-v3.2

# Quants (Stage 5)
build/bin/transcribe-quantize models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf \
  models/smart-turn-v3.2/smart-turn-v3.2-F16.gguf --quant F16
build/bin/transcribe-quantize models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf \
  models/smart-turn-v3.2/smart-turn-v3.2-Q8_0.gguf --quant Q8_0

# Speed (Stage 6, profile turn-publication-v1; shared build for the Python binding)
cmake -B build-shared -DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_BUILD_EXAMPLES=OFF
cmake --build build-shared --target transcribe
uv run --project scripts/envs/smart_turn scripts/turn/bench.py --variant smart-turn-v3.2 \
  --library build-shared/src/libtranscribe.dylib
uv run scripts/catalog/ingest_perf.py

# Accuracy (Stage 7): Pipecat test set per quant, scored against the fp32 ONNX column
cmake --build build --target transcribe-turn-eval
for q in F32 F16 Q8_0; do
  uv run --project scripts/envs/smart_turn scripts/turn/eval_cpp.py \
    --model models/smart-turn-v3.2/smart-turn-v3.2-$q.gguf \
    --ref reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl \
    --out reports/turn/smart-turn-v3.2-$q.smart-turn-data-v3.2-test.jsonl
  uv run --project scripts/envs/smart_turn scripts/turn/score.py \
    reports/turn/smart-turn-v3.2-$q.smart-turn-data-v3.2-test.jsonl --ref-column p_fp32
done
uv run scripts/catalog/ingest_accuracy.py --models smart-turn-v3.2
uv run scripts/catalog/render.py
```

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|---|---|---|---|---|---|
| Clip score | `transcribe_turn_run` | `validate.py` (`head.prob` + tensors) | within `tests/tolerances/smart_turn.json`; decision equal to `turn.json` on every case | MUST PASS | **PASS**: 16/16 tensors on all 8 cases, max abs dp 3.3e-6 |
| 8 s window | crop / left pad | golden `jfk`, `noise`, `turn-incomplete-en-long` (crop); `ja`, `turn-*` (pad) | as above; audio before the last 8 s does not change p | MUST PASS | **PASS**: golden cases; `smart_turn_real_smoke` (jfk tail == whole), `smart_turn_smoke` (toy: crop and pad identities) |
| Test-set parity | decision vs onnxruntime | `scripts/turn/eval_cpp.py` (via `tools/transcribe-turn-eval`) | decision agreement and accuracy vs the fp32 baseline above | MUST PASS | **PASS**: 31,527 clips, 100% decision agreement, 93.71% accuracy (= reference), max abs dp 2.4e-5 |
| Threshold | `transcribe_turn_params::threshold` | unit test | default 0.5 from the GGUF; override honoured | MUST PASS | **PASS**: `smart_turn_real_smoke`, `turn_dispatch_unit` |
| Streaming | `transcribe_turn_stream_*` | - | NOT_IMPLEMENTED (`frame_samples` 0) | OUT OF SCOPE — the model scores clips | verified (`smart_turn_smoke`) |
| Speech track | `has_speech` | - | empty | OUT OF SCOPE — the model has no speech head | verified (`smart_turn_smoke`) |
| Transcribe / translate / timestamps | ASR | - | UNSUPPORTED_ROLE | OUT OF SCOPE — not an ASR model | verified |
| Batch (offline) | serial | - | - | OUT OF SCOPE — TURN has no batch API | n/a |

## Notes

- Stage 4 (C++): `src/arch/smart_turn/` reuses `whisper::build_encoder_graph`,
  `build_whisper_encoder_weights` and `read_whisper_frontend_hparams` (split out
  of the Whisper loader, behaviour unchanged) and the Whisper log-mel. The
  waveform normalisation is computed in double on the host. Manual attention by
  default (`TRANSCRIBE_FLASH_ATTN` overrides). `validate.py` compares the TURN
  role's `turn.json` (p within the manifest tolerance, decision).
- Tolerances finalized: see `tests/tolerances/smart_turn.json` `_comment`
  (widened entries and the massive-activation mechanism).
- Speed: the catalog's cells (profile `turn-publication-v1`) were measured on
  an Apple M1 standing in for the unreachable M4 reference machine; see the
  [model page](../../models/smart-turn-v3.2.md#performance). onnxruntime on the
  same M1: fp32 83 / 28 ms per 8 s window (1 / 4 threads), int8 40 / 15 ms.
- Quant policy: `head.cls.1.weight` (a LayerNorm scale whose name the generic
  norm rules miss) and the one-row projections `head.pool.2.weight` /
  `head.cls.6.weight` are pinned to the Norm bucket (F32) in
  `tools/transcribe-quantize/policy.cpp` and `reference_dtype_for`; the conv
  stem is in the Conv bucket (F32 under Q8_0), `enc.pos_emb.weight` and
  `frontend.*` in Norm. Q8_0 (`build/bin/transcribe-quantize ... --quant
  Q8_0`) has 27 Q8_0 and 54 F32 tensors, 10.7 MB. Shipped: F32, F16 and Q8_0;
  test-set results are on the [model page](../../models/smart-turn-v3.2.md#accuracy).

## Reproducibility

The Mac-built GGUFs are canonical: their sha256 are in
`catalog/smart-turn-v3.2.json` (downloads) and the convert manifests.
Converting on another platform gives identical bytes for every upstream
weight and for the metadata. A converter-computed constant can differ in the
last f32 bit: `frontend.mel_filterbank` (transformers' `mel_filter_bank`),
because float64 / float32 libm `exp` / `log` round differently across
platforms. On the Ryzen 7 PRO 4750U the effect on outputs is about 1e-7.
Cross-platform byte reproducibility is not a goal (owner decision).
