# Smart Turn v3.2 (end of turn)

Status: supported (TURN role). F32, F16 and Q8_0 uploaded privately to
Hugging Face on 2026-11-25. Usage: [`docs/turn.md`](../../turn.md);
downloads, accuracy and speed: the
[model page](../../models/smart-turn-v3.2.md).

Smart Turn answers "has the speaker finished their turn?" for one clip of
16 kHz audio (architecture pattern `encoder-classifier`). It scores the last
8 s: the whisper-tiny encoder, attention pooling over its 400 positions and a
small MLP give one probability. It does not stream and does not read text.

## Identity

- Family key `smart_turn`, variant `smart-turn-v3.2`,
  `general.architecture = smart_turn`.
- Upstream module: `SmartTurnV3Model` in smart-turn's `train.py`
  (github.com/pipecat-ai/smart-turn @ `4e12f2b`), exported to ONNX.
- Hugging Face: `pipecat-ai/smart-turn-v3` at
  `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`; source file
  `smart-turn-v3.2-gpu.onnx` (fp32; sha256 on the model page). The `-cpu.onnx`
  is upstream's static int8 quantization; it is only scored for comparison.
- License: BSD-2-Clause. The `turn-*` golden clips from
  `pipecat-ai/smart-turn-data-v3.2-test` are CC-BY-4.0 (`samples/README.md`).
- Parameters: 8,000,386.

## References

- Canonical: onnxruntime 1.23.2 on the fp32 ONNX, run as smart-turn's
  `inference.py` `predict_endpoint` does (`ORT_SEQUENTIAL`, inter-op 1,
  `ORT_ENABLE_ALL`; CPU, 1 thread). Front end: transformers 4.48.2
  `WhisperFeatureExtractor(chunk_length=8)` on its numpy path. Decision
  `p > 0.5`. Versions pinned in `scripts/envs/smart_turn/pyproject.toml`
  (those smart-turn's `requirements.txt` exports with).
- Instrumented: `scripts/dump_reference_smart_turn_onnx.py`, the same graph
  with the dump points as outputs. That stops a few fusions, so its p is
  within 6e-8 of the canonical session (bounded at 1e-6; the canonical p is
  written as `head.prob`).
- Cross-check: a PyTorch twin, `SmartTurnV3Model` copied from `train.py`
  with the ONNX weights mapped in (`scripts/lib/smart_turn_onnx.py`), torch
  2.9.0. The dumper writes its tensors to `<out>/torch/` and the differences
  to `<out>/bridge.json`.

## Front end

```
window = audio[-128000:]  or  [zeros(128000 - n) | audio]   left padded
x      = (window - mean) / sqrt(var + 1e-7)      whole window, padding included
mel    = Whisper log-mel of x: reflect pad 200, periodic Hann 400, hop 160,
         |X|^2, slaney 80-bin filterbank, log10(max(., 1e-10)),
         drop the last frame, max(., max - 8), (. + 4) / 4      -> [80, 800]
```

transformers switches to a float32 torch STFT when torch is importable.
`inference.py` has no torch and Pipecat ships a numpy copy
(`_whisper_features.py`), so the reference pins the numpy path (vs the torch
path: 4.0e-5 in the mel, 4.2e-7 in p; vs Pipecat's copy: 1.2e-7, 1.8e-6). The
filterbank is transformers' `mel_filter_bank(slaney)`; the GGUF stores it.

## Forward pass

```
h  = GELU(conv1(mel)) ; GELU(conv2(h)) ; + pos_emb[400]     whisper-tiny stem
h  = 4 x pre-LN block (6 heads x 64, k without bias, FFN 1536, erf GELU)
h  = LayerNorm(h)                                            [400, 384]
s  = Linear(256,1)(tanh(Linear(384,256)(h)))                 [400]
w  = softmax_t(s) ; pooled = sum_t w[t] h[t]                 [384]
z  = Linear(64,1)(GELU(Linear(256,64)(GELU(LayerNorm(Linear(384,256)(pooled))))))
p  = sigmoid(z)
```

All LayerNorm eps are 1e-5. The export computes attention as torch SDPA:
q and k are each scaled by 64^-0.25.

The positional table must be loaded, not rebuilt: `train.py` sets 400
positions, so whisper-tiny's 1500-row table was re-initialised with
`sinusoids()` on the training device, and it differs from a CPU
`sinusoids(400, 384)` by up to 3.1e-5.

Whisper's massive activations (from block 2, channel 375, then 16, 128,
247, reach |x| 80 to 153) make fp32 reduction order visible in
`enc.block.{2,3}.out` and `enc.final` (twin vs onnxruntime up to 2.0e-2);
the final LayerNorm and the pooling contract it again (p within 5.4e-6).
Mechanism and widened entries: `tests/tolerances/smart_turn.json`.

## Tensor map

The ONNX stores 26 Linear weights as anonymous MatMul initializers (`val_N`)
in `[in, out]` order. `map_onnx_weights` names each by the bias `Add` that
consumes the MatMul (for the bias-free k projection, by the
`self_attn_layer_norm` feeding it) and transposes it. Loaded strictly into
the twin, the map reproduces onnxruntime's p to 5.4e-7 on random waveforms
and 5.4e-6 on the golden clips, as do the GGUF's tensors with a mel from the
GGUF's own filterbank and window.

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

(T) = transposed by the converter. Every tensor is F32 in the F32 GGUF; ggml
`ne` is the reverse of the numpy shape.

## GGUF contract

Keys: `stt.variant`, `stt.smart_turn.source_sha256`,
`stt.capability.{lang_detect, translate, timestamps, streaming}` (all false),
`stt.smart_turn.encoder.{n_layers 4, d_model 384, n_heads 6, ffn_dim 1536,
num_mel_bins 80, max_source_positions 400, activation "gelu"}`,
`stt.smart_turn.{max_audio_samples 128000, threshold 0.5,
waveform_normalize true, waveform_normalize_eps 1e-7}`, and
convert-whisper.py's `stt.frontend.*` with `chunk_length` 8, `n_samples`
128000, `nb_max_frames` 800. `general.*` carries the 23 languages, the
licence and the source URL.

Quant policy: `head.cls.1.weight` (a LayerNorm scale the generic norm rules
miss) and the one-row projections `head.pool.2.weight` / `head.cls.6.weight`
are pinned F32 in `tools/transcribe-quantize/policy.cpp` and
`reference_dtype_for`; the conv stem, `enc.pos_emb.weight` and `frontend.*`
stay F32 too. Q8_0 has 27 Q8_0 and 54 F32 tensors.

## C++ implementation (`src/arch/smart_turn/`)

Reuses `whisper::build_encoder_graph`, `build_whisper_encoder_weights`,
`read_whisper_frontend_hparams` and the Whisper log-mel. The waveform
normalisation runs on the host in double. Manual attention by default
(`TRANSCRIBE_FLASH_ATTN` overrides). `AUTO` resolves to the CPU.

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

`silence` normalises to zeros (variance 0), so the mel is constant and p sits
near the cut-off. Dumps (`build/validate/smart_turn/smart-turn-v3.2/<case>/ref/`)
are named as the gates in `tests/tolerances/smart_turn.json`, plus
`enc.pos_emb` and `head.pool_scores` (debug only) and `turn.json` (p, the
decision, the int8 ONNX's p).

Test-set baseline (`scripts/turn/run_reference_smart_turn_onnx.py`,
`p > 0.5`; FPR / FNR are percent of all clips, as upstream reports them):

| ONNX | all 31,527 clips | published |
|---|---:|---:|
| fp32 (`-gpu`) | 93.71% (FPR 3.51, FNR 2.78) | 93.71% (3.51, 2.78) |
| int8 (`-cpu`) | 92.62% (FPR 4.76, FNR 2.63) | 92.63% (4.73, 2.64) |

The fp32 numbers reproduce upstream's exactly, which pins the front end and
the decision rule. int8 (onnxruntime on arm64 here; upstream's host is not
recorded) agrees with fp32 on 96.12% of decisions (mean |dp| 0.042, max
0.95).

## Commands

```bash
uv run scripts/validate.py ref --family smart_turn             # reference dumps
uv run --project scripts/envs/smart_turn scripts/turn/run_reference_smart_turn_onnx.py \
  --out reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl
uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py
uv run scripts/preflight.py --family smart_turn --variant smart-turn-v3.2 --gate B
uv run scripts/validate.py all --family smart_turn --variant smart-turn-v3.2
build/bin/transcribe-quantize models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf \
  models/smart-turn-v3.2/smart-turn-v3.2-Q8_0.gguf --quant Q8_0   # and F16

# Speed (profile turn-publication-v1; shared build for the Python binding)
uv run --project scripts/envs/smart_turn scripts/turn/bench.py --variant smart-turn-v3.2 \
  --library build-shared/src/libtranscribe.dylib
uv run scripts/catalog/ingest_perf.py

# Accuracy: Pipecat test set per GGUF, against the fp32 ONNX column
cmake --build build --target transcribe-turn-eval
uv run --project scripts/envs/smart_turn scripts/turn/eval_cpp.py \
  --model models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf \
  --ref reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl \
  --out reports/turn/smart-turn-v3.2-F32.smart-turn-data-v3.2-test.jsonl
uv run --project scripts/envs/smart_turn scripts/turn/score.py \
  reports/turn/smart-turn-v3.2-F32.smart-turn-data-v3.2-test.jsonl --ref-column p_fp32
uv run scripts/catalog/ingest_accuracy.py --models smart-turn-v3.2
uv run scripts/catalog/render.py
```

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|---|---|---|---|---|---|
| Clip score | `transcribe_turn_run` | `validate.py` (`head.prob` + tensors) | within `tests/tolerances/smart_turn.json`; decision equal to `turn.json` on every case | MUST PASS | **PASS**: 14/14 gated tensors on all 8 cases (plus 2 debug-only dumps, `enc.pos_emb` and `head.pool_scores`), max abs dp 3.3e-6 |
| 8 s window | crop / left pad | golden `jfk`, `noise`, `turn-incomplete-en-long` (crop); `ja`, `turn-*` (pad) | as above; audio before the last 8 s does not change p | MUST PASS | **PASS**: golden cases; `smart_turn_real_smoke` (jfk tail == whole), `smart_turn_smoke` (toy: crop and pad identities) |
| Test-set parity | decision vs onnxruntime | `scripts/turn/eval_cpp.py` (via `tools/transcribe-turn-eval`) | decision agreement and accuracy vs the fp32 baseline above | MUST PASS | **PASS**: 31,527 clips, 100% decision agreement, 93.71% accuracy (= reference), max abs dp 2.4e-5 |
| Threshold | `transcribe_turn_params::threshold` | unit test | default 0.5 from the GGUF; override honoured | MUST PASS | **PASS**: `smart_turn_real_smoke`, `turn_dispatch_unit` |
| Streaming | `transcribe_turn_stream_*` | - | NOT_IMPLEMENTED (`frame_samples` 0) | OUT OF SCOPE — the model scores clips | verified (`smart_turn_smoke`) |
| Speech track | `has_speech` | - | empty | OUT OF SCOPE — the model has no speech head | verified (`smart_turn_smoke`) |
| Transcribe / translate / timestamps | ASR | - | UNSUPPORTED_ROLE | OUT OF SCOPE — not an ASR model | verified |
| Batch (offline) | serial | - | - | OUT OF SCOPE — TURN has no batch API | n/a |
| Non-CPU backends (Metal, CUDA, Vulkan) | `transcribe_turn_run` on a GPU device | golden cases on the device | p within tolerance of the CPU path | OUT OF SCOPE — the model is small and `AUTO` selects the CPU; a per-backend tolerance and test-set run would bring it in | ACCEPTED GAP — unvalidated: Metal drifts about 1e-3 in p (`jfk` 0.948773 vs CPU 0.948620); decisions held on the golden clips |

## Reproducibility

The Mac-built GGUFs are canonical (sha256 in `catalog/smart-turn-v3.2.json`
and the convert manifests). Elsewhere the converter writes identical
upstream weights and metadata, but `frontend.mel_filterbank` can differ in
the last f32 bit (libm `exp` / `log`); on the Ryzen 7 PRO 4750U the effect on
outputs is about 1e-7. Cross-platform byte reproducibility is not a goal
(owner decision).
