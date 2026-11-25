# SAVA (streaming end of turn and speech)

Status: supported (TURN and VAD roles). F32, F16 and Q8_0 uploaded privately
to Hugging Face on 2026-11-25. Usage: [`docs/turn.md`](../../turn.md),
[`docs/vad.md`](../../vad.md); downloads, accuracy and speed: the
[model page](../../models/sava.md).

SAVA reads 16 kHz audio in 640-sample (40 ms) frames. Every frame it gives
an end-of-turn probability and, 3 frames later, a speech probability
(architecture pattern `encoder-classifier`: a causal streaming encoder with
two per-frame heads). There is no WER: acceptance is per-frame probability
parity with the reference and the same clip decisions.

## Identity

- Family key `sava`, variant `sava`, `general.architecture = sava`.
- Upstream code: <https://github.com/inworld-ai/sava> tag `v1.0.0`
  (`b4221a5`); PyPI `inworld-sava==1.0.0` is byte-identical to its
  `src/inworld/sava`.
- Weights: HF `inworld/sava` at `1beae7c1dd0df6b0f92030644fc905eb087dc6a0`,
  `sava.safetensors` (77 tensors, all bf16, 25,521,558 parameters; sha256 on
  the model page and in the golden manifest). The configuration is
  safetensors metadata (`format=sava`, `config=<json>`); there is no
  `config.json`.
- License: Apache-2.0 for code and weights.

## References

- Canonical: `inworld.sava.run(model, audio)` for clips, `sava.Session`
  push / flush for streams, models loaded with `sava.load` (bf16 upcast to
  fp32). torch 2.14.1, fp32, CPU, one thread (`scripts/envs/sava`).
- Instrumented: `scripts/dump_reference_sava_author.py`: forward hooks and
  method wrappers on the same `sava.run` call; it asserts the captured logits
  reproduce `sava.run`'s curves bit for bit.
- Upstream's golden test (`tests/inworld/sava/test_golden.py`, logits from
  Inworld's training code, atol 2e-4) passes in `scripts/envs/sava`; our
  dumps are within 3.3e-5 of that fixture (`golden.json`).

## Forward pass

A frame `t` is samples `[640 t, 640 t + 640)`; a step is one pass over one
frame. All math is fp32. `rms(x)` is `x / sqrt(mean(x^2) + 1e-6)`, times the
weight when there is one.

```
ctx      = the 240 samples before the frame (zeros before the stream)
window j = [ctx | frame][160 j : 160 j + 400], j = 0..3     ends where hop j ends
mel_j    = (log10(max(mel_fb . |DFT(hann . window)|^2, 1e-10)) + 4) / 4   128 bins
row_t    = [mel_0 | mel_1 | mel_2 | mel_3 | ln(max(rms(frame), 1e-5))]    513
mixer_t  = rms(mixer.proj @ [row_{t-1} | row_t]) * norm_out             row_{-1} = 0
trunk    = 3 x Mamba-2 block, 1 x attention block, each
           x += mix(rms(x) * norm_mix); x += down(silu(gate(h)) * up(h)), h = rms(x) * norm_mlp
           then rms(x) * trunk.norm_out
v        = rms(blk(head.vad.down @ (rms(trunk) * norm_in))) * norm_out
logit    = head.vad.readout . v + b            on time: frame t
logit_la = head.vad.readout_la . v + b         judges frame t - 3; -30 at steps 0..2
speech_in_t = sigmoid(logit_{t-1})             the on-time logit; 0 at step 0
turn_in  = [rms(trunk) * head.turn.norm_in | speech_embed * speech_in_t + b]   576
turn     = head.turn.readout @ rms(blk(head.turn.down @ turn_in)) * norm_out + b
```

`turn` has 4 horizons (0, 5, 10, 20 frames); column 0 is the released
end-of-turn score. There is no Whisper `max - 8` clamp, dither or
pre-emphasis. The turn head depends on the previous step only through
`speech_in`, so a block can run the VAD head, the sigmoid, a one-step shift
and the turn head as one batch.

The front end runs on the host in fp64 (mixed-radix FFT, 400 = 2^4 * 25)
and matches an fp64 NumPy derivation to 6e-8. The reference's fp32 DFT GEMM
is cancellation-bound near the 1e-10 floor (7.9e-4 off on `dots-full`), so
the `fe.mel` drift is the reference's. The C++ does not read
`frontend.dft_basis`.

### Mamba-2 block

```
u       = ssm.in @ h                     z [0,1024) | xBC [1024,2304) | dt [2304,2320)
xBC     = silu(causal_conv_k4(xBC) + conv1d.bias)
x, B, C = xBC[0:1024] as [16 heads, 64], xBC[1024:1152], xBC[1152:1280]    1 group
h_t     = exp(softplus(dt + dt.bias) * A) h_{t-1} + (x * dt) B^T      A = ssm.a, per head
y       = (h_t C + x * ssm.d) * silu(z)
out     = ssm.out @ (rms(y over 1024, no weight) * ssm.norm_gated)
```

- `ggml_ssm_conv(sx, conv1d.weight)` with `sx` ne `[3 + T, 1280]` (the conv
  state, the last 3 raw xBC columns, then `transpose(xBC)`). The kernel reads
  the weight as F32.
- `ggml_ssm_scan` with `A` ne `[1, 16]` takes the scalar-per-head Mamba-2 path
  and applies softplus itself (threshold 20, as torch). It is the exact
  recurrence; the reference's offline path is the chunked SSD (chunk 64),
  which agrees to rounding (`ssm_y`, `ssm_state` dumps).

### Attention block (trunk block 3 and both heads)

```
q, k, v = split(attn.qkv @ h)           q 8 x 64 | k 2 x 64 | v 2 x 64
q, k    = rope(rms(q) * q_norm, p), rope(rms(k) * k_norm, p)       per head
o_h     = softmax(q_h . K_g / 8) V_g     g = h // 4, keys at p-511..p
o_h    -= (o_h . v_g(p)) / max(|v_g(p)|^2, 1e-12) * v_g(p)        XSA, own v, not roped
out     = attn.out @ concat(o_h)
```

- RoPE is NeoX rotate-half over 64 dims with `rope.freqs` (torch's fp32
  table; `ggml_rope_ext`'s iterated `theta_scale` rounds differently), so the
  host builds cos / sin and the rotation is plain ggml ops.
- Positions are the absolute step index since the stream started; flush
  steps advance it, nothing resets it within a stream.
- The reference computes fp32 angles, so its own outputs drift with stream
  age (on `dots.wav` at an offset, vs offset 0: 3.3e-5 after 5 min, 4.2e-4
  after 1 h, 3.0e-3 after 11 h, 5.7e-2 after 4.6 days). The model is shift
  invariant in exact arithmetic. Owner decision: the C++ uses fp64 angles
  from the int64 position, `(double) p * (double) rope.freqs[i]`, rounded to
  fp32 cos / sin, and never rebases. On `dots-full` it departs from the
  reference by the reference's drift (speech 4e-6 in the first 1,000 frames,
  3.9e-5 after 6,000; fp32 angles would stay at 1.2e-5).

## Streaming alignment

- Step `t` emits the turn scores of frame `t` (no lag) and the lookahead
  speech verdict on frame `t - 3` (`sigmoid(-30)` for `t < 3`, dropped).
- `Session.flush()` zero-pads a partial frame, then steps 3 frames of
  silence, even with no audio pushed. They release the last 3 speech verdicts
  and 3 turn rows (`flush_turns`) that the library does not return (they
  score padding).
- `sava.run(audio)` is push(all) then flush: `n = ceil(samples / 640)`
  frames, `speech[t] = out_la[t + 3]`, `turn[t] = out_turn[t]`.
  `transcribe_turn_run`'s `p` is `turn[n - 1, 0]`. Empty audio gives empty
  tracks.
- In the reference, run and stream differ by up to 3.1e-6 (chunked SSD and
  512-frame attention windows offline, the recurrence and a per-frame turn
  head in a Session). The C++ is bit-identical by construction (below).

Inworld's turn rule (`segmenter.py`, `Policy()`) is the caller's: see
[`docs/turn.md`](../../turn.md) and `examples/turn_live`. Its thresholds are
compared as Python floats, which is why the GGUF stores them as float64
(0.9 in fp32 is 0.89999998). Its segments can overlap and run up to 3 frames
past the audio, and on the last 3 frames it reads the flush rows, so a live
caller may differ there.

## GGUF contract

KVs (F32 unless noted; u32 counts; f64 thresholds):

- Identity: `stt.variant`, `stt.sava.{source_sha256, source_revision}`,
  `stt.capability.streaming = true`.
- Front end: `stt.frontend.{sample_rate 16000, n_fft 400, win_length 400,
  hop_length 160, num_mels 128, window "hann"}`, `stt.vad.frame_samples 640`,
  `stt.sava.{frame_samples 640, context_samples 240, mel_floor 1e-10,
  log_energy_floor 1e-5}`.
- Topology: `stt.sava.{hidden_size 512, ffn_size 2048, rms_eps 1e-6, layers
  ["mamba" x 3, "attention"], mixer_context_frames 2, turn_horizons
  [0,5,10,20], vad_embed_size 64, lookahead_frames 3, no_speech_logit -30}`;
  `stt.sava.mamba.{expand 2, head_dim 64, state_size 128, num_groups 1,
  conv_kernel 4, chunk_size 64}`; `stt.sava.attention.{num_heads 8,
  num_kv_heads 2, head_dim 64, sliding_window 512, rope_theta 1e4, xsa
  (bool), xsa_alpha 1}`; `stt.sava.{vad_branch,turn_branch}.{hidden_size
  512, ffn_size 1536, attention.*}`.
- Decisions: `stt.sava.threshold` (f64) 0.9, the turn cut-off. Also
  written, not read: `stt.sava.speech_threshold` (f64) 0.5 and Inworld's
  rule, `stt.sava.policy.*`.

Tensors (86; `ne` in ggml order; "F32" means F32 in every GGUF):

| name | ne | from upstream |
|---|---|---|
| `frontend.dft_basis` | [400, 402] | `mel.dft_basis().T` (F32; unused, kept so the published GGUFs stay byte-identical) |
| `frontend.mel_filterbank` | [201, 128] | `mel.mel_filterbank()` (F32) |
| `rope.freqs` | [32] | `modules.rope_frequencies(64, 1e4)` (F32) |
| `mixer.proj.weight`, `mixer.norm_out.weight` | [1026, 512], [512] | `mixer.projection`, `mixer.norm` |
| `trunk.{0,1,2}.norm_mix.weight` | [512] | `mixer_norm` |
| `trunk.{0,1,2}.ssm.in.weight` | [512, 2320] | `mixer.in_proj` (rows z / xBC / dt) |
| `trunk.{0,1,2}.ssm.conv1d.{weight,bias}` | [4, 1280], [1280] | `mixer.conv_{weight,bias}` (weight F32) |
| `trunk.{0,1,2}.ssm.{dt.bias,a,d}` | [16], [1, 16], [16] | `dt_bias`, `-exp(a_log)` (F32), `d_skip` (F32) |
| `trunk.{0,1,2}.ssm.norm_gated.weight`, `.ssm.out.weight` | [1024], [1024, 512] | `mixer.norm`, `mixer.out_proj` |
| `trunk.3.norm_attn.weight` | [512] | `attn_norm` |
| `<attn>.attn.qkv.weight` | [512, 768] | `attention.qkvr` (rows q / k / v) |
| `<attn>.attn.{q,k}_norm.weight`, `.attn.out.weight` | [64], [512, 512] | `attention.{q,k}_norm`, `attention.wo` |
| `<blk>.norm_mlp.weight` | [512] | `mlp_norm` |
| `<blk>.mlp.{gate,up}.weight`, `.mlp.down.weight` | [512, F], [F, 512] | `mlp.w13` (gate rows first), `mlp.w2` |
| `trunk.norm_out.weight` | [512] | `final_norm` |
| `head.vad.{norm_in,norm_out}.weight`, `head.vad.down.weight` | [512], [512, 512] | `vad_norm`, `vad_branch.{out_norm,down}` |
| `head.vad.readout{,_la}.{weight,bias}` | [512, 1], [1] | `vad_branch.readout`, `vad_la_readout` |
| `head.turn.{norm_in,norm_out}.weight`, `head.turn.down.weight` | [512], [576, 512] | `turn_norm`, `turn_branch.{out_norm,down}` |
| `head.turn.speech_embed.{weight,bias}` | [64] | `vad_embed` (Linear 1 -> 64; weight F32) |
| `head.turn.readout.{weight,bias}` | [512, 4], [4] | `turn_branch.readout` |

`<attn>` is `trunk.3`, `head.vad.blk` or `head.turn.blk`; `<blk>` is any
trunk or head block; F is 2048 in the trunk, 1536 in the heads.

Names avoid quant-policy collisions: norm scales use the `norm_` prefix
(kept F32 by the existing rule), and the heads are `head.vad.*` because
`vad.*` belongs to parakeet-ultra's conv rule. The F32 rules for
`frontend.dft_basis`, `rope.freqs`, `*.ssm.{a,d,conv1d.weight}` and
`*.speech_embed.weight` live in `tools/transcribe-quantize/policy.cpp` and
`scripts/lib/gguf_common.py`, pinned by
`scripts/lib/test_quant_policy_sync.py`.

## C++ implementation (`src/arch/sava/`)

`weights.cpp` reads the hparams and binds the weights; `graph.cpp` holds the
step graph, the stream state and the host front end; `model.cpp` the load,
the TURN and VAD ops tables and the validation dumps. `AUTO` resolves to the
CPU; Metal runs on request (jfk `p` 0.536095 vs 0.536394 on CPU), unvalidated.

Steps run in blocks of up to 64 frames, one graph per block, one column per
frame, at least 4 columns (padding after the real ones):

- ggml's CPU matmul takes tinyBLAS only from 4 columns on NEON; every other
  op computes a column the same way whatever the width. The attention query
  is made contiguous so K.Q takes tinyBLAS too.
- So every frame is bit-identical whatever the chunking or thread count:
  stream == run on the CPU (`transcribe_sava_smoke`,
  `transcribe_sava_real_smoke`). Conditions: no BLAS backend in the
  scheduler, and `d_state` a multiple of `ggml_ssm_scan`'s SIMD step (the
  scalar remainder rounds a graph's first token differently; the toy fixture
  uses 64). On CPUs with int8 matmul instructions (i8mm, AVX-VNNI) ggml may
  pick per-width kernels: re-run `transcribe_sava_real_smoke` there.

State (about 3.4 MB per session) lives in a backend buffer the graph writes
after its last read: the mixer's last row, the last on-time speech
probability, per Mamba block the conv tail (`sx[n_real : n_real + 3]`) and
the SSM state (`ggml_ssm_scan` with `K = n - n_real + 1` snapshots), and per
attention layer K and V rings of `window + 64` slots. Position `p` goes to
slot `p % ring` via `ggml_set_rows` before the block's attention; an additive
-inf mask selects `[p - 511, p]`, so the reduction order depends on `p` only.
The V ring is stored transposed so the value product reads it in place. Rings
are zeroed at reset.

`transcribe_turn_stream_feed_batch` (`step` in `graph.cpp`) steps one frame of
N sessions in graphs of up to 64 sessions (one graph exhausts ggml-alloc's
free blocks past ~120; fewer for a trunk deeper than 4 layers): weight matmuls, norms and element-wise ops run once
over `max(N, 4)` columns; the conv, scan, ring writes, attention (query
repeated to 4 columns) and state write-backs run per session in the shape of
a 1-frame step, so each session gets exactly its own feed's values
(`transcribe_sava_smoke`: 1-8 and 130 sessions, ragged chunks, 1 and 4 threads;
`transcribe_sava_real_smoke`: 4 users, F32 / F16 / Q8_0). Each chunk's graph
lives in its first session, keyed by process-unique `Stream::uid`s, and is
rebuilt when the list changes. Sessions share the model's CPU backend
(`work_data`, `n_threads`), so the one-compute-per-model rule of
[roles](../../roles.md) is required for safety. With `TRANSCRIBE_DUMP_DIR`
set, the batch steps sessions one by one so each trace stays complete.

A streamed frame computes 4 columns for 1. Dropping that needs a ggml change
giving every width the same kernel (about 25% faster, bit-exact) or giving
up stream == run (1.4e-6); both were declined for now.

## Validation data

Dumps: `build/validate/sava/sava/<case>/ref/`, per-step tensors `[S, ...]`
with `S = n_frames + 3`, named as the gates in
`tests/tolerances/sava.json` (front end, mixer, every trunk block, the SSM
outputs and states, both heads), plus `sava.{speech,turn}`,
`stream.{speech,turn}` (a Session pushed 13 / 1280 / 777 samples
cyclically), `stream.json`, `golden.json` and `turn.json` (the clip's `p`
and decision, compared with the CLI by `validate.py`). `validate.py` needs
no SAVA code: per-case `"stages": ["encoder"]` and `"role": "turn"`.

Cases (`tests/golden/sava/sava.manifest.json`): `sava-speech` (upstream's
fixture, 574 frames, wraps the attention window), `jfk`, `jobs-silence`,
`ja`, `dots`, `silence` (digital zeros: mel and energy floors, partial last
frame), `noise`, and `dots-full` (306 s, 7,648 frames, for drift).

## Quantization

Shipped: F32 (the reference), F16 and Q8_0 (`scripts/lib/quant_policy.py`).
Each stays bit-identical between stream and run. `mixer.proj`'s 1026-wide
rows fall back to F16 under Q8_0.

| | F16 | Q8_0 | BF16 (not shipped) |
|---|---|---|---|
| File | 51.9 MB | 28.5 MB | 51.9 MB (`convert-sava.py --outtype bf16`) |
| Weights vs the bf16 source | exact except 1,065 values (<= 3e-8) | 8-bit blocks | exact |
| Matmul activations | fp32 | int8 blocks | bf16 |
| Golden per-frame max abs vs F32, speech / turn | 1.2e-6 / 1.5e-6 | 0.050 / 0.066 | 0.013 / 0.033 |
| Golden turns vs F32 | 8 of 8 identical | 7 of 8; `dots-full` 6 of 59 change | 7 of 8; `dots-full` 5 of 59 change |
| F32 tolerance gate | passes | - | - |
| Pipecat flips vs F32 at 0.5 / 0.9 | 0 / 0 of 31,527 | 76 / 126 of 31,527 | 1 / 3 of 3,000 |
| Speed vs F32 (M1) | slower per clip (no tiled F16 x F32 kernel on NEON) | fastest streamed frame | 15-22x slower (no native BF16) |

Every `dots-full` change sits at a score within 0.004 of 0.9 or 0.03 of 0.5.
BF16 is lossless storage but lossy compute, so the 16-bit tier is F16. K
tiers were not evaluated.

## Upstream claims checked

- 25.5M parameters, 51 MB bf16: yes. Speech 120 ms late, turn without lag:
  yes. Golden parity to 2e-4: yes (3.3e-5).
- Pipecat test-set accuracy and AUC: yes (per-clip PyTorch reference
  93.36%, AUC 0.97946; published 93.35%, 0.97946).
- "3.8 ms per 40 ms frame on an M3 Max": not reproduced; on an M1, torch
  2.14.1 takes 10.5 ms per streamed frame (1 thread). Different machine, not
  investigated.
- The VAD frame AUC benchmark was not checked (needs its dataset).

## Commands

```bash
uv sync --project scripts/envs/sava
uv run scripts/validate.py ref --family sava --variant sava     # all 8 cases (~4 min)
uv run --project scripts/envs/sava scripts/convert-sava.py      # models/sava/sava-F32.gguf
uv run scripts/preflight.py --family sava --variant sava --gate B
uv run scripts/validate.py all --family sava --variant sava
ctest --test-dir build -R sava                                  # toy fixture
TRANSCRIBE_SAVA_GGUF=$PWD/models/sava/sava-F32.gguf ctest --test-dir build -R sava_real

# Pipecat test set: per-clip PyTorch reference (~17 min on 6 workers),
# then each GGUF scored against it (--ref is Smart Turn's run: the clip list)
uv run --project scripts/envs/sava --with pyarrow scripts/turn/run_reference_sava_torch.py \
  --out reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl
uv run --project scripts/envs/smart_turn scripts/turn/eval_cpp.py \
  --model models/sava/sava-F32.gguf \
  --ref reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl \
  --out reports/turn/sava-F32.smart-turn-data-v3.2-test.jsonl
uv run --project scripts/envs/smart_turn scripts/turn/score.py \
  reports/turn/sava-F32.smart-turn-data-v3.2-test.jsonl \
  --ref-jsonl reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl
uv run scripts/catalog/ingest_accuracy.py --models sava
```

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
|---|---|---|---|---|---|
| Whole-clip turn scores (`transcribe_turn_run`) | offline | `validate.py all` | `sava.turn[:,0]`, `sava.speech` and `p` within `tests/tolerances/sava.json` | MUST PASS | PASS (8 cases; p within 2.7e-6) |
| Turn segments | - | - | - | OUT OF SCOPE - removed from the API in review | SKIP - passed before removal (all 8 cases, 59 turns on `dots-full`) |
| Streaming (`transcribe_turn_stream_*`) | live | `transcribe_sava_smoke`, `transcribe_sava_real_smoke` | any chunking bit-identical to run on CPU; within tolerance of `stream.*` | MUST PASS | PASS (1 sample to whole clip, 1 and 4 threads, jfk and dots) |
| Speech track through `transcribe_vad_*` | live / offline | the same smokes | the same speech scores as the TURN speech track, lookahead 3 | MUST PASS | PASS (bit-identical) |
| `transcribe_vad_run` segments | offline | one-off check (below) | Silero's segmentation applied to `sava.speech` on 40 ms frames | MUST PASS (proposed) | PASS (jfk, sava-speech, dots, dots-full, ja: identical) |
| Long stream (5 min) | offline | `dots-full` case | within tolerance, same decision | MUST PASS (proposed) | PASS (drift documented) |
| Horizons 200 / 400 / 800 ms (turn columns 1-3) | - | dumped only | - | OUT OF SCOPE - not in the API | SKIP - not exposed by runtime (gated as dump tensors) |
| On-time speech output | - | dumped only | - | OUT OF SCOPE - internal (feeds the turn head) | SKIP - not exposed by runtime (gated as dump tensors) |
| F16 / Q8_0 GGUFs | offline | Pipecat test set (`scripts/turn/eval_cpp.py`), golden cases | accuracy and AUC vs F32 | MUST PASS | PASS (F16 = F32; Q8_0 93.36%, AUC 0.97935) |
| Metal / Vulkan | - | - | - | OUT OF SCOPE - CPU first (`AUTO` selects the CPU) | ACCEPTED GAP - Metal runs, unvalidated |
| Batch (`run_batch`) | - | - | - | OUT OF SCOPE - not an ASR role | SKIP - not exposed by runtime |

The `transcribe_vad_run` check applied silero-vad 6.2.3's
`get_speech_timestamps_from_probs` (window 640 instead of 512) to the
reference `sava.speech` and compared with the C++ segments, once; it is not
wired into a test.

## Reproducibility

The Mac-built GGUFs are canonical (sha256 in `catalog/sava.json` and the
convert manifests). Elsewhere the converter writes identical upstream
weights and metadata, but `frontend.mel_filterbank` and `ssm.a` can differ in
the last f32 bit (libm `exp` / `log`); on the Ryzen 7 PRO 4750U the effect
on outputs is about 1e-7. Cross-platform byte reproducibility is not a goal
(owner decision).
