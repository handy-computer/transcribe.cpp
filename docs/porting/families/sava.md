# SAVA (Inworld): streaming speech and end-of-turn detection

Status: validation (Stages 1-4: intake, oracle, F32 GGUF, C++ in `src/arch/sava/`;
`validate.py all` green on the 8 golden cases).

SAVA reads 16 kHz mono audio in 640-sample (40 ms) frames. For every frame it
gives a speech probability and an end-of-turn probability. It serves the TURN
role (`docs/turn.md`) and the VAD role with lookahead. The architecture pattern
is `encoder-classifier`: a causal streaming encoder with two per-frame heads.
There is no WER. Acceptance is per-frame probability parity with the
reference and the same clip decisions.

## Identity

- Family key `sava`, variant `sava`, GGUF `general.architecture = sava`.
- Upstream: <https://github.com/inworld-ai/sava>, tag `v1.0.0`
  (`b4221a54458ea60df8c777911d0567418994cacc`). PyPI `inworld-sava==1.0.0` is
  byte-identical to that tag's `src/inworld/sava`.
- Weights: HF `inworld/sava` at `1beae7c1dd0df6b0f92030644fc905eb087dc6a0`,
  `sava.safetensors`, 51,051,668 bytes, sha256 `aa610ee9...4be34` (the value
  pinned in `inworld.sava.registry`). The file holds 77 tensors, all bf16,
  25,521,558 parameters. The configuration is stored as safetensors metadata
  (`format=sava`, `config=<json>`); there is no `config.json`.
- License: Apache-2.0 for code and weights.

## References

- Canonical reference: `inworld.sava.run(model, audio)` for whole recordings,
  `sava.Session` push/flush for streams, and `sava.segment(curves,
  sava.Policy(), 3)` / `sava.TurnSegmenter` for turns. The model is loaded with
  `sava.load` (bf16 upcast to fp32, CPU). Pins: torch 2.14.1, fp32, one thread
  (`scripts/envs/sava`).
- Instrumented reference: `scripts/dump_reference_sava_author.py`. It puts
  forward hooks and method wrappers on the same `sava.run` call, so every
  dumped intermediate is the reference's own value. It checks that the
  captured logits reproduce `sava.run`'s curves bit for bit.
- Upstream's golden test (`tests/inworld/sava/test_golden.py`, logits from
  Inworld's training code, atol 2e-4) passes in `scripts/envs/sava`. Our
  dumps differ from that fixture by at most 3.3e-5 in logits (`golden.json`).
- Cross-check: `build/validate/sava/check_sava_gguf.py` is a frame-by-frame
  NumPy model that reads only the F32 GGUF and its KVs. It matches the dumps
  to 2e-6 in probability, 2e-5 in logits and 1e-3 on `trunk.3.out` (p99 151),
  on `sava-speech` and `silence`. This validates the tensor layout and the
  math below.

## Forward pass (exact)

Notation: a frame `t` is samples `[640 t, 640 t + 640)`. A step is one pass of
the model over one frame. All math is fp32. `rms(x)` is
`x / sqrt(mean(x^2) + 1e-6)`, followed by a multiply by the weight when there
is one.

Front end (`mel.py`, no parameters; the GGUF stores the tables):

```
ctx      = the 240 samples before the frame (zeros before the stream)
window j = [ctx | frame][160 j : 160 j + 400], j = 0..3   # ends where hop j ends
[re; im] = frontend.dft_basis @ window                    # 402 x 400, Hann folded in
power    = re^2 + im^2                                    # 201 bins
mel_j    = (log10(max(frontend.mel_filterbank @ power, 1e-10)) + 4) / 4   # 128
fe.mel   = [mel_0 | mel_1 | mel_2 | mel_3]                # 512, hop-major
fe.log_energy = ln(max(sqrt(mean(frame^2)), 1e-5))       # natural log
```

There is no Whisper `max - 8` clamp, no dither and no pre-emphasis.

The C++ runs the front end on the host in fp64: a mixed-radix FFT
(400 = 2^4 * 25), the power, the mel bank from `frontend.mel_filterbank`,
and log10. It matches an fp64 NumPy derivation to 6e-8. The reference's
fp32 DFT GEMM is cancellation-bound in bins near the 1e-10 floor: it is off
from exact by up to 7.9e-4 on `dots-full` and 2.7e-4 on `jfk`. The C++ does
not read `frontend.dft_basis`, which stays in the GGUF as the reference's
fp32 table.

Mixer: `row_t = [fe.mel_t | fe.log_energy_t]` (513 values).
`mixer.out_t = rms(mixer.proj @ [row_{t-1} | row_t]) * mixer.norm_out`.
`row_{-1}` is all zeros, including the energy slot. It is not the row of a
silent frame.

Trunk: blocks 0-2 are Mamba-2, block 3 is attention. Each block computes
`x += mix(rms(x) * norm_mix|norm_attn)`, then
`x += down(silu(gate(h)) * up(h))` with `h = rms(x) * norm_mlp`.
After the trunk, `trunk.norm = rms(x) * trunk.norm_out`.

VAD head:

```
v       = rms(blk(head.vad.down @ (rms(trunk.norm) * head.vad.norm_in))) * head.vad.norm_out
logit   = head.vad.readout . v + b         # on time: frame t
logit_la= head.vad.readout_la . v + b      # judges frame t - 3; -30 for steps 0..2
```

Turn head:

```
speech_in_t = sigmoid(logit_{t-1})  (fp32; 0 at step 0)   # the ON-TIME logit
embed       = head.turn.speech_embed.weight * speech_in_t + bias      # 64
turn.input  = [rms(trunk.norm) * head.turn.norm_in | embed]           # 576
th          = rms(blk(head.turn.down @ turn.input)) * head.turn.norm_out
turn logits = head.turn.readout @ th + b    # 4 horizons: 0, 5, 10, 20 frames
```

Column 0 is the released end-of-turn score. The turn head depends on the
previous step only through `speech_in`, which comes from the VAD head. A
chunk can therefore run VAD head, then sigmoid, then a one-step shift (with
the carried value first), then the turn head as one batch. Upstream streams
the turn head frame by frame, but that is not required.

### Mamba-2 block and its ggml mapping

```
u        = ssm.in @ h                     rows: z [0,1024) | xBC [1024,2304) | dt [2304,2320)
xBC_c    = silu(depthwise_conv_k4(xBC, ssm.conv1d) + ssm.conv1d.bias)   # causal, raw xBC tail kept
x, B, C  = xBC_c[0:1024] as [16 heads, 64], xBC_c[1024:1152], xBC_c[1152:1280]   (1 group)
dt       = softplus(dt + ssm.dt.bias)     # torch threshold 20
h_t      = exp(dt * A) h_{t-1} + (x * dt) B^T     A = ssm.a = -exp(A_log), per head
y        = h_t C                          # [16, 64]
y        = (y + x * ssm.d) * silu(z)      # D per head
out      = ssm.out @ (rms(y over all 1024, no weight) * ssm.norm_gated)
```

How the steps map to ggml (`ggml/include/ggml.h`; CPU kernel in
`ggml/src/ggml-cpu/ops.cpp`, checked):

- `ggml_ssm_conv(sx, c)`. `sx` is ne `[3 + T, 1280, 1]`: the conv state (the
  last 3 raw xBC columns) concatenated with `transpose(xBC)`, time first.
  `c` is `trunk.i.ssm.conv1d.weight` with ne `[4, 1280]`. The output is
  `sum_k sx[t + k] * c[k]`, which matches upstream's `shift_conv`. The bias
  and silu are separate ops. The kernel reads `c` as F32, so the tensor is
  F32 in every GGUF.
- `ggml_ssm_scan(s, x, dt, A, B, C, ids, K=1)` with:
  - `s` ne `[128, 64, 16, 1]`: the state, laid out as the reference's
    `[H, P, N]`.
  - `x` ne `[64, 16, T, 1]`.
  - `dt` ne `[16, T, 1]`: raw `dt + dt.bias`. The kernel applies softplus
    itself (`x > 20 ? x : logf(1 + expf(x))`; torch uses `log1p`).
  - `A` ne `[1, 16]` (`trunk.i.ssm.a`). The scalar-per-head Mamba-2 path is
    taken because `A->ne[0] == 1`.
  - `B` and `C` ne `[128, 1, T, 1]`.
  - The kernel evaluates in the reference's order: `dA = expf(dt_sp * A)`,
    then `s = s * dA + B * (x * dt_sp)`, then `y = sum(s * C)`. It
    vectorises d_state 128 on NEON/AVX. Head dim 64, state 128 and one group
    are all supported. The output is `y` followed by the final state.
- D skip, `silu(z)` gating, the RMSNorm over 1024 values (eps 1e-6) and the
  `norm_gated` weight are ordinary ops. This is llama.cpp's Mamba-2 order.
- Offline, the reference uses the chunked SSD algorithm (`chunk_size` 64). A
  stream with one-frame pushes uses the exact recurrence. Both agree to
  rounding (`trunk.i.ssm_y`, `trunk.i.ssm_state` dumps). ggml's scan is the
  recurrence.

### Attention block (trunk block 3 and both heads)

```
q, k, v = split(attn.qkv @ h)          rows q [0,512) 8x64 | k [512,640) 2x64 | v [640,768) 2x64
q, k    = rope(rms(q) * q_norm, p), rope(rms(k) * k_norm, p)    # per-head RMS, eps 1e-6
o_h     = softmax(q_h . K_g / 8) V_g    g = h // 4, keys at positions p-511..p (incl. self)
o_h    -= alpha * (o_h . v_g(p)) / max(|v_g(p)|^2, 1e-12) * v_g(p)    # XSA, alpha 1
out     = attn.out @ concat(o_h)
```

- The KV cache holds normed and rotated keys, plus values, for the last 512
  positions. Positions below 0 (cache slots not yet filled) are masked.
- The GQA mapping `h // 4` is ggml's `mul_mat` broadcast order.
- XSA (exclusive self attention) uses the frame's own value vector: before
  RoPE, and not normed. ggml has no op for it. Build it as `mul` +
  `sum_rows` (dot, then squared norm), `clamp(min 1e-12)`, `div`, then a
  broadcast `mul` and `sub` on views `[64, 4, 2, T]` / `[64, 1, 2, T]`.
- RoPE is NeoX rotate-half over all 64 dims:
  `[x1 cos - x2 sin, x2 cos + x1 sin]` with
  `freq_i = 10000^(-i/32)`, i < 32.
  - Positions: the absolute step index since the stream started (int64 in
    torch, converted to fp32 for the angle). Every attention layer uses the
    same index. Nothing resets it within a stream, and flush steps advance
    it.
  - Angles: torch computes `fl32(fl32(p) * freq_i)` with `freq_i` from
    `rope_frequencies` in fp32. `rope.freqs` is exactly that table; NumPy
    `powf` differs from it in some entries.
  - `ggml_rope_ext` builds angles by repeated multiplication with
    `theta_scale = powf(base, -2/64)`. That rounds differently from the
    reference, by a few ULP of the angle.
  - Exact parity needs cos/sin tables built on the host from `rope.freqs`
    (`(float)p * freq[i]`, `cosf`/`sinf`), with the rotation done in ggml ops.
- Stream age changes the reference's own outputs, because fp32 angles lose
  precision as `p` grows. The model is mathematically shift invariant:
  attention depends only on `p_q - p_k`, and fp64 angles at any offset match
  offset 0 to 3e-6. Measured on `dots.wav` run at a position offset, against
  offset 0:

  | offset (frames) | stream age | speech max abs | turn max abs |
  |---:|---|---:|---:|
  | 7,650 | 5 min | 3.3e-5 | 1.1e-5 |
  | 90,000 | 1 h | 4.2e-4 | 1.1e-4 |
  | 1,000,000 | 11 h | 3.0e-3 | 1.0e-3 |
  | 10,000,000 | 4.6 days | 5.7e-2 | 1.1e-2 |

  Decision (owner): the C++ computes the angles in fp64 from the absolute
  int64 position, `(double) p * (double) rope.freqs[i]`, and rounds cos / sin
  to fp32. Positions are not rebased. This is the shift-invariant value the
  reference approximates, so the C++ departs from the reference by the
  reference's own drift: on `dots-full` (positions up to 7,650) speech
  differs by 4e-6 in the first 1,000 frames and 3.9e-5 after 6,000. With
  fp32 angles in the C++ the drift stays at 1.2e-5. Parity is tested on the
  short cases and on `dots-full` at the measured tolerance.

## Streaming alignment (what `run`, `feed` and `flush` must reproduce)

- `Session.push(pcm)` keeps any partial frame and steps every whole frame.
  Step `t` emits three values:
  - `speech[t]`: the lookahead verdict on frame `t - 3`. It is
    `sigmoid(-30) = 9.36e-14` for `t < 3`.
  - `turn[t, 0:4]`: whether the turn has ended by frame `t`. No lag.
  - The on-time speech probability, used internally only (it feeds the next
    step's `speech_in`).
- `Session.flush()` zero-pads the partial frame (when there is one), then
  steps 3 frames of silence. Those steps release the last 3 speech verdicts
  and emit 3 more turn rows (`flush_turns`). Flush always steps the 3
  silent frames, even when no audio was pushed. After flush the reference
  Session is closed.
- `sava.run(audio)`:
  - It runs `n = ceil(samples / 640)` frames. The audio is zero-padded to
    `(n + 3) * 640` samples and sent through one causal forward.
  - It returns `speech[t] = out_la[t + 3]`, `turn[t] = out_turn[t]` and
    `speech_on_time[t]` for `t < n`, and `flush_turns = out_turn[n:n+3]`.
  - Steps 0..2 of the lookahead output are dropped.
  - This is exactly push(all) followed by flush.
  - For `transcribe_turn_run`: `p = turn[n - 1, 0]`, the turn score of the
    last frame (which is zero-padded when partial). Per-frame scores are
    `turn[:, 0]` and `speech`.
  - Empty audio gives empty curves and no flush steps.
- Run and stream are not bit-identical in the reference. Offline uses the
  chunked SSD scan and 512-frame attention windows; Session steps use the
  per-frame recurrence and run the turn head frame by frame. Measured
  differences in probability, all 8 cases, odd chunks / 640-sample pushes /
  one push:
  - up to 3.1e-6 overall;
  - 1.4e-6 / 2.0e-6 / 6.0e-7 on `sava-speech`.

  The C++ should make them bit-identical by design: the same graph shapes
  and kernels for every chunking, as the Silero port does.
- `speech_in` at step `t` uses the on-time logit of step `t - 1`, including
  across chunk boundaries and into the flush steps.

## Reference turn policy (`segmenter.py`, `Policy()` defaults)

The library does not apply this rule: the API review removed turn segments,
so callers apply it to the stream's score tracks, as `examples/turn_live`
does (`docs/turn.md`). It is recorded here because it defines how the scores
are meant to be used. Before the removal, `transcribe_turn_run` returned
these turns and matched `sava.segment` on all 8 cases.

`sava.segment(curves, policy, 3)` replays the stream. Frame `j`'s speech
verdict comes out at step `now = j + 3`, and the turn score used with it is
the one from step `now`, column 0. For the last 3 frames that is a
`flush_turns` row. Per judged frame `j`:

1. Thresholds:
   - `is_speech = speech[j] >= 0.5`;
   - `confident = speech[j] >= 0.9`;
   - turn check: `turn[now] >= 0.9`.

   Each threshold is compared as a Python float (double) against an fp32
   value, so the GGUF stores the thresholds as float64.
2. When no turn is open:
   - a non-speech frame is ignored;
   - a speech frame opens a turn at `start = max(0, j - round(800 / 40))`.

   `round` is Python's half-to-even (`_frames(ms) = round(ms / 40)`). Only
   pre-roll goes through it.
3. A speech frame resets the pause and adds 40 ms of speech (and 40 ms of
   confident speech when confident).
4. A silent frame adds 40 ms to the pause. Then:
   - if `pause_ms >= 1600`, the turn closes at `end = now + 1`, reason
     `silence`;
   - otherwise, if `pause_ms >= 0` (min silence before check) and the turn
     score is `>= 0.9`, it closes at `end = now + 1`, reason `turn`.
5. `finish()` closes an open turn at `last_judged + 1 = n`, reason `end`.
6. A closed turn is `kept` if `speech_ms >= 400` or `confident_ms >= 200`.
   `segment()` returns dropped turns too, with `kept=false`.

Consequences for a caller copying the rule:

- Segments can overlap: `end = now + 1` is up to 3 frames past the judged
  frame, and the next turn's pre-roll is not clamped. Measured: 33 of 59
  turns overlap on `dots-full`, 6 of 7 on `dots`, 1 of 2 on `sava-speech`.
- `end` can run up to 3 frames past the audio. Map it to samples as
  `min(end * 640, n_samples)`.
- The flush steps' turn scores feed the segmenter. A stream's flush does
  not return them (they score padding), so a live caller copying Inworld's
  rule may differ on the last 3 frames.
- `turn_segments.json` holds both the offline turns and the live `TurnSegmenter`
  run over the dumped stream; they agree on every case.

## Per-session state (fp32)

| state | size |
|---|---:|
| mel context (previous 240 samples) | 240 |
| mixer previous row | 513 |
| 3 x Mamba conv tail (3 x 1280) | 11,520 |
| 3 x Mamba SSM state (16 x 64 x 128) | 393,216 |
| 3 x attention K and V (reference: 512 x 2 x 64 each; C++ ring: 576 x 2 x 64 each) | 393,216 (C++ 442,368) |
| previous on-time speech probability | 1 |
| step counter / RoPE position (int64) | - |
| pending partial frame | < 640 samples |

That is about 3.2 MB (3.4 MB in the C++). The lookahead needs no buffer: the speech verdict for
frame `t - 3` is read directly from step `t`'s `readout_la`.

## GGUF contract

KVs (F32 unless noted; u32 for counts; f64 for thresholds):

- Identity: `general.architecture = sava`, `stt.variant = sava`, and
  `stt.sava.source_sha256` / `stt.sava.source_revision`.
- Front end:
  - `stt.frontend.{sample_rate 16000, n_fft 400, win_length 400,
    hop_length 160, num_mels 128, window "hann"}`;
  - `stt.vad.frame_samples 640`;
  - `stt.sava.{frame_samples 640, context_samples 240, mel_floor 1e-10,
    log_energy_floor 1e-5}`.
- Topology:
  - `stt.sava.{hidden_size 512, ffn_size 2048, rms_eps 1e-6,
    layers ["mamba","mamba","mamba","attention"], mixer_context_frames 2}`;
  - `stt.sava.mamba.{expand 2, head_dim 64, state_size 128, num_groups 1,
    conv_kernel 4, chunk_size 64}`;
  - `stt.sava.attention.{num_heads 8, num_kv_heads 2, head_dim 64,
    sliding_window 512, rope_theta 1e4, xsa (bool) true, xsa_alpha 1}`;
  - `stt.sava.{vad_branch,turn_branch}.{hidden_size 512, ffn_size 1536}` and
    `.attention.*` with the same fields;
  - `stt.sava.{turn_horizons [0,5,10,20], vad_embed_size 64,
    lookahead_frames 3, no_speech_logit -30}`.
- Decisions:
  - `stt.sava.threshold` (f64) 0.9: the turn cut-off, `Policy.turn_threshold`;
  - `stt.sava.speech_threshold` (f64) 0.5;
  - `stt.sava.policy.{pre_roll_ms 800, min_silence_before_check_ms 0,
    max_silence_ms 1600, min_speech_ms 400, min_confident_speech_ms 200}`
    (u32);
  - `stt.sava.policy.confident_speech_threshold` (f64) 0.9;
  - `stt.capability.streaming = true`.

Tensors (86). `ne` is in ggml order, `[cols, rows]`. "F32 always" means F32
in every GGUF and every quant preset, because of the policy rules for these
names.

| name | ne | from upstream |
|---|---|---|
| `frontend.dft_basis` | [400, 402] | `mel.dft_basis().T`: cos rows 0..200, sin rows 201..401, Hann folded in (F32 always); not read by the C++, kept so the published GGUFs stay byte-identical |
| `frontend.mel_filterbank` | [201, 128] | `mel.mel_filterbank()` (F32 always) |
| `rope.freqs` | [32] | `modules.rope_frequencies(64, 1e4)` (F32 always) |
| `mixer.proj.weight` | [1026, 512] | `mixer.projection` (input `[row_{t-1} \| row_t]`) |
| `mixer.norm_out.weight` | [512] | `mixer.norm` |
| `trunk.{0,1,2}.norm_mix.weight` | [512] | `mixer_norm` |
| `trunk.{0,1,2}.ssm.in.weight` | [512, 2320] | `mixer.in_proj`, rows z / xBC / dt |
| `trunk.{0,1,2}.ssm.conv1d.weight` | [4, 1280] | `mixer.conv_weight` (F32 always) |
| `trunk.{0,1,2}.ssm.conv1d.bias` | [1280] | `mixer.conv_bias` |
| `trunk.{0,1,2}.ssm.dt.bias` | [16] | `mixer.dt_bias` |
| `trunk.{0,1,2}.ssm.a` | [1, 16] | `-exp(mixer.a_log)` in fp32 (F32 always) |
| `trunk.{0,1,2}.ssm.d` | [16] | `mixer.d_skip` (F32 always) |
| `trunk.{0,1,2}.ssm.norm_gated.weight` | [1024] | `mixer.norm` |
| `trunk.{0,1,2}.ssm.out.weight` | [1024, 512] | `mixer.out_proj` |
| `trunk.3.norm_attn.weight` | [512] | `attn_norm` |
| `<attn>.attn.qkv.weight` | [512, 768] | `attention.qkvr`, rows q / k / v |
| `<attn>.attn.{q,k}_norm.weight` | [64] | `attention.{q,k}_norm` |
| `<attn>.attn.out.weight` | [512, 512] | `attention.wo` |
| `<blk>.norm_mlp.weight` | [512] | `mlp_norm` |
| `<blk>.mlp.{gate,up}.weight` | [512, F] | `mlp.w13` split in half: gate rows first |
| `<blk>.mlp.down.weight` | [F, 512] | `mlp.w2` |
| `trunk.norm_out.weight` | [512] | `final_norm` |
| `head.vad.norm_in.weight` | [512] | `vad_norm` |
| `head.vad.down.weight` | [512, 512] | `vad_branch.down` |
| `head.vad.blk.*` | as above, F = 1536 | `vad_branch.block` |
| `head.vad.norm_out.weight` | [512] | `vad_branch.out_norm` |
| `head.vad.readout{,_la}.weight` / `.bias` | [512, 1] / [1] | `vad_branch.readout` / `vad_la_readout` |
| `head.turn.norm_in.weight` | [512] | `turn_norm` |
| `head.turn.speech_embed.weight` / `.bias` | [64] / [64] | `vad_embed` (Linear 1 -> 64; weight F32 always) |
| `head.turn.down.weight` | [576, 512] | `turn_branch.down` |
| `head.turn.blk.*` | as above, F = 1536 | `turn_branch.block` |
| `head.turn.norm_out.weight` | [512] | `turn_branch.out_norm` |
| `head.turn.readout.weight` / `.bias` | [512, 4] / [4] | `turn_branch.readout` |

In the table:

- `<attn>` is `trunk.3`, `head.vad.blk` or `head.turn.blk`.
- `<blk>` is any trunk block or head block. F is 2048 in the trunk.
- `{q,k}_norm` are per head (64 values) and shared by all heads.

Naming notes:

- Norm scales use the `norm_` prefix, so the existing quant-policy rule keeps
  them F32.
- The heads are named `head.vad.*`, not `vad.*`. The `vad.*` names belong to
  parakeet-ultra's conv rule.
- New F32 rules, added to both `tools/transcribe-quantize/policy.cpp` and
  `scripts/lib/gguf_common.py` and pinned in
  `scripts/lib/test_quant_policy_sync.py`: `frontend.dft_basis`,
  `rope.freqs`, `*.ssm.a`, `*.ssm.d`, `*.ssm.conv1d.weight`,
  `*.speech_embed.weight`.

GGUF sizes: F32 is 102.8 MB, and every tensor matches the reference exactly.
BF16 (`--outtype bf16`) is 51.9 MB with 36 matrices in bf16. It is lossless
storage of the source, but bf16 matmuls would not be.

## Reference dumps

`build/validate/sava/sava/<case>/ref/`. Per-step tensors are `[S, ...]` with
`S = n_frames + 3` (the run's padded steps), time-major, f32:

- Front end and mixer: `fe.mel [S,512]`, `fe.log_energy [S]`,
  `mixer.out [S,512]`.
- Trunk:
  - `trunk.{0..3}.mix [S,512]`: the sublayer output before the residual add;
  - `trunk.{0..2}.ssm_y [S,1024]`: the scan output before the D skip,
    `h * 64 + p`;
  - `trunk.{0..2}.ssm_state [16,64,128]`: the state after step S-1;
  - `trunk.{0..3}.out [S,512]` and `trunk.norm [S,512]`.
- Heads:
  - `head.vad.hidden [S,512]`, `head.vad.logit [S]`, `head.vad.logit_la [S]`
    (-30 at steps 0..2);
  - `head.turn.speech_in [S]`, `head.turn.input [S,576]`,
    `head.turn.embed [S,64]`, `head.turn.hidden [S,512]`,
    `head.turn.logits [S,4]`.
- Products:
  - `sava.speech [n]` (released), `sava.speech_on_time [n]`, `sava.turn [n,4]`
    and `sava.flush_turns [3,4]`;
  - `stream.speech [S]` and `stream.turn [S,4]`: a Session pushed 13 / 1280 /
    777 samples cyclically, as emitted.
- Sidecars:
  - `stream.json`: odd chunks, 640-sample pushes and a single push, each
    compared with run;
  - `turn_segments.json`: turns with frames, samples, `speech_ms`, reason and
    `kept`; the live-equals-offline flag; the run's `p` and `ended`;
  - `golden.json`: `sava-speech` only, compared with upstream's fixture.
  - `turn.json`: the clip's `p` and decision, and the kept turns in samples
    with the API's reason names. This is what `validate.py` compares with the
    CLI.

Cases are in `tests/golden/sava/sava.manifest.json`:

| case | frames | turns | notes |
|---|---:|---:|---|
| `sava-speech` | 574 | 2 | upstream fixture; wraps the 512-frame attention window |
| `jfk` | 275 | 1 | |
| `jobs-silence` | 134 | 0 | |
| `ja` | 181 | 1 | reason `end` |
| `dots` | 884 | 7 | |
| `silence` | 201 | 0 | digital zeros; hits the mel and energy floors; partial last frame |
| `noise` | 375 | 0 | |
| `dots-full` | 7,648 | 59 | 306 s, for drift; a pause-frame turn score is 2.9e-6 from 0.9 |

`samples/sava-speech.wav` is upstream's `tests/fixtures/speech.wav`
(Apache-2.0; the street recording is public domain).
`samples/silence.wav` is 128,160 zero samples.

## C++ implementation (`src/arch/sava/`)

- `weights.cpp`: hparams from the KVs (generic in every size) and weight
  binding.
- `graph.cpp`: the step graph, the stream state and the host front end.
- `model.cpp`: load, the TURN and VAD ops tables, the validation dumps.
  AUTO resolves to the CPU backend. Metal runs on request (jfk `p` 0.536095
  vs 0.536394 on CPU), but it is not validated.

Steps run in blocks of up to 64 frames, one graph per block, one column per
frame. A block of `n_real` frames runs as `n = max(n_real, 4)` columns; the
padding columns come after the real ones. Why:

- ggml's CPU matmul takes tinyBLAS only from 4 columns (NEON) and another
  kernel below that, with a different reduction order.
- Every other op (`rms_norm`, `soft_max`, `ssm_conv`, `ssm_scan`) computes a
  column the same way whatever `n` is.
- The attention query is made contiguous (`ggml_cont`) before K.Q, so that
  matmul takes tinyBLAS too. Its per-element order no longer depends on the
  column count either, and it is 9% faster for 1 user, 15% for batches of
  4-8 (turn-perf's "variant B").
- So each frame is computed bit for bit the same whatever the chunking:
  stream == run on the CPU, at any thread count. `transcribe_sava_smoke` and
  `transcribe_sava_real_smoke` check this bit for bit.

The state lives in a backend buffer per session, and the graph writes it
after its last read:

- `mixer_tail`: the last real row.
- `prev_speech`: sigmoid of the last real on-time logit.
- Per Mamba block, the conv state (`sx[n_real : n_real + 3]`) and the SSM
  state. `ggml_ssm_scan` is called with `K = n - n_real + 1` snapshots, so
  the state after the last real column is snapshot slot `n - n_real`.
- Per attention layer (trunk block 3, VAD head, turn head), K and V rings of
  `window + 64` slots. The K ring is ne `[KV hd, ring]`. The V ring is
  stored transposed, ne `[ring, KV hd]`, written with one-element
  `set_rows`, so the value product reads it in place; before, a transposed
  copy of every ring was made per step (11% of a 1-frame step). Outputs are
  byte-identical to the untransposed ring.
  - position `p` is written to slot `p % ring` with `ggml_set_rows` before
    the block's attention, without evicting a key any query of the block
    still reads;
  - query `p` reads the slots holding positions `[p - window + 1, p]`
    through an additive -inf mask;
  - the slot order, and with it the softmax and value reduction order,
    depends on `p` alone;
  - the ring is zeroed at reset, so masked slots stay finite.

Conditions for stream == run, documented here:

- CPU backend, and no BLAS accelerator in the scheduler (`GGML_BLAS` is off
  in the default build).
- `d_state` a multiple of `ggml_ssm_scan`'s SIMD step (16 on NEON, 32 on
  AVX, 64 on AVX-512; SAVA's is 128). The scan's scalar remainder loop
  rounds a graph's first token differently from the others. The toy fixture
  uses `d_state` 64 for this reason.

### Batched streams (`TurnOps::step_batch`)

`transcribe_turn_stream_feed_batch` steps one frame of each of N sessions
in one graph (`stream_step_batch` in `graph.cpp`). Columns are sessions.

- **Shared work:** the weight matmuls, norms and element-wise ops run once
  over `n = max(N, 4)` columns. Padding columns repeat session 0 and write no
  state.
- **Per-session work,** each on that session's own state buffer and in the
  shape a 1-frame `stream_step` uses:
  - the Mamba-2 conv and scan (1 token);
  - the K / V ring writes;
  - attention, with the query repeated to 4 columns so K.Q and the value
    product take the same kernels;
  - the state write-backs.
- **Result:** every session gets bit for bit the values of its own feed.
  `transcribe_sava_smoke` checks 1, 2, 3, 5 and 8 sessions with ragged
  chunks, a rotating first session and 1 and 4 threads.
  `transcribe_sava_real_smoke` checks 4 users with F32, F16 and Q8_0.
- **Where the graph lives:** in the session listed first (`TurnSession::batch`:
  its own context and scheduler, freed with the session's scratch). It is
  reused while the list of streams is the same, keyed by `Stream::uid`, which
  is unique for the process lifetime, so a freed session's graph is never
  reused. A different list rebuilds it, which costs about 0.5 ms at N = 8.
- **No model-level scratch, but one compute per model still holds:** the
  graphs and scratch are per session, yet every session shares the model's
  CPU backend, whose `work_data` and `n_threads` are per backend. Two
  computes on sessions of the same model (single feeds or batches) race on
  them, so the rule in [roles](../../roles.md) (at most one compute in flight
  per model) is required for safety.
- **Debug dumps:** with `TRANSCRIBE_DUMP_DIR` set, the batch steps sessions
  one by one, with the same values, so each stream's validation trace stays
  complete.

### Speed

Apple M1, idle machine, ms per 40 ms frame (clip: `jfk.wav`; stream:
640-sample feeds):

| | clip, 1 thread | clip, 4 threads | stream, 1 thread | stream, 4 threads |
|---|---:|---:|---:|---:|
| F32 | 0.90 | 0.38 | 4.6 | 3.3 |
| F16 | 1.81 | 0.59 | 7.2 | 2.9 |
| Q8_0 | 0.70 | 0.28 | 3.0 | 1.5 |

8 users batched, ms per tick (one frame each):

| | 1 thread | 4 threads |
|---|---:|---:|
| F32 | 10.1 | 6.3 |
| F16 | 17.2 | 6.9 |
| Q8_0 | 8.1 | 4.9 |

When users' frame phases differ (their feeds complete frames on different
ticks), the list of stepping streams changes between ticks and the batch
graph is rebuilt; that cost about 7% in a test feeding 320-sample ticks.

Before the contiguous query and the transposed V ring, a streamed frame took
5.4 ms (F32) and 3.9 ms (Q8_0) at 1 thread.

For comparison, the reference (torch 2.14.1) on the same machine with 1
thread takes 0.37 ms per frame for a whole clip and 10.5 ms per frame
streaming.

A streamed frame still runs a 4-column graph for 1 real column. Dropping
the padding needs either:

- a ggml change that gives every column count the same kernel (turn-perf's
  "variant E", about 25% faster for 1 user, bit-exact), or
- accepting stream != run (up to 1.4e-6).

Both were decided against for now. On ARM the F16 file runs 1.6-1.9x
slower than F32: ggml's F16 x F32 matmul has no tiled kernel there.

## Quantization (Stage 5)

Shipped: F32 (the reference), F16 and Q8_0. `scripts/lib/quant_policy.py`
gives `sava` the preset list `("F16", "Q8_0")`. Each tier stays
bit-identical between stream and run: `transcribe_sava_real_smoke` runs F32,
F16 and Q8_0.

| | F16 | Q8_0 | BF16 (not shipped) |
|---|---|---|---|
| File | 51.9 MB, `transcribe-quantize --quant F16` | 28.5 MB, `--quant Q8_0` | 51.9 MB, `convert-sava.py --outtype bf16` |
| Weights vs the bf16 source | exact except 1,065 tiny values (<= 3e-8) | 8-bit blocks; `mixer.proj` (1026-wide rows) F16 | exact |
| Matmul activations | fp32 (`vec_dot_type` F32) | int8 blocks | rounded to bf16 |
| Golden per-frame max abs vs F32, speech / turn | 1.2e-6 / 1.5e-6 | 0.050 / 0.066 | 0.013 / 0.033 |
| Golden turns vs F32 | identical, 8 of 8 | 7 of 8; `dots-full` 6 of 59 change | 7 of 8; `dots-full` 5 of 59 change |
| F32 tolerance gate (`validate.py --gguf`) | passes, 32/32 tensors x 8 | - | - |
| Pipecat AUC (F32 0.97946) | 0.97946 | 0.97935 | 0.97991 on the first 3,000 clips (F32 0.97990) |
| Pipecat decision flips vs F32, at 0.5 / 0.9 | 0 / 0 of 31,527 | 76 / 126 of 31,527 (at 8f97ab5b; 75 / 124 before the batching commit) | 1 / 3 of 3,000 (F16 0 / 0, Q8_0 9 / 11 on those) |
| Clip frame, ms, 1 / 4 threads (F32 0.90 / 0.38; from Speed) | 1.81 / 0.59 | 0.70 / 0.28 | 20.7 / 5.7 (before 6b130f91) |
| Streamed frame, ms, 1 / 4 threads (F32 4.6 / 3.3; from Speed) | 7.2 / 2.9 | 3.0 / 1.5 | 82 / 22 (before 6b130f91) |

The `dots-full` turn changes all sit at frames where a score is within 0.004
of 0.9 or 0.03 of 0.5 (details in `docs/models/sava.md`; per-case numbers in
`build/quant/sava/golden_{F16,BF16,Q8_0}_vs_f32.json`).

Speed notes (Apple M1, idle machine):

- On NEON, ggml has no tiled kernel for F16 weights x F32 activations, so F16
  per clip frame is slower than F32. On the Ryzen 7 PRO 4750U (AVX2, F16C)
  F16 is not faster than F32 either for a whole clip on one thread (245.7 vs
  221.6 ms for 8 s); it is faster for a streamed frame (8.4 vs 9.8 ms).
  Q8_0 is the fastest streamed frame on both machines.
- BF16 has no native support on the M1; CPUs with AVX512-BF16 or ARMv8.6
  BF16 were not measured.
- Bit-identity (stream == run, batched == alone) was checked on M1 NEON and
  on the Ryzen 7 PRO 4750U (AVX2) for F32, F16 and Q8_0. On CPUs with int8
  matmul instructions (i8mm, AVX-VNNI) ggml may pick per-column-count
  kernels; re-run `transcribe_sava_real_smoke` there.

Recommendation: ship F16, not BF16.

- F16 is the 16-bit file that reproduces F32 (0 flips, F32 gate passed).
- BF16's storage is lossless, but its compute is lossy (bf16 activations)
  and far slower on CPUs without BF16 hardware.
- For speed, Q8_0 is the tier.

K tiers were not evaluated.

## Upstream claims checked

- 25.5M parameters: yes (25,521,558). Weights 51 MB bf16: yes.
- Speech emitted 120 ms late: yes (3 frames). Turn has no lag: yes.
- Golden parity to 2e-4 on the training reference's logits: yes, 3.3e-5.
- Speed: "3.8 ms per 40 ms frame on an M3 Max CPU; a 12 s recording in
  0.04 s" was not reproduced on this Apple M1. With torch 2.14.1 it measured
  10.5 ms per frame with 1 thread (22 ms with 4 threads) and 0.11 s for
  12 s. That is a different machine; not investigated further.
- Turn clip accuracy and AUC on the Pipecat test set: yes. The per-clip
  PyTorch reference (`scripts/turn/run_reference_sava_torch.py`, inworld-sava
  1.0.0 CPU fp32) gives 93.36% and AUC 0.97946 (published: 93.35%, 0.97946);
  F32 and F16 make its decision on all 31,527 clips at 0.5 and 0.9 (max |dp|
  1.1e-5 / 1.2e-5), Q8_0 on 31,451 at 0.5 and 31,401 at 0.9 (max |dp| 0.17).
- The VAD frame AUC benchmark was not checked: it needs the benchmark
  dataset.

## Commands

```bash
uv sync --project scripts/envs/sava
uv run --project scripts/envs/sava scripts/dump_reference_sava_author.py encoder \
  --audio samples/sava-speech.wav --out build/validate/sava/sava/sava-speech/ref \
  [--expected-npz <sava clone>/tests/fixtures/speech_expected.npz]
uv run scripts/validate.py ref --family sava --variant sava     # all 8 cases (~4 min)
uv run --project scripts/envs/sava scripts/convert-sava.py      # models/sava/sava-F32.gguf
uv run --project scripts/envs/sava scripts/convert-sava.py --outtype bf16 --out /tmp/sava-BF16.gguf
uv run scripts/preflight.py --family sava --variant sava --gate B
uv run --project scripts/envs/sava python build/validate/sava/check_sava_gguf.py \
  models/sava/sava-F32.gguf samples/sava-speech.wav build/validate/sava/sava/sava-speech/ref

# Pipecat test set: per-clip PyTorch reference (~17 min on 6 workers), then
# each GGUF scored against it
uv run --project scripts/envs/sava --with pyarrow scripts/turn/run_reference_sava_torch.py \
  --out reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl
uv run --project scripts/envs/smart_turn scripts/turn/score.py \
  reports/turn/sava-F32.smart-turn-data-v3.2-test.jsonl \
  --ref-jsonl reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl

# C++
cmake --build build --target transcribe-cli
uv run scripts/validate.py all --family sava --variant sava     # ~3 min
ctest --test-dir build -R sava    # toy fixture smoke; real smoke with the GGUF
TRANSCRIBE_SAVA_GGUF=$PWD/models/sava/sava-F32.gguf ctest --test-dir build -R sava_real
build/bin/transcribe-cli -m models/sava/sava-F32.gguf samples/sava-speech.wav
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

## Notes

- `scripts/validate.py` needs no SAVA-specific code:
  - per-case `"stages": ["encoder"]` runs the dumper;
  - `"role": "turn"` compares the CLI's `turn:` line (p and decision) with
    the dumper's `turn.json`;
  - the C++ dumps the reference's tensor names when `TRANSCRIBE_DUMP_DIR` is
    set, written when a stream ends.
- `head.vad.logit_la` holds -30 at steps 0..2 on both sides.
- The `transcribe_vad_run` segment check: apply silero-vad 6.2.3's
  `get_speech_timestamps_from_probs`, with its 512-sample window replaced by
  640, to the reference `sava.speech`, and compare with the C++
  `transcribe_vad_run` segments. Done once; it is not wired into a test.

## Reproducibility

The Mac-built GGUFs are canonical: their sha256 are in `catalog/sava.json`
(downloads) and the convert manifests. Converting on another platform gives
identical bytes for every upstream weight and for the metadata. A few
converter-computed constants can differ in the last f32 bit, because
float64 / float32 libm `exp` / `log` round differently across platforms:
`frontend.mel_filterbank` (`inworld.sava.mel.mel_filterbank`) and `ssm.a`
(`-exp(A_log)`, torch float32 `exp`). On the Ryzen 7 PRO 4750U the effect on
outputs is about 1e-7. Cross-platform byte reproducibility is not a goal
(owner decision).
