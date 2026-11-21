# Forward map - sava

Reference: inworld-sava 1.0.0 (github.com/inworld-ai/sava v1.0.0, b4221a5), `src/inworld/sava/`
Closest in-tree analog: src/arch/silero_vad/ (streaming VAD, fixed graph shapes), src/arch/smart_turn/ (TURN ops)

Per-step tensors are `[S, C]` with S = n_frames + 3 (the run's flush steps), one column per
frame in the graph (`ne [C, n]`). All gates live in `tests/tolerances/sava.json`.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| framing, 240-sample causal context | `modules.frame_audio`, `mel.MelFrontend.forward_chunk` | 4 windows of 400 per frame | - | host: `Stream::context`, windows end on each 160 hop | silero_vad `build_windows` |
| Hann DFT, power, Slaney mel, `(log10(max(x, 1e-10)) + 4) / 4` | `mel.features`, `mel.log_compress` | [S, 512] hop-major | `fe.mel` | host fp64: mixed-radix FFT (400 = 2^4 * 25), mel bank from `frontend.mel_filterbank` | transcribe-mel.cpp (fp32 FFT) |
| log RMS energy | `FrameMixer.rows` | [S] | `fe.log_energy` | host fp64 sum, fp32 `log(max(rms, 1e-5))` | - |
| frame mixer `rms(W [row_{t-1} \| row_t])` | `FrameMixer._mix` | [S, 512] | `mixer.out` | `concat(tail, rows[:, :n-1])`, `concat(prev, row, 0)`, `mul_mat`, `rms_norm`; tail state = last real row | - |

## Encoder (trunk)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Mamba-2 in_proj, causal conv k4 + SiLU | `Mamba2Mixer._split`, `shift_conv` | [S, 2320] | - | `mul_mat`; `ggml_ssm_conv(concat(conv_state, xBC^T), conv1d)`; conv state = `sx[n_real : n_real + 3]` | llama.cpp mamba2 |
| SSD scan | `ssd.chunked_scan` / `recurrent_step` | [S, 1024] | `trunk.{0,1,2}.ssm_y`, `.ssm_state` | `ggml_ssm_scan` (A ne [1, 16], softplus inside), K = padding + 1 snapshots; state = slot n - n_real | llama.cpp mamba2 |
| D skip, `silu(z)` gate, grouped RMSNorm, out_proj | `Mamba2Mixer._output` | [S, 512] | `trunk.{0,1,2}.mix` | `mul`, `add`, `rms_norm` over inner / groups, `mul_mat` | llama.cpp mamba2 |
| sliding-window GQA, q/k RMSNorm, RoPE, XSA | `Attention.forward_chunk`, `windowed_attention`, `reject_own_value` | [S, 512] | `trunk.3.mix` | KV ring per layer (`set_rows`, slot = p % (window + 64)), -inf mask from p, `soft_max_ext`; rotate-half with host fp64 cos/sin; XSA from `mul`/`sum_rows`/`div`/`sub` | whisper encoder attention |
| SwiGLU FFN + residuals | `SwiGLU`, `MambaBlock` / `AttentionBlock` | [S, 512] | `trunk.{i}.out` | `mul_mat`, `silu`, `mul` | - |
| final RMSNorm | `SavaModel.final_norm` | [S, 512] | `trunk.norm` | `rms_norm` | - |

## Decoder (heads)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| VAD branch (norm, down, attention block, norm) | `Branch.features` | [S, 512] | `head.vad.hidden` | as trunk attention, own ring | - |
| on-time / lookahead readouts | `vad_branch.readout`, `vad_la_readout`, `_mask_lookahead` | [S], [S] | `head.vad.logit`, `head.vad.logit_la` | `mul_mat` + bias; steps < 3 set to -30 on the host | - |
| speech input: sigmoid of the previous step's on-time logit | `SavaModel._heads` | [S] | `head.turn.speech_in` | `concat(prev_speech state, sigmoid(logit)[:n-1])`; state = last real column | - |
| speech embedding, concat | `SavaModel._turn_input` | [S, 64], [S, 576] | `head.turn.embed`, `head.turn.input` | `repeat * speech_in + b`, `concat` | - |
| turn branch, readout | `Branch.features`, `turn_branch.readout` | [S, 512], [S, 4] | `head.turn.hidden`, `head.turn.logits` | as VAD branch; batched over the block | - |
| probabilities | `streaming._probabilities`, `run` | [n], [n, 4] | `sava.speech`, `sava.turn`, `stream.*` | host sigmoid | - |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| stream state across calls | `StreamState`, `Session.push` | - | stream == run (smokes) | persistent backend buffer per session: mixer tail, prev speech, conv / SSM states, 3 KV rings; written by the graph after every read | silero_vad (host state) |
| block graph | `SavaModel.step` | - | stream == run (smokes) | blocks of <= 64 frames, padded to >= 4 columns so every matmul takes the same kernel | silero_vad `k_min_graph_frames` |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| whole clip | `sava.run` + `sava.segment` | `transcribe_turn_run`: p, ended, per-frame turn and speech, kept turns | Whole-clip turn scores; Turn segments |
| live stream | `Session.push` / `flush` | `transcribe_turn_stream_feed` / `flush` | Streaming |
| speech only | lookahead output | `transcribe_vad_*` (lookahead 3, threshold 0.5) | Speech track through `transcribe_vad_*` |

## Deviations From Closest Analog

- Front end on the host in fp64, not a ggml graph: the reference's fp32 DFT GEMM is
  cancellation-bound near the 1e-10 mel floor; fp64 removes the C++'s share of that error.
- RoPE angles in fp64 from the absolute position (owner decision); the reference uses fp32
  angles and drifts with stream age.
- The state lives in a backend buffer that the graph writes (silero_vad keeps it on the host).

## Variant Notes

- `sava`: the only variant.
