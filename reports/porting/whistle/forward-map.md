# Forward map - whistle

Reference: closed Cactus Needle engine `libneedle3.dylib` (cactus_needle-3.2.0,
Cactus-Compute/needle3@2ae11323) running `whistle.cact` (Cactus-Compute/whistle@b358ddad).
Open partial reference: cactus-compute/needle@ef3cf754 `needle/model/architecture.py`
(Needle-3 text stack: ZCRMSNorm, MultiHeadAttention + taps, HadamardMLP, mHC, Engram).
Closest in-tree analog: src/arch/cohere/ (encoder-decoder).

How rows were settled. The engine has no source and exposes only `needle_embed`
(encoder output) and `needle_transcribe` (text, language, per-word start/end/probability).
Every row without an open-source location was settled by weight surgery: patched copies
of `whistle.cact` (identity/selector CQ matrices, gates forced open/shut) run through the
public C API, compared against a numpy prototype under explicit hypotheses. The identified
forward pass is `scripts/lib/whistle_ref.py` (the reference for intermediate gate tensors,
user decision 2026-10-09). The probe experiments themselves (`t*.py` / `w*.py`, surgery
harness `probe.py`) ran from the sandbox `tmp/whistle-probe/` and are not part of the repo;
the metrics they produced are recorded per row below. Metric: per-frame
cosine / median abs error vs the engine, or mean |Δ word probability|. The engine runs
8-bit activations + int8 KV, so its own readout noise floor is ~0.01 (normalized mel) and
cos ~0.9999 (encoder output); every accepted row below sits at that floor, and every
rejected alternative is clearly above it.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Framing | probe t7/t8: frames = n_samples // 160; frame i centered at sample 160·i (400-tap window, zero pad 200 each side), rfft 512 | [T=N//160, 257] | `enc.mel.in` (C++ only) | CPU frontend, real FFT | whisper mel |
| Window | probe t29: Hann **symmetric** (N-1) best on 3 clips; periodic is within noise (0.0123 vs 0.0125) | 400 | — | precomputed table | — |
| Power + mel | probe t7: \|X\|² @ shipped Slaney fb [257,80] (`frontend.mel_filterbank`) | [T, 80] | — | matmul | — |
| Log | probe t25–t38: `ln(mel + 1e-8 · max(mel))` (floor relative to the clip max mel power; scale-invariant, t34: jfk×0.01 / ×3 readouts identical). Rejected: fixed eps, peak/RMS waveform normalization, Whisper clamp | [T, 80] | — | two-pass (max then log) | — |
| Normalize | probe t6: per-utterance, per-bin mean/std (readout std = 1.000) | [T, 80] | — | two-pass over T | — |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Stem | probes t4/t10 (killA, cos 0.99991): Conv2d 1→128 3×3 s2 pad1 → SiLU → depthwise 3×3 s2 pad1 → pw 128→128 → SiLU → depthwise 3×3 s2 pad1 → pw → SiLU; input [T, F=80] with kernel [kt, kf]; no biases | [T/8, 10, 128] | `enc.stem` | conv_2d / conv_2d_dw + mul_mat | parakeet subsampling |
| Stem flatten + out | probe t4: channel-major `c*10 + f` (fc order repeats with period 5) → Linear 1280→512 | [T', 512] | `enc.stem` | permute + reshape + mul_mat | parakeet |
| mHC lanes | architecture.py `_ScanBody.advance` + `Stack`; probe t11 (cos 0.99992): 4 lanes broadcast from stem out, per-layer pre/post/res with Sinkhorn(20, log-space), `pre_off=8·lane−4`, `post_off=−4·(1−lane)`, lane = layer % 4, `y = block(u) − u`; end: lane mean → `enc.final_norm` | [T', 4, 512] | `enc.blk.N.out` | small matmuls + Sinkhorn on [T,4,4] | none (novel) |
| Block order | probe t15 (all 24 orders; best cos 0.99994): HMLP_0 → attention → conv → HMLP | | | | cohere conformer |
| HMLP_0 / HMLP | architecture.py `HadamardMLP`; probe t11/t15: `u + 0.5·HMLP(ZCRMS(u))` (macaron half step), norms `pre_hada_norm_0` / `pre_hada_norm`; shipped `hada.perm1/2` = RandomState(11/13) | | | Kronecker 16×32 as two batched matmuls + get_rows perm | none (novel) |
| Self-attention | architecture.py `MultiHeadAttention` (no taps); probes t12/t13: `u + σ(attn_gate)·ZCRMS_post(Attn(ZCRMS_in(u)))`; GQA 8q/2kv, qk 48, v 64, q/k ZCRMSNorm per head, **RoPE θ=1e5** half-split (amplified probe: 0.9998 vs 0.9970 none), bidirectional, scale 1/√48, output × σ(gate_proj(x)) | | | flash_attn_ext / mul_mat + soft_max | cohere |
| Conv module | probe t14 (cos 0.99991): ZCRMS(conv_norm) → pw1 512→1024 → GLU `a·σ(b)` → depthwise k9 centered → ZCRMS(conv_out_norm) → SiLU → pw2 → residual (scale 1) | | | conv_1d_dw / ssm_conv | parakeet conformer conv |
| Final | lane mean → ZCRMSNorm(`enc.final_norm`) = `needle_embed` output (probe t0: scale −1 zeroes it) | [T', 512] | `enc.final` | | |

Full numpy encoder vs engine `enc.final` (t39): median cos 0.99991–0.99995 on jfk,
noise, jobs-silence, fleurs-de/fr/es.

## Decoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Memory | probes t18–t21: `mem = enc + σ(pe_gate)·[sin(p·ω), cos(p·ω)]`, ω_i = 10000^(−2i/512), p from 0 (cos/sin swap and p+1 rejected; Whisper-style ω spacing is within noise and immaterial at σ(−6.11)=0.002) | [T', 512] | `dec.mem` | precomputed table + add | — |
| Embedding | architecture.py `_input_embeddings`: `emb[tok]·√512` | | | get_rows + scale | cohere |
| Engram (sites 3, 7) | architecture.py `engram_indices`, `Engram`, `Block` (alpha gate) | | | host-side hashing + get_rows; dilated 4-tap causal conv carried in state | none (novel) |
| Self-attention | architecture.py `MultiHeadAttention` with `qkv_conv_taps=3` (causal taps on q/k/v projections), RoPE θ=1e5, causal | | | mul_mat + taps state + KV cache | cohere |
| Block order | probe t22 (all 6 orders): self → cross → HMLP (Δp 0.0016; next best 0.042) | | | | |
| Cross-attention | probe t17/t22: `u + σ(cross_gate)·ZCRMS_post_cross(XAttn(ZCRMS_cross(u), mem))`; 8 heads (MHA), qk 48, v 64, q/k ZCRMSNorm, **no RoPE**, output × σ(gate_proj(x)) | | | cross KV computed once | cohere |
| HMLP | architecture.py `Block`: `u + HMLP(ZCRMS(u))` (scale 1) | | | | |
| mHC + final | architecture.py `Stack`; lane mean → `dec.final_norm` → tied head over 8199 | [T, 8199] | `dec.logits_raw` | mul_mat with token_embd | cohere |

Evidence: teacher-forced word probabilities on jfk match the engine to mean |Δ| 0.0016
(t22); greedy decode reproduces the engine text exactly (t16).

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Prompt | probe t22/t23: `<s>` (2) then `<|lang|>` (8192+i), then text until `</s>` (1); no dummy prefix on the first text piece | | | | cohere prompt |
| Auto language | probe t22: argmax over the 7 language tokens after `<s>` (matches engine on jfk/de/fr/es) | | | | |
| Search | engine default: 5-beam (no knob). Port: Whisper-style beam (5 beams, top beam+1 per hypothesis, stop at 5 finished, rank by sum log p / length). 300-row LibriSpeech sample: greedy 224/300, beam 283/300 byte-identical to the engine; raw-sum / GNMT 0.6 / 2.0 rankings fit worse; residual diffs are near-ties (casing, commas, homophones). An int8 QAT activation fake-quant emulation was tried and fit worse (279/300), so it was not kept | | host-side beam over a batched step graph; beam reorder = get_rows on every per-hypothesis cache | none (first beam search in tree) |
| Step state | taps need the last 2 pre-tap q/k/v projections per layer; Engram needs last 2 tokens + 9 previous value rows (dilation 3 × 3 taps) | | `dec.logits_raw.gen8` | | |
| Max length | header max_seq_len 320 | | | | |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Explicit language | `<s><|lang|>` forced | `run_params.language` -> `<|lang|>` token; unsupported codes rejected | Transcribe explicit |
| Auto language | argmax over language tokens after `<s>` | same; `detected_language` set | Transcribe auto / Language detection |
| No speech | Gate upstream of the model (t46: unchanged with stem output zeroed, decoder perturbed). Scale-invariant, energy-dynamics based. Best rule found (t49, separates 46/46 labeled clips): speech iff p99 − p10 of per-frame mel energy (dB, 10 ms frames) > ~4.6 dB (gap 4.26–5.05; proposed 10·log10(3)=4.77). **Approximation**, pending user sign-off. REF LibriSpeech: 0/2611 empty. | gate before the encoder: OK status, empty text, no language | No-speech |
| Word timestamps | Probed (w1/w2): only decoder layers 4-7 move the engine's times (perturbing cross-attn K of layers 0-3 never does; every head of 4-7 does). Recipe fitted (w4-w10): rows = positions 0..n+1 (`<s>`, each text token's predicting position, the `</s>`-predicting position), per-head standardization over the token axis, head mean, no median filter, Whisper DTW on -matrix; token k spans [jump(k), jump(k+1)); a word ends at the jump after its last non-punctuation token; first word clamped to an energy onset (first 40 ms block within 19.5 dB of the loudest, ceil to 80 ms) | teacher-forced 1-hypothesis pass after the beam search; same recipe | Word timestamps |

## Deviations From Closest Analog

- 4-lane mHC residual stream with Sinkhorn mixing replaces the plain residual (encoder and decoder).
- HadamardMLP (Kronecker Walsh factors + fixed permutations + rank-8 softmax conditioning) replaces the FFN.
- Decoder self-attention has causal q/k/v taps (3) and Engram n-gram memory at layers 3 and 7.
- Frontend log floor is relative to the clip's max mel power; normalization is per-utterance per-bin.
- No biases anywhere.
- The conv module pads with `ggml_concat` (Metal's PAD has no left padding).

## Variant Notes

- `whistle`: family baseline.
