# Voxtral

Status: research

Mistral's Voxtral audio models. Two distinct sub-architectures share this
family key:

- **Voxtral 2507** (`voxtral`): offline audio-LLM — a Whisper-large-v3
  bidirectional encoder + a Ministral/Mistral-Small text decoder with
  audio-token injection. Variants: `voxtral-mini-3b-2507`,
  `voxtral-small-24b-2507`.
- **Voxtral Realtime 2602** (`voxtral_realtime`): a *streaming* audio-LLM
  with a causal RoPE + sliding-window audio encoder, a streaming log-mel
  frontend, and delay-token latency conditioning. Variant:
  `voxtral-mini-4b-realtime-2602`. Architecturally distinct from 2507 — it
  shares only the projector shape and the tekken tokenizer.

## Identity

- Family key: `voxtral`
- Upstream architecture strings: `voxtral` (2507),
  `voxtral_realtime` (2602)
- Hugging Face repos:
  - `mistralai/Voxtral-Mini-3B-2507` (pinned to `3060fe34b35ba5d44202ce9ff3c097642914f8f3`)
  - `mistralai/Voxtral-Small-24B-2507` (pinned to `da5b42409f279fdd92febee0511a6c32828569c1`)
  - `mistralai/Voxtral-Mini-4B-Realtime-2602` (pinned to `2769294da9567371363522aac9bbcfdd19447add`)
- License: Apache-2.0 (all three)
- Variants:
  - `voxtral-mini-3b-2507` — Ministral-3B text backbone (~9.5 GB bf16); first target
  - `voxtral-small-24b-2507` — Mistral-Small-24B text backbone (~55 GB bf16)
  - `voxtral-mini-4b-realtime-2602` — ~3.4B LM + ~970M causal audio encoder; streaming

## Architecture

### Voxtral 2507 (`voxtral`)

- Pattern: **audio-llm** (audio encoder + causal LM with audio-token
  injection — no cross-attention, no transducer).
- Audio encoder (`VoxtralEncoder`, = Whisper-large-v3 encoder):
  `conv1` (128→1280, k3 s1, pad1) → GELU → `conv2` (1280→1280, k3 s2,
  pad1) → GELU (halves time: 3000 mel frames → 1500); add fixed
  sinusoidal `embed_positions` (1500×1280, **synthesized by the
  converter, kept F32**); 32 layers of **bidirectional full** attention
  (d_model 1280, 20 heads, head_dim 64, ffn 5120, GELU; `k_proj` has
  **no bias**, q/v/out do); pre-LN blocks + a final `layer_norm`.
- Projector (`VoxtralMultiModalProjector`): the encoder output
  (B,1500,1280) is **reshaped (B,375,5120)** — concatenating 4 consecutive
  frames (4× time downsample) — then `Linear(5120→H)` → GELU →
  `Linear(H→H)`, both **bias=False**, where H = text hidden size. Yields
  **375 audio tokens per 30 s chunk**.
- Text decoder (`LlamaForCausalLM`): standard Ministral/Mistral-Small.
  GQA 32 q / 8 kv heads, head_dim 128, SwiGLU (silu), RMSNorm (eps 1e-5),
  **NEOX RoPE** (theta 1e8, q/k pre-permuted by the converter),
  `attention_bias=false`, **untied lm_head**, `max_position_embeddings`
  131072.
- Fusion: audio embeddings `masked_scatter`'d into the LM input embedding
  at `audio_token_id=24` positions; the placeholder run + control tokens
  are produced by mistral-common (`encode_transcription` /
  `apply_chat_template`).
- Output contract: transcript text only (first port). Auto language
  detection by default; no segment/word timestamps.

### Voxtral Realtime 2602 (`voxtral_realtime`)

- Pattern: **audio-llm**, *streaming* (additive fusion).
- Audio encoder (`voxtral_realtime_encoder`): **causal** conv stem
  (left-pad-only `CausalConv1d`, conv2 stride 2) → 32-layer **causal +
  sliding-window (750)** transformer with **RoPE** (theta 1e6, head_dim
  64), RMSNorm pre-norm, SwiGLU/silu MLP (`activation_function=gelu` is
  dead, used only in the conv stem), 32 heads MHA, `k_proj` no bias,
  FFN bias only on the down proj.
- Projector: reshape (B,T,1280)→(B,T/4,5120) (`downsample_factor=4`,
  seq%4 truncated) → `Linear(5120→3072)` → GELU → `Linear(3072→3072)`,
  bias-free. Audio embeds are **added** onto text embeds (not scattered).
- Streaming schedule: `audio_length_per_tok=8` mel frames = 4 encoder
  frames = 1 audio token = 80 ms (12.5 Hz). One text token per audio
  slot, output hard-clamped to `ceil(mel_frames/8)`. Audio fed
  `downsample_factor=4` encoder frames per decode step.
- Delay-token conditioning: `default_num_delay_tokens=6` (480 ms,
  configurable 80 ms–2.4 s via `transcription_delay_ms` in tekken.json)
  drives both the prompt pad run (`[BOS] + [STREAMING_PAD=32]*(32+delay)`)
  and a sinusoidal `time_embedding` feeding a per-layer **adaptive RMSNorm
  on the FFN branch** (`post_ln(h) * (1 + ada_rms_norm(t_cond))`).
- Text decoder (`voxtral_realtime_text`): 26-layer Mistral-style, GQA
  32/8, head_dim 128, SwiGLU, **sliding_window=8192**, RoPE theta 1e6,
  **tied lm_head**.

## Frontend

Both sub-architectures use a 16 kHz log-mel (128 mels, n_fft 400,
hop 160, win 400, periodic Hann, slaney mel filterbank 0–8000 Hz, drop
last STFT frame). The normalization differs and is load-bearing:

- **2507** (`WhisperFeatureExtractor`): **per-utterance** log10 norm —
  `max(log, log.max()-8)`, then `(log+4)/4`. center=True reflect pad.
  30 s / 480000-sample chunking.
- **Realtime** (`VoxtralRealtimeFeatureExtractor`, no
  `preprocessor_config.json`): **fixed global** max
  (`global_log_mel_max=1.5`, *not* the per-utterance max) so per-frame
  mel is causal; `center=True` only for the first chunk; conv padding
  cache carries continuity across streaming chunks.

## Tokenizer

Tekken (tiktoken-style BPE) via **mistral-common** — no HF
`tokenizer.json`/`tokenizer_config.json`. BOS=1, EOS=2, PAD=11; 2507 uses
`audio_token_id=24`, Realtime uses `STREAMING_PAD=32` (n_special=1000,
vocab id = token id − 1000). The transcription/chat prompt (control
tokens + audio placeholder run, and for Realtime the delay schedule) is
built inside mistral-common, not in transformers — the C++ port must
reproduce its exact token layout.

## References

- Canonical reference: HuggingFace **transformers** —
  `VoxtralForConditionalGeneration` (>=4.54) and
  `VoxtralRealtimeForConditionalGeneration` (>=5.x). Tokenizer +
  prompt/transcription template: **mistral-common** (tekken), which
  transformers delegates to.
- Instrumented reference: same transformers modeling code
  (`refs/huggingface/transformers/src/transformers/models/voxtral{,_realtime}/`).
- Cross-check references: MLX
  `refs/mlx/mlx-audio/mlx_audio/stt/models/voxtral{,_realtime}/`.

## Commands

Reference run (Stage 2 finalizes the entrypoint):

```bash
uv run --project scripts/envs/voxtral scripts/reference/voxtral_run.py \
  --variant voxtral-mini-3b-2507 samples/jfk.wav
```

Reference dumps:

```bash
uv run scripts/validate.py reference --family voxtral --variant voxtral-mini-3b-2507
```

Conversion:

```bash
uv run scripts/convert-voxtral.py --repo-id mistralai/Voxtral-Mini-3B-2507 \
  --out models/voxtral-mini-3b-2507/
```

Validation:

```bash
uv run scripts/validate.py all --family voxtral --variant voxtral-mini-3b-2507
```

Benchmarks:

```bash
uv run scripts/bench/run.py --family voxtral --variant voxtral-mini-3b-2507
```

## Capability Validation

Targets below are **user-signed (2026-06-04)**. Acceptance dataset:
**LibriSpeech test-clean** (English-only — user decision). Port order:
**`voxtral-mini-3b-2507` → `voxtral-small-24b-2507` → realtime**. The
Realtime port is **deferred** (intake done; C++ port starts only after
the 2507 audio-LLM path is proven) — its rows are inactive this effort.
`Status` is filled at Stage 4.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/voxtral-mini-3b-2507/Voxtral-Mini-3B-2507-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | **PASS** — byte-exact vs ref on jfk; full test-clean WER 1.87% (== Oracle 1.87%) |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/voxtral-mini-3b-2507/Voxtral-Mini-3B-2507-BF16.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | **PASS** — jfk → correct English on the no-`lang:` prompt path |
| Translate (speech→text) | 2507 only | `build/bin/transcribe-cli -m .../Voxtral-Mini-3B-2507-BF16.gguf --translate --target-language en samples/german.wav` | non-empty English transcript on non-English audio | MUST PASS (2507) — user-signed | **PASS** — german.wav → fluent English translation via the instruct template (synthesized "Translate this to English." + tekken BPE, no `[TRANSCRIBE]`) |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model .../Voxtral-Mini-3B-2507-BF16.gguf --samples-dir samples/wer/<dataset> --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | **ACCEPTED GAP — serial fallback** — no batched fast path this port (`run_batch=nullptr`); the dispatcher runs the per-utterance serial loop, so batched output is byte-identical by construction. Full-set WER confirms it: b1 == b8 == 1.8707% (|Δ|=0). A batched encoder/decoder fast path is a follow-up. |
| Transcribe (multilingual) | non-English audio | `build/bin/transcribe-cli -m .../Voxtral-Mini-3B-2507-BF16.gguf --language de samples/german.wav` | non-empty plausible non-English transcript | OUT OF SCOPE — English-only acceptance (user); multilingual WER not gated this port. Revisit if a multilingual acceptance set is added | **PASS (not gated)** — german.wav → correct German transcript; demonstrated, not WER-gated |
| Audio understanding / Q&A / summarization | 2507 only | chat request | non-empty answer grounded in audio | OUT OF SCOPE — chat capability, not ASR; not exposed by the transcribe CLI | ACCEPTED GAP — the instruct mechanism is proven (translate uses it); a general free-text `--prompt` CLI flag is the remaining wiring to expose arbitrary Q&A |
| Function calling from voice | 2507 (esp. 24B) | chat request with tools | tool call emitted | OUT OF SCOPE — not an ASR observable; not exposed by the transcribe CLI | SKIP — not exposed by runtime |
| Segment timestamps | — | n/a | — | OUT OF SCOPE — model emits no timestamp tokens | SKIP — model emits no timestamp tokens |
| Word timestamps | — | n/a | — | OUT OF SCOPE — model emits no timestamp tokens | SKIP — model emits no timestamp tokens |
| Streaming | realtime variant only (`capabilities.streaming`) | `build/bin/transcribe-cli -m models/voxtral-mini-4b-realtime-2602/...-BF16.gguf --stream-chunk-ms <N> --backend cpu --threads 1 samples/jfk.wav` | byte-equal transcript vs reference streaming runner | MUST PASS (forced) — **realtime port DEFERRED (user-signed); activates at the realtime Stage 4** | N/A this variant — `capabilities.streaming=false` for 2507; contract activates at the realtime Stage 4 |
| Configurable transcription delay | realtime only | `--stream-chunk-ms` / delay param | output at the requested 80 ms–2.4 s delay | OUT OF SCOPE — realtime port deferred; revisit after streaming lands | N/A this variant — realtime deferred |

## Notes

- **Port order (user-signed 2026-06-04):** `voxtral-mini-3b-2507` first
  (proves the 2507 audio-LLM path), then `voxtral-small-24b-2507` (same
  architecture, larger decoder), then `voxtral-mini-4b-realtime-2602`.
- **Realtime deferred (user-signed):** the streaming model is intaken now
  but its C++ port does not start until the 2507 path is proven. Its
  forced-`MUST PASS` streaming row stays as the contract that activates at
  the realtime port's Stage 4. The realtime encoder's offline path may be
  numerically equivalent to streaming (fixed frontend max + causal
  convs) — verify at that stage before relying on it.
- **Translation is MUST PASS for 2507 (user-signed):** speech→text
  translation must be a gated capability for the 2507 variants. This
  requires the runtime/CLI to expose a translation request path (a
  Stage 4 implementation requirement, since translation is a
  mistral-common chat/transcription-request mode, not a bare ASR flag).

### Converter tensor-name / dtype decisions (Stage 3, `scripts/convert-voxtral.py`)

GGUF architecture string is `voxtral`. HF → GGUF tensor mapping and the
non-obvious dtype/tokenizer choices the C++ loader (Stage 4) must match:

- **Conv stem → `enc.conv.0/1.{weight,bias}`** (named with `.conv.` so
  `gguf_common.reference_dtype_for` routes the kernels to **F16** — the
  loader has no BF16 conv kernel; biases stay F32). Source Conv1d weights
  are passed through in PyTorch `[out, in, k]` order.
- **Sinusoidal positional embedding** `audio_tower.embed_positions.weight`
  → `enc.pos_emb.weight`, kept **F32** (it is already F32 in the
  checkpoint — *not* synthesized here, unlike the HF convert script).
- **Final encoder LayerNorm** `audio_tower.layer_norm` → `enc.ln_post.*`
  (F32). Encoder blocks: `enc.blocks.N.{norm_attn,attn.{q,k,v,out},
  norm_ffn,ffn.{fc1,fc2}}` — **`attn.k` has no bias** (Whisper); q/v/out do.
- **Projector** (no biases): `multi_modal_projector.linear_1/2.weight` →
  `proj.linear_1/2.weight`. The 4× frame-grouping reshape is **not** a
  tensor — it's `stt.voxtral.projector.downsample_factor=4` /
  `input_dim=5120`; Stage 4 reshapes encoder output before `proj.linear_1`.
- **Text decoder** (Llama, no attention biases): `dec.blocks.N.{norm_attn,
  norm_ffn,attn.{q,k,v,o},ffn.{gate,up,down}}`, `dec.token_embd.weight`,
  `dec.output_norm.weight`. **lm_head is UNTIED** → emitted as a separate
  `dec.output.weight` (do not tie to `dec.token_embd`). NEOX RoPE
  theta 1e8 is applied in C++; q/k weights are stored as-is (already
  permuted to HF split-halves layout in the checkpoint).
- **Tokenizer (tekken → llama.cpp gpt2 BPE):** `tokenizer.ggml.model=
  "gpt2"`, `tokenizer.ggml.pre="tekken"`, 131072 tokens (ids 0–999
  CONTROL, 1000+ NORMAL byte-level), **269443 reconstructed merges**
  (tekken is rank-based/tiktoken with no explicit merges, so they are
  synthesized from the mergeable ranks exactly as llama.cpp's
  `MistralVocab` does), bos=1/eos=2/unk=0/pad=11, add_bos=true.
  Stage 4's C++ needs gpt2 byte-level BPE + the tekken pretokenizer regex.
- **Frontend buffers baked in:** `frontend.mel_filterbank` (slaney) and
  `frontend.window` (periodic Hann) as F32 tensors, plus the full
  `stt.frontend.*` KV block (per-utterance Whisper log-mel).
