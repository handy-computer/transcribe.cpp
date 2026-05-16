# Granite NAR (NLE Editor)

Status: research

## Identity

- Family key: `granite_nar`
- Upstream architecture string: `nle` (`NLENARDecoder`)
- Hugging Face repo: `ibm-granite/granite-speech-4.1-2b-nar` (pinned `7d20732df04d097262c4ecd8fe7f34ec2b3e6c42`)
- License: Apache-2.0
- Variants:
  - `granite-speech-4.1-2b-nar`: Non-autoregressive editor; en/fr/de/es/pt transcription only.

This is a sibling release to the `granite` audio-LLM family but with a fundamentally different decoding pipeline (encoder-CTC plus bidirectional Granite LLM as a single-pass editor). Kept in its own family so the transcribe.cpp `src/arch/granite_nar/` implementation does not need to share code with the AR audio-LLM family.

## References

- Canonical reference: `transformers` with `trust_remote_code=True`. The repo ships its own `configuration_nle.py`, `feature_extraction_nle.py`, and `modeling_nle.py` (registered via `auto_map`); no IBM author PyPI package exists. Model card pins `transformers==4.57.6` or `5.5.3`, plus `flash-attn==2.8.3`.
- Instrumented reference: same as canonical. Configured under `scripts/envs/granite_nar/` at Stage 2.
- Cross-check references: the AR sibling family (`granite`) shares the underlying Conformer encoder and the Granite-4.0-1b-base LLM weights; encoder-side tensor-name conventions can be reused, but the LLM operates without a causal mask here so its forward graph differs.

## Architecture

Pattern: encoder-ctc with a bidirectional-LLM editor head on top. The four canonical porting patterns do not directly capture the editor step, so Stage 4 will need a custom flow.

- **Audio encoder**: Conformer with block-attention, identical dims to the AR `granite_speech` encoder: 16 layers, `hidden_dim=1024`, 8 heads at `dim_head=128`, depthwise `conv_kernel_size=15`, `conv_expansion_factor=2`, `feedforward_mult=4`, `input_dim=160` (= 80 logmels with a 2-frame stack), `output_dim=348`, `max_pos_emb=512` (Shaw relpos), `context_size=200`. Adds NAR-specific extras: `bpe_output_dim=100353`, `bpe_pooling_window=4`, `self_conditioning_layer=8`, `attn_type=block`, `loss_lambda=0.2`.
- **Projector**: `nle_projector` (NOT a Q-Former). MLP-with-attention module: `block_size=15`, `downsample_rate=5`, `encoder_dim=1024`, `hidden_size=2048`, `llm_dim=2048`, `mlp_ratio=2`, `num_layers=2`, `num_heads=32`, `num_encoder_layers=4`, `attn_bias=true`, `layernorm_eps=1e-06`. Consumes multi-layer encoder features at `encoder_layer_indices=[4, 8, 12, -1]`. `scale_projected_embeddings=true` (the projector output is scaled before the LLM; exact factor lives in modeling_nle.py and must be read at oracle time).
- **Text LM**: `granite-4.0-1b-base` (`model_type=granite`), same dims as in the AR family (40 layers, 2048 hidden, 16 query / 4 KV heads, RMS eps 1e-05, RoPE theta 10000, SiLU, Granite scalar multipliers). Differences here:
  - `tie_word_embeddings=true` (vs false in the base AR variants).
  - `vocab_size=100352` (vs 100353 in AR; no `audio_token_index` because there is no LLM token-injection).
  - **Causal mask disabled**: `attn_implementation=flash_attention_2` with `is_causal=False`. The LLM is used as a bidirectional editor; the ggml graph must not apply the standard causal triangular mask.
  - LoRA adapters are hinted by the model card but `has_lora_adapter` is unset in config. Stage 3 (convert) must determine whether deltas are merged into the safetensors or shipped separately.
- **Fusion**: no audio-token scatter. The LLM receives projected encoder outputs directly as input embeddings.
- **Decode**: a single forward pass through encoder + projector + bidirectional LLM produces logits over the LLM vocab; CTC greedy decode on those logits yields the final text. No autoregressive token loop, no generation_config.
- **CTC tokenizer**: a 348-entry `char2idx` embedded inline in config covering ASCII 32-127 + Latin-1 supplement 128-255 + Katakana 0x30A1-0x30FB (indices 256-347). Used at the encoder-side CTC head; the LLM-output CTC decode uses the same tokenizer as the AR variants (BPE, vocab 100352, identical sha256). The Katakana entries appear vestigial since Japanese is not in the supported language list.

## Frontend

`NLEFeatureExtractor` with flat `preprocessor_config.json` fields (no `melspec_kwargs` nesting). Numerically identical to the AR `granite_speech` variants:

- `sample_rate=16000`, mono.
- `n_mels=80`, `hop_length=160`, `win_length=400`, `n_fft=512` (window zero-padded to 512 before FFT).
- Window/center/padding/mel-norm not declared; values follow `torchaudio.transforms.MelSpectrogram` defaults (Hann periodic, `center=True`, reflect padding, htk-style mel scale, no filter normalization). Stage 2 (oracle) must confirm by running the reference processor end-to-end.

The encoder consumes `80 mels x 2-frame stack = input_dim=160`; the 2-frame stack is implicit in the encoder definition.

## Commands

Reference run:

```bash
TODO
```

Reference dumps:

```bash
TODO
```

Conversion:

```bash
TODO
```

Validation:

```bash
TODO
```

Benchmarks:

```bash
TODO
```

## Capability Validation

Allowed statuses: `PASS`, `SKIP - not exposed by runtime`, `ACCEPTED GAP - <reason>`.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-bf16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript produced by a single NAR editor pass | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/granite-speech-4.1-2b-nar/granite-speech-4.1-2b-nar-bf16.gguf samples/jfk.wav` | non-empty plausible transcript | TODO |

## Notes

- Single-variant family at intake time. Future NAR releases from IBM would slot in here.
- Upstream LibriSpeech test-clean WER target for porting-7-wer: 1.29 (ref-dtype C++ must score <= 1.30 on the same manifest).
- The bidirectional-LLM-with-disabled-causal-mask + LoRA-adapter situation is the highest-risk part of this port. Stage 4 should bring up the encoder + projector + CTC head against the AR family's existing encoder code first, then bolt on the bidirectional LLM forward graph as a distinct step.
