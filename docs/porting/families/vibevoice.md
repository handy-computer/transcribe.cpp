# VibeVoice

Status: research

## Identity

- Family key: `vibevoice`
- Upstream architecture string: `VibeVoiceForASRTraining` (model_type `vibevoice`)
- Hugging Face repo: `microsoft/VibeVoice-ASR`
- Hugging Face revision: `d0c9efdb8d614685062c04425d91e01b6f37d944`
- License: MIT
- Variants: `vibevoice-asr` (~9B params, ~17.4 GB BF16)

## References

- Canonical reference: `github.com/microsoft/VibeVoice` — the `vibevoice` Python package
  (`vibevoice.modular.modeling_vibevoice_asr.VibeVoiceASRForConditionalGeneration`,
  `vibevoice.processor.vibevoice_asr_processor.VibeVoiceASRProcessor`). The HF repo
  ships weights + config.json only; no modeling code, processor, tokenizer, or
  generation_config.
- Instrumented reference: same package (`demo/vibevoice_asr_inference_from_file.py`).
- Cross-check references: tech report arXiv:2601.18184; `github.com/localai-org/vibevoice.cpp`
  (community ggml port — targets VibeVoice **TTS** and small Qwen2.5-0.5B variants, useful
  as a causal-conv / tokenizer reference only, not a drop-in for the 9B ASR model).

LM backbone is Qwen2.5-7B (decoder_config: hidden 3584, 28 layers, 28 heads / 4 KV,
intermediate 18944, vocab 152064, rope_theta 1e6, max_position 131072). The tokenizer
is the stock `Qwen/Qwen2.5-7B` byte-level BPE, loaded by the processor at runtime.

Frontend is **not** a spectrogram. Raw 24 kHz waveform feeds two parallel causal-conv
VAE encoders (acoustic vae_dim 64, semantic vae_dim 128), each with cumulative 3200x
downsampling → ~7.5 Hz token rate.

## Commands

Reference run:

```bash
# from a checkout of github.com/microsoft/VibeVoice
uv run --project scripts/envs/vibevoice \
  python demo/vibevoice_asr_inference_from_file.py \
  --model_path microsoft/VibeVoice-ASR --audio_files samples/jfk.wav
```

Reference dumps:

```bash
TODO  # porting-2-oracle: instrument vibevoice.modular.modeling_vibevoice_asr
```

Conversion:

```bash
TODO  # porting-3-convert: scripts/convert-vibevoice.py
```

Validation:

```bash
uv run scripts/validate.py all --family vibevoice --variant vibevoice-asr
```

Benchmarks:

```bash
TODO  # porting-6-bench
```

## Capability Validation

Proposed `Target` values below; non-forced rows require user sign-off at intake.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/vibevoice-asr/vibevoice-asr-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/vibevoice-asr/vibevoice-asr-BF16.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | TODO |
| Translate | only if exposed | n/a | — | OUT OF SCOPE — model transcribes (incl. code-switching) but does not translate; not advertised | TODO |
| Segment timestamps | structured Who/When/What output | `<TODO structured-output parse>` | segment-level timestamps present in parsed output | MUST PASS | TODO |
| Word timestamps | only if exposed | n/a | — | OUT OF SCOPE — model emits segment-level, not word-level timestamps | TODO |
| Speaker diarization | structured Who/When/What output | `<TODO diarization parse>` | speaker labels present in parsed output, DER/cpWER scored | MUST PASS | TODO |
| Hotwords / context bias | prompt-level biasing | `<TODO hotword prompt>` | biased transcript on domain terms | OUT OF SCOPE — first port uses the default prompt; hotword prompt path deferred | TODO |
| Streaming | non-streaming model | n/a | — | OUT OF SCOPE — model is non-streaming (60-min single pass); `capabilities.streaming: false` | TODO |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/vibevoice-asr/vibevoice-asr-BF16.gguf --samples-dir samples/wer/librispeech-test-clean --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | TODO |

## Notes

- Acceptance dataset: LibriSpeech test-clean (English supported; publisher does not report
  LibriSpeech, so the gate is the measured Oracle reference baseline, not a publisher score).
  Publisher-reported English WER: 7.99% MLC-Challenge, 18.81% AMI-IHM.
- The model always emits structured Who/When/What output. Plain-WER scoring requires a
  normalizer that strips speaker tags + timestamps from the hypothesis (Stage 2 / Stage 7).
- Segment timestamps and speaker diarization are **in scope (MUST PASS)** for this port:
  Stage 2/Stage 4 must parse the structured output and Stage 7 must score it (timestamps +
  DER/cpWER), in addition to plain transcript WER.
- Long-form (60-minute / 64K-token) inference is deferred; first port targets short utterances
  (jfk / LibriSpeech segments). See intake `known_risks` for the full list, including the
  learned-VAE frontend, dual-tokenizer fusion, the unused diffusion head, and KV-memory at
  long context.
