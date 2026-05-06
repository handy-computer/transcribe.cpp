# FunASR-Nano

Status: research

## Identity

- Family key: `funasr_nano`
- Upstream architecture string: `FunASRNano` — `SenseVoiceEncoderSmall` (frozen) + 2-layer audio adaptor (Transformer, 512 → 1024) + Qwen3-0.6B LLM (frozen) + 5-layer CTC decoder (auxiliary, training)
- Hugging Face repo: `FunAudioLLM/Fun-ASR-Nano-2512`
- Hugging Face revision: `a7088d620f755dcdca575b63db184c3ad55b2865` (pinned at intake)
- License: see https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE — confirm redistribution terms before publishing converted GGUFs
- Variants:
  - `fun-asr-nano-2512` — ~800 M params on the trainable side; bundled Qwen3-0.6B brings total checkpoint to 1.97 GB (mixed-precision: encoder F32, LLM BF16). Headline languages zh / en / ja, plus Chinese dialects and accents.
  - (sibling, not yet ported) `FunAudioLLM/Fun-ASR-MLT-Nano-2512` — same architecture, 31 languages incl. Southeast Asian + European.

## References

- Canonical reference: **author_repo_funasr** — `funasr.AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", trust_remote_code=True)`. Under the hood FunASR loads the `FunASRNano` class via its model registry; the class definition lives in the FunASR repo (modelscope/FunASR) and combines `SenseVoiceEncoderSmall`, an audio adaptor, the bundled Qwen3-0.6B LLM, and a CTC head.
- Instrumented reference: same — Stage 2 will hook `FunASRNano.inference` / `FunASRNano.generate` / `WavFrontend.forward` for tensor dumps in the same shape used for sensevoice.
- Cross-check references:
  - `refs/models/sensevoice/` (when populated) — `SenseVoiceEncoderSmall` is shared verbatim; encoder weights are likely identical to sensevoice-small.
  - FunASR upstream toolkit https://github.com/modelscope/FunASR — authoritative for `WavFrontend`, SAN-M, and the FunASRNano class.
  - Qwen3-0.6B Hugging Face repo (Qwen/Qwen3-0.6B) — authoritative for the LLM tokenizer + chat-template behavior. Note that Fun-ASR-Nano bundles the LLM weights inside `model.pt`; the `Qwen3-0.6B/` subfolder in the checkpoint ships only tokenizer files.

There is no transformers-mainline path; FunASR is the only framework
that can run this model end-to-end.

## Commands

Reference run:

```bash
TODO  # Stage 2 will add: uv run scripts/validate.py ref --family funasr_nano --variant fun-asr-nano-2512
```

Reference dumps:

```bash
TODO  # Stage 2 will add:
#   uv run --project scripts/envs/funasr_nano \
#     scripts/dump_reference_funasr_nano_funasr.py decode \
#     --model FunAudioLLM/Fun-ASR-Nano-2512 \
#     --audio samples/jfk.wav \
#     --out build/validate/funasr_nano/fun-asr-nano-2512/jfk/decode/ref
```

Conversion:

```bash
TODO  # Stage 3 will add:
#   uv run --project scripts/envs/funasr_nano \
#     scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-Nano-2512
```

Validation:

```bash
TODO  # Stage 4 will add:
#   uv run scripts/validate.py all --family funasr_nano --variant fun-asr-nano-2512
```

Benchmarks:

```bash
TODO  # Stage 6 will add:
#   uv run scripts/bench/run.py --models fun-asr-nano-2512 --quants q8_0,q4_k_m \
#     --samples jfk,dots --backends metal,cpu --iters 3 --warmup 1 \
#     --name fun-asr-nano-2512-publication
```

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | TODO |
| Transcribe | explicit language hint (zh) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language zh samples/zh.wav` | non-empty plausible Mandarin transcript | TODO |
| Transcribe | explicit language hint (ja) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language ja samples/ja.wav` | non-empty plausible Japanese transcript | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf samples/jfk.wav` | non-empty plausible transcript without language flag | TODO |
| ITN (inverse text normalization) | flag-driven | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --itn samples/jfk.wav` (or library-level params) | output contains digits / capitalization / punctuation that the no-ITN path omits | TODO |
| Translate | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Segment timestamps | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Word timestamps | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Streaming / real-time | non-streaming decode despite README marketing | n/a | n/a | SKIP — not advertised at API level |
| VAD | not part of FunASRNano (FunASR pipeline uses separate fsmn-vad) | n/a | n/a | SKIP — out of family scope |
| Diarization | not advertised | n/a | n/a | SKIP — not advertised |

## Notes

- The encoder is `SenseVoiceEncoderSmall`, frozen during FunASRNano training. Per the intake known-risks list, Stage 3 should sha256-compare each shared encoder tensor against the corresponding `sensevoice-small-F32.gguf` tensor. Stage 4 should refactor SenseVoice encoder code into a shared module (`src/arch/_shared/` or similar) reused by both `src/arch/sensevoice/` and `src/arch/funasr_nano/`, rather than duplicating the SAN-M block code.
- The CTC head with `ctc_weight=1.0` in the YAML config is auxiliary loss during training; whether it is also consumed at inference (e.g. first-pass decoding, rescoring) needs to be confirmed by reading FunASR's `FunASRNano.inference` / `generate` call sites at Stage 2. The intake currently treats inference as LLM-driven only.
- Two tokenizers ship in the checkpoint: `Qwen3-0.6B/tokenizer.json` (BPE, 151 936-vocab) for LLM-driven inference output, and `multilingual.tiktoken` (Whisper-style tiktoken) for the auxiliary CTC head. The GGUF must round-trip the Qwen3 chat template so the C++ loader can rebuild the prompt (system / user formatting + audio token placement).
- ITN is exposed at the Python API as `itn=True`; sensevoice handled this via prefix tokens, but Fun-ASR-Nano almost certainly drives it via the LLM prompt template. The on/off prompt difference must be captured at Stage 2 and round-tripped into the C++ public API, mirroring the `transcribe_sensevoice_params { use_itn }` pattern landed earlier on the sensevoice branch.
