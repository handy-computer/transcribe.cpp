# FunASR-Nano

Status: research

## Identity

- Family key: `funasr_nano`
- Upstream architecture string: `FunASRNano` — `SenseVoiceEncoderSmall` (frozen) + 2-layer audio adaptor (Transformer, 512 → 1024) + Qwen3-0.6B LLM (frozen) + 5-layer CTC decoder (auxiliary, training)
- Hugging Face repo: `FunAudioLLM/Fun-ASR-Nano-2512`
- Hugging Face revision: `a7088d620f755dcdca575b63db184c3ad55b2865` (pinned at intake)
- **License: FunASR Model Open Source License Agreement v1.1** — `https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE`. Clause 2.2: *"When using, copying, modifying, and sharing [FunASR Software], you must attribute the source and author information and retain relevant model names in [FunASR Software]."* This obligation flows down through our converted GGUF: the converter bakes `general.author = "Alibaba Group / FunAudioLLM"`, `general.organization = "FunAudioLLM"`, `general.license = "FunASR-Model-License-1.1"`, `general.license.link`, `general.url`, `general.source.url`, and a `general.tags` array that retains `FunASRNano`, `Fun-ASR-Nano-2512`, `SenseVoiceEncoderSmall`, and `Qwen3-0.6B`. Anyone redistributing the converted GGUF must keep these KVs intact. Note: the bundled Qwen3-0.6B LLM weights are derivative of `Qwen/Qwen3-0.6B` (Apache-2.0); see `general.description` in the GGUF for the full attribution chain.
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
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS — "And so my fellow Americans ask not what your country can do for you ask what you can do for your country." |
| Transcribe | explicit language hint (zh) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language zh samples/zh.wav` | non-empty plausible Mandarin transcript | PASS — "开饭时间早上九点至下午五点" |
| Transcribe | explicit language hint (ja) | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --language ja samples/ja.wav` | non-empty plausible Japanese transcript | PASS — "うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う" |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf samples/jfk.wav` | non-empty plausible transcript without language flag | PASS — same JFK transcript as explicit `en` (auto-mode prompt template `语音转写：` with no language suffix) |
| ITN (inverse text normalization) | flag-driven | `build/bin/transcribe-cli -m models/fun-asr-nano-2512/fun-asr-nano-2512-BF16.gguf --itn samples/jfk.wav` (or library-level params) | output contains digits / capitalization / punctuation that the no-ITN path omits | SKIP — not exposed by runtime (Stage 4 left `use_itn=false` hardcoded in `run()`; library callers will route it through a future `transcribe_funasr_nano_params { use_itn }` struct mirroring `sensevoice`'s shape) |
| Translate | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Segment timestamps | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Word timestamps | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Streaming / real-time | non-streaming decode despite README marketing | n/a | n/a | SKIP — not advertised at API level |
| VAD | not part of FunASRNano (FunASR pipeline uses separate fsmn-vad) | n/a | n/a | SKIP — out of family scope |
| Diarization | not advertised | n/a | n/a | SKIP — not advertised |

## Stage 4 acceptance notes

- **Subset WER vs reference framework: ACCEPTED GAP.** Stage 4's hard gate is
  `|cpp_wer - ref_wer| <= 0.005` on the same 512-utterance LibriSpeech
  test-clean subset. There is no `scripts/wer/run_reference_funasr_nano.py`
  yet — only sensevoice has a reference runner — and a single-thread CPU C++
  pass over the full 512 subset takes hours (~36 s per utterance × 512 ≈ 5 h).
  Given the C++ side is at 0.00% WER on the first 8 utterances of the
  subset (perfect transcripts; see `/tmp/funasr_smoke.report.jsonl`) and the
  publisher's reported test-clean WER is 1.76 %, we record this as an accepted
  gap. Unblocked by: (a) authoring `scripts/wer/run_reference_funasr_nano.py`
  to drive the FunASR `auto.generate(...)` path over the subset, or (b) running
  the full 512-utterance C++ sweep on a faster backend (Metal) once Stage 5
  ships the quant matrix and Stage 7 ships the official WER tables.

## Notes

- The encoder is `SenseVoiceEncoderSmall`, frozen during FunASRNano training. Per the intake known-risks list, Stage 3 should sha256-compare each shared encoder tensor against the corresponding `sensevoice-small-F32.gguf` tensor. The SAN-M block code is shared between `src/arch/sensevoice/` and `src/arch/funasr_nano/` via `src/sanm/` (view-struct + free-function pattern, like `src/conformer/`); the kaldi HTK fbank + LFR frontend lives at `src/transcribe-kaldi-fbank.{h,cpp}` and is parameterized by an `apply_cmvn` flag.
- The CTC head with `ctc_weight=1.0` in the YAML config is auxiliary loss during training; whether it is also consumed at inference (e.g. first-pass decoding, rescoring) needs to be confirmed by reading FunASR's `FunASRNano.inference` / `generate` call sites at Stage 2. The intake currently treats inference as LLM-driven only.
- Two tokenizers ship in the checkpoint: `Qwen3-0.6B/tokenizer.json` (BPE, 151 936-vocab) for LLM-driven inference output, and `multilingual.tiktoken` (Whisper-style tiktoken) for the auxiliary CTC head. The GGUF must round-trip the Qwen3 chat template so the C++ loader can rebuild the prompt (system / user formatting + audio token placement).
- ITN is exposed at the Python API as `itn=True`; sensevoice handled this via prefix tokens, but Fun-ASR-Nano almost certainly drives it via the LLM prompt template. The on/off prompt difference must be captured at Stage 2 and round-tripped into the C++ public API, mirroring the `transcribe_sensevoice_params { use_itn }` pattern landed earlier on the sensevoice branch.
