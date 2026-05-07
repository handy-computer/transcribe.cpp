# FunASR-Nano

Status: supported with manifest-driven numerical validation against
FunASR 1.3.1 for both shipped variants. C++ validation passes for
`fun-asr-nano-2512` and `fun-asr-mlt-nano-2512`; the CLI/API expose
ITN through the family prompt path. Stage 7 LibriSpeech test-clean WER
passes against FunASR reference baselines: Nano BF16 1.78% vs reference
1.79%, MLT BF16 1.74% vs reference 1.76%.

## Identity

- Family key: `funasr_nano`
- Upstream architecture string: `FunASRNano` — `SenseVoiceEncoderSmall` (frozen) + 2-layer audio adaptor (Transformer, 512 → 1024) + Qwen3-0.6B LLM (frozen) + 5-layer CTC decoder (auxiliary, training)
- Hugging Face repo: `FunAudioLLM/Fun-ASR-Nano-2512`
- Hugging Face revision: `a7088d620f755dcdca575b63db184c3ad55b2865` (pinned at intake)
- **License: FunASR Model Open Source License Agreement v1.1** — `https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE`. Clause 2.2: *"When using, copying, modifying, and sharing [FunASR Software], you must attribute the source and author information and retain relevant model names in [FunASR Software]."* This obligation flows down through our converted GGUF: the converter bakes `general.author = "Alibaba Group / FunAudioLLM"`, `general.organization = "FunAudioLLM"`, `general.license = "FunASR-Model-License-1.1"`, `general.license.link`, `general.url`, `general.source.url`, and a `general.tags` array that retains `FunASRNano`, `Fun-ASR-Nano-2512`, `SenseVoiceEncoderSmall`, and `Qwen3-0.6B`. Anyone redistributing the converted GGUF must keep these KVs intact. Note: the bundled Qwen3-0.6B LLM weights are derivative of `Qwen/Qwen3-0.6B` (Apache-2.0); see `general.description` in the GGUF for the full attribution chain.
- Variants:
  - `fun-asr-nano-2512` — `FunAudioLLM/Fun-ASR-Nano-2512`, ~800 M params on the trainable side; bundled Qwen3-0.6B brings total checkpoint to 1.97 GB (mixed-precision: encoder F32, LLM BF16). Headline languages zh / en / ja, plus Chinese dialects and accents.
  - `fun-asr-mlt-nano-2512` — `FunAudioLLM/Fun-ASR-MLT-Nano-2512`, identical architecture and ~1.97 GB checkpoint, trained on a smaller corpus (hundreds of thousands of hours vs Nano's tens of millions). Covers 31 languages: zh, en, yue, ja, ko, vi, id, th, ms, tl, ar, hi, bg, hr, cs, da, nl, et, fi, el, hu, ga, lv, lt, mt, pl, pt, ro, sk, sl, sv. Validated through the same runtime path as a sibling variant.

## References

- Canonical reference: **author_repo_funasr** — `funasr.AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", trust_remote_code=True)`. Under the hood FunASR loads the `FunASRNano` class via its model registry; the class definition lives in the FunASR repo (modelscope/FunASR) and combines `SenseVoiceEncoderSmall`, an audio adaptor, the bundled Qwen3-0.6B LLM, and a CTC head.
- Instrumented reference: same — `scripts/dump_reference_funasr_nano_funasr.py` hooks `FunASRNano.inference` / `FunASRNano.generate` / `WavFrontend.forward` for tensor dumps in the same shape used for sensevoice.
- Cross-check references:
  - `refs/models/sensevoice/` (when populated) — `SenseVoiceEncoderSmall` is shared verbatim; encoder weights are likely identical to sensevoice-small.
  - FunASR upstream toolkit https://github.com/modelscope/FunASR — authoritative for `WavFrontend`, SAN-M, and the FunASRNano class.
  - Qwen3-0.6B Hugging Face repo (Qwen/Qwen3-0.6B) — authoritative for the LLM tokenizer + chat-template behavior. Note that Fun-ASR-Nano bundles the LLM weights inside `model.pt`; the `Qwen3-0.6B/` subfolder in the checkpoint ships only tokenizer files.

There is no transformers-mainline path; FunASR is the only framework
that can run this model end-to-end.

## Environment

Python env: `scripts/envs/funasr_nano/pyproject.toml`
(funasr==1.3.1, transformers==4.57.6, torch/torchaudio, soundfile,
librosa, gguf, sentencepiece, kaldiio, tiktoken). FunASR 1.3.1 is
pinned because the `FunASRNano` registry entry, state-dict layout, and
`generate()` signature are framework-version specific.

## Golden Manifest

- `tests/golden/funasr_nano/fun-asr-nano-2512.manifest.json`
- `tests/golden/funasr_nano/fun-asr-mlt-nano-2512.manifest.json`

## Artifacts

| Artifact | Path |
| --- | --- |
| Intake (Nano) | `reports/porting/funasr_nano/fun-asr-nano-2512/intake.json` |
| Intake (MLT) | `reports/porting/funasr_nano/fun-asr-mlt-nano-2512/intake.json` |
| Forward map | `reports/porting/funasr_nano/forward-map.md` |
| Tolerances | `tests/tolerances/funasr_nano.json` |
| Converter reports | `reports/convert/fun-asr-{nano,mlt-nano}-2512-BF16.json` |
| Reference dump root | `build/validate/funasr_nano/` |
| Bench reports | `reports/perf/apple-m4-max/fun-asr-{nano,mlt-nano}-2512-publication_*.json` |
| WER scores | `reports/wer/fun-asr-{nano,mlt-nano}-2512-{REF,BF16,F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M}.test-clean.score.json` |
| WER summaries | `reports/wer/fun-asr-{nano,mlt-nano}-2512.test-clean.summary.md` |

## Commands

Full numerical validation:

```bash
uv run scripts/validate.py all --family funasr_nano --variant fun-asr-nano-2512
uv run scripts/validate.py all --family funasr_nano --variant fun-asr-mlt-nano-2512
```

Step by step:

```bash
for VARIANT in fun-asr-nano-2512 fun-asr-mlt-nano-2512; do
  uv run scripts/validate.py ref     --family funasr_nano --variant ${VARIANT}
  uv run scripts/validate.py cpp     --family funasr_nano --variant ${VARIANT}
  uv run scripts/validate.py compare --family funasr_nano --variant ${VARIANT}
done
```

Reference dump (single audio):

```bash
uv run --project scripts/envs/funasr_nano \
  scripts/dump_reference_funasr_nano_funasr.py decode \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --audio samples/jfk.wav \
    --language en \
    --out build/validate/funasr_nano/fun-asr-nano-2512/jfk/decode/ref
```

Conversion (reference dtype = BF16):

```bash
uv run --project scripts/envs/funasr_nano \
  scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-Nano-2512 \
    --revision a7088d620f755dcdca575b63db184c3ad55b2865 \
    --repo-id FunAudioLLM/Fun-ASR-Nano-2512 \
    --variant fun-asr-nano-2512

uv run --project scripts/envs/funasr_nano \
  scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --revision cf67a938bf2829959d08fdfb84e186eff02a67ff \
    --repo-id FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --variant fun-asr-mlt-nano-2512
```

Quantize (run once per target preset):

```bash
for Q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  build/bin/transcribe-quantize \
    models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf \
    models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-${Q}.gguf \
    --quant ${Q}

  build/bin/transcribe-quantize \
    models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf \
    models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-${Q}.gguf \
    --quant ${Q}
done
```

Bench (Stage 6 publication runs):

```bash
uv run scripts/bench/run.py \
  --models Fun-ASR-Nano-2512 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name fun-asr-nano-2512-publication

uv run scripts/bench/run.py \
  --models Fun-ASR-MLT-Nano-2512 \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name fun-asr-mlt-nano-2512-publication
```

WER (Stage 7 — full LibriSpeech test-clean sweep + reference baselines):

```bash
# Reference baselines (FunASR).
uv run --project scripts/envs/funasr_nano \
  scripts/wer/run_reference_funasr_nano.py \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-nano-2512-REF.test-clean.jsonl \
    --torch-threads 12
uv run scripts/wer/score.py reports/wer/fun-asr-nano-2512-REF.test-clean.jsonl

uv run --project scripts/envs/funasr_nano \
  scripts/wer/run_reference_funasr_nano.py \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-mlt-nano-2512-REF.test-clean.jsonl \
    --model    FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --revision cf67a938bf2829959d08fdfb84e186eff02a67ff \
    --torch-threads 12
uv run scripts/wer/score.py reports/wer/fun-asr-mlt-nano-2512-REF.test-clean.jsonl

# transcribe.cpp ports.
for PRESET in BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  uv run scripts/wer/run.py \
    --model models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-${PRESET}.gguf \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-nano-2512-${PRESET}.test-clean.jsonl
  uv run scripts/wer/score.py reports/wer/fun-asr-nano-2512-${PRESET}.test-clean.jsonl

  uv run scripts/wer/run.py \
    --model models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-${PRESET}.gguf \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out      reports/wer/fun-asr-mlt-nano-2512-${PRESET}.test-clean.jsonl
  uv run scripts/wer/score.py reports/wer/fun-asr-mlt-nano-2512-${PRESET}.test-clean.jsonl
done
```

## Architecture summary

- Pattern: `audio-llm` (frozen SenseVoice encoder + audio adaptor +
  bundled Qwen3-0.6B causal LM). Transcripts come from the LLM decoder.
- Frontend: FunASR `WavFrontend` — 80-bin Kaldi-style fbank
  (16 kHz, 25 ms frame length, 10 ms shift, hamming window), LFR stack
  (`lfr_m=7`, `lfr_n=6`) to 560-d features, **no CMVN**
  (`stt.frontend.apply_cmvn=false`).
- Encoder: `SenseVoiceEncoderSmall` SAN-M body, shared shape with
  SenseVoice but without the SenseVoice prefix embeddings or CTC decode:
  projection block, 49 main residual SAN-M blocks, LayerNorm, 20
  `tp_encoders`, final LayerNorm.
- Adaptor: 2-layer Transformer-style adaptor maps 512-d encoder frames
  to 1024-d Qwen embeddings. `use_low_frame_rate=true` means only the
  computed `fake_token_len` prefix of adaptor frames is spliced into
  the LLM prompt.
- Decoder: bundled Qwen3-0.6B BF16 weights, 28 layers, 16/8 GQA, tied
  lm_head. The chat-template prompt contains `<|startofspeech|>` /
  `<|endofspeech|>` markers; C++ replaces the intervening placeholder
  rows with adaptor audio embeddings.
- Tokenizers: Qwen3 byte-level BPE drives inference. The published
  checkpoint also carries a partial auxiliary CTC path, but required CTC
  tensors are absent, so the port intentionally skips CTC inference.
- Language and ITN controls are prompt-template driven. Explicit
  language sets `语音转写成<lang>...`; default ITN off appends
  `，不进行文本规整`, while `use_itn=true` omits that suffix.

## Capability Validation

One row per advertised capability. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not advertised` — capability is not advertised upstream for
  this model family.
- `SKIP — not exposed by runtime` — capability exists upstream but the
  public CLI/API does not surface it.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe (nano) | explicit language hint (en) | `build/bin/transcribe-cli -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS — "And so my fellow Americans ask not what your country can do for you ask what you can do for your country." |
| Transcribe (nano) | explicit language hint (zh) | `build/bin/transcribe-cli -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf --language zh samples/zh.wav` | non-empty plausible Mandarin transcript | PASS — "开饭时间早上九点至下午五点" |
| Transcribe (nano) | explicit language hint (ja) | `build/bin/transcribe-cli -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf --language ja samples/ja.wav` | non-empty plausible Japanese transcript | PASS — "うちの中学は弁当制で持っていけない場合は五十円の学校販売のパンを買う" |
| Transcribe (nano) | auto / no language hint | `build/bin/transcribe-cli -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf samples/jfk.wav` | non-empty plausible transcript without language flag | PASS — same JFK transcript as explicit `en` (auto-mode prompt template `语音转写：` with no language suffix) |
| Transcribe (mlt) | explicit language hint (en) | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS — "and so my fellow americans ask not what your country can do for you ask what you can do for your country" |
| Transcribe (mlt) | explicit language hint (zh) | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf --language zh samples/zh.wav` | non-empty plausible Mandarin transcript | PASS — "開放時間早上九點至下午五點" (note: MLT defaults to traditional Han glyphs vs Nano's simplified "开饭时间") |
| Transcribe (mlt) | explicit language hint (ja) | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf --language ja samples/ja.wav` | non-empty plausible Japanese transcript | PASS — "うちの中学は弁当制で、持っていけない場合は、五十円の学校販売のパンを買う" (with ITN-style commas) |
| Transcribe (mlt) | explicit language hint (ko) | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf --language ko samples/ko.wav` | non-empty plausible Korean transcript | PASS — "조금만 생각을 하면서 살면 훨씬 편할 거야" (MLT-specific; Nano does not advertise ko) |
| Transcribe (mlt) | explicit language hint (yue) | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf --language yue samples/yue.wav` | non-empty plausible Cantonese transcript | PASS — "呢幾個字都表達唔到我想講嘅意思" (uses Cantonese particles 呢/嘅; MLT-specific first-class language) |
| Transcribe (mlt) | auto / no language hint | `build/bin/transcribe-cli -m models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-BF16.gguf samples/jfk.wav` | non-empty plausible transcript without language flag | PASS — same English transcript as explicit `en` (auto-mode prompt template applied) |
| ITN (inverse text normalization) | prompt flag, both variants | `build/bin/transcribe-cli -m models/Fun-ASR-Nano-2512/Fun-ASR-Nano-2512-BF16.gguf --language zh --itn samples/zh.wav` (repeat with `Fun-ASR-MLT-Nano-2512`; library: `transcribe_funasr_nano_params{ .use_itn = true }`) | runtime selects the upstream ITN prompt path; transcript remains plausible | PASS — `--itn` is wired through CLI and API; Nano zh renders `开饭时间早上九点至下午五点。`, and both variants share the same prompt builder |
| Translate | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Segment timestamps | not exposed upstream | n/a | n/a | SKIP — README explicitly lists `Support returning timestamps` as TODO |
| Word timestamps | not exposed upstream | n/a | n/a | SKIP — not advertised |
| Streaming / real-time | non-streaming decode despite README marketing | n/a | n/a | SKIP — not advertised at API level |
| VAD | not part of FunASRNano (FunASR pipeline uses separate fsmn-vad) | n/a | n/a | SKIP — out of family scope |
| Diarization | not advertised | n/a | n/a | SKIP — README explicitly lists `Support speaker diarization` as TODO |

## Acceptance notes

- **Stage 7 WER gap closed.** `scripts/wer/run_reference_funasr_nano.py`
  now drives the FunASR `auto.generate(...)` path over the full 2620-utterance
  LibriSpeech test-clean manifest for both variants. See
  `reports/wer/fun-asr-nano-2512.test-clean.summary.md` and
  `reports/wer/fun-asr-mlt-nano-2512.test-clean.summary.md`.
- **Ref-dtype hard gate: PASS.** Nano BF16 is 1.78% WER vs 1.79%
  reference (-0.01 pp). MLT BF16 is 1.74% WER vs 1.76% reference
  (-0.02 pp). Both are far inside the ±0.005 absolute Stage 7 gate.
- **ITN exposure closed.** The public CLI flag `--itn` and library
  `transcribe_funasr_nano_params.use_itn` now select the FunASR prompt
  path that omits `，不进行文本规整`.

## Notes

- The encoder is `SenseVoiceEncoderSmall`, frozen during FunASRNano training. The SAN-M block code is shared between `src/arch/sensevoice/` and `src/arch/funasr_nano/` via `src/sanm/` (view-struct + free-function pattern, like `src/conformer/`); the Kaldi HTK fbank + LFR frontend lives at `src/transcribe-kaldi-fbank.{h,cpp}` and is parameterized by an `apply_cmvn` flag.
- The CTC head with `ctc_weight=1.0` in the YAML config is auxiliary loss during training. The released checkpoints omit required CTC tensors (`ctc_decoder` layers 2..4 and `ctc.ctc_lo.*`), so runtime inference is LLM-driven only.
- Two tokenizers ship in the checkpoint: `Qwen3-0.6B/tokenizer.json` (BPE, 151 936-vocab) for LLM-driven inference output, and `multilingual.tiktoken` (Whisper-style tiktoken) for the auxiliary CTC head. The GGUF round-trips the Qwen3 chat template so the C++ loader can rebuild the prompt (system / user formatting + audio token placement).
- ITN is prompt-template driven. `transcribe_funasr_nano_params.use_itn`
  mirrors the SenseVoice family-param shape but affects FunASR Nano by
  omitting the no-normalization suffix from the LLM prompt; the CLI
  `--itn` flag fills both family param structs and each handler consumes
  only its own pointer.
