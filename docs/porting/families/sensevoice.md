# SenseVoice

Status: research

## Identity

- Family key: `sensevoice`
- Upstream architecture string: `SenseVoiceSmall` (encoder `SenseVoiceEncoderSmall` + CTC head)
- Hugging Face repo: `FunAudioLLM/SenseVoiceSmall`
- Hugging Face revision: `3eb3b4eeffc2f2dde6051b853983753db33e35c3` (pinned at intake)
- License: `model-license` (see https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE — confirm redistribution terms before publishing converted GGUFs)
- Variants:
  - `sensevoice-small` — 234M params, F32, 5 advertised languages (zh, yue, en, ja, ko), audio capped at 30 s per direct-inference call

## References

- Canonical reference: **author_repo_funasr** — `funasr.AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf")` is the published entry path on the HF model card. The alternate direct path uses `from model import SenseVoiceSmall; SenseVoiceSmall.from_pretrained(...)` from the [SenseVoice repo's `model.py`](https://github.com/FunAudioLLM/SenseVoice), which itself depends on FunASR for SAN-M attention, `WavFrontend`, `SinusoidalPositionEncoder`, `SentencepiecesTokenizer`, and the CTC head.
- Instrumented reference: same — FunASR (with hooks added to `SenseVoiceSmall.inference` / `WavFrontend.forward` for tensor dumps in Stage 2).
- Cross-check references:
  - SenseVoice repo's standalone `model.py` — useful to read alongside the FunASR class hierarchy because it inlines the encoder and CTC head in one file.
  - FunASR upstream toolkit at https://github.com/modelscope/FunASR — authoritative for `WavFrontend`, the SAN-M layer, and the kaldifeat fbank invocation.

There is no HF Transformers / NeMo / ESPnet equivalent for this model.
There is no `trust_remote_code` shim (the HF repo carries no `config.json`
or `modeling_*.py`). FunASR is the only framework that can run it.

## Commands

Reference run:

```bash
TODO  # Stage 2 will add: uv run scripts/validate.py ref --family sensevoice --variant sensevoice-small
```

Reference dumps:

```bash
TODO  # Stage 2 will add:
#   uv run --project scripts/envs/sensevoice \
#     scripts/dump_reference_sensevoice_funasr.py decode \
#     --model FunAudioLLM/SenseVoiceSmall \
#     --audio samples/jfk.wav \
#     --out build/validate/sensevoice/sensevoice-small/jfk/ref
```

Conversion:

```bash
TODO  # Stage 3 will add:
#   uv run --project scripts/envs/sensevoice \
#     scripts/convert-sensevoice.py FunAudioLLM/SenseVoiceSmall
```

Validation:

```bash
TODO  # Stage 4 will add:
#   uv run scripts/validate.py all --family sensevoice --variant sensevoice-small
```

Benchmarks:

```bash
TODO  # Stage 7 will add:
#   uv run scripts/wer.py --family sensevoice --variant sensevoice-small \
#     --dataset librispeech-test-clean
```

## Architecture summary

- Pattern: `encoder-ctc` (non-autoregressive; CTC head over a 25,055-token SentencePiece vocab; no decoder, no cross-attention).
- Frontend: FunASR `WavFrontend` — 80-bin kaldifeat fbank (sample_rate=16000, frame_length=25 ms = 400 samples, frame_shift=10 ms = 160 samples, window=hamming), per-feature CMVN read from `am.mvn`, then **low-frame-rate (LFR) stacking** (lfr_m=7 consecutive 80-bin frames concatenated with stride lfr_n=6) producing 560-d features at 60 ms frame period as the encoder input. Center / padding / dither / mel filterbank norm need confirmation against `funasr/frontends/wav_frontend.py` at Stage 2 (recorded as intake gaps).
- Generation contract: a 4-token prefix `[language, event, emotion, textnorm]` is **prepended as embeddings** to the front of the encoder input. Each prefix slot indexes a small `Embedding(7+6+2, input_size)` keyed by `lid_dict`/`event_dict`/`emo_dict`/`textnorm_dict`. These are not vocabulary tokens passed through an LM-head decoder. The output language/event/emotion/ITN labels live in the *output* CTC vocabulary at separate token IDs (e.g. `<|en|>=24885`, `<|HAPPY|>=25001`, `<|withitn|>=25016`).
- Encoder: `SenseVoiceEncoderSmall` (SAN-M = Self-Attention Network with Memory). Two-tier depth: `encoders` = 50 SAN-M blocks, `tp_encoders` = 20 SAN-M blocks. d_model=512, attention_heads=4, linear_units=2048, FSMN kernel_size=11, sinusoidal positional encoding, pre-norm.
- Output head: CTC over vocab_size=25055. Blank id is 0 (`<unk>`). Decoding is greedy/beam CTC.
- Tokenizer: SentencePiece BPE (`chn_jpn_yue_eng_ko_spectok.bpe.model`), 25,055 pieces. SP-level specials: `<unk>=0`, `<s>=1`, `</s>=2`. The vocabulary also carries language tokens (24884–24992), event tokens (24993+), emotion tokens (25001+), and ITN tokens (25016/25017) — these are emitted by the CTC head at decode time.
- Audio length contract: the model's direct inference path is capped at **30 seconds per call** (model card explicit). Long-form inference in the upstream demo uses an external `fsmn-vad` model to chunk audio before SenseVoice; that chunker is a separate FunASR model and is **out of scope for this port**.

## Capabilities (from intake)

- Languages: 5 advertised — zh, yue, en, ja, ko. (Note: HF model-card YAML lists `[en, zh, ja, ko]` and is missing `yue`; the language-token vocabulary and the README body both confirm Cantonese support — treat the HF YAML as an upstream metadata bug.)
- Language detection: yes (`language="auto"`; falls back to `<|nospeech|>` for non-speech).
- Translation: no.
- Timestamps: none — non-AR CTC has no segment/word timestamp head.
- Streaming: no (not advertised).
- VAD: no (the SenseVoice model itself emits a `<|nospeech|>` label and event tags, but frame-level VAD is bolted on externally via `fsmn-vad`).
- Diarization: no.
- Beyond schema fields: speech emotion recognition (SER), audio event detection (AED), and inverse text normalization (ITN) are advertised — they emerge from the same CTC head as separate vocabulary tokens. Stage 4 will need to decide whether the runtime exposes these as user-visible outputs or strips them.

## Upstream benchmarks (from model card)

The model card cites benchmarks against Whisper on AISHELL-1, AISHELL-2,
Wenetspeech, LibriSpeech, and Common Voice — but publishes the
**numerical scores only as PNG figures** (`image/asr_results1.png`,
`image/asr_results2.png`). There is no machine-readable WER/CER number
to gate against.

Acceptance dataset for Stage 7 WER gate: **LibriSpeech test-clean**
(default per the porting skill; SenseVoice supports English so it is a
valid target). Treat the absence of an upstream baseline as
"no parity number to match" — Stage 7 reports our measured WER without
a publisher-claimed gate. Complementary tests on AISHELL-1 (Mandarin,
CER) and Common Voice are recommended because Mandarin is SenseVoice's
strongest case.

## Known risks

See `reports/porting/sensevoice/sensevoice-small/intake.json::known_risks`. Highlights:

1. **FunASR-native checkpoint, not HF Transformers.** Loader is `SenseVoiceSmall.from_pretrained` (FunASR class registry) operating on a PyTorch pickle (`model.pt`). State_dict keys come from the FunASR class hierarchy and need an explicit converter mapping; there is no `safetensors_index.json` / `config.json` / `preprocessor_config.json` to lean on.
2. **kaldifeat fbank frontend, not a torchaudio mel-spectrogram.** Window=hamming, HTK-style mel filterbank, per-feature CMVN from `am.mvn` (560-dim). Whisper-style hann_periodic + slaney mels + log-mel compression are NOT applicable. Center/padding/dither defaults need confirmation from the source.
3. **Low-frame-rate (LFR) stacking is the hidden frontend trap.** Mels (80-bin) at 10 ms stride are stacked into 560-d features at 60 ms stride before the encoder. Skipping or misaligning this step silently breaks every downstream tensor shape.
4. **SAN-M attention with two-tier depth.** 50 main + 20 "tp" blocks. Each layer is MHA + FSMN-memory branch (kernel_size=11). Replicating the FSMN convolution and the two-tier masking is non-trivial.
5. **4-embedding prefix is NOT vocabulary tokens.** Language/event/emotion/textnorm prefixes are direct embeddings indexed by small lid/event/emo/textnorm dicts and prepended to the encoder input. The vocabulary IDs of `<|en|>` etc. are how the *output* CTC head labels emit those classes — confusing input-prefix indices with output vocab IDs will break decoding.
6. **30-second direct-inference cap.** Long-form requires external VAD chunking (fsmn-vad), which is out of scope.
7. **Publisher reports no numerical WER.** Benchmark scores are PNG-only on the model card.
8. **License is `model-license`, not Apache-2.0.** Confirm redistribution terms before publishing converted GGUFs to handy-computer/* on the Hub.

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint (en) | `build/bin/transcribe-cli -m models/sensevoice-small/sensevoice-small-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS |
| Transcribe | explicit language hint (zh) | `build/bin/transcribe-cli -m models/sensevoice-small/sensevoice-small-F32.gguf --language zh samples/zh.wav` | non-empty plausible Mandarin transcript | PASS — `开放时间早上九点至下午五点` |
| Transcribe | explicit language hint (ja, ko, yue) | `build/bin/transcribe-cli -m … --language <ja|ko|yue> samples/<ja|ko|yue>.wav` | non-empty plausible transcript in the requested language | PASS |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/sensevoice-small/sensevoice-small-F32.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | PASS |
| Language detection | LID emitted via raw CTC tokens | `build/bin/transcribe-cli --raw-tokens -m … samples/jfk.wav` | language label `<\|en\|>` / `<\|zh\|>` / etc. present in the transcript text field | PASS — `<\|en\|>` emitted on jfk.wav, `<\|zh\|>` on zh.wav, etc. |
| Translate | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Segment timestamps | n/a | non-AR CTC has no segment-timestamp head | n/a | SKIP — not advertised |
| Word timestamps | n/a | non-AR CTC has no word-timestamp head; runtime does not expose ctc_forced_align in v1 | n/a | SKIP — not exposed by runtime |
| Streaming | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Voice activity detection | n/a | external (fsmn-vad), out of scope for this port | n/a | SKIP — not exposed by runtime |
| Speaker diarization | n/a | not advertised by upstream | n/a | SKIP — not advertised |
| Speech emotion recognition (SER) | emotion token in output | `build/bin/transcribe-cli --raw-tokens -m … samples/jfk.wav` | one of `<\|EMO_UNKNOWN\|>` / `<\|NEUTRAL\|>` / `<\|HAPPY\|>` / `<\|SAD\|>` / `<\|ANGRY\|>` etc. present in raw CTC output | PASS — `<\|EMO_UNKNOWN\|>` emitted on jfk.wav |
| Audio event detection (AED) | event token in output | `build/bin/transcribe-cli --raw-tokens -m … samples/jfk.wav` | one of `<\|Speech\|>` / `<\|BGM\|>` / `<\|Applause\|>` etc. present in raw CTC output | PASS — `<\|Speech\|>` emitted on jfk.wav |
| Inverse text normalization (ITN) | use_itn=True path | `build/bin/transcribe-cli --itn -m … samples/jfk.wav` (or library: `transcribe_sensevoice_params{ .use_itn = true }` via `transcribe_params::sensevoice`) | numbers/punctuation rendered in formal form when ITN enabled; `<\|withitn\|>` rides along in raw output | PASS — ITN-on jfk renders as `And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.`; ITN-on zh renders `九点` → `9点` and adds period |

Stage 4 will replace `<actual supported command>` placeholders with the
real CLI invocations once the runtime exposes (or doesn't expose) raw
CTC token visibility, ITN, and per-utterance language/event/emotion
labels.

## Notes

- This is the first FunASR-native port in the repo. Existing ports
  (Whisper, Parakeet, Cohere, Qwen3-ASR) all live under HF Transformers
  or NeMo, both of which expose `config.json` / preprocessor / tokenizer
  files mechanically. Expect intake → preflight → converter scaffolding
  to need a FunASR-aware code path that reads `config.yaml`, `am.mvn`,
  and the SP model directly.
- Use `samples/jfk.wav` for the English smoke test. Cantonese (`yue`),
  Mandarin (`zh`), Japanese (`ja`), and Korean (`ko`) example clips
  ship with the HF repo at `example/{yue,zh,ja,ko}.mp3` — pull these
  into `samples/` (or reference them by repo path) for the multilingual
  capability rows.
- Stage 7's WER target is LibriSpeech test-clean, but the model is
  optimized for Mandarin. AISHELL-1 (CER) is the more representative
  complementary check; flag if absolute LibriSpeech WER comes in
  noticeably worse than Whisper-Small without that being a port defect.
