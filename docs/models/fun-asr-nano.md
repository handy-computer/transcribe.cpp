# Fun-ASR-Nano

Alibaba / FunAudioLLM's
[Fun-ASR-Nano](https://huggingface.co/FunAudioLLM) family ported to
transcribe.cpp. Both variants share the same architecture — a frozen
**SenseVoiceEncoderSmall** (50 SAN-M main blocks + 20 transformer
blocks), a 2-layer audio adaptor (512 → 1024), and a bundled
**Qwen3-0.6B** LLM (28 layers, 16/8 GQA, BF16) that produces the
transcript autoregressively — and differ only in which corpus they
were trained on. Pick on language coverage.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/funasr_nano.md`](../porting/families/funasr_nano.md).

## Choosing a variant

- **Mandarin-heavy workloads.** `fun-asr-nano-2512` — zh / en / ja,
  plus 7 Chinese dialects (Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin)
  and 26 regional Mandarin accents. Trained on the larger
  Mandarin-skewed corpus; the right pick when Chinese accuracy is the
  primary axis.
- **Broad multilingual coverage.** `fun-asr-mlt-nano-2512` — 31
  languages (zh / en / yue / ja / ko / vi / id / th / ms / tl / ar /
  hi / bg / hr / cs / da / nl / et / fi / el / hu / ga / lv / lt / mt /
  pl / pt / ro / sk / sl / sv). Trained on a smaller, broader corpus;
  use this when you need East/Southeast Asian or European coverage
  beyond zh/en/ja.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix and per-language WER/CER on the language each variant
targets.

| Variant | Params | Q8_0 size | WER (Q8_0) | Languages | Doc |
| --- | ---: | ---: | ---: | --- | --- |
| `fun-asr-nano-2512`     | ~800M | 850 MB | 1.79% | zh, en, ja + 7 dialects | [fun-asr-nano-2512.md](fun-asr-nano-2512.md) |
| `fun-asr-mlt-nano-2512` | ~800M | 850 MB | 1.74% | 31 languages            | [fun-asr-mlt-nano-2512.md](fun-asr-mlt-nano-2512.md) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

Both variants are bounded by the 40,960-token decoder context, but they admit
different amounts of audio because of their frame rates: **Fun-ASR-Nano** accepts
up to about **5.4 hours** per call, **Fun-ASR-MLT-Nano** up to about **41
minutes**. These ceilings bound memory and sit far beyond normal clips; audio
past them is rejected up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than
silently truncated. `transcribe_session_get_limits()` reports the exact
per-session value (and it drops if you lower `--n-ctx`). See the
[input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/fun-asr-nano-2512/fun-asr-nano-2512-Q8_0.gguf \
  samples/jfk.wav
```

Pass `--itn` to enable inverse text normalization (digits,
capitalization, punctuation) at the model level.

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream FunAudioLLM checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Fun-ASR-Nano variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Inverse text normalization** (digits / capitalization /
  punctuation) via `--itn` on the CLI, or
  `transcribe_funasr_nano_params { use_itn = true }` via the library
  API.
- **Hotword / context biasing** via `--context` on the CLI or
  `transcribe_run_params::context` in the API. The text fills the
  hotword slot of the trained prompt template (upstream
  `FunASRNano.get_prompt` hotwords path), so comma-separated
  keyword lists match the trained shape best — e.g.
  `--context "pull buoy, catch-up freestyle"`.

What's not supported (consistent across the family): translation,
real-time streaming, long-form chunking, timestamps, VAD, speaker
diarization. See the family doc for the full runtime contract.
