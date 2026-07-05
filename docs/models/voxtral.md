# Voxtral (2507)

Mistral's **Voxtral** offline audio-LLMs ported to transcribe.cpp. Each is
a Whisper-large-v3 bidirectional audio encoder (32 layers, `d_model=1280`)
feeding a 4-frame-group projector (375 audio tokens per 30 s chunk) into a
Mistral/Ministral causal LM via audio-token injection at `audio_token_id=24`.
Both variants share that encoder, projector, log-mel frontend, and tekken
tokenizer — they differ only in the text decoder.

Offline speech-to-text and speech-to-text **translation** from a 16 kHz mono
WAV, via greedy decoding. Auto language detection or an explicit `--language`
hint (English, French, German, Spanish, Italian, Portuguese, Dutch, Hindi).
Licensed Apache-2.0.

For Mistral's **streaming** sibling, see
[Voxtral Realtime](voxtral-realtime.md).

## Variants

| Variant | Text decoder | BF16 WER (test-clean) | Card | GGUF |
| --- | --- | ---: | --- | --- |
| `voxtral-mini-3b-2507` | Ministral-3B (30L, `d=3072`) | 1.88% | [card](voxtral-mini-3b-2507.md) | [HF](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-GGUF) |
| `voxtral-small-24b-2507` | Mistral-Small-24B (40L, `d=5120`) | 1.56% | [card](voxtral-small-24b-2507.md) | [HF](https://huggingface.co/handy-computer/Voxtral-Small-24B-2507-GGUF) |

WER on the full LibriSpeech `test-clean` split (2620 utterances), Whisper
English normalizer. Both match the HuggingFace `transformers` reference
within rounding (3B 1.87%, 24B 1.57%). See each variant's card for the full
quant matrix, per-quant WER, and quick-start commands.

## Input limits

Both variants accept up to about **2.9 hours** of 16 kHz mono audio in a single
call — the 131,072-token decoder context is the binding limit. That ceiling
bounds memory and is far longer than any normal clip; audio past it is rejected
up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` rather than silently truncated.
Lowering `--n-ctx` lowers the limit, and `transcribe_session_get_limits()`
reports the exact per-session value. See the
[input-length contract](../input-limits.md).

## Notes

- The 24B is the larger sibling — same architecture, scaled decoder. It is a
  GPU-class model (BF16/F16 need ~50 GB) and should be run at **batch size
  ≤ 8**; see its [card](voxtral-small-24b-2507.md) for details.
- **No context biasing**: Voxtral's transcription template has no slot for
  `transcribe_run_params::context` (a supplied context warns and is
  ignored). The model's free-text instruct mode exists upstream but changes
  the task rather than biasing recognition, so it is not mapped onto the
  core context field; if exposed later it would be a voxtral run extension.
- Upstream: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507),
  [`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507).
