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

<!-- catalog:family variants=voxtral-mini-3b-2507,voxtral-small-24b-2507 -->
| Variant                  | Params | Languages                 | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `voxtral-mini-3b-2507`   |   4.7B | 8 languages + auto-detect |   5.00 GB | LibriSpeech test-clean (WER) | 1.87% | translate    | [voxtral-mini-3b-2507.md](voxtral-mini-3b-2507.md) |
| `voxtral-small-24b-2507` |  24.3B | 8 languages + auto-detect |  25.81 GB | LibriSpeech test-clean (WER) | 1.56% | translate    | [voxtral-small-24b-2507.md](voxtral-small-24b-2507.md) |
<!-- /catalog -->

WER on the full LibriSpeech `test-clean` split (2620 utterances), Whisper
English normalizer. Both match the HuggingFace `transformers` reference
within rounding. See each variant's card for the full
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
- Upstream: [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507),
  [`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507).
