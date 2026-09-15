# Qwen3-ASR

Alibaba's [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr)
family ported to transcribe.cpp. An audio-LLM design: a bidirectional
audio encoder feeds audio tokens into a Qwen3 causal LM that emits the
transcript autoregressively. Both shipped variants share the contract
— 16 kHz mono PCM in, transcript text out — and auto-detect across
30 languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/qwen3_asr.md`](../porting/families/qwen3_asr.md).

## Choosing a variant

- **Smaller, faster, near-realtime CPU.** `qwen3-asr-0.6b` pairs an
  18-layer encoder with a Qwen3 LM at `hidden_size=1024`.
- **Accuracy headroom.** `qwen3-asr-1.7b` widens both halves of the
  model (24-layer encoder, LM `hidden_size=2048`,
  `intermediate_size=6144`) for ~0.5pp WER improvement at ~2.5× the
  storage and decode cost.
- Language coverage is identical across the two variants — pick on
  the accuracy/cost axis, not on languages.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix.

<!-- catalog:family variants=qwen3-asr-0.6b,qwen3-asr-1.7b -->
| Variant          | Params | Languages                  | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `qwen3-asr-0.6b` |   782M | 30 languages + auto-detect |    850 MB | LibriSpeech test-clean (WER) | 2.11% | -            | [qwen3-asr-0.6b.md](qwen3-asr-0.6b.md) |
| `qwen3-asr-1.7b` |     2B | 30 languages + auto-detect |   2.19 GB | LibriSpeech test-clean (WER) | 1.62% | -            | [qwen3-asr-1.7b.md](qwen3-asr-1.7b.md) |
<!-- /catalog -->

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

Both variants accept up to about **87 minutes** of 16 kHz mono audio in a single
call — the binding limit is the 65,536-token decoder context, shared across the
family. That ceiling is there to bound memory and sits far beyond any normal
clip; audio past it is rejected up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG`
rather than silently truncated. Lowering `--n-ctx` lowers the limit (and the
KV-cache footprint), and `transcribe_session_get_limits()` reports the exact
per-session value. See the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/qwen3-asr-0.6b/qwen3-asr-0.6b-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream Qwen checkpoint via the per-variant doc's reproduction
section.

## Capabilities

All Qwen3-ASR variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Auto language detection** across 30 languages.

What's not supported (consistent across the family): translation,
real-time streaming, VAD, speaker diarization, timestamps. See the
family doc for the full runtime contract.
