# Granite Speech 5.0 TurboCTC

IBM's Granite Speech 5.0 TurboCTC ported to transcribe.cpp. A 470M-parameter
Granite Conformer encoder with a self-conditioned CTC head.

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only, and it does not translate.

Unlike Granite Speech 4 / 4.1, there is no LLM head here — just an encoder and
a CTC layer, so decoding is a single pass and there is no timestamp surface.

For the architecture, validation contract and porting notes see
[`docs/porting/families/granite5_ctc.md`](../porting/families/granite5_ctc.md).

## Choosing a variant

Two variants, same architecture and the same speed. Pick on licence:

- **`granite-speech-5.0-470m-turboctc`** — Apache-2.0. Use this for anything
  commercial. 1.33% WER on LibriSpeech test-clean.
- **`granite-speech-5.0-470m-turboctc-nc`** — CC-BY-NC-SA-4.0, research and
  non-commercial only. Trained on more data (~75,000 h vs ~60,000 h) and
  slightly more accurate: 1.29% WER. ShareAlike means anything you derive from
  it carries the same terms.

The accuracy gap is inside the measurement's confidence interval, so take the
Apache one unless you specifically want the NC weights.

## All variants

| Variant | Licence | WER (test-clean) | Card |
| --- | --- | ---: | --- |
| `granite-speech-5.0-470m-turboctc` | Apache-2.0 | 1.33% | [card](granite-speech-5.0-470m-turboctc.md) |
| `granite-speech-5.0-470m-turboctc-nc` | CC-BY-NC-SA-4.0 | 1.29% | [card](granite-speech-5.0-470m-turboctc-nc.md) |

Both ship BF16, F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M, from 948 MB down to 279 MB.
Download links are on the per-variant cards.

## Quick start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc-Q8_0.gguf \
  samples/jfk.wav
```

## Capabilities

Transcription only. No streaming, no translation, no language detection, and no
timestamps — CTC emission peaks are not word boundaries, so timestamp requests
are rejected rather than answered with a guess. `--language` is accepted but
reaches nothing.
