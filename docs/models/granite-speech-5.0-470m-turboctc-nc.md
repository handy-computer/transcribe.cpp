# Granite Speech 5.0 470M TurboCTC NC

IBM's [`ibm-granite/granite-speech-5.0-470m-turboctc-nc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc)
ported to transcribe.cpp. A 470M-parameter Granite Conformer encoder with a
self-conditioned CTC head.

> **Non-commercial.** Licensed
> [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), not
> Apache-2.0. Research and non-commercial use only, and ShareAlike means these
> GGUFs carry the same terms. For commercial use take the Apache-2.0 sibling,
> [`granite-speech-5.0-470m-turboctc`](granite-speech-5.0-470m-turboctc.md).

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript. Not a streaming model. English only, and it does not translate.

Same architecture as the Apache-2.0 sibling, trained on more data (~75,000 h vs
~60,000 h). IBM reports 4.85% aggregate WER across the 8 Open ASR leaderboard
test sets for this model, against 5.00% for the sibling.

Licensed CC-BY-NC-SA-4.0. Ported from upstream commit
[`0eb7b4f`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc/commit/0eb7b4fe726a294815dc45d342860465b5af68ef),
pinned 2026-09-12.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [granite-speech-5.0-470m-turboctc-nc-BF16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-BF16.gguf)     | 948 MB | 1.29% |
| F16    | [granite-speech-5.0-470m-turboctc-nc-F16.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-F16.gguf)       | 949 MB | 1.29% |
| Q8_0   | [granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf)     | 506 MB | 1.30% |
| Q6_K   | [granite-speech-5.0-470m-turboctc-nc-Q6_K.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q6_K.gguf)     | 392 MB | 1.28% |
| Q5_K_M | [granite-speech-5.0-470m-turboctc-nc-Q5_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q5_K_M.gguf) | 336 MB | 1.29% |
| Q4_K_M | [granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf/resolve/main/granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf) | 279 MB | 1.33% |

Measured on the full LibriSpeech test-clean split (2620 utterances), greedy CTC
decoding, no external LM. Reference baseline (transformers 5.17.0, F32, CPU):
**1.29%**, 95% CI [1.15%, 1.42%]. Every tier falls inside that CI, so the
ordering between them is not meaningful. Q4_K_M is the weakest at 1.33%; if you
want a small file without that, use Q5_K_M. Clean read speech only, not checked
on noisy or accented audio.

## Quick Start

```bash
build/bin/transcribe-cli \
  -m models/granite-speech-5.0-470m-turboctc-nc/granite-speech-5.0-470m-turboctc-nc-Q8_0.gguf \
  samples/jfk.wav
```

## Performance

Wall-clock latency (mean over 3 iterations after 1 warmup), with speedup over
realtime in parentheses.

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  | 37.0 ms (297×) | 38.1 ms (289×) |
| Metal   | dots (35.3s) | 85.3 ms (414×) | 87.2 ms (405×) |
| CPU     | jfk (11.0s)  |  233 ms (47×)  |  233 ms (47×)  |
| CPU     | dots (35.3s) |  696 ms (51×)  |  687 ms (52×)  |

macOS 26.6.2, transcribe.cpp `144ccad`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  631 ms (17×) |  647 ms (17×) |
| Vulkan  | dots (35.3s) | 1.49 s (24×)  | 1.52 s (23×)  |
| CPU     | jfk (11.0s)  |  702 ms (16×) |  654 ms (17×) |
| CPU     | dots (35.3s) | 2.27 s (16×)  | 2.26 s (16×)  |

Fedora 43, transcribe.cpp `3a5ed01`. Vulkan device: `AMD Radeon Graphics (RADV RENOIR)`.

### Apple M4

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  |  100 ms (110×) |  103 ms (107×) |
| Metal   | dots (35.3s) |  271 ms (131×) |  278 ms (127×) |
| CPU     | jfk (11.0s)  |  383 ms (29×)  |  414 ms (27×)  |
| CPU     | dots (35.3s) |  1.17 s (30×)  |  1.26 s (28×)  |

macOS 26.5.1, transcribe.cpp `54b241e`.

Q8_0 is usually a little faster than Q4_K_M despite being 1.8× the size, so
pick Q4_K_M for footprint rather than speed. Cost is linear in audio length, so
realtime factor holds up on long files. Batching is supported and gives
identical output, but buys almost nothing here.

## Details

Conversion, validation and benchmark procedure for this family live in
[docs/porting/families/granite5_ctc.md](../porting/families/granite5_ctc.md).
