# ARK-ASR candidate port

This candidate reuses the ARK implementation and converter published by
harshav in [ARK-ASR-3B-GGUF](https://huggingface.co/harshav/ARK-ASR-3B-GGUF).
The source archive `transcribe.cpp-arkasr-b838a2b.tar.gz` has SHA-256
`776c2a7a690dc21dc629aa79d03003a0a9f9a463f1e0a4f3a4cefc6ee59dbe97`.
The runtime additions are MIT licensed; original weights are Apache-2.0.
The original model and inference implementation are maintained by
[AutoArk](https://github.com/AutoArk/open-audio-opd).

The source was adapted to the current device-selection API and gate/up packing
status contract. Optional Q/K/V projection biases were added to the shared
causal decoder, preserving null defaults for other families. The converter
now records variant identity, supported languages and automatic language
detection, accepts a pinned source revision, and checks variant dimensions.

## Variants

`ark-asr-0.6b` uses a 896-dimension Qwen2 decoder with 24 layers, 14 attention
heads and 2 KV heads. `ark-asr-3b` uses the existing 2048-dimension decoder.
Both have a Whisper-style encoder with partial interleaved RoPE and a merge-4
MLP adapter. The 0.6B name describes the decoder, not the full checkpoint.

## Conversion

```sh
uv run scripts/convert-arkasr-to-gguf.py \
  --input Edge0/ARK-ASR-0.6B \
  --revision 45776b56d58cdfb2e2eb632f7e110f38684633e0 \
  --variant ark-asr-0.6b --outtype q8_0 \
  --output ark-asr-0.6b-Q8_0.gguf
```

The reused converter's Q8 path retains the tied embedding/output matrix and
convolutions in F16, and norms/biases/frontend tensors in F32. This is a
candidate conversion path; moving quantization to the standard C++ policy is
part of upstream acceptance work.

## Runtime contract

16 kHz mono PCM; offline automatic-language transcription; cancellation;
no timestamps, translation, streaming or recognition hints. A language option
does not steer this port. Long audio is processed in windows of at most 30
seconds, with low-energy boundaries, overlap and transcript stitching.

```sh
build/bin/transcribe-cli --backend metal -m ark-asr-0.6b-Q8_0.gguf samples/jfk.wav
build/bin/transcribe-cli --backend cpu -m ark-asr-0.6b-Q8_0.gguf samples/jfk.wav
uv run scripts/verify-arkasr-reference.py --model /path/to/pinned/checkpoint \
  --audio samples/jfk.wav --device mps
```

## Acceptance status

This is a draft candidate for
[issue 190](https://github.com/handy-computer/transcribe.cpp/issues/190).
It is not a canonical published model family. The existing default test suite,
real-model transcription comparisons and original-checkpoint oracle provide
initial evidence. The intake signoff, standard golden tensor manifest/dumper,
tolerance profile, family fixture/structural/ABI smokes, full numerical gates,
WER dataset evaluation and canonical artifact publication remain required
before upstream acceptance/publication. Transcript comparisons must not be
reported as human-ground-truth WER.
