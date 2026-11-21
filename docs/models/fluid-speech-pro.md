# Fluid Speech Pro

<!-- catalog:intro -->
Upstream: [`FluidVoice-Labs/fluid-speech-pro`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) at [`46027a3`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/commit/46027a3).

FluidVoice's retrain of parakeet-tdt-0.6b-v3. Turns speech into text in 25
European languages and works offline. It adds one small block between the
encoder's subsampling and its first Conformer layer, so it needs a
transcribe.cpp build that knows that block. Give it a 16 kHz mono WAV and it
gives you back text with punctuation and capitals. Same languages and API as
v3. Does not stream and does not translate. The files also carry
FluidVoice's clean-mode adapter (edited, non-verbatim text); transcribe.cpp
does not apply it yet, so output is verbatim.
<!-- /catalog -->

## What it does

Speech to text in 25 European languages, offline. It is
`parakeet-tdt-0.6b-v3` retrained by FluidVoice, with one extra block: a small
residual layer (LayerNorm, projection, 15-tap depthwise convolution over
time, GELU, projection) between the encoder's subsampling and its first
Conformer layer. Same languages and API as v3; about 2 M more parameters.

It does not stream and it does not translate. FluidVoice's optional clean
mode (edited, non-verbatim text; FluidVoice: it removes repeated words and
filler words) is a rank-32 low-rank adapter. These files carry it
(`enc.blocks.N.<module>.clean_mode.{a,b}` for 11 modules per Conformer
layer, plus two bias deltas on the predictor and joint, and the
`stt.parakeet.clean_mode.{rank,alpha,modules}` keys), at F32 in the F32 file
and F16 in every other file (about 55 MB). transcribe.cpp checks the adapter
when it loads the file but does not apply it yet: output is verbatim, the
same as without it.

<!-- catalog:pin -->
Licensed CC-BY-4.0. Ported from upstream commit [`46027a3`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/commit/46027a3), pinned 2026-11-21. Validated against the NeMo (FluidVoice fluid_speech.py loader) reference at transcribe.cpp commit [`1281464`](https://github.com/handy-computer/transcribe.cpp/tree/1281464) on 2026-11-22.
<!-- /catalog -->

## Download

<!-- catalog:downloads -->
| Quantization | Download |    Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| F32          | [fluid-speech-pro-F32.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-F32.gguf) | 2.63 GB | - |
| F16          | [fluid-speech-pro-F16.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-F16.gguf) | 1.32 GB | - |
| Q8_0         | [fluid-speech-pro-Q8_0.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-Q8_0.gguf) |  797 MB | - |
| Q6_K         | [fluid-speech-pro-Q6_K.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-Q6_K.gguf) |  667 MB | - |
| Q5_K_M       | [fluid-speech-pro-Q5_K_M.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-Q5_K_M.gguf) |  606 MB | - |
| Q4_K_M       | [fluid-speech-pro-Q4_K_M.gguf](https://huggingface.co/handy-computer/fluid-speech-pro-gguf/resolve/main/fluid-speech-pro-Q4_K_M.gguf) |  542 MB | - |
<!-- /catalog -->

<!-- catalog:recipe -->
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
Greedy transducer decoding, verbatim mode. FluidVoice's leaderboard
numbers use beam search with 4 beams, and on SPGISpeech their optional
clean mode, which these files carry but transcribe.cpp does not apply yet.
<!-- /catalog -->

<!-- catalog:accuracy -->
<!-- /catalog -->

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/fluid-speech-pro/fluid-speech-pro-Q8_0.gguf \
  samples/jfk.wav
```

Audio has to be 16 kHz mono WAV. Convert anything else first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

It figures out the language on its own. Add `--language de` (or any of the
25 codes) if you want to force one.

## Performance

### Apple M4 Max

<!-- catalog:perf machine=m4-max -->
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u -->
<!-- /catalog -->

Rerun the benchmark yourself:

```bash
uv run scripts/bench/run.py --profile --models fluid-speech-pro
```

## How we checked it

Compared tensor by tensor against NeMo running FluidVoice's own `.nemo`
through their `fluid_speech.py` loader (fetched at the pinned revision, clean
mode off), using the v3 tolerances plus one entry for the extra block's
output. Everything matches and the test transcript is identical. The extra
block's own contribution matches NeMo to about 5e-5 relative; the rest of its
output error is inherited from the subsampler, as on v3. Batches of 2, 4 and
8 give the same text as one at a time.

| Field | Value |
| --- | --- |
| Reference | NeMo, `FluidVoice-Labs/fluid-speech-pro` `fluid-speech-pro.nemo` via `fluid_speech.py` |
| Dump script | `scripts/dump_reference_parakeet_nemo.py` |
| Manifest | `tests/golden/parakeet/fluid-speech-pro.manifest.json` |
| Tolerances | `tests/tolerances/parakeet.json` |
| Command | `uv run scripts/validate.py all --family parakeet --variant fluid-speech-pro` |

## Build it yourself

```bash
# get the source checkpoint
hf download FluidVoice-Labs/fluid-speech-pro fluid-speech-pro.nemo \
  --revision 46027a3bffc5817ce8f9b29add0a062610134a90 --local-dir models/fluid-speech-pro

# convert to F32 GGUF
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py models/fluid-speech-pro/fluid-speech-pro.nemo \
    models/fluid-speech-pro/fluid-speech-pro-F32.gguf --repo-id FluidVoice-Labs/fluid-speech-pro

# make the smaller files
uv run scripts/quantize-all.py models/fluid-speech-pro/fluid-speech-pro-F32.gguf

# check it against NeMo
uv run scripts/validate.py all --family parakeet --variant fluid-speech-pro

# measure WER
uv run scripts/wer/ingest.py librispeech
uv run scripts/wer/run.py \
  --model models/fluid-speech-pro/fluid-speech-pro-F32.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/fluid-speech-pro-F32.librispeech-test-clean.b1.jsonl
uv run scripts/wer/score.py reports/wer/fluid-speech-pro-F32.librispeech-test-clean.b1.jsonl
```

A third-party `TalkMate/fluid-speech-pro-gguf` Q8_0 exists. It uses the same
metadata keys and tensor names, so it loads in this build, but it was
converted from an earlier FluidVoice upload (2026-10-30); FluidVoice's
2026-11-05 update changed the predictor and joint weights, which these files
include.

## Limits

- No streaming.
- No translation.
- Clean mode is in the files but not applied yet; output is verbatim.
- Timestamps are per token, not per word.
- Long files go through in one pass. Nothing cuts them up for you.
- Needs a transcribe.cpp build with the pre-Conformer block. Older builds
  load these files but ignore the block's tensors and run plain v3 math on
  Fluid Speech Pro weights, which transcribes worse without any error.

## Licence

CC BY 4.0. Credit it as **Fluid Speech Pro by FluidVoice**. Built on
[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(CC BY 4.0, NVIDIA Corporation) and
[Moondream Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)
(CC BY 4.0, Moondream).
