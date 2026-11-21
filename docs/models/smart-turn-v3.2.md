# Smart Turn v3.2

<!-- catalog:intro -->
Upstream: [`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3) at [`f766f81`](https://huggingface.co/pipecat-ai/smart-turn-v3/commit/f766f81).

End-of-turn detection by Pipecat (Daily): has the speaker finished their
turn, or only paused? Not a transcription model: the TURN role scores the
current turn's audio (the last 8 s) with one probability that the turn is
complete. Run it when a VAD hears a pause. 23 languages. CPU is the default
backend.
<!-- /catalog -->

API and examples: [turn detection](../turn.md). Architecture and validation:
[family note](../porting/families/smart_turn.md).

Source weights: `smart-turn-v3.2-gpu.onnx` (the fp32 export) from
[`pipecat-ai/smart-turn-v3`](https://huggingface.co/pipecat-ai/smart-turn-v3)
at `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` (SHA256
`ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be`).
Upstream code: [`pipecat-ai/smart-turn`](https://github.com/pipecat-ai/smart-turn).
BSD-2-Clause.

## How to use it

Pass the audio of the current turn, from where the speaker started, when a
VAD (for example Silero, with Pipecat's 0.2 s stop) hears a pause. The model
scores the last 8 s; shorter audio is left padded with zeros. If the speaker
continues before you act, score the whole turn again with the new audio.
`p >= threshold` (default 0.5, upstream's cut-off) sets `ended`.

```bash
build/bin/transcribe-cli -m smart-turn-v3.2-F32.gguf turn.wav
# turn: p=0.990633 ended=1 threshold=0.50
```

## Downloads

<!-- catalog:downloads metric=false -->
| Quantization | Download |  Size |
| --- | --- | ---: |
| F32          | [smart-turn-v3.2-F32.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-F32.gguf) | 32 MB |
| F16          | [smart-turn-v3.2-F16.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-F16.gguf) | 17 MB |
| Q8_0         | [smart-turn-v3.2-Q8_0.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-Q8_0.gguf) | 11 MB |
<!-- /catalog -->

To convert it yourself:

```bash
uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py
```

## Accuracy

Pipecat's test set `pipecat-ai/smart-turn-data-v3.2-test` (31,527 clips, 23
languages), upstream's decision rule `p > 0.5`:

| Model | Accuracy | False positives | False negatives |
| --- | ---: | ---: | ---: |
| transcribe.cpp F32 (CPU) | 93.71% | 3.51% | 2.78% |
| transcribe.cpp F16 (CPU) | 93.70% | 3.51% | 2.78% |
| transcribe.cpp Q8_0 (CPU) | 93.68% | 3.53% | 2.78% |
| upstream fp32 ONNX (onnxruntime) | 93.71% | 3.51% | 2.78% |
| upstream int8 ONNX (onnxruntime) | 92.62% | 4.76% | 2.63% |

The F32 port makes the same decision as the fp32 ONNX on every clip; the
largest probability difference is 2.4e-5 (mean 2.9e-7). F16 makes the same
decision as F32 on 99.99% of clips (2 differ, both with F32 p within 3.4e-4
of the cut-off; mean |dp| 4.6e-5, max 0.0038). Q8_0 makes the same
decision as F32 on 99.84% of clips (49 differ, all with F32 p within 0.07 of
the cut-off; mean |dp| 0.0019, 99th percentile 0.034, max 0.13) and loses
0.03 points of accuracy; upstream's int8 ONNX loses 1.09 points against its
fp32 ONNX and agrees with it on 96.12% (mean |dp| 0.042, max 0.95). Rows:
`reports/turn/smart-turn-v3.2-{F32,F16,Q8_0}.smart-turn-data-v3.2-test.jsonl`
(`scripts/turn/eval_cpp.py`).

F16 and Q8_0 store the encoder's attention and FFN matrices and the head's
hidden projections (384 to 256, 256 to 64) at that type; the conv stem,
positional table, front-end buffers, norms, biases and the two one-row
projections stay F32.

## Performance

One 8 s window on an Apple M1, CPU, F32: 106 ms with one thread, 32 ms with
the default threads (front end 2 ms of it). onnxruntime on the same machine:
83 ms / 28 ms (fp32 ONNX, 1 / 4 threads), 40 ms / 15 ms (int8 ONNX).

`AUTO` selects the CPU: the model is small, runs once per pause, and the CPU
path is the validated one.

**On ARM CPUs, the F16 file currently runs 1.6-1.9x slower than F32**: 224
vs 117 ms per window with 1 thread and 80 vs 46 ms with 4 threads on an M1.
ggml multiplies F16 weights by fp32 activations without its fast tiled
kernel on ARM. CPU users should prefer F32 or Q8_0.
