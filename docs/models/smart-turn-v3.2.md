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

<!-- catalog:pin -->
Licensed BSD-2-Clause. Ported from upstream commit [`f766f81`](https://huggingface.co/pipecat-ai/smart-turn-v3/commit/f766f81), pinned 2026-11-21. Validated against the onnxruntime 1.23.2 on smart-turn-v3.2-gpu.onnx (CPU, F32) reference at transcribe.cpp commit [`3465deb3`](https://github.com/handy-computer/transcribe.cpp/tree/3465deb3) on 2026-11-25.
<!-- /catalog -->

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

<!-- catalog:downloads -->
| Quantization | Download |  Size | Accuracy (Pipecat Smart Turn v3.2 test) |
| --- | --- | ---: | ---: |
| F32          | [smart-turn-v3.2-F32.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-F32.gguf) | 32 MB | 93.71% |
| F16          | [smart-turn-v3.2-F16.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-F16.gguf) | 17 MB | 93.70% |
| Q8_0         | [smart-turn-v3.2-Q8_0.gguf](https://huggingface.co/handy-computer/smart-turn-v3.2-gguf/resolve/main/smart-turn-v3.2-Q8_0.gguf) | 11 MB | 93.68% |
<!-- /catalog -->

To convert it yourself:

```bash
uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py
```

## Accuracy

<!-- catalog:recipe -->
Accuracy on the full Pipecat Smart Turn v3.2 test split (31,527 clips), a clip called complete when p > 0.5, scored on cpu. Measured at transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

<!-- catalog:prose field=wer.notes -->
A clip's score is `transcribe_turn_run`'s `p` on the whole clip (the
model sees its last 8 s); the clip is called complete when `p > 0.5`,
upstream's benchmark rule. False positives (incomplete clips called
complete) and false negatives (the reverse) are shares of all clips, as
Pipecat publishes them. The reference is upstream's fp32 ONNX
(`smart-turn-v3.2-gpu.onnx`) under onnxruntime on the same clips.
Pipecat publishes 93.71% for it and 92.63% for its int8 ONNX.
<!-- /catalog -->

<!-- catalog:turn reference="the fp32 ONNX" -->
| GGUF | Accuracy (95% CI) | False positives | False negatives |     AUC | Same decision as the fp32 ONNX | Max abs p difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32  | 93.71% (93.45-93.97) | 3.51% | 2.78% | 0.98252 | 31,527 / 31,527 | 2.4e-05 |
| F16  | 93.70% (93.44-93.97) | 3.51% | 2.78% | 0.98252 | 31,525 / 31,527 | 0.0038 |
| Q8_0 | 93.68% (93.43-93.96) | 3.53% | 2.78% | 0.98249 | 31,478 / 31,527 | 0.13 |

Measured at transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

Upstream's ONNX files on the same clips (onnxruntime, CPU):

| Model | Accuracy | False positives | False negatives |
| --- | ---: | ---: | ---: |
| upstream fp32 ONNX | 93.71% | 3.51% | 2.78% |
| upstream int8 ONNX | 92.62% | 4.76% | 2.63% |

Against the fp32 ONNX, the F32 port's mean probability difference is
2.9e-7. Against F32, F16 makes the same
decision as F32 on 99.99% of clips (2 differ, both with F32 p within 3.4e-4
of the cut-off; mean |dp| 4.6e-5, max 0.0038). Q8_0 makes the same
decision as F32 on 99.84% of clips (49 differ, all with F32 p within 0.07 of
the cut-off; mean |dp| 0.0019, 99th percentile 0.034, max 0.13) and loses
0.03 points of accuracy; upstream's int8 ONNX loses 1.09 points against its
fp32 ONNX and agrees with it on 96.12% (mean |dp| 0.042, max 0.95). Rows:
`reports/turn/smart-turn-v3.2-{F32,F16,Q8_0}.smart-turn-data-v3.2-test.jsonl`
from `scripts/turn/eval_cpp.py`, scored by `scripts/turn/score.py`.

F16 and Q8_0 store the encoder's attention and FFN matrices and the head's
hidden projections (384 to 256, 256 to 64) at that type; the conv stem,
positional table, front-end buffers, norms, biases and the two one-row
projections stay F32.

## Performance

CPU on two machines: an Apple M1 (Mac mini), standing in for the Apple M4
reference machine, which was unreachable, and an AMD Ryzen 7 PRO 4750U.
`jfk-8s` is one 8 s window: `transcribe_turn_run` on the first 8 s of
`samples/jfk.wav` (shorter audio is padded to the same window, so it costs
the same). `-1t` runs on one thread; without it, the library's default
thread count (8 threads on both machines: the default is the usable CPUs, capped at 8).

### Apple M1 (proxy for the M4)

<!-- catalog:perf machine=m1 dp_ms=1 -->
Compute latency (front end + model), speedup over realtime in parentheses; profile `turn-publication-v1`: mean over 20 iterations after 3 warmup.

| Backend | Sample           |               F32 |               F16 |              Q8_0 |
| ------- | ---------------- | ----------------: | ----------------: | ----------------: |
| CPU     | jfk-8s (8.0s)    | 33.1 ms (241.74×) | 54.1 ms (147.77×) | 38.2 ms (209.57×) |
| CPU     | jfk-8s-1t (8.0s) | 106.3 ms (75.23×) | 212.5 ms (37.64×) |  90.1 ms (88.83×) |

Apple M1 (proxy for the Apple M4 reference machine, which was unreachable): transcribe.cpp `8f97ab5b` on 2026-11-24.
<!-- /catalog -->

### AMD Ryzen 7 PRO 4750U

<!-- catalog:perf machine=ryzen-4750u dp_ms=1 -->
Compute latency (front end + model), speedup over realtime in parentheses; profile `turn-publication-v1`: mean over 20 iterations after 3 warmup.

| Backend | Sample           |               F32 |               F16 |              Q8_0 |
| ------- | ---------------- | ----------------: | ----------------: | ----------------: |
| CPU     | jfk-8s (8.0s)    | 40.6 ms (196.81×) | 43.1 ms (185.80×) | 38.9 ms (205.81×) |
| CPU     | jfk-8s-1t (8.0s) | 148.2 ms (53.97×) | 167.0 ms (47.90×) | 170.7 ms (46.86×) |

AMD Ryzen 7 PRO 4750U (Radeon RADV RENOIR): transcribe.cpp `8415e8e2` on 2026-11-24.
<!-- /catalog -->

onnxruntime on the M1: 83 ms / 28 ms per window (fp32 ONNX, 1 / 4
threads), 40 ms / 15 ms (int8 ONNX).

`AUTO` selects the CPU: the model is small, runs once per pause, and the CPU
path is the validated one.

**F16 and Q8_0 are not faster than F32 everywhere.** On the M1 (ARM), F16
runs about 2x slower than F32 on one thread: ggml multiplies F16 weights by
fp32 activations without its fast tiled kernel on ARM. Q8_0 is the fastest
there. On the Ryzen (x86), F16 and Q8_0 are both slower than F32 on one
thread and about as fast with the default threads. On ARM prefer F32 or
Q8_0; on x86, F32 is as fast as either.
