# SAVA

<!-- catalog:intro -->
Upstream: [`inworld/sava`](https://github.com/inworld-ai/sava) at [`1beae7c`](https://github.com/inworld-ai/sava/commit/1beae7c).

Streaming end-of-turn and voice activity detection by Inworld: every 40 ms
it gives the probability that the speaker's turn has ended and, 120 ms
later, the probability that the frame held speech. Not a transcription
model: it serves the TURN role (live stream or whole clip, with Inworld's
turn rule cutting a clip into turns) and the VAD role. Any language. CPU is
the default backend.
<!-- /catalog -->

API and examples: [turn detection](../turn.md), [voice activity detection](../vad.md).
Architecture and validation: [family note](../porting/families/sava.md).

Source weights: `sava.safetensors` (bf16) from
[`inworld/sava`](https://huggingface.co/inworld/sava) at
`1beae7c1dd0df6b0f92030644fc905eb087dc6a0` (SHA256
`aa610ee9b5d3d1be96f70581a8a3affe7cf20d06816b4ada1efdd8ab6274be34`).
Upstream code: [`inworld-ai/sava`](https://github.com/inworld-ai/sava)
(`inworld-sava` 1.0.0). Apache-2.0.

## How to use it

**Live audio.** Feed 16 kHz audio to one TURN stream in chunks of any size.
Each call returns:

- the turn score of every 640-sample frame it completed;
- the speech score of every frame the model has judged. A frame is judged
  3 frames (120 ms) after its audio.

The caller decides when a turn ends. Inworld's reference rule:

1. A turn opens on the first frame with speech `>= 0.5`, 800 ms earlier.
2. While it is open, every silent frame checks the turn score of the step
   that judged it (`>= 0.9` closes the turn).
3. 1.6 s of silence closes it anyway.

`examples/turn_live` (`transcribe-turn-live --turn sava-Q8_0.gguf in.wav`)
runs that rule in 20 ms chunks.

**A whole recording.** `transcribe_turn_run` scores the clip as one stream
and returns:

- `p` and `ended`: the last frame's turn score, and `p >= 0.9`;
- every frame's turn and speech scores;
- the turns Inworld's rule finds (`sava.segment`, dropped turns left out).

```bash
build/bin/transcribe-cli -m sava-Q8_0.gguf samples/sava-speech.wav
# turn: p=0.998900 ended=1 threshold=0.90
#   segment: 0 start=85760 end=204160 t0=5.360 t1=12.760 end=model
#   segment: 1 start=197120 end=251520 t0=12.320 t1=15.720 end=silence
```

**Speech only.** The VAD role (`transcribe_vad_*`) returns the same speech
scores (lookahead 3 frames, threshold 0.5). `transcribe_vad_run` cuts them
into segments with Silero's rule on 40 ms frames.

## Downloads

<!-- catalog:downloads metric=false -->
| Quantization | Download |   Size |
| --- | --- | ---: |
| F32          | [sava-F32.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-F32.gguf) | 103 MB |
| F16          | [sava-F16.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-F16.gguf) | 52 MB |
| Q8_0         | [sava-Q8_0.gguf](https://huggingface.co/handy-computer/sava-gguf/resolve/main/sava-Q8_0.gguf) |  28 MB |
<!-- /catalog -->

The release weights are bf16, and upstream runs them in fp32:

- **F32** holds them exactly.
- **F16** holds them exactly except 1,065 of 25.5M tiny values (each off by
  at most 3e-8). It computes like F32, because ggml keeps the activations of
  an F16 matmul in fp32. Its scores match F32 to 4e-6.
- **Q8_0** is the smallest and fastest file.

To convert and quantize it yourself:

```bash
uv run --project scripts/envs/sava scripts/convert-sava.py
build/bin/transcribe-quantize models/sava/sava-F32.gguf models/sava/sava-F16.gguf --quant F16
build/bin/transcribe-quantize models/sava/sava-F32.gguf models/sava/sava-Q8_0.gguf --quant Q8_0
```

## Accuracy

Pipecat's test set `pipecat-ai/smart-turn-data-v3.2-test` (31,527 clips, 23
languages). The score of a clip is the turn score of its last frame
(`transcribe_turn_run`'s `p`). Accuracy uses Inworld's benchmark rule
`p > 0.5`; the API's default cut-off is 0.9.

| Model | AUC | Accuracy (0.5) | Accuracy (0.9) | Best accuracy (threshold) |
| --- | ---: | ---: | ---: | ---: |
| transcribe.cpp F32 (CPU) | 0.97946 | 93.36% | 88.29% | 93.51% (0.45) |
| transcribe.cpp F16 (CPU) | 0.97946 | 93.36% | 88.29% | 93.51% (0.45) |
| transcribe.cpp Q8_0 (CPU) | 0.97935 | 93.37% | 88.24% | 93.48% (0.45) |
| Inworld's published SAVA (fp32, A100) | 0.97946 | 93.35% | - | 93.51% (0.45) |
| Inworld's published smart-turn v3.2 | 0.98250 | 93.71% | - | 93.81% (0.36) |

Details:

- **F32 vs Inworld's published SAVA:** the AUC matches to 1e-7. On real
  recordings accuracy is 95.84% (Inworld: 95.86%); on synthetic ones 92.79%
  (Inworld: 92.78%).
- **F16 vs F32:** the same decision on every clip at 0.5 and at 0.9. The
  clip score differs by at most 3.9e-6.
- **Q8_0 vs F32:** the same decision on 99.76% of clips at 0.5 (75 flips)
  and 99.61% at 0.9 (124 flips). The clip score differs by 2.8e-3 on average
  (p99 3.5e-2).
- **Rows:** `reports/turn/sava-{F32,F16,Q8_0}.smart-turn-data-v3.2-test.jsonl`,
  from `scripts/turn/eval_cpp.py`. Its `p_fp32` / `p_int8` columns are Smart
  Turn's.

On the 8 golden recordings (`tests/golden/sava/`):

- **F32 vs the reference:** within 4e-5 in probability, with the same turns
  as `sava.segment`.
- **F16 vs F32:** within 1.5e-6 per frame, with the same turns on all 8.
  F16 also passes the F32 tolerance gate (`validate.py` with
  `--gguf sava-F16.gguf`).
- **Q8_0 vs F32:** per-frame scores differ by up to 0.065 (mean 1e-3 to
  4e-3 on speech clips). The turns are identical on 7 of the 8. On the 5 min
  `dots-full` 6 of 59 turns change: Q8_0 finds 57, with turns merged at
  59.4 s and 104.1 s, ends moved at 101.8 s, 145.7 s and 174.4 s, and one
  start moved by a frame. Each change is a score within 0.004 of the 0.9
  cut-off, or a speech score within 0.03 of 0.5.

## Performance

Apple M1, CPU, `jfk.wav`, idle machine, ms of compute per 40 ms frame:

| File | Whole clip, 1 thread | Whole clip, 4 threads | Stream (640-sample feeds), 1 thread | Stream, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.94 | 0.38 | 5.4 | 3.4 |
| F16 | 1.80 | 0.59 | 8.1 | 3.0 |
| Q8_0 | 0.75 | 0.30 | 3.9 | 1.7 |

F16 is slower than F32 on this CPU. ggml multiplies F16 weights by fp32
activations without its fast tiled kernel on ARM; x86 with F16C has one (not
measured). Choose a file by need:

- **Q8_0:** the fastest and smallest.
- **F16:** half the size of F32 with F32's results.
- **F32:** the reference.

A whole clip runs in blocks of up to 64 frames. A streamed frame runs a
4-frame graph: the padding makes a stream bit-identical to a whole-clip run
for any chunking, at the cost of computing 4 columns for 1. This holds for
F32, F16 and Q8_0. For comparison, upstream's PyTorch (1 thread) takes 0.37
ms per clip frame and 10.5 ms per streamed frame on the same machine.

`AUTO` selects the CPU, the validated path.

A BF16 file (`convert-sava.py --outtype bf16`) is not published:

- Its storage is lossless, but ggml rounds a BF16 matmul's activations to
  bf16. It moves 5 of the 59 `dots-full` turns and flips 1 of 3,000 Pipecat
  clips at 0.5 (3 at 0.9).
- Without native BF16 it is 15-22x slower than F32: 20.7 / 5.7 ms per clip
  frame and 82 / 22 ms per streamed frame (1 / 4 threads, M1).
