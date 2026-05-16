# Canary-Qwen 2.5B

NVIDIA's [`nvidia/canary-qwen-2.5b`](https://huggingface.co/nvidia/canary-qwen-2.5b)
ported to transcribe.cpp. A NeMo SALM (Speech-Augmented Language Model):
a 32-layer FastConformer audio encoder (`d_model=1024`, 16 heads) feeds
audio embeddings into a Qwen3-1.7B causal LM (28 layers,
`hidden_size=2048`, `intermediate_size=6144`) via audio-token injection
at a sentinel position in the prompt.

## What it's for

Offline English speech-to-text. Takes a 16 kHz mono WAV and produces a
transcript via greedy decoding. English only; no translation, no
explicit PnC toggle (SALM applies punctuation and capitalization
implicitly when the audio supports it).

See NVIDIA's [model card](https://huggingface.co/nvidia/canary-qwen-2.5b)
for training data, intended use, and upstream evaluation.

Licensed CC-BY-4.0. Ported from upstream commit
[`b1469e1`](https://huggingface.co/nvidia/canary-qwen-2.5b/commit/b1469e1bba1cfe140205529c79c434ca47180960),
pinned 2026-05-15.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [canary-qwen-2.5b-BF16.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-BF16.gguf)     | 4.73 GB | 1.63% |
| F16    | [canary-qwen-2.5b-F16.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-F16.gguf)       | 4.73 GB | 1.63% |
| Q8_0   | [canary-qwen-2.5b-Q8_0.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q8_0.gguf)     | 2.61 GB | 1.63% |
| Q6_K   | [canary-qwen-2.5b-Q6_K.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q6_K.gguf)     | 2.06 GB | 1.63% |
| Q5_K_M | [canary-qwen-2.5b-Q5_K_M.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q5_K_M.gguf) | 1.85 GB | 1.63% |
| Q4_K_M | [canary-qwen-2.5b-Q4_K_M.gguf](https://huggingface.co/handy-computer/canary-qwen-2.5b-gguf/resolve/main/canary-qwen-2.5b-Q4_K_M.gguf) | 1.62 GB | 1.63% |

WER measured on the full LibriSpeech `test-clean` split (2620 utterances)
with the Whisper-style English text normalizer and jiwer 3.x. The
same-machine NeMo SALM reference run (CPU torch, dither=0.0, greedy
`model.generate`) lands at **1.61%** with 95% bootstrap CI [1.47%,
1.75%]; NVIDIA's published number is 1.60% (within the same CI). All six
GGUF presets land at exactly 1.63% (`+0.02pp` over our same-machine REF
run). The remaining `+0.02pp` C++ vs REF gap is BF16 weight-precision
cascade noise: of 2620 utterances, only 21 (0.8%) differ post-normalizer,
all classic small-margin token flips (homophones, word-boundary flips,
function-word substitutions).

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/canary-qwen-2.5b/canary-qwen-2.5b-Q8_0.gguf \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags:

- `-l en` (or omit): English is the only supported language; passing
  any other code returns `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`.
- No `--task`, `--target-language`, or `--pnc` toggles. SALM is
  ASR-only, English-only, and applies PnC implicitly.

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup), with
speedup over realtime in parentheses. Units: `ms` below 1 s, `s` above (2
decimal places).

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |             Q8_0 |           Q4_K_M |
| ------- | ------------ | ---------------: | ---------------: |
| Vulkan  | jfk (11.0s)  |   2.41 s (4.6×)  |   2.11 s (5.2×)  |
| Vulkan  | dots (35.3s) |   9.72 s (3.6×)  |   8.48 s (4.2×)  |
| CPU     | jfk (11.0s)  |   4.73 s (2.3×)  |   3.43 s (3.2×)  |
| CPU     | dots (35.3s) |  18.42 s (1.9×)  |  13.51 s (2.6×)  |

Fedora Linux 43, transcribe.cpp `51db32d`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models canary-qwen-2.5b \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name canary-qwen-2.5b-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against NeMo SALM
(`nemo.collections.speechlm2.SALM` 2.7.3) on `samples/jfk.wav` with the
strict CPU backend, BF16 weights promoted to F32 at load time. All 16
checkpointed tensors fall within family tolerance, and the BF16
transcript matches the reference verbatim
(`And so my fellow Americans ask not what your country can do for you ask what you can do for your country`).
Tolerances are pinned in `tests/tolerances/canary_qwen.json` with a
detailed `_comment` block naming the precision regime, the two
implementation gotchas (BF16 mel filterbank in NeMo's preprocessor,
forced F32 promotion of F16 depthwise conv kernels on CPU), and the
mechanism behind every widened entry. Last validated at commit
[`6f6c699`](https://github.com/handy-computer/transcribe.cpp/tree/6f6c699).

| Field | Value |
| --- | --- |
| Reference | NeMo SALM 2.7.3 (`nvidia/canary-qwen-2.5b`) |
| Dump script | `scripts/dump_reference_canary_qwen_nemo.py` |
| Manifest | `tests/golden/canary_qwen/canary-qwen-2.5b.manifest.json` |
| Tolerances | `tests/tolerances/canary_qwen.json` |
| Command | `uv run scripts/validate.py all --family canary_qwen --variant canary-qwen-2.5b` |

Selected tensors (observed on CPU, strict backend; see tolerance file
for budgets):

| Tensor | Shape | Max abs diff | Mean abs diff | Notes |
| --- | --- | ---: | ---: | --- |
| `enc.mel.in`         | `[128,1101]`  | `6.724e-01` | `5.242e-05` | NeMo preprocessor's BF16 fb/window matrices vs C++ F32 STFT |
| `enc.pre_encode.out` | `[138,1024]`  | `2.610e+02` | `5.795e-01` | Output of the conv subsampler; large extreme-bin spikes from the mel difference are absorbed here, then drained by the next LayerNorm |
| `enc.block.0.out`    | `[138,1024]`  | `1.653e+01` | `2.448e-02` | First FastConformer block |
| `enc.block.16.out`   | `[138,1024]`  | `2.528e+01` | `7.975e-02` | Mid-encoder |
| `enc.block.31.out`   | `[138,1024]`  | `7.921e-01` | `1.875e-02` | Final FastConformer block |
| `enc.final`          | `[1024,138]`  | `7.921e-01` | `1.875e-02` | Encoder output (transposed) |
| `perception.proj.out`| `[138,2048]`  | `3.520e+00` | `4.314e-02` | Audio→LM width projection |
| `dec.token_emb`      | `[15,2048]`   | `0.000e+00` | `0.000e+00` | Pure embedding lookup |
| `dec.audio_injected` | `[152,2048]`  | `3.520e+00` | `3.917e-02` | Audio-tokens scattered into the prompt sequence |
| `dec.block.0.out`    | `[152,2048]`  | `3.355e+00` | `4.365e-02` | First Qwen3 LM block |
| `dec.block.14.out`   | `[152,2048]`  | `1.506e+01` | `1.190e-01` | Mid-LM |
| `dec.block.27.out`   | `[152,2048]`  | `1.112e+02` | `9.333e-01` | Final LM block (accumulated) |
| `dec.out_before_head`| `[152,2048]`  | `1.566e+01` | `4.152e-02` | Pre-head hidden state |
| `dec.logits_raw.gen0`| `[151936]`    | `5.226e-01` | `6.571e-02` | Greedy step 0 logits |
| `dec.logits_raw.gen8`| `[151936]`    | `1.514e+00` | `2.116e-01` | Greedy step 8 logits (mid-generation, exercises KV cache write/read) |

For the full porting writeup including the SALM trace, the
audio-injection scatter contract, and the BF16-vs-F32 weight precision
investigation, see
[`docs/porting/families/canary_qwen.md`](../porting/families/canary_qwen.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/canary_qwen \
  scripts/convert-canary-qwen.py nvidia/canary-qwen-2.5b \
  --revision b1469e1bba1cfe140205529c79c434ca47180960
```

### Quantize

```bash
uv run scripts/quantize-all.py models/canary-qwen-2.5b/canary-qwen-2.5b-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family canary_qwen --variant canary-qwen-2.5b
```

### Score WER

```bash
PRESET=BF16
uv run scripts/wer/run.py \
  --model models/canary-qwen-2.5b/canary-qwen-2.5b-${PRESET}.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/canary-qwen-2.5b-${PRESET}.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/canary-qwen-2.5b-${PRESET}.librispeech-test-clean.jsonl
```

### Score WER against the NeMo SALM reference

```bash
uv run --project scripts/envs/canary_qwen \
  scripts/wer/run_reference_canary_qwen_nemo.py \
    --model nvidia/canary-qwen-2.5b \
    --manifest samples/wer/test-clean.manifest.jsonl \
    --out reports/wer/canary-qwen-2.5b-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/canary-qwen-2.5b-REF.librispeech-test-clean.jsonl
```
