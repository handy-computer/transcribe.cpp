# MOSS-Transcribe-Diarize

OpenMOSS's [`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
ported to transcribe.cpp. A 0.9B audio-LLM: a 24-layer Whisper-Medium audio
encoder (`d_model=1024`, GELU, LayerNorm) feeds a 4x temporal merge
(1024 -> 4096) and a VQAdaptor MLP bridge into a Qwen3-0.6B causal decoder
(28 layers, `hidden_size=1024`, GQA 16/8 heads, `rope_theta=1e6`) via
audio-token injection at `<|audio_pad|>` positions.

## What it's for

Offline English and Chinese speech-to-text with optional speaker attribution.
The model generates the canonical diarized format `[start][Sxx]text[end]`
(e.g. `[0.48][S01]Welcome[1.66]`) via greedy decoding. The runtime parses those
emergent text markers into clean `full_text`, segment rows, and—when
`diarize=ON`—speaker IDs and speaker-turn rows. Built for long-form,
multi-speaker audio. No translation; not a streaming model.

See OpenMOSS's [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
for training data, intended use, and upstream evaluation. All of OpenMOSS's
published metrics are Chinese multi-speaker diarization CER/cpCER; LibriSpeech
test-clean is used here only as an English acceptance set.

Licensed Apache-2.0. Ported from upstream commit
[`d7231bb`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/commit/d7231bbae2587a4af278735eb765b318c4f64edd),
pinned 2026-07-12.

## Memory and length

MOSS keeps the whole recording in memory while it transcribes, so RAM use grows
with how long the audio is. Plan for roughly **85 MB of extra memory per minute
of audio**, on top of the model file itself — about **2.5 GB for a 30-minute
clip** and **~5 GB for an hour**. It can handle recordings up to a couple of
hours if you have the memory; anything longer is rejected with a clear error
instead of being silently cut off. If you run low on memory, split long audio
into shorter pieces.

## Download

| Quantization | Download | Size | WER (LibriSpeech test-clean) |
| --- | --- | ---: | ---: |
| BF16   | [MOSS-Transcribe-Diarize-BF16.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-BF16.gguf)     | 1.83 GB | 2.08% |
| F16    | [MOSS-Transcribe-Diarize-F16.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-F16.gguf)       | 1.83 GB | 2.07% |
| Q8_0   | [MOSS-Transcribe-Diarize-Q8_0.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q8_0.gguf)     | 987 MB  | 1.93% |
| Q6_K   | [MOSS-Transcribe-Diarize-Q6_K.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q6_K.gguf)     | 768 MB  | 1.96% |
| Q5_K_M | [MOSS-Transcribe-Diarize-Q5_K_M.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q5_K_M.gguf) | 700 MB  | 1.99% |
| Q4_K_M | [MOSS-Transcribe-Diarize-Q4_K_M.gguf](https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf/resolve/main/MOSS-Transcribe-Diarize-Q4_K_M.gguf) | 617 MB  | 2.59% |

These WER values describe this dataset only, not a general quality ranking. A
quant that scores slightly better here is not necessarily better in real-world
use; dataset-specific decoding near-ties can make quantization noise help or
hurt individual utterances.

WER measured on the full LibriSpeech `test-clean` split (2620 utterances) with
the Whisper-style English normalizer and jiwer 3.x. MOSS emits the diarized
format `[start][Sxx]text[end]`; the bracket spans are metadata and are
de-diarized to a space (for both hypothesis and reference) before scoring,
matching the author-repo reference runner. The same-manifest MOSS author-repo
reference (bf16, greedy) lands at **2.07%**, 95% bootstrap CI [1.82%, 2.40%];
the BF16 port lands at 2.08%, within `+0.01pp` of the reference and well inside
the CI. Q4_K_M's higher 2.59% is not broad degradation but a handful of 4-bit
tail failures (6 empty outputs, 5 English->Chinese language-drift utterances,
1 timestamp-token repetition loop); prefer Q5_K_M or higher if those matter.
The runtime applies the same marker removal to `full_text`, so WER scoring and
the public transcript agree. The pre-parsed inline marker string is not exposed
by the current result API.

## Quick Start

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-Q8_0.gguf \
  --diarize \
  samples/jfk.wav
```

If your audio is not already 16 kHz mono WAV, convert it first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

CLI flags:

- `-l en` / `-l zh` (or omit for `auto`): English and Chinese are supported.
- No `--task` / `--target-language`: the model is ASR-only, no translation.
- Speaker attribution is off by default. Pass `--diarize` to populate
  `speaker_id` and speaker-turn rows; `--no-diarize` is the explicit off form.
- Timestamp selection is independent: `--timestamps segment` or `auto` keeps
  parsed turn timing; `--timestamps none` returns attribution with zero times.
- `full_text` is always clean marker-free text after a successful parse.

## Performance

Cells are wall-clock latency (mean over 3 iterations after 1 warmup),
with speedup over realtime in parentheses. Units: `ms` below 1 s, `s`
above (2 decimal places).

### Apple M4 Max

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Metal   | jfk (11.0s)  | 388 ms (28.3×) | 369 ms (29.8×) |
| Metal   | dots (35.3s) | 1.27 s (27.8×) | 1.17 s (30.1×) |
| CPU     | jfk (11.0s)  | 2.06 s (5.3×)  | 2.37 s (4.6×)  |
| CPU     | dots (35.3s) | 5.71 s (6.2×)  | 5.84 s (6.0×)  |

macOS 26.5.1, transcribe.cpp `e745720`.

### AMD Ryzen 7 PRO 4750U

| Backend | Sample       |          Q8_0 |        Q4_K_M |
| ------- | ------------ | ------------: | ------------: |
| Vulkan  | jfk (11.0s)  |  3.88 s (2.8×) |  3.68 s (3.0×) |
| Vulkan  | dots (35.3s) | 11.38 s (3.1×) | 10.68 s (3.3×) |
| CPU     | jfk (11.0s)  |  7.49 s (1.5×) |  7.06 s (1.6×) |
| CPU     | dots (35.3s) | 21.20 s (1.7×) | 19.22 s (1.8×) |

Fedora Linux 43, transcribe.cpp `e745720`. Vulkan device: `AMD Radeon
Graphics (RADV RENOIR)`.

Benchmark reproduction:

```bash
uv run scripts/bench/run.py \
  --models moss-transcribe-diarize \
  --quants q8_0,q4_k_m \
  --samples jfk,dots \
  --backends metal,cpu,vulkan \
  --iters 3 --warmup 1 \
  --name moss-transcribe-diarize-publication
```

## Numerical Validation

transcribe.cpp is validated tensor-by-tensor against the MOSS author repo
(`scripts/dump_reference_moss_author.py`, `trust_remote_code`) on
`samples/jfk.wav` with the strict CPU backend. The reference runs BF16 (torch,
eager attention); the C++ path dequantizes BF16 weights to F32 and computes in
F32, so C++ is the *more* precise side and the residual gap is a constant
~1-3% relative bf16-vs-f32 drift, not a bug. The transcript compare is
`dediarized` (bracket metadata stripped to a space). Confirmed WER-neutral: on
the first 100 test-clean utterances the C++ ref-dtype WER (1.40%) is
bit-identical to the Oracle reference on the same subset (1.40%). Tolerances
are pinned in `tests/tolerances/moss.json` with a `_comment` block naming the
precision regime, the large-pre-normalization-activation maxes, and the encoder
padding-trim contract. Last validated at commit
[`3f5e15c`](https://github.com/handy-computer/transcribe.cpp/tree/3f5e15c).

| Field | Value |
| --- | --- |
| Reference | MOSS author repo (`OpenMOSS-Team/MOSS-Transcribe-Diarize`) |
| Dump script | `scripts/dump_reference_moss_author.py` |
| Manifest | `tests/golden/moss/moss-transcribe-diarize.manifest.json` |
| Tolerances | `tests/tolerances/moss.json` |
| Command | `uv run scripts/validate.py all --family moss --variant moss-transcribe-diarize` |

Selected tensors (observed on CPU, strict backend; see the tolerance file for
budgets and per-tensor notes):

| Tensor | Max abs diff | Mean abs diff | Notes |
| --- | ---: | ---: | --- |
| `enc.mel.in`          | `9.872e-05` | `5.150e-06` | C++ MelFrontend vs reference feature extractor |
| `enc.pos_add.out`     | `5.000e-02` | `2.200e-03` | Encoder input + positional embedding |
| `enc.block.0.out`     | `1.200e-01` | `4.000e-03` | First Whisper encoder block |
| `enc.block.23.out`    | `3.100e+03` | `8.000e-02` | Pre-final-LN residual; ref \|max\| ~3.6e3, bf16 rel error dominates max_abs (renormalized by `enc.ln_post`) |
| `enc.ln_post.out`     | `1.300e+01` | `3.000e-03` | Encoder output LayerNorm |
| `enc.merge.out`       | `1.500e+00` | `3.000e-03` | 4x temporal merge |
| `enc.adaptor.out`     | `4.000e-01` | `1.200e-02` | VQAdaptor decoder handoff (rel_mean ~0.96%) |
| `dec.audio_injected`  | `4.000e-01` | `7.000e-03` | Audio tokens scattered into the prompt |
| `dec.block.0.out`     | `7.500e-01` | `9.500e-03` | First Qwen3 decoder block |
| `dec.block.27.out`    | `3.200e+02` | `2.600e-01` | Pre-final-RMSNorm residual (bf16 accumulation over 28 layers) |
| `dec.out_before_head` | `8.500e+00` | `8.000e-02` | Pre-head hidden state |
| `dec.logits_raw`      | `4.800e-01` | `7.200e-02` | Prefill logits; argmax preserved (transcript exact) |
| `dec.logits_raw.gen8` | `5.000e-01` | `6.500e-02` | Greedy step 8 logits (KV-cache decode coverage) |
| `dec.token_emb`       | `0.000e+00` | `0.000e+00` | Pure embedding lookup (pinned exact) |

For the full porting writeup, see
[`docs/porting/families/moss.md`](../porting/families/moss.md).

## Reproduction

### Convert

```bash
uv run --project scripts/envs/moss \
  scripts/convert-moss.py OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision d7231bbae2587a4af278735eb765b318c4f64edd
```

### Quantize

```bash
uv run scripts/quantize-all.py models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-BF16.gguf
```

### Validate

```bash
uv run scripts/validate.py all --family moss --variant moss-transcribe-diarize
```

### Score WER

```bash
PRESET=BF16
uv run scripts/wer/run.py \
  --model models/MOSS-Transcribe-Diarize/MOSS-Transcribe-Diarize-${PRESET}.gguf \
  --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
  --out reports/wer/MOSS-Transcribe-Diarize-${PRESET}.librispeech-test-clean.jsonl
uv run scripts/wer/score.py \
  reports/wer/MOSS-Transcribe-Diarize-${PRESET}.librispeech-test-clean.jsonl \
  --dediarize
```

### Score WER against the MOSS author-repo reference

```bash
uv run --project scripts/envs/moss \
  scripts/wer/run_reference_moss_author.py \
    --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --revision d7231bbae2587a4af278735eb765b318c4f64edd \
    --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
    --out reports/wer/moss-transcribe-diarize-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py \
  reports/wer/moss-transcribe-diarize-REF.librispeech-test-clean.jsonl \
  --dediarize
```
