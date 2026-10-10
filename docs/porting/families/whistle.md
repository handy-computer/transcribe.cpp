# Whistle

Status: validation (Stage 4: ref-dtype validate.py green, tolerances user-accepted, capability table resolved; WER +0.023 pp over the engine accepted by the user as within the engine's own noise floor; Metal fixed; signed off 2026-10-09)

## Identity

- Family key: `whistle`
- Upstream architecture string: `WhistleForSpeechRecognition` (`model_type: whistle`, `weights_version: 2.0.0`)
- Hugging Face repo: `Cactus-Compute/whistle`
- Hugging Face revision: `b358ddadd89b7a713b5aa131f23032d3cca1b251`
- License: Apache-2.0 (`LICENSE` in the HF repo)
- Variants: `whistle` (~55M params; 16.9 MB deployed at 2/4-bit CQ; English, German, French, Spanish, Italian, Dutch, Polish)

Architecture pattern: **encoder-decoder**. 16 kHz audio → 80-bin log-mel
(n_fft 512, win 400, hop 160; Slaney-normalized filterbank shipped in the
`.cact`) → 3-stage depthwise/pointwise Conv2D stem (128 ch, 8× time
downsampling, 1280 → 512 projection, 80 ms frames) → 8 encoder blocks
(d_model 512; GQA self-attention 8q/2kv heads, qk 48 / v 64, q/k
ZCRMSNorm, gate_proj, sigmoid `attn_gate`; depthwise conv module k=9 with
pw1 512→1024 / pw2; two HadamardMLPs; 4-lane mHC hyper-connections) →
8 Needle-3 decoder blocks (self-attention with causal q/k/v taps of width 3,
RoPE θ=1e5, gated cross-attention to the encoder with sigmoid `cross_gate`,
HadamardMLP, 4-lane mHC) plus Engram hashed n-gram embeddings at decoder
sites 3 and 7 → tied 8199-way head (8192 SentencePiece text pieces + 7
`<|lang|>` tokens at 8192–8198). Hard cap of 30 s per pass and 320 decoder
tokens.

The decoder is laddered: any depth ≥ 2 is deployable upstream via
`--audio-depth`. This port targets the full 8-layer depth.

## References

- Canonical reference: `author_repo_cactus_needle_engine`, the closed-source
  prebuilt Needle engine (`Cactus-Compute/needle3@2ae11323dc000f5e70c49f7403efa6af12ba9e67`
  platform binaries / `cactus-needle` 3.2.0 wheel) running
  `whistle.cact`. It is a **black-box oracle**: it exposes the transcript,
  detected language, word times (`needle_transcribe`) and the encoder
  output per 80 ms frame (`needle_embed`), and no intermediate tensors.
  - **There is no first-party source for the audio path** (mel details,
    stem, encoder block, cross-attention, `pe_gate`, decode loop). The
    user accepted the engine-only oracle on 2026-10-09 over pausing for
    Cactus to publish code or adopting a third-party port, and with it
    explicitly waived the CLAUDE.md "do not guess" rule for those
    components. Every reconstruction is confirmed against engine outputs.
- Instrumented reference: none. Stage 2 dumps are limited to what the
  engine emits (encoder embedding, transcript, language, word times).
- Partial first-party source: `refs/cactus/needle` (`git@github.com:cactus-compute/needle.git@ef3cf754`),
  `needle/model/architecture.py` (ZCRMSNorm, tapped GQA attention,
  HadamardMLP, mHC + Sinkhorn, Engram, RoPE, depth ladder) and
  `needle/model/export.py` / `quantize.py` (`.cact` layout, CQ
  dequantization, SentencePiece reference tokenizer). Authoritative for
  the shared decoder blocks and the weight format.
- Cross-check references (non-authoritative):
  - `refs/cactus/needle` branch `origin/experiments@646ebdc` `src/model.py`:
    April 2026 speech encoder-decoder prototype (stem, encoder block,
    gated cross-attention).
  - `mrfakename/whistle-ONNX@14fc92cb`: a third-party PyTorch
    reimplementation on dequantized `.cact` weights. Not adopted as the
    reference.

### Weights

The reference-dtype GGUF comes from the deployed `whistle.cact`, not from
`checkpoints/whistle.safetensors`. The safetensors file holds F32 QAT
masters (`run.stage=qapt`, 2/4-bit weights, int8 KV and activations), and
the shipped, benchmarked model is the CQ-quantized `.cact` (user decision,
2026-10-09). CQ matmul weights (145 × 2-bit, 7 × 4-bit, group 128) are
dequantized to F32. The FP16-shipped tensors are stored F32 as well. Most
of them are in the loader's Norm bucket, which requires F32, and every FP16
value is exact in F32, so no value changes. Reference dtype label: `F32`.

### Conversion (Stage 3)

- **Converter:** `scripts/convert-whistle.py`, with the archive reader and
  CQ dequantizer in `scripts/lib/cact.py`.
- **Output:** `models/whistle/whistle-F32.gguf` (679 tensors, 220.9 MB).
  Provenance is in `reports/convert/whistle-F32.json`.
- **Tensor identity.** `.cact` records have no names, so identity is
  positional.
  - The text-decoder order follows Needle's `export.py::_tensors`.
  - The audio-side order (speech config, cross-attention, `pe_gate`,
    encoder layers, encoder mHC, encoder final norm, stem, filterbank) is
    not published. It was derived by matching each record against the
    named F32 QAT masters in `checkpoints/whistle.safetensors`.
  - Every conversion re-proves the map: 524/524 FP16 records equal
    `float16(master)` exactly, and 152/152 CQ records re-quantize from
    their masters to identical codes and norms. Details are in
    `reports/convert/whistle-F32.verify.json`.
- **Names stay close to upstream.** Ambiguous blocks keep their upstream
  suffixes instead of asserting semantics:
  - `hmlp` ↔ `hadamard_mlp` and `hmlp_0` ↔ `hadamard_mlp_0`
  - `norm_hmlp` ↔ `pre_hada_norm` and `norm_hmlp_0` ↔ `pre_hada_norm_0`
  - `norm_in` ↔ `ZCRMSNorm_0`

  The encoder record order is `norm_hmlp_0`, `norm_in`, `norm_post_attn`,
  `norm_conv`, `norm_conv_out`, `norm_hmlp`, then attention, the conv
  module, `hmlp_0` and `hmlp`. That suggests a macaron block, but it is
  unverified until Stage 4 bring-up.
- **Not tensors.** The 13-value speech config vector becomes
  `stt.whistle.speech_config` (fields 0–1, `3246, 1`, are unidentified).
  The tokenizer becomes `tokenizer.ggml.*` (`model=bpe`, scores,
  byte_fallback, no space prefix).
- **Layout conventions.**
  - Matrices are `[out, in]` as stored in the archive (ggml `ne = [in, out]`).
  - `mhc.phi_*` is `[layers × lanes, 4 × d_model]`.
  - `engram.N.tables` is `[n_tables × slots, sub_dim]`.
  - Conv kernels (`conv.dw`, `stem.conv.*`) are `[kh·kw, C]`. Which axis
    is time and which is frequency is for Stage 4 to determine.
- **Quant policy.** Whistle rules are in `policy.cpp::classify_tensor` and
  `reference_dtype_for`, pinned in `test_quant_policy_sync.py`. They keep
  `.hmlp*`, `*_taps`, `*_gate`, `hada.perm*`, `.mhc.a_*` and `.mhc.b_*` in
  the Norm bucket (F32). A throwaway Q8_0 run requantized exactly the 152
  matrices upstream itself shipped as CQ.

## Commands

The engine is loaded via ctypes from `scripts/lib/whistle_engine.py`, not
through the `cactus-needle` wrapper. That avoids telemetry and keeps writes
out of `~/.cache/cactus-needle`. On first use the pinned wheel
(`Cactus-Compute/needle3@2ae11323`, `cactus_needle-3.2.0`) is extracted
to `../refs/cactus/needle3-engine/3.2.0/wheel/`. `WHISTLE_ENGINE_LIB` and
`WHISTLE_CACT` override the library and weights paths. The converter is
created at Stage 3.

Acceptance manifest (≤30 s subset). The engine rejects longer audio, and
`samples/wer/` is gitignored, so regenerate the subset after
`scripts/wer/ingest.py librispeech`:

```bash
uv run --no-project python -c "
import json, wave
with open('samples/wer/librispeech-test-clean.manifest.jsonl') as f, \
     open('samples/wer/librispeech-test-clean-le30s.manifest.jsonl', 'w') as g:
    for l in f:
        w = wave.open(json.loads(l)['audio'])
        if w.getnframes() / w.getframerate() <= 30.0: g.write(l)
"   # 2611 of 2620 rows
```

Reference run (engine oracle):

```bash
uv run --project scripts/envs/whistle \
  scripts/wer/run_reference_whistle_engine.py \
    --manifest samples/wer/librispeech-test-clean-le30s.manifest.jsonl \
    --out reports/wer/whistle-REF.librispeech-test-clean-le30s.jsonl
uv run scripts/wer/score.py reports/wer/whistle-REF.librispeech-test-clean-le30s.jsonl
```

Reference dumps (engine-observable outputs only; all manifest cases):

```bash
uv run scripts/validate.py ref --family whistle --variant whistle
# per case: uv run --project scripts/envs/whistle scripts/dump_reference_whistle_engine.py \
#   {encoder|decode} --audio samples/jfk.wav --language en \
#   --out build/validate/whistle/whistle/jfk/ref
```

Conversion:

```bash
uv run --project scripts/envs/whistle \
  scripts/convert-whistle.py --repo-id Cactus-Compute/whistle \
    --revision b358ddadd89b7a713b5aa131f23032d3cca1b251
```

Hadamard-domain 2-bit file (`Q2_K_HR`, 21.3 MB; see the note below and
`docs/tools/quantization.md`). The `HR-F32` file is an intermediate and is
not published:

```bash
uv run --project scripts/envs/whistle scripts/convert-whistle.py --hadamard-domain   # -> whistle-HR-F32.gguf
build/bin/transcribe-quantize models/whistle/whistle-HR-F32.gguf \
  models/whistle/whistle-Q2_K_HR.gguf --quant Q2_K_HR
uv run scripts/wer/run.py --model models/whistle/whistle-Q2_K_HR.gguf \
  --manifest samples/wer/librispeech-test-clean-le30s.manifest.jsonl --backend cpu --batch-size 8 \
  --out reports/wer/whistle-Q2_K_HR.librispeech-test-clean-le30s.jsonl
```

Validation (Stage 4 regime: F32 GGUF, `--backend cpu --threads 1`, C++ frontend):

```bash
uv run scripts/validate.py all --family whistle --variant whistle   # 13 tensors x 4 cases + transcripts
# real-model contract test (TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON):
TRANSCRIBE_WHISTLE_GGUF=models/whistle/whistle-F32.gguf build/bin/transcribe_whistle_real_smoke
# batch parity
uv run scripts/batch_parity.py --model models/whistle/whistle-F32.gguf --list <wav list> --language en \
  --batch-sizes 2,4,8 --backend cpu --golden-in tests/golden/batch/whistle.cpu.json
uv run scripts/batch_tensor_parity.py --model models/whistle/whistle-F32.gguf --wav samples/jfk.wav --batch 4 --backend cpu
# full ref-dtype WER
for B in 1 8; do
  uv run scripts/wer/run.py --model models/whistle/whistle-F32.gguf \
    --manifest samples/wer/librispeech-test-clean-le30s.manifest.jsonl --backend cpu --batch-size $B \
    --out reports/wer/whistle-F32.librispeech-test-clean-le30s.b$B.jsonl
  uv run scripts/wer/score.py reports/wer/whistle-F32.librispeech-test-clean-le30s.b$B.jsonl
done
```

The `ref` stage now writes two kinds of tensors: engine outputs (`enc.final`,
transcripts) and intermediates from the numpy reference model
`scripts/lib/whistle_ref.py` (`enc.mel.in`, `enc.stem.out`, `enc.blk.<i>.out`,
`dec.logits_raw`, `dec.logits_raw.gen8`).

Benchmarks:

```bash
uv run scripts/bench/run.py \
  --models whistle \
  --quants f32,q8_0,q4_k_m \
  --samples jfk \
  --backends cpu,metal \
  --iters 3 --warmup 1 \
  --name whistle-publication
```

## Capability Validation

One row per advertised capability. `Target` is the Stage 1 scope decision
(user-signed for every row that is not forced); `Status` is filled at
Stage 4.

- `MUST PASS`: in scope. Stage 4 must resolve the row to `PASS`.
- `OUT OF SCOPE — <reason>`: deferred. Stage 4 may resolve it to SKIP or
  ACCEPTED GAP. The reason names what would bring it back in scope.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript matching the engine oracle | MUST PASS | PASS — jfk byte-identical to the engine; LibriSpeech le30s 2438/2611 byte-identical |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | PASS — jfk: `detected-language: en`, same transcript |
| Language detection | auto, non-English | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/fleurs-de.wav` (also `fleurs-fr`, `fleurs-es`) | `detected-language: de` / `fr` / `es` line matching the engine's `language` | MUST PASS | PASS — fleurs-de/fr/es detect de/fr/es; transcripts byte-identical to the engine |
| No-speech handling | silence / steady noise | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf samples/noise.wav` | empty transcript and no detected language, as the engine returns | MUST PASS | PASS — noise.wav: empty text, no language (also with `-l en`); jobs-silence: "Thank you." [en] as the engine. Gate is a fitted approximation (see Notes) |
| Translate | n/a | n/a | n/a | OUT OF SCOPE — not advertised upstream (transcription only) | SKIP — not exposed by runtime (`supports_translate = false`) |
| Segment timestamps | n/a | n/a | n/a | OUT OF SCOPE — upstream emits word-level times only | SKIP — not exposed by runtime (one whole-utterance segment spanning the words) |
| Word timestamps | `--timestamps word` | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --timestamps word samples/jfk.wav` | per-word start/end within ±1 frame (80 ms) of the engine's `words` | MUST PASS | PASS — jfk 22/22 words within ±1 frame; LibriSpeech le30s (2438 text-identical utterances, 47,399 words): 99.28% of words with both ends within ±1 frame, 99.63% of boundaries, 97.6% exact |
| Keyword biasing | `--vocabulary` | `build/bin/transcribe-cli -m models/whistle/whistle-F32.gguf --vocabulary "<terms>" samples/product-names.wav` | biased terms survive as the engine's `keywords=` run does | OUT OF SCOPE — engine bias weights/algorithm undocumented and decode-time (not prompt-based); back in scope once characterized | SKIP — not exposed by runtime (`--vocabulary` not wired for this family) |
| Encoder embedding | n/a | n/a | per-80 ms encoder rows | OUT OF SCOPE — transcribe.cpp exposes no embedding API; used internally only as the Stage 2 oracle checkpoint | SKIP — not exposed by runtime |
| Decoder depth ladder | n/a | n/a | N-layer rung (N ≥ 2) transcript | OUT OF SCOPE — port ships the full 8-layer depth; back in scope if the runtime gains a depth knob or ladder rungs ship as variants | SKIP — not exposed by runtime (full 8-layer depth only) |
| Streaming | n/a | n/a | n/a | OUT OF SCOPE — model is an offline 30 s window (`capabilities.streaming: false`); the engine's live mode is a host-side re-decode/agreement policy | SKIP — not exposed by runtime (offline only) |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/whistle/whistle-F32.gguf --list <200 le30s wavs> --language en --batch-sizes 2,4,8 --backend cpu --golden-in tests/golden/batch/whistle.cpu.json` | byte-identical hypotheses + CPU tensor parity | MUST PASS | PASS — explicit batched `run_batch` (one beam search over utterances × beams); text parity 200/200 at batch 2/4/8 vs serial and golden `tests/golden/batch/whistle.cpu.json`; encoder tensor parity bit-exact; full le30s batch 8 byte-identical to batch 1 (2611/2611, WER 4.3247% both) |

## Notes

- **Stage 4 bring-up (2026-10-09).** `src/arch/whistle/`. Every op was
  identified by probing the closed engine; provenance per op, with the
  measured metric, is in `reports/porting/whistle/forward-map.md`.
  - **How the engine was probed.** Patched copies of `whistle.cact` (gates
    forced open/shut, CQ matrices rewritten as exact one-hot selectors via
    Hadamard sign patterns) run through `needle_embed` / `needle_transcribe`,
    compared to a float64 numpy model under explicit hypotheses. The engine
    accepts relocated records but rejects re-typed ones (CQ -> FP16).
  - **Reference for intermediates** (user decision 2026-10-09): the numpy
    model, now `scripts/lib/whistle_ref.py`. It matches the engine's encoder
    output at the engine's own 8-bit noise floor (median per-frame cosine
    >= 0.9999 on six clips). `enc.final` and transcripts stay engine-sourced.
  - **Frontend.** `n // 160` frames centered at `160·i`, symmetric Hann 400 in
    a 512 rfft, power, shipped filterbank, `ln(mel + 1e-8·max(mel))`,
    per-utterance per-bin mean/std. Scale-invariant (verified with jfk ×0.01
    / ×3). Hann symmetric vs periodic is within the engine's noise.
  - **Search.** The engine's 5-beam search is reproduced as Whisper-style beam
    search ranked by length-normalized log-probability (user decision:
    implement beam). Greedy matched the engine on 224/300 sampled rows, beam
    on 283/300. Near-tie flips remain (casing, commas, homophones). An int8
    QAT activation emulation was tried and fit worse; not kept.
  - **No-speech gate** (user-signed approximation). The engine gates audio
    before the model (unchanged with the stem output zeroed). Fitted rule:
    speech iff p99 − p10 of per-frame mel energy exceeds 10·log10(3) dB;
    separates all 46 probe clips; gates 0/2611 LibriSpeech rows.
  - **Word timestamps.** Weight-surgery probes showed the engine aligns on
    cross-attention of decoder layers 4–7, all heads. Recipe in the forward
    map (Whisper DTW with token-axis standardization, punctuation-trimmed word
    ends with quote exceptions, energy onset clamp for the first word).
    Engine archive for the comparison:
    `reports/wer/whistle-REF.librispeech-test-clean-le30s.words.jsonl`
    (text, language, words with start/end/probability for all 2611 rows);
    C++ counterpart `reports/wer/whistle-F32.librispeech-test-clean-le30s.words.jsonl`.
    Held out from all rule fitting (1684 utterances, 32,552 words): 99.26%
    of words with both ends within ±1 frame (in-sample 99.28%). The user
    confirmed this rate satisfies the MUST PASS row (2026-10-09). FLEURS
    de/fr/es golden clips: 116/116 boundaries within ±1 frame (114 exact).
  - **Tolerances** (`tests/tolerances/whistle.json`): every reference-model
    tensor within its magnitude budget; `enc.final` (engine-sourced) widened
    ~3000× for the engine's activation quantization, documented in the file.
    Reviewed and accepted by the user (2026-10-09).
  - **Input / memory contract.** Hard 30 s cap (`max_audio_samples`)
    rejected before decode with `TRANSCRIBE_ERR_INPUT_TOO_LONG`; `n_ctx`
    lowers the 320-token decoder context (never raises it) and a hit returns
    `TRANSCRIBE_ERR_OUTPUT_TRUNCATED` with the partial transcript; cache
    allocation failure returns `TRANSCRIBE_ERR_OOM`. Covered by
    `tests/whistle_real_smoke.cpp`.
  - **Backends: CPU and Metal.** Metal initially produced garbage
    transcripts. Root cause, an upstream ggml bug (still on ggml-org/ggml
    master 2026-10-05; came in with llama.cpp #28948, MoE/SSM_CONV fusion),
    fixed downstream by `patches/ggml/0003-metal-alloc-deps-after-reorder.patch`
    (`ggml_backend_metal_graph_optimize`): Metal's fusion code
    (`ggml_metal_fusion_add_alloc_deps`) registered scheduler "keep `tensor`
    alive until node `until`" dependencies *before* Metal's concurrency
    reorder. The pattern match is by op type only, so `until` need not depend
    on the kept tensor (here: a `SCALE` on a ZCRMSNorm weight). The reorder
    hoisted it above the kept tensor's producer, the scheduler placed the
    dependency node there, and `ggml_gallocr` pre-allocated the kept tensor
    before it was computed; in-place reuse then handed encoder node #293's
    memory to #331 early and re-allocated #293 over live tensors (with
    asserts enabled `ggml-alloc.c` aborts on `hn->addr.offset == 0`; release
    builds corrupt silently). The fix computes the deps on the final,
    reordered node order (kept tensors of a pattern node are then always
    produced before it). No in-tree family uses `ggml_mul_mat_id` (the deps'
    intended MoE case); 14 other models across 13 families give
    bit-identical Metal dumps and text before/after the fix. Whistle on Metal
    (F32, LibriSpeech le30s, b8): 4.3227% WER vs CPU 4.3247%, 2/2611 raw
    transcripts differ. Separately, the conv module's left pad is
    unsupported on Metal and fell back to a CPU split (now `ggml_concat`,
    bit-identical on CPU). Vulkan/CUDA are untested. Repro harnesses:
    `tmp/metal-cmp/` in the sandbox (not in the repo).
  - **Ref-dtype WER** (LibriSpeech le30s, CPU): batch 1 4.3247% (2264
    errors), batch 8 4.3247% (byte-identical), engine Oracle 4.3017% (2252).
    +0.023 pp exceeds the +0.01 pp gate. 2438/2611 transcripts byte-identical
    to the engine; of the 100 that differ after normalization C++ is worse on
    46, better on 35, equal on 19 (net +12 words; sign test not significant).
    Mechanism: the engine's int8 activations / KV vs the port's F32 tipping
    near-tie beam hypotheses. The user ruled this a blocker (2026-10-09);
    follow-up diagnosis:
    - Teacher-forcing both texts of the 173 raw-differing utterances
      through the C++ decoder: C++ prefers its own text on 152 (median
      margin 0.002 nats/token, i.e. numeric near-ties) and scores the
      engine's text higher on 21 (median margin 0.007; beam pruning
      differences).
    - Search variants do not help: 2×beam candidates per hypothesis,
      a 10-hypothesis finished pool, and HF-style end-of-sequence ranking
      either change nothing or lower agreement on a 300-row sample
      (283 -> 272 / 278 of 300). The Whisper-style rule remains the best fit.
    - An 8-bit QAT activation emulation fit worse (279/300).
    - Engine noise floor: re-running the engine and the port on two
      ±1-LSB dithered copies of the set (inaudible, int16 PCM) moves the
      engine to 4.2788% / 4.3151% (127 / 129 raw transcripts change vs the
      undithered engine) and the port to 4.3227% / 4.3227% (11 / 13 change).
      Port minus engine per input: +0.023 / +0.044 / +0.008 pp (mean +0.025,
      every bootstrap 95% CI includes 0). The engine's own spread across
      inaudibly different inputs (0.036 pp) exceeds the 0.01 pp gate, so the
      gate cannot discriminate between the port and the engine at this set
      size; a residual ~+0.02 pp tilt (port marginally worse) can't be
      separated from noise.
    Closing the gap would need the engine's exact runtime quantization.
    **User decision (2026-10-09): gap accepted** as a numerics difference
    within the engine's own noise floor; not a blocker.

- **Stage 6 CPU latency work (2026-10-10).** Batch-1, 100 LibriSpeech
  clips (929 s audio), Q8_0, M4 Max, word timestamps on (the engine always
  computes them): 124 ms/clip before (4 threads, without word timing) ->
  ~60-63 ms/clip at 4-6 threads; Cactus engine 38.5 ms/clip on ~3.8 cores.
  All Whistle-local, no ggml patch; Metal keeps the generic graph path
  (`WhistleAux::cpu_fused_ops` gates the CPU-only pieces):
  - load-time folds: ZCRMSNorm `1 + scale`, HadamardMLP weight transposes;
  - fused CPU custom ops (`ggml_custom_4d` / `ggml_map_custom1`): Sinkhorn,
    HadamardMLP (register-blocked 16x32 Kronecker), mHC pre/post mixing,
    q/k/v causal taps, post-norm gated residual;
  - CPU repack layout for every matmul-only linear, plus a zero-padded
    8200-row copy of the tied head (8199 rows cannot be repacked);
  - load-time row fusion of projections sharing an input (self-attn
    q|k|v|gate, cross q|gate, mHC phi pre|post|res per layer), streamed
    straight from the GGUF so nothing is allocated twice;
  - f32 GEMM eligibility: contraction axes zero-padded to a multiple of 4,
    GQA query heads and the beams of one utterance grouped as matmul columns;
  - beam reorder moves only filled positions; persistent model-owned CPU
    threadpool (compute is serialized per model, transcribe.h); mel frames
    split across threads; vectorized beam log-softmax;
  - word timing reads the cross-attention captured during the beam search
    (back-pointer walk) instead of a second teacher-forced decode (identical
    word lists on 400 clips); debug-dump runs keep the teacher-forced pass
    for the `dec.logits_raw.gen8` gate.
  Validation after the changes: F32 full le30s WER 4.3247% at batch 1 and 8
  with 2611/2611 transcripts identical to the accepted run; batch parity
  200/200 vs the unchanged golden; `validate.py all` exit 0; Metal
  0/100 (F32) and 2/100 (Q8_0, near-ties) transcript differences vs CPU.
  Remaining gap: at 4 threads a decode step is ~420 small ggml ops and
  per-op thread sync is most of its 0.7 ms; parity would need hand-written
  whole-layer kernels. `GGML_NATIVE=ON` (i8mm repack kernels) measured no
  gain. Q4_K_M is slower than Q8_0 (~77-82 ms/clip).

- **Scope sign-off (2026-10-09).** The user signed off two MUST PASS rows
  whose upstream mechanism is closed:
  - Word timestamps: DTW over cross-attention, with layers and heads
    undocumented.
  - No-speech handling: mechanism undocumented.

  Stage 2 must characterize both from engine outputs. If Stage 4 can't
  reproduce them, that is a user-signed blocker, not a downgrade.
  Keyword biasing and streaming are OUT OF SCOPE.

- Acceptance dataset: LibriSpeech test-clean restricted to ≤30 s
  (`librispeech-test-clean-le30s`, 2611 of 2620 rows; user decision
  2026-10-09). The publisher reports 4.31% on the full split. That number
  is context only; gates use the measured engine Oracle baseline on the
  subset.
- **Oracle WER baseline (Stage 2).**
  - Score: **4.30%** (95% CI 4.06–4.56%) on `librispeech-test-clean-le30s`.
  - Run: 2611 utterances, 0 errors, all detected `en`.
  - Settings: forced `en` per row, engine default search, Whisper
    `EnglishTextNormalizer`.
  - Breakdown: Sub/Del/Ins 1758/206/288.
  - Forced `en` and auto-detect produce byte-identical hypotheses on all
    2611 rows, so the language prompt does not change the decode here.
  - The header must carry `"language": "en"`. `score.py` routes the
    normalizer by header language, and a placeholder value silently falls
    back to `BasicTextNormalizer` (an earlier run scored 4.76% that way).
  - Files: `reports/wer/whistle-REF.librispeech-test-clean-le30s.{jsonl,score.json}`.
  - Publisher full-split figure is 4.31% (−0.01 pp delta; within noise).
    Context only. This measured run is the Stage 4 / Stage 7 gate.
  - **Stage 7 does not pick up the subset automatically.** Its LibriSpeech
    branch hard-codes `librispeech-test-clean.manifest.jsonl`. Override
    `$MANIFEST` with `samples/wer/librispeech-test-clean-le30s.manifest.jsonl`
    for this family.
- **Oracle observables (Stage 2).**
  - `needle_embed` returns `ceil(seconds / 0.08)` rows × 512, e.g. 138 for
    the 11.0 s jfk. It is dumped as `enc.final`. Whether it is taken
    before or after `encoder/final_norm` is undocumented.
  - `needle_transcribe` returns text, language, word times and the token
    count. There are no token IDs, logits or mel.
  - Output is deterministic across repeat calls.
  - Audio over 30 s errors with `audio limit is 30 s`.
  - The C API, the dylib exports and the CLI offer no beam-width or greedy
    knob. The only decode knobs are `--depth` (ladder) and `--threads`.
  - `noise.wav` gives empty text and empty language.
  - `jobs-silence.wav` (5.3 s) gives "Thank you." with language `en`.
- **Beam vs greedy.** The engine defaults to 5-beam search, and
  transcribe.cpp decodes greedily. Stage 2 found no way to run the engine
  greedy, so the REF baseline is beam search. Stage 4 either implements the
  engine's search or records the greedy-vs-beam WER delta as an accepted
  gap. Exact `transcript_compare` will likely fail for greedy decoding;
  Stage 4 chooses the compare mode.
- **Numerics.** The engine runs int8 KV and 8-bit activations (QAT
  targets), so the F32 ggml port cannot be bit-exact against the oracle.
  Parity is judged on the encoder embedding, transcripts and WER. The
  Stage 2 `enc.final` tolerance is a provisional magnitude budget that the
  port will not meet; Stage 4 replaces it with a measured, justified one.
- **Frontend unknowns.** Window type, log flavor/clamp, normalization,
  STFT centering/padding and inference dither are engine-internal. They
  are fixed at Stage 4 bring-up by matching `needle_embed` (`enc.final`). The filterbank itself
  is copied verbatim from the `.cact`.
- **Novel ops.**
  - HadamardMLP: Kronecker 16×16 ⊗ 32×32 Walsh-Hadamard with fixed
    permutations and a rank-8 correction.
  - mHC: 4 lanes, Sinkhorn-normalized lane mixing.
  - Engram: host-side n-gram hashing plus a dilated 4-tap conv.
  - Causal q/k/v taps: the KV state also carries 2 previous projections.
- Long-form audio over 30 s needs a chunking policy. The engine's own
  long-audio path is its streaming API.
- **Acceptance set exceeds the 30 s cap.** 9 of 2620 LibriSpeech
  test-clean utterances run past 30 s (max 34.95 s). The engine baseline
  and the C++ port must handle them the same way, or the WER gate
  compares different inputs.
- **Reference-run hygiene (Stage 2).**
  - Set `NEEDLE_TELEMETRY=0`, because the `cactus-needle` wrapper sends
    usage pings and re-fetches `config.json` on every load.
  - Point `NEEDLE_WHISTLE_WEIGHTS` at the HF-cache `whistle.cact` and
    `NEEDLE3_LIB_PATH` at an engine under `refs/`, so nothing is written
    to `~/.cache/cactus-needle`.
- **Hadamard-domain 2-bit file, `Q2_K_HR` (2026-10-10).** The `.cact` stores
  each 128-weight group as codebook indices × an FP16 norm, with
  `W_group = (cb[idx] · norm) @ H128`. Rotated back to the original domain
  the weights are Gaussian-like, and stock Q2_K is 30% weight error (n512
  WER 3.83% vs 3.00% at F32). In the rotated domain each group has exactly 4
  values, and Q2_K holds them to 4.8%; that is about the floor for any type
  with 4 evenly spaced levels, since the Lloyd-Max levels are uneven.
  - `convert-whistle.py --hadamard-domain` writes `C = cb[idx] · norm` for
    every CQ record (152 tensors) and sets `stt.whistle.hadamard_group = 128`.
    The runtime applies `H128` to the input of every such matmul
    (`hada_in`: a fused fast Walsh-Hadamard op on CPU, a `mul_mat` by H
    elsewhere), to the embedding rows after lookup, and to the head input.
    Engram needs no rotation: rotated table rows feed rotated key/value
    projections block for block, and `H·H = I`.
  - `Q2_K_HR` (`transcribe-quantize`) mirrors Cactus's allocation: Q2_K
    where they use CQ2 (Engram tables stored two rows per 256-wide row,
    `WhistleEngram::paired`), Q4_K where they use CQ4 (`token_embd`, mHC
    phi), Q8_0 for the two 128-wide stem projections. 21.3 MB vs their
    16.9 MB: Q2_K is 2.625 vs 2.125 bits/weight, and the converter writes
    their F16 small tensors as F32.
  - CPU load expands Q2_K (and Q4_K, in Hadamard-domain files only) to Q8_0:
    ggml has no repacked Q2_K kernel on Arm, and the plain Q2_K path was
    about 2x slower than the whole Q8_0 model.
  - Rotated F32 vs plain F32: max relative drift about 1e-6 on every dump
    tensor; paired vs unpaired tables are bit-identical.
  - Full le30s, CPU, batch 8: **4.2960% (2249 errors)**, vs engine 4.3017%
    (2252) and F32 4.3247% (2264); 2162/2611 transcripts byte-identical to
    the engine (F32: 2438; the rest are near-tie flips). n512 ablation, in
    errors (F32 323): rotated Q2_K linears 326, unrotated 413.
  - Speed, batch 1, 100 clips, M4 Max CPU: 145 ms/clip at 1 thread and
    67 ms at 4 (Q8_0: 135 / 62). Peak RSS about the same as Q8_0 (the
    weights are Q8_0 in memory); load +7 ms.
