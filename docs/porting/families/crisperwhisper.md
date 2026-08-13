# CrisperWhisper

Status: research

## Identity

- Family key: `crisperwhisper`
- Upstream architecture string: `whisper` (`model_type`); `config.architectures`
  declares `WhisperForConditionalGenerationWithAttentionLoss`, a training-time
  subclass that exists nowhere in the published inference package and is never
  resolved — HF Auto classes dispatch on `model_type`.
- Hugging Face repo: `nyralabs/CrisperWhisper2.0_small` (first port; per-variant
  intakes pin each variant's repo + revision under
  `reports/porting/crisperwhisper/<variant>/intake.json`)
- Hugging Face revision: `57750c47fde52dc1b016ec2bd4bf4704944cf3df`
- License: **dual, and not Apache-2.0.** Inference code is MIT (Part A).
  Model weights **and their outputs** are under the nyra health Non-Commercial
  Research License v1.0 (Part B): Non-Commercial Use only, share-alike onto
  derivative works (a GGUF is one), and redistribution requires shipping the
  agreement, a prescribed `NOTICE` file, and contractually binding recipients.
  See [Licensing](#licensing) — this is a maintainer decision before Stage 3.
- Variants (open, non-commercial research license):

  | Variant | HF repo | d_model | enc / dec layers | heads | mels | vocab | params |
  |---|---|---:|---:|---:|---:|---:|---:|
  | `crisperwhisper-2.0-small` | `nyralabs/CrisperWhisper2.0_small` | 768 | 12 / 12 | 12 | 80 | 51896 | 241.8 M |
  | `crisperwhisper-2.0-medium` | `nyralabs/CrisperWhisper2.0_medium` | 1024 | 24 / 24 | 16 | 80 | 51896 | 763.9 M |
  | `crisperwhisper-2.0-large` | `nyralabs/CrisperWhisper2.0_large` | 1280 | 32 / 32 | 20 | 80 | 51896 | 1543.3 M |
  | `crisperwhisper-2.0-turbo` | `nyralabs/CrisperWhisper2.0_turbo` | 1280 | 32 / **4** | 20 | **128** | **51897** | 808.9 M |

  All four are uniformly BF16 (no F32 norms or biases).

- Excluded variants:
  - `nyralabs/CrisperWhisper2.0_{small,medium,large,turbo}_pro` — `gated: manual`,
    no model card, **commercial license only**. Same shapes as their open
    counterparts. Out of scope.
  - `nyrahealth/CrisperWhisper` (v1) — deprecated by the publisher, a
    whisper-large-v2 finetune with a genuinely remapped ("changed") tokenizer
    the reference still special-cases. Different contract from 2.0. Out of scope.

### The turbo variant is not a size variant

`small`, `medium`, and `large` are whisper-v2-generation derivatives: 80 mel
bins, 99 language tokens, `vocab_size` 51896. `turbo` is a
whisper-large-v3-turbo derivative: 128 mel bins, 100 language tokens (adds
`<|yue|>`), `vocab_size` 51897.

**Every special-token id above 50357 shifts by +1 on turbo.**

| Token | small / medium / large | turbo |
|---|---:|---:|
| `<|translate|>` | 50358 | 50359 |
| `<|transcribe|>` | 50359 | 50360 |
| `<|notimestamps|>` | 50363 | 50364 |
| timestamp base `<\|0.00\|>` | 50364 | 50365 |
| `[UM]` (first added event token) | 51865 | 51866 |
| `<ehtx>` (last added token) | 51895 | 51896 |

`turbo` additionally ships `begin_suppress_tokens: [220, 50256]` and
`forced_decoder_ids: null`, where the other three ship
`forced_decoder_ids: [[1, 50259], [2, 50359]]` and
`begin_suppress_tokens: null`. Nothing may hardcode a token id; every id is
read from `generation_config` / `tokenizer.json` per variant. This is the same
80-vs-128-mel / v2-vs-v3 split the `whisper` family already handles, so it is a
known shape, not a new one.

## References

- Canonical reference: **`author_repo_crisperwhisper`** — the publisher's
  `crisperwhisper` PyPI package (MIT,
  <https://github.com/nyrahealth/CrisperWhisper>). The model card sets
  `library_name: crisperwhisper`; this is the only documented entrypoint.
- Instrumented reference: **`crisperwhisper.transformers_engine.TransformersEngine`**
  — the package's pure-torch backend. It loads stock
  `AutoModelForSpeechSeq2Seq` with `attn_implementation="eager"` (already
  required, because word timing needs `output_attentions`), so it is directly
  hookable for Stage 2 tensor dumps and reuses the class set our `whisper`
  family already instruments.
- Cross-check references:
  - `docs/porting/families/whisper.md` + `src/arch/whisper/` — the encoder,
    decoder, KV cache, and mel frontend are the same graph. This port is a
    decode-harness port, not a graph port.
  - `refs/ggml-org/whisper.cpp` — ggml-native cross-check for encoder graph
    shape and mel constants, as for `whisper`.
  - `ctranslate2-crisperwhisper` (the `ct2` backend) — **not** a reference.
    Linux/CUDA-only fork with int8/fp16 fused kernels and speculative decoding;
    the publisher's own DOCS warn about fp16 batched-GEMM rounding divergence
    between it and the transformers backend.

### Bridge validation

The canonical and instrumented references are the same package, so no bridge is
needed for the decode contract. One bridge does exist and must be recorded: the
**auto / no-language-hint** path is not reachable through
`crisperwhisper.transcribe()` at all (`language` defaults to `"en"` and a lang
token is always forced). Its reference is stock `transformers`
`generate(language=None)` against the same checkpoint, which is a different
harness from the one every other row uses.

## Commands

Reference run (Stage 2, shipped):

```bash
uv run scripts/validate.py ref --family crisperwhisper --variant crisperwhisper-2.0-small
```

Reference dumps (what the above invokes per case):

```bash
uv run --project scripts/envs/crisperwhisper \
  scripts/dump_reference_crisperwhisper_author.py decode \
  --model nyralabs/CrisperWhisper2.0_small \
  --audio samples/jfk.wav \
  --mode verbatim \
  --language en \
  --out build/validate/crisperwhisper/crisperwhisper-2.0-small/jfk/ref \
  --torch-threads 1
```

The dumper never enables `hallucination_mitigation`, `early_eot_recovery`, or
`temperature_fallback`. All three default to `True` in the publisher's
`transcribe()` and all three can rewind and re-decode, which makes
greedy-argmax parity untestable, so there is no flag to turn them on here —
the dumper drives `engine.generate_with_repair(..., hallucination_mitigation=
False)`, which reduces to a plain greedy `generate`.

Oracle catalog + provisional tolerances (one-shot, re-run after any dump change):

```bash
uv run scripts/build_crisperwhisper_oracle.py
```

Conversion (Stage 3, shipped):

```bash
uv run --project scripts/envs/crisperwhisper \
  scripts/convert-crisperwhisper.py nyralabs/CrisperWhisper2.0_small \
  --revision 57750c47fde52dc1b016ec2bd4bf4704944cf3df
```

Writes `models/crisperwhisper-2.0-small/crisperwhisper-2.0-small-BF16.gguf`
and nothing else; the quant matrix is Stage 5. Provenance (SHA-256, source
revision, converter commit) lands in
`reports/convert/crisperwhisper-2.0-small-BF16.json`.

Validation:

```bash
uv run scripts/validate.py all --family crisperwhisper --variant crisperwhisper-2.0-small
```

Reference WER baseline (Stage 2, shipped — one-time run, not repeated downstream):

```bash
# Gate: intended mode
uv run --project scripts/envs/crisperwhisper \
  scripts/wer/run_reference_crisperwhisper_author.py \
    --model nyralabs/CrisperWhisper2.0_small \
    --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
    --mode intended --torch-threads 5 \
    --out reports/wer/crisperwhisper-2.0-small-REF.librispeech-test-clean.jsonl

# Reported alongside: verbatim mode
uv run --project scripts/envs/crisperwhisper \
  scripts/wer/run_reference_crisperwhisper_author.py \
    --model nyralabs/CrisperWhisper2.0_small \
    --manifest samples/wer/librispeech-test-clean.manifest.jsonl \
    --mode verbatim --torch-threads 5 \
    --out reports/wer/crisperwhisper-2.0-small-REF-verbatim.librispeech-test-clean.jsonl

uv run scripts/wer/score.py reports/wer/crisperwhisper-2.0-small-REF.librispeech-test-clean.jsonl
uv run scripts/wer/score.py reports/wer/crisperwhisper-2.0-small-REF-verbatim.librispeech-test-clean.jsonl
```

The runner also defaults the three rewind features off, for the same reason and
so the WER baseline matches what the port will actually implement. Pass
`--rewind-features` to measure the publisher's out-of-the-box behaviour instead;
that run is not the gate.

## Capability Validation

One row per advertised capability. Each row carries two human/agent columns
that are filled at different stages:

- **`Target`** — the *scope decision*, set at Stage 1 and signed off by the
  user. It declares whether this port is obligated to deliver the capability.
  This is the contract Stage 4 implements against.
- **`Status`** — the *observed outcome*, filled at Stage 4 after running the
  command. Stage 1 leaves it `TODO`.

Allowed `Target` values (Stage 1):

- `MUST PASS` — in scope for this port. Stage 4 must resolve `Status` to
  `PASS`; it may not be downgraded to SKIP/ACCEPTED GAP without the user
  re-signing the scope change.
- `OUT OF SCOPE — <reason>` — explicitly deferred by the user at intake.
  Stage 4 may resolve it to SKIP or ACCEPTED GAP. The reason names what would
  bring it back in scope.

Allowed `Status` values (Stage 4): `PASS`, `SKIP — not exposed by runtime`,
`ACCEPTED GAP — <reason>`.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/crisperwhisper-2.0-small/crisperwhisper-2.0-small-BF16.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | **PASS** |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/crisperwhisper-2.0-small/crisperwhisper-2.0-small-BF16.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | **PASS** |
| Verbatim mode | `[verbatim_1..5]` prompt prefix | `build/bin/transcribe-cli -m <gguf> --language en --mode verbatim samples/disfluency.wav` | retains the filler `[UH]`, the cutoff, and the `from the from the` repetition; WER vs `nyra-disfluency-en-verbatim` within gate of the reference's 3.54 % | MUST PASS | **PASS** |
| Intended mode | `[intended_1..5]` prompt prefix | `build/bin/transcribe-cli -m <gguf> --language en --mode intended samples/disfluency.wav` | no event tokens; gated on LibriSpeech (reference 3.22 %), **not** on `nyra-disfluency-en-intended` — see [Disfluency evaluation set](#disfluency-evaluation-set) | MUST PASS | **PASS** |
| Word timestamps | Viterbi over supervised cross-attention | `build/bin/transcribe-cli -m <gguf> --language en --timestamps word samples/jfk.wav` | per-word `(start, end)` within tolerance of the reference aligner's output | MUST PASS | **PASS** |
| Longform (>30 s) | `<ctx>` continuation | `build/bin/transcribe-cli -m <gguf> --language en samples/whole-earth.wav` | no duplicated or dropped words at 26 s-stride window seams; WER vs reference within gate | MUST PASS | **PASS** |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/crisperwhisper-2.0-small/crisperwhisper-2.0-small-BF16.gguf --samples-dir samples/wer/librispeech-test-clean --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | **PASS** |
| Segment timestamps | whole-audio span | `build/bin/transcribe-cli -m <gguf> --language en --timestamps segment samples/jfk.wav` | `transcribe_returned_timestamp_kind` == `SEGMENT`, one segment covering the audio | PARTIAL | **HONORED, COARSE** — the ABI requires a coarser-than-max request to be answered at that granularity, so `SEGMENT` returns the whole-audio span rather than `NONE`. The span is not a real boundary: the model is always decoded with `<\|notimestamps\|>` and emits no timestamp tokens to segment on. Real boundaries would have to be synthesized from word timings under a segmentation rule that has not been agreed. |
| Translate | inherited `<\|translate\|>` | — | — | OUT OF SCOPE — `<\|translate\|>` (50358) and `task_to_id.translate` survive in the checkpoint, but the author package exposes no translate path and neither the card nor DOCS claims it. Back in scope if a reference translate path is established and measured. | SKIP — not exposed by runtime |
| Streaming | n/a | — | — | OUT OF SCOPE — `capabilities.streaming: false`; longform is offline overlapped-window continuation, not a streaming contract. | SKIP — not exposed by runtime |
| Hotword boosting | `<htx> … <ehtx>` | — | — | OUT OF SCOPE — Pro-models-only feature; the open checkpoints were never trained with hotword prompts and the reference raises a `UserWarning` and warns of degradation. Back in scope only with a commercially-licensed Pro checkpoint. | SKIP — not exposed by runtime |
| Verbatimize | `<vtx> clean transcript <evtx>` | — | — | OUT OF SCOPE — second-order task (audio + trusted clean transcript in, verbatim transcript out); no runtime surface for supplying a prior transcript. Back in scope once the CLI/API can take a reference transcript. | SKIP — not exposed by runtime |
| Forced alignment | `forced_align()` | — | — | OUT OF SCOPE — depends on the word-timing aligner plus a teacher-forced attention pass; deferred until word timestamps are `PASS`. | SKIP — not exposed by runtime |
| Hallucination mitigation | rewind/escape repair | — | — | OUT OF SCOPE — **no working reference to port against**, see [Hallucination repair](#hallucination-repair-deferred-with-evidence). Revisit at Stage 7 once the upstream import bug is resolved and repair is measured on the disfluency set. | SKIP — not exposed by runtime |
| Temperature fallback | collapse recovery | — | — | OUT OF SCOPE — `fallback.py` genuinely **samples** (`do_sample`, escalating temperature ladder, incrementing per-attempt seed). Bit-exact reproduction needs an RNG-stream contract shared with the reference, which this repo does not have. Back in scope only with a seeded-sampling contract. | SKIP — not exposed by runtime |
| Early-EOT recovery | truncation recovery | — | — | OUT OF SCOPE — deferred, not blocked. Unlike temperature fallback it is fully deterministic (bans EOT for `min_new_tokens` steps, measures `stop_prob`), so it is portable; ~200 lines coupled to the longform path. Back in scope once longform is `PASS`. | SKIP — not exposed by runtime |
| Dual-mode (`transcribe_dual`) | batched verbatim+intended | — | — | OUT OF SCOPE — CTranslate2 backend only; no transformers reference exists. | SKIP — not exposed by runtime |
| Speculative decoding | draft + verify | — | — | OUT OF SCOPE — CTranslate2 backend only; a throughput feature, not a capability of the checkpoint. | SKIP — not exposed by runtime |

## Architecture summary

- Pattern: `encoder-decoder`. **The graph is stock Whisper.** Encoder, decoder,
  cross-attention, KV cache, positional encodings, and the mel frontend are
  identical to `docs/porting/families/whisper.md`. The only weight-level
  differences are BF16 storage and a 31-row-wider (32 on turbo) tied
  embedding/output head.
- The port is therefore a **decode-harness port**, not a graph port. Everything
  novel lives above the graph:

### Prompt contract

The decoder input is, literally:

```
encode("[verbatim_1][verbatim_2][verbatim_3][verbatim_4][verbatim_5]"
       [+ " <ctx> " + last N confirmed words + " <ectx>"]
       [+ " <htx> " + hotwords + " <ehtx>"])          # Pro only
+ [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]
```

Three things about this differ from every Whisper prompt we already handle:

1. The mode tags come **before** the Whisper prefix, not after it, and there is
   no `<|startofprev|>` wrapper.
2. Context comes **before** hotwords. That is the training order; emitting them
   the other way round silently changes the output rather than erroring.
3. `<|notimestamps|>` is always present. The model never emits timestamp tokens.

Swap `[verbatim_N]` for `[intended_N]` to get the clean transcript.

### Added tokens (31, or 32 on turbo)

Ids 51865–51895 (51866–51896 on turbo), **all with `special: false`**, so
`skip_special_tokens` does not strip them:

- 15 vocal-event tokens: `[UM]`, `[UH]`, `[laughter]`, `[sniff]`,
  `[throatclearing]`, `[cough]`, `[sigh]`, `[breath]`, `[lipsmack]`, `[yawn]`,
  `[noise]`, `[crying]`, `[fart]`, `[scream]`, `[sneeze]`. These are **intended
  output** in verbatim mode and must survive decoding.
- 5 `[verbatim_N]` + 5 `[intended_N]` mode tags. Stripped from output.
- 6 markers: `<vtx>`, `<evtx>`, `<ctx>`, `<ectx>`, `<htx>`, `<ehtx>`. Stripped.

The reference strips the second and third groups by explicit id list plus regex
(`crisperwhisper/prompt.py::strip_prompt_artifacts`), not by the tokenizer's
special-token machinery.

The base BPE vocab is untouched: `vocab_sha256` is byte-identical to
`openai/whisper-small`'s. Only `added_tokens` differ.

### Word timestamps

Not Whisper's median-filtered DTW. `crisperwhisper/word_timing.py` (~23 KB) runs
a **Viterbi alignment** over the cross-attention rows of the
`generation_config.alignment_heads` (10 `(layer, head)` pairs, different per
variant), with blank states derived from **mel-band energy** — which is why the
engine keeps the 10 ms-hop numpy mel around alongside the model input tensor.
The transformers backend captures attention inline during generation via
`generate(output_attentions=True, return_dict_in_generate=True)`, which is why
it must load with `attn_implementation="eager"`.

### Longform

`<ctx>`-based **conditional continuation**, not timestamp-token stitching:
30 s windows at a 26 s stride (4 s overlap), first chunk decoded without
context, each subsequent chunk prompted with the last `context_words=12`
confirmed words. At non-final boundaries, trailing words are dropped
overlap-aware — a word is dropped only when its audio starts inside the overlap
region, capped at `drop_words=2`. Word timings are mapped chunk-local → global
and monotonized across seams.

**The `whisper` family's long-form code does not transfer.** That path is built
on timestamp tokens and `condition_on_prev_tokens`; this model emits neither.

Two alternative strategies exist in the reference (`chunked_lcs`, `token_lcs`)
and are not part of this port.

## Capabilities (from intake)

- Languages: 99 in the tokenizer and `lang_to_id` (100 on turbo, adding `yue`).
  The model card declares only `en` + `de`; the published benchmark covers 10
  languages. Declared as the full 99 so GGUF `general.languages` is not
  narrower than the checkpoint — but per-language accuracy beyond en/de is
  unmeasured. See intake `intake_gaps`.
- Language detection: inherited (`forced_decoder_ids: [[1, null]]` plus all 99
  `<|lang|>` tokens). Never exercised by the author harness.
- Translation: `null` — inherited token, no reference path, unvalidated.
- Timestamps: ceiling is **word**. `AUTO` resolves to `WORD`; `SEGMENT` is
  honored as a single whole-audio span, not a real boundary (see above).
- Streaming: no.
- Diarization: no.

## Upstream benchmarks (from model card)

The publisher benchmarks verbatimness and timing, **not WER**:

- Nyra Verbatim Speech Benchmark, disfluency F1, 10-language average: **87.8**
  (Pro: 93.5). Reported for "CrisperWhisper 2.0" as a line, not per size.
- TIMIT read-speech mean absolute word-boundary error: **29.6 ms**.
- LibriSpeech test-clean: **not reported for 2.0**. For scale only, v1
  (`nyrahealth/CrisperWhisper`, large-v2 finetune, deprecated) logged 1.82 % on
  the Open ASR Leaderboard in 2024-08 under the leaderboard's text normalizer.

Acceptance dataset for the Stage 7 WER gate: **LibriSpeech test-clean**, scored
against the measured Oracle reference baseline (there is no publisher number to
compare against).

## Measured reference baseline (Stage 2)

Reference: `crisperwhisper==2.0.2` transformers backend, transformers 4.57.6,
F32 compute from BF16 storage, CPU, greedy, all three rewind features OFF.
Dataset: LibriSpeech test-clean, 2620 utterances, 0 errors. Scored with
`scripts/wer/score.py --language en` (Whisper `EnglishTextNormalizer`).

| Mode | WER | 95% CI | Sub | Del | Ins | Role |
|---|---:|---|---:|---:|---:|---|
| `intended` | **3.22 %** | [3.02, 3.43] | 1296 | 230 | 184 | **the Stage 7 gate** |
| `verbatim` | 3.60 % | [3.11, 4.34] | 1351 | 219 | 340 | reported only |

Artifacts: `reports/wer/crisperwhisper-2.0-small-REF.librispeech-test-clean.{jsonl,score.json}`
and `…-REF-verbatim.…` for the verbatim run. Per-utterance hypotheses are kept
so Stage 4/7 can diff C++ against reference `hyp_text` by `id` instead of
re-running the reference.

No publisher LibriSpeech number exists for CrisperWhisper 2.0, so there is no
measured-vs-published delta to report. For scale only: openai/whisper-small, the
same size class, publishes 3.43 % on this split.

### The verbatim/intended gap is not what it looks like

The 0.38 pp gap is **not** mostly disfluency insertion. Exactly one utterance
(`5639-40744-0036`) enters a runaway repetition loop in verbatim mode —
`"…left her to her to her to her to him to her to him…"`, 437 % WER on a
38-word reference — and that single utterance contributes **0.31 pp** of the
0.38 pp. Genuine disfluency insertion accounts for roughly 0.07 pp.

Two consequences:

1. **LibriSpeech is a weak discriminator for this model's headline feature.**
   Clean read speech has almost no disfluencies for verbatim mode to preserve,
   so the two modes produce near-identical content. The WER gate is a
   regression guard, not evidence the verbatim path works. The
   verbatim/intended capability rows must be exercised on disfluent audio.
2. **Expect Stage 4 to reproduce that loop.** Looping is precisely what the
   publisher's `hallucination_mitigation` rewind/escape repair exists to fix,
   and this port has it `OUT OF SCOPE`. The C++ runtime, having no repair
   either, should loop on the same utterance. That is reference-faithful
   behaviour, not a C++ defect — check `hyp_text` by `id` before treating a
   verbatim-mode outlier as a regression. The `intended` gate run has zero
   utterances above 100 % WER.

**WER gate semantics (decided at intake sign-off): report both modes, gate on
`intended`.** Default `verbatim` mode emits `[UM]`/`[UH]`/`[laughter]` and
preserves stutters and false starts by design; scored against LibriSpeech's
clean references with the standard normalizer, that inflates WER for *correct*
behavior. So:

- `docs/models/crisperwhisper.md` records **two** numbers per variant:
  `mode=intended` under the standard normalizer, and `mode=verbatim` under a
  normalizer that additionally strips the 15 bracketed vocal-event tokens.
- The Stage 7 gate binds only on the `intended` number, which is
  apples-to-apples with every other family.
- Whichever mode is being scored, the Oracle reference baseline and the C++ run
  must use the identical mode and the identical normalizer.

## Hallucination repair (deferred, with evidence)

This row was originally deferred for a bad reason. The Stage 1 note said repair
"must be disabled in the reference for parity, so it cannot be validated by the
same runs" — which conflates two separate things. Tensor dumps must be
repair-off for determinism; that says nothing about whether the shipped runtime
should carry repair, which would be validated behaviourally against a repair-ON
reference. And the algorithm is cheap: `find_token_loop` (~25 lines: earliest
consecutive n-gram repeat, n ∈ 1..5, thresholds `{1:8, 2:8, 3:4, 4:3, 5:3}`),
plus rewind-to-`keep_reps` and a one-step ban on the loop starter. Perhaps 150
lines of deterministic C++, well below the Viterbi aligner or `<ctx>` longform
already in scope.

It stays out of scope on evidence, not on that reasoning:

**1. Upstream bug — repair is unreachable on the transformers backend.**
`crisperwhisper/hallucination.py` does a module-level `import ctranslate2`, and
`transformers_engine.generate_with_repair` imports from that module whenever
repair is enabled. With only the documented `crisperwhisper[transformers]`
extra installed, `hallucination_mitigation=True` raises `ModuleNotFoundError:
No module named 'ctranslate2'`. The helpers the transformers path actually
needs (`find_token_loop`, `DEFAULT_REPAIR_THRESHOLDS`) are pure Python; the CT2
symbol is used only in a CT2-only helper. Upstream `DOCS.md` lists
"Hallucination mitigation (rewind/escape repair)" as supported on **both**
backends. It is not. Worth reporting upstream.

**2. Forced through, it regresses.** Installing stock `ctranslate2` purely to
satisfy the spurious import and re-decoding the 25 worst-WER verbatim
utterances from the LibriSpeech baseline:

| | WER over those 25 |
|---|---:|
| repair OFF | 1.4444 |
| repair ON | **1.9383** |

19 of 25 unchanged, the 4 nominally "improved" were sub-0.001 differences, and
the one genuine runaway loop (`5639-40744-0036`) got **worse**: 4.368 → 6.474.
Repair does fire — punctuation shifts, so it rewinds and re-decodes — then
loops again on a different tokenisation until `max_repairs=3` is exhausted.

Two caveats kept deliberately visible: 25 utterances of clean read speech is not
the domain repair is built for, and forcing an import the packaging does not
support may mean measuring a broken backend rather than an ineffective
algorithm. Both are reasons to re-measure on
[the disfluency set](#disfluency-evaluation-set) at Stage 7, not reasons to port
an unvalidated feature now. **We cannot port what we cannot first validate
against a working reference.**

## Disfluency evaluation set

LibriSpeech cannot exercise this model's headline capability: clean read speech
has no disfluencies for verbatim mode to preserve, so both modes produce
near-identical output (the measured gap is 0.38 pp, and 0.31 pp of that is a
single hallucination loop).

`nyralabs/disfluency_speech_english` is the publisher's own benchmark data and
solves this. **Apache-2.0** — permissive, unlike the weights it benchmarks.
Derived from `amaai-lab/DisfluencySpeech` ([arXiv:2406.08820](https://arxiv.org/abs/2406.08820)).

| Property | Value |
|---|---|
| Test split | 249 utterances |
| Duration | 3.6–10.7 s, mean 6.5 s — **all inside the 30 s window**, so every utterance is a clean single-chunk case with no longform interaction |
| References | paired `verbatim_transcript` **and** `intended_transcript` |
| Divergent pairs | 237 / 249 |

Ingest:

```bash
uv run scripts/wer/ingest.py nyra-disfluency --split test
```

This is the only source in `scripts/wer/ingest.py` that writes **two** manifests
over one WAV directory (`nyra-disfluency-en-{verbatim,intended}.manifest.jsonl`),
because scoring both modes against a single clean reference cannot distinguish
them. Each mode is scored against its own ground truth.

Reference conventions worth knowing before writing any comparison logic: fillers
are bracketed (`[UH]`, `[laughter]`, `[breath]`), **cutoffs are marked with a
trailing asterisk** (`bam*`, `Neil*`) rather than the hyphen form the model card
shows (`th-`), and repetitions are written out (`from the from the`).

### Measured on the disfluency test split (249 utterances, 0 errors)

Each mode scored against **its own** reference column, same audio, same
normalizer, rewind features off:

| Mode | Reference column | WER | 95% CI | Sub | Del | Ins |
|---|---|---:|---|---:|---:|---:|
| `verbatim` | `verbatim_transcript` | **3.54 %** | [2.97, 4.15] | 120 | 47 | 32 |
| `intended` | `intended_transcript` | 9.77 % | [8.53, 11.13] | 108 | 103 | **262** |

**The verbatim number is the real result: 3.54 % on genuinely disfluent
spontaneous speech**, and it is the evidence that the headline capability works.
That is the number the verbatim `MUST PASS` row should be gated on.

**Do not gate on the 9.77 % intended number.** It measures a reference
convention mismatch, not model quality. This dataset's `intended_transcript` is
not "the same utterance with disfluencies removed" — it is an abridged
semantic core that deletes entire clauses the speaker actually said. Hence 262
insertions against only 32 for verbatim. Three examples, with the model's
verbatim hypothesis shown to prove the audio really contains the dropped text:

| | |
|---|---|
| verbatim ref | `I've got a Bachelor's in electrical engineering, so I'm not, like, a hugely advanced degree or any of that stuff.` |
| intended ref | `I've got a Bachelor's in electrical engineering.` |
| model verbatim hyp | `I've got a bachelor's in electrical engineering, so and I'm not like a hugely advanced degree or any of that stuff.` |

| | |
|---|---|
| verbatim ref | `But it, it was not necessarily local, it was. But I, I mean, so what's.` |
| intended ref | `It was not necessarily local.` |
| model verbatim hyp | `But it it was not necessarily local, it was. But I I mean, so what's` |

CrisperWhisper's `intended` mode removes disfluencies while keeping every
content clause, which is a different — and defensible — definition from this
dataset's. The model is right and the reference is simply answering a different
question. **LibriSpeech test-clean stays the gate for `intended` mode**
(3.22 %); this set is the gate for `verbatim`. The
`nyra-disfluency-en-intended` manifest is still worth keeping for C++-vs-
reference `hyp_text` parity diffs by `id`, just not as an accuracy gate.

Artifacts: `reports/wer/crisperwhisper-2.0-small-REF-{verbatim,intended}.nyra-disfluency-en.{jsonl,score.json}`.

`samples/disfluency.wav` is utterance `DISFLUENCY_TEST_000003` from this split
(7.14 s, Apache-2.0), copied in as a golden-manifest case because it carries a
filler, a cutoff, and a repetition in one clip:

- verbatim: `Well, what about [UH] papyrus, you know, made out of bam* you know, bamboo stuff, from the from the banks of the Nile.`
- intended: `What about papyrus, made out of bamboo stuff, from the from the banks of the Nile.`

## Licensing

Not Apache-2.0. The repo ships two licenses:

| Component | License |
|---|---|
| Inference code (the `crisperwhisper` package we read) | MIT — imposes nothing on us |
| Model weights, config, tokenizer, **and Outputs** | nyra health Non-Commercial Research License v1.0 |

Consequences for this repo:

1. **Non-Commercial Use only** — and the restriction reaches the *outputs*
   (Sec. 4.1). Transcripts produced by this model may not be commercially
   exploited, and may not train any model intended for commercial use.
2. **A converted GGUF is a Derivative Work** (Sec. 1, Sec. 3.1) and inherits the
   agreement in its entirety. It may **not** be released under MIT/Apache/BSD
   (Sec. 3.2).
3. **Redistribution is permitted for Non-Commercial Use** (Sec. 2.1(c)) but
   Sec. 3.3 requires all three of: (a) shipping a complete copy of the
   agreement, (b) contractually binding the recipient *before or upon access*,
   and (c) a `NOTICE` file carrying the prescribed attribution text. (b) means
   a published `handy-computer/*` GGUF repo would need to be **gated**, not
   merely private.
4. The `_pro` checkpoints are commercial-license-only and are excluded here.

**Decision (intake sign-off): proceed with the port, publish no GGUF.**
transcribe.cpp ships C++ support and the converter; users convert their own
checkpoints locally. Nothing is uploaded to `handy-computer/*` for this family,
which keeps us out of Sec. 3.3 entirely (no distribution, so no agreement copy,
no `NOTICE` file, no recipient-binding obligation). Revisit only if the family
is later wanted as a published artifact, at which point the repo must be gated.

Note that (1) still applies to us as a *user* of the weights: outputs from this
family are Non-Commercial Use only, including any use of transcripts as training
data.

## Known risks

See `reports/porting/crisperwhisper/crisperwhisper-2.0-small/intake.json::known_risks`
(14 entries). Highlights:

1. `turbo` shifts every special-token id by +1 and uses 128 mel bins — it is not
   a size variant of the other three.
2. The weights are non-commercial-research licensed, and so are their outputs.
3. The decode contract is not Whisper's: no timestamp tokens, no
   `condition_on_prev_tokens` long-form, prompt tags before the Whisper prefix.
4. 31 added tokens with `special: false`; the vocal-event ones are intended
   output, the mode/marker ones must be stripped.
5. Uniformly BF16 including norms and biases — the converter's existing BF16
   branch demotes the conv stem to F16 for want of a BF16 conv kernel, which is
   now on the accuracy-reference path rather than an edge case. **Measured at
   Stage 3: bit-exact for `small`**, see
   [BF16 → F16 conv demotion](#bf16--f16-conv-demotion-is-lossless-here);
   still open for `medium` / `large` / `turbo`.
6. `transcribe()` defaults `hallucination_mitigation`, `early_eot_recovery`, and
   `temperature_fallback` all to `True`; all three rewind, and all three must be
   pinned off for greedy parity.
7. `max_new_tokens` defaults to 256, and prompt + `<ctx>` context eat into the
   448-position decoder budget before generation starts.

## Conversion (Stage 3)

`scripts/convert-crisperwhisper.py` emits GGUF architecture `crisperwhisper`
with tensor names **byte-identical to `convert-whisper.py`'s**, so
`src/arch/crisperwhisper/` can reuse the whisper graph code and the Stage 2
oracle dumps line up name-for-name. The decode contract that is *not* Whisper's
rides along as `stt.crisperwhisper.*` KV.

### Tensor-mapping decisions

**1. `encoder_blank_head.{weight,bias}` is dropped.** All four checkpoints ship
a `[1, d_model]` + `[1]` head that stock Whisper does not have (481 tensors
against whisper-small's 479). It is untrained training scaffold, not an
inference weight:

- unreferenced by the publisher's `crisperwhisper` package and by transformers'
  Whisper modelling code. The Viterbi aligner's blank probabilities come from
  `word_timing.blank_logp_from_mel_energy` / `blank_logp_from_space_attention`,
  never from a learned head;
- `weight.std() = 0.0207` against `config.init_std = 0.02`; `mean ≈ 1e-3`;
- `bias` is exactly `0.0`, still at its zero-init;
- `config.architectures` names `WhisperForConditionalGenerationWithAttentionLoss`,
  a training-time subclass that exists nowhere in the published inference
  package.

Emitting it would put an unused, untrained tensor into every quant of every
variant. The converter lists it in `DROPPED_TENSORS` with the reason and
**fails** on any other unconsumed safetensors key, so a future checkpoint that
grows a real extra tensor stops the conversion instead of printing a warning.

**2. Nothing is hardcoded by id.** Mode tags are discovered by regex
(`[verbatim_N]` / `[intended_N]`, sorted by `N`, so the upstream-configurable
tag counts are read rather than assumed); event tokens and markers resolve by
literal content and raise if absent. This is what makes `turbo` — where every
id above 50357 shifts by +1 — a config change rather than a code change.

**3. `general.languages` is ordered by language-token id** (`en, zh, de, …`,
Whisper's canonical order), not by `generation_config.lang_to_id` key order
(alphabetical). The token-id order is what the intake and golden manifest
declare, so Gate B's language comparison is order-clean. The `whisper` family
still warns here.

**4. The mel filterbank is rebuilt, not read.** Unlike whisper tiny…large-v2,
`preprocessor_config.json` carries no `mel_filters` array, so the converter
reproduces `transformers.audio_utils.mel_filter_bank(..., norm="slaney",
mel_scale="slaney")` exactly as `WhisperFeatureExtractor` would at load time.

### BF16 → F16 conv demotion is lossless here

Known risk 5 flagged that the loader has no BF16 conv kernel, so
`reference_dtype_for` demotes `enc.conv.{0,1}.weight` to F16 — and that on this
family the demotion sits on the accuracy-reference path rather than being an
edge case. Measured against the safetensors: **both conv kernels round-trip
bit-exactly** (`max_abs_err = 0.0` over the full tensors). BF16's 8 mantissa
bits fit inside F16's 10 whenever the exponent is in F16 range, and no value in
either kernel underflows. Spot-checked BF16 and F32 tensors are likewise exact.
The risk is retired for `small`; re-check it on `medium` / `large` / `turbo`,
where the value range could differ.

### Dtype layout of the shipped GGUF

| Bucket | Count | Stored as |
|---|---:|---|
| Linear / embedding | 193 | BF16 (reference dtype) |
| Norms, biases, positional tables, frontend buffers | 286 | F32 |
| Conv stem (`enc.conv.{0,1}.weight`) | 2 | F16 |

`scripts/lib/test_quant_policy_sync.py` passes unchanged — every name this
family emits was already covered by `reference_dtype_for` (the `whisper`
family's Breeze-ASR-25 BF16 finetune exercises the same rules), so neither
`policy.cpp::classify_tensor` nor the test corpus needed a new entry.

### Preflight Gate B: one expected WARN

```
gate B WARN: crisperwhisper/crisperwhisper-2.0-small
  PASS: dtype_consistency
  WARN: frontend_config — declared.normalization=none != gguf.normalization=whisper_logmel
  PASS: tokenizer_alignment
  PASS: architecture_sanity
  PASS: capabilities
```

Both sides are correct; the comparator is matching two different vocabularies.
`intake.frontend.normalization` is the *statistical* normalization enum
(`per_feature` / `global` / `per_utterance` / `none`), and the checkpoint
applies none of those, which is what Gate A validated against the reference
preprocessor. `stt.frontend.normalize` is the *loader dispatch tag* for mel
post-processing, and `whisper_logmel` is the only value that names Whisper's
`log10 → max(x, x.max() − 8) → (x + 4) / 4` compression (intake
`known_risks[0]`). The
schema enum has no value that expresses it, and `whisper_logmel` is what
`src/arch/whisper/weights.cpp` requires. Neither artifact was weakened to
silence the comparison; the `whisper` family sidesteps it only because its
intakes leave `normalization` null.

## Capability validation evidence (Stage 4 Step 8)

Every `MUST PASS` row resolved to PASS. What each was actually checked against:

| Row | Evidence |
|---|---|
| Transcribe, explicit hint | byte-exact vs reference on jfk + disfluency, both modes; 5,738-utterance hypothesis diff |
| Transcribe, auto / no hint | language detected from audio and matched against the bridge reference (stock `transformers` `detect_language`) on 6 languages: en, de, ja, ru, zh, ko — all correct |
| Verbatim mode | keeps `[UH]`, the cutoff, and `from the from the`; 100 % identical to the reference on all 249 disfluency utterances at F32 |
| Intended mode | strips events, fillers and the repetition; byte-exact vs reference on both samples; full-manifest diff at 99.81 % / 99.60 % |
| Word timestamps | 0.0 ms deviation on all 23 disfluency words; 2.27 ms mean on jfk |
| Longform (>30 s) | byte-exact vs reference on whole-earth.wav: 206 words, 4 windows, matching per-window counts and contexts |
| Batch (offline) | real `run_batch()` parallel path; WER-neutral vs batch 1 within noise (+0.0019 / +0.0056 pp over 2620 utterances). NOT byte-identical: 15 / 11 utterances differ, all punctuation near-ties. See [Batching](#batching-is-wer-neutral-not-byte-identical). |

### The auto-language row needed real detection, not a default

The publisher's `transcribe()` has no auto path at all: `language` defaults to
`"en"` and a language token is always forced. The first implementation here
mirrored that by falling back to `<|en|>`, which *looked* fine — on
`samples/german.wav` a forced `<|en|>` still returns correct German, so the
no-hint path passes a casual check while being silently wrong for any language
the model would have had to actually identify.

The row is a forced `MUST PASS`, so it now runs a real detection forward: prefix
`[mode tags] <|startoftranscript|>`, one decoder pass, argmax restricted to the
99 `<|lang|>` tokens. Its reference is the bridge named at intake (stock
`transformers`, not the author package, which cannot reach this path).

### Segment timestamps is an ACCEPTED GAP, not a silent one

`caps.max_timestamp_kind` is capped at `WORD`, but the enum orders
`SEGMENT(2) < WORD(3)`, so a `--timestamps segment` request is *accepted* by the
dispatcher rather than rejected. `include/transcribe.h` is explicit about what
must happen next: a coarser-or-equal request is answered at the requested
granularity, and only a finer one returns `ERR_UNSUPPORTED_TIMESTAMPS`. The
runtime therefore reports `returned_timestamp_kind == SEGMENT` and hands back a
single segment spanning the whole audio. That is the only honest answer
available — the model is always decoded with `<|notimestamps|>` and emits no
timestamp tokens to segment on — but callers should know the span is not a real
boundary. Long-form runs report the same single span; the 26 s-stride window
seams are decode geometry, not linguistic boundaries, and are deliberately not
surfaced as segments.

`AUTO` follows the same contract from the other end: it means "equal to the
model's `max_timestamp_kind`", so it resolves to `WORD` here and runs the
Viterbi aligner. That is why `transcribe-cli` (which defaults to `AUTO`) emits
word timings with no flags, and why `--batch-size` under `AUTO` falls back to
the serial path — the batched decode graph captures no cross-attention. Callers
that want batched throughput and no timings should pass `--timestamps none`,
which is what `scripts/wer/run.py` defaults to.

## Batching is WER-neutral, not byte-identical

`scripts/batch_parity.py` reports byte-identical output vs serial at batch
2/4/8 — but on a 32-utterance subset. **That result does not hold at scale**,
and citing it as "batch parity passes" would overstate what was measured.

Full LibriSpeech test-clean, BF16, batch 8 vs batch 1:

| Mode | batch 1 | batch 8 | b8 − b1 | b8 − reference | identical |
|---|---:|---:|---:|---:|---:|
| intended | 3.23030 | 3.23220 | +0.0019 pp | +0.0075 pp | 2605/2620 (99.43 %) |
| verbatim | 3.60560 | 3.61120 | +0.0056 pp | +0.0094 pp | 2609/2620 (99.58 %) |

Both stay inside the `+0.01 pp` gate against the oracle, and the b8−b1 delta is
inside the ~0.01 pp dataset-noise threshold, so the `Batch (offline)` row is
PASS on its stated criterion ("within 0.01 % of WER of batch size 1, over a
large set").

The 26 differing utterances are all the same near-tie class as the F32
divergences — punctuation and hyphenation (`Number 10, Fresh Nelly` vs
`Number 10 fresh Nelly`, `three wire` vs `three-wire`). The cause is expected:
the batched decoder runs `build_step_graph_batched`, a different graph topology
with a different reduction order, so marginal argmax decisions occasionally
flip. Stage 4's own guidance anticipates this ("the GPU flash-attention path is
expected to drift ~1e-3 and is not the Stage 4 regime").

Worth recording what this says about the parity tool: 32 utterances were not
enough to hit a single near-tie. A subset that small can only prove the batched
path is not *grossly* broken, not that it is bit-exact.

**Needs human sign-off** per Stage 4 Step 11 (batch 1 vs batch 8 is
human-reviewed).

## Numerical validation (Stage 4)

### The gate pairs an F32 GGUF with the F32 oracle, while the family ships BF16

This is the one non-obvious decision in the port, and it was made on
measurement rather than convention.

The Stage 2 oracle runs BF16 storage upcast to F32 compute
(`compute_type="float32"`). The dumper's original note claimed that regime "is
exactly what ggml does with a BF16 GGUF". **It is not.** ggml's BF16 path sets
`vec_dot_type = GGML_TYPE_BF16` with `from_float = ggml_cpu_fp32_to_bf16`, so it
rounds the F32 *activations* to BF16 before every dot product; torch at F32
upcasts the weights and leaves activations alone.

CrisperWhisper's encoder makes that difference loud, because Whisper's outlier
channels put `max|x|` around **800** where `p99_abs` is **4.3**. Measured on
`enc.block.6.out`, all three pairings, same graph:

| oracle compute | C++ GGUF | `enc.block.6.out` max_abs | `dec.embed_sum` |
|---|---|---:|---|
| F32 | **F32** | **1.66e-02** | **0.0 exact** |
| F32 | BF16 | 12.76 | 0.0 exact |
| BF16 | BF16 | 71.13 | 7.3e-04 |

BF16-on-both-sides is the **worst** pairing, not the matched one: torch at
`dtype=bfloat16` stores every activation in BF16 including the residual stream
and the embedding table, while ggml keeps the graph in F32 and rounds only
`mul_mat`'s src1. ggml is the higher-precision of the two, so moving the oracle
to BF16 moves it further away. It also destroys the exact `0.0` that
`dec.embed_sum` / `dec.token_emb` / `dec.pos_emb` / `enc.pos_emb` must hold,
because a BF16 oracle re-rounds the embedding table.

So the gate measures **graph correctness** (F32 oracle vs F32 GGUF, identical
weight values since BF16 → F32 is lossless), and the precision cost of the
shipped BF16 artifact is measured where precision cost belongs: the Step 11 WER
gate. The pairing is pinned in the golden manifest under `validation`
(`gguf_quant: F32`, `kv_type: f32`) so `validate.py` cannot silently pick the
BF16 file, whose default preference would otherwise win.

`scripts/convert-crisperwhisper.py --force-reference-dtype f32` builds the gate
GGUF. It is a diagnostic flag: the shipped artifact is always the detected
dtype.

### Result

**39/39 tensors within tolerance** across both cases, nothing provisional, and
**both transcripts byte-exact** including the `[UH]` event token and the
`from the from the` repetition.

Word timings match the reference aligner: **exact to 0.0 ms on all 23 words** of
the disfluency case, and 2.27 ms mean (one 50 ms boundary) on jfk.

Four entries were widened against the Stage 2 provisional budget
(`enc.block.{6,8,11}.out` 35x, `enc.final` 4.8x). That is **not** extra drift:
the provisional formula scales by `p99_abs`, which under-represents these
tensors by ~180x. Relative to actual peak magnitude the observed drift is
**1.9e-05** (`enc.final` 7.0e-05), tighter than the 1e-4 relative budget the
formula targets. Absolute drift is flat across blocks 6/8/11 (1.53e-02 each),
consistent with a fixed rounding difference riding the residual stream rather
than accumulation.

`dec.xattn.align` gates the word-timing path: head-averaged cross-attention,
one row per **predicted** token including the row that emits EOT (hence one row
longer than the generated text, matching the reference's attention pass). Being
post-softmax it agrees to ~6e-07.

`dec.intended.logits_raw` gates the mode contract via a second prompt pass in
the opposite mode, mirroring the reference dumper's sibling pass. It is what
proves the `[verbatim_N]`/`[intended_N]` tags are a real control surface rather
than decoration.

### Cross-attention capture in the shared component

`src/whisper_graph/decoder.{h,cpp}` grew one optional parameter, `const
AlignHeads * align_heads`, defaulting to null. Listing a `(layer, head)` pair
forces **that layer only** onto the manual `mul_mat` + `soft_max_ext` path,
because `ggml_flash_attn_ext` fuses the softmax and never materializes the
probabilities. That is also the path the reference runs
(`attn_implementation="eager"`), so a capture layer is closer to the oracle, not
further from it. The `whisper` family passes null and was re-verified
bit-identical (42/42 tensors, transcripts exact) after the change.

### `enc.mel.in` layout fix

The Stage 2 dumper wrote `enc.mel.in` mel-major `[80, 3000]` (HF's layout).
Whisper's convention, and what the encoder tensor actually holds
(ggml `ne=[n_mels, T]` means n_mels is innermost), is `[3000, 80]`. The dumper
now transposes. A mel-major buffer fed to the encoder does not error, it decodes
as silence: the model emits EOT immediately and the bracketed non-speech event
tokens rank just behind it. That is worth knowing as a debugging signature.

## Long-form `<ctx>` continuation (Stage 4)

Byte-exact with the reference on `samples/whole-earth.wav`: **206 words across
4 windows**, identical per-window word counts (85 / 50 / 64 / 7) and identical
continuation contexts.

### The `<ctx>` prompt carries space tokens the obvious construction omits

The reference builds ONE string and lets the HF tokenizer split it on added
tokens, so the spaces around the markers survive as their own tokens:

```
[verbatim_1..5]  ' '  <ctx>  ' was' ' sort' … ' Google'  ' '  <ectx>  <|sot|> <|en|> <|transcribe|> <|notimestamps|>
```

Our tokenizer does not split on added tokens, so the sequence is assembled by
hand — and emitting `<ctx>` directly after the tag block, without the space
token, makes the prompt two tokens shorter than the reference's.

That is not cosmetic. On `whole-earth.wav` the short prompt moved the window-1
decode **off** the reference's early-EOT stop: the C++ transcribed 73 words
where the reference transcribed 50, and recovered a clause
("On the back cover of their final issue was a photograph…") that the reference
drops. The fuller text was *more* faithful to the audio and *less* faithful to
the oracle — a parity bug wearing a flattering disguise. With the space tokens
restored, both implementations stop at the same place and the transcripts match
exactly.

### That dropped clause is expected, not a defect

The reference's window 1 ends early at a sentence-final pause because a
continuation context is present. That is precisely the pathology
`early_eot_recovery` exists to fix, and this port has it `OUT OF SCOPE`
(deferred, not blocked — it is deterministic and portable, see the capability
table). With recovery disabled on both sides, the clause falls between the
window-1 early stop and the window-2 start at 52 s and is lost. Reproducing
that loss is correct behaviour for this port; anything else would mean the
C++ is not running the same algorithm.

## Reference parity, measured (Stage 4 Step 11)

Every utterance of both acceptance sets, both modes, both dtypes: 5,738
comparisons per dtype, 0 errors. Hypotheses are diffed per utterance against
the Stage 2 reference run (`scripts/wer/compare_hyps.py`), not just scored,
because two runs can land on the same aggregate WER while individual
utterances differ.

| Set | Mode | dtype | identical | ref WER | C++ WER | delta |
|---|---|---|---:|---:|---:|---:|
| LibriSpeech test-clean (2620) | intended | F32 | 2615/2620 (99.81 %) | 3.22470 | 3.22650 | +0.0018 pp |
| LibriSpeech test-clean (2620) | intended | BF16 | 2598/2620 (99.16 %) | 3.22470 | 3.23030 | +0.0056 pp |
| LibriSpeech test-clean (2620) | verbatim | F32 | 2619/2620 (99.96 %) | 3.60180 | 3.60750 | +0.0057 pp |
| LibriSpeech test-clean (2620) | verbatim | BF16 | 2603/2620 (99.35 %) | 3.60180 | 3.60560 | +0.0038 pp |
| nyra-disfluency-en (249) | verbatim | F32 | **249/249 (100 %)** | 3.54410 | 3.54410 | 0.0000 pp |
| nyra-disfluency-en (249) | verbatim | BF16 | 244/249 (97.99 %) | 3.54410 | 3.50850 | -0.0356 pp |
| nyra-disfluency-en (249) | intended | F32 | 248/249 (99.60 %) | 9.77270 | 9.75210 | -0.0206 pp |
| nyra-disfluency-en (249) | intended | BF16 | 247/249 (99.20 %) | 9.77270 | 9.73140 | -0.0413 pp |

All eight clear the `+0.01 pp` gate; the worst positive delta is +0.0057 pp.

### What the 7 F32 divergences actually are

Out of 5,738 utterances, seven differ, and every one is a single decoding
decision flipping on a near-tie — exactly what ~2.5e-04 logit drift predicts:

| id | reference | C++ |
|---|---|---|
| `4077-13754-0004` | `…talked of. Why not…` | `…talked of, why not…` |
| `7127-75947-0022` | `…flatters me, whoever…` | `…flatters me. Whoever…` |
| `2300-131720-0029` | `Addison electrolytic meter` | `Edison electrolytic meter` |
| `1188-133604-0018` | `which are in many respects` | `which earn many respects` |
| `7127-75946-0003` | `whereupon Saint-Agnan` | `We're upon Saint-Agnan` |
| `7176-92135-0027` | `…suddenly ands toward him.` | `…suddenly and` (early EOT) |

Four are punctuation or homophone ties; one is an early EOT. None indicates a
different algorithm, and the ground truth for `7176-92135-0027` is
`AND TURNS TOWARDS HIM`, which neither side gets.

### BF16 storage costs no measurable accuracy

This is what the Stage 4 tolerance-regime decision was betting on, now
measured. The shipped BF16 GGUF carries **12.76 max_abs** drift at
`enc.block.6.out` against the oracle, ~1000x the F32 build. That buys a WER
delta of at most **+0.0056 pp**, and on the disfluency set BF16 scores
*better* than the reference. Gating tensors at F32 and the shipped artifact on
WER was the right split: the tensor gate stays tight enough to catch a real
regression, and the number that matters is measured where it matters.

## Notes

- Reuse `src/arch/whisper/` for the graph. The encoder, decoder blocks, KV
  cache, and mel frontend are the same; budget the work above the graph.
- The `whisper` family's `.en`-variant `bos_token_id` quirk does not apply here
  (these are all multilingual checkpoints with a standard `<|endoftext|>` at
  50257 / 50256 respectively).
- Variant order (decided at intake sign-off): **all four ship, `small` first.**
  `crisperwhisper-2.0-small` (241.8 M, ~480 MB BF16) is the cheapest variant to
  iterate on locally and shares the 80-mel / 51896-vocab layout with `medium`
  and `large`, so those two follow as pure size changes. `turbo` is last,
  because it is the only one exercising the 128-mel / +1-token-offset path.

## Intake sign-off decisions

Recorded 2026-08-07. These are the contract Stages 2 onward implement against.

| Decision | Outcome |
|---|---|
| Proceed given the non-commercial license? | Yes, **but publish no GGUF**. See [Licensing](#licensing). |
| Non-forced capability scope | All four `MUST PASS`: verbatim mode, intended mode, word timestamps, longform `<ctx>`. |
| Stage 7 WER gate | Report `intended` **and** `verbatim`; gate on `intended`. |
| Variant coverage | All four open variants. Order: `small`, `medium`, `large`, `turbo`. |

Stage 2 additions (2026-08-07):

| Decision | Outcome |
|---|---|
| Disfluency evaluation data | Ingest `nyralabs/disfluency_speech_english` (Apache-2.0) now; it unblocks the verbatim/intended `MUST PASS` rows that LibriSpeech cannot exercise. See [Disfluency evaluation set](#disfluency-evaluation-set). |
| Golden-manifest cases | Two: `jfk` (clean read speech) and `disfluency` (filler + cutoff + repetition). |
| Hallucination mitigation | Stays `OUT OF SCOPE`, but the **recorded reason is corrected**: not "makes parity untestable" (wrong), rather no working reference exists to port against. See [Hallucination repair](#hallucination-repair-deferred-with-evidence). Revisit at Stage 7. |
| Temperature fallback | Stays `OUT OF SCOPE`, reason upgraded from hand-wave to hard blocker: the reference samples, so parity needs an RNG-stream contract we do not have. |

Stage 3 additions (2026-08-08):

| Decision | Outcome |
|---|---|
| GGUF architecture key | Its own, `crisperwhisper` — not a `whisper` variant. The graph is shared but the decode contract is not, and this repo already gives every derived family its own arch dir (`canary_qwen`, `moonshine_streaming`, `voxtral_realtime`, `granite_nar`). Tensor names stay identical to `whisper`'s so the graph code is reusable. |
| `encoder_blank_head.{weight,bias}` | **Dropped**, on evidence it is untrained training scaffold. See [Conversion](#conversion-stage-3). |
| Long-form geometry in KV | Emitted as `stt.crisperwhisper.longform.*` (30 s / 26 s / 12 words / 2 words, pinned against `crisperwhisper==2.0.2`). These live in the package rather than the checkpoint, but the 4 s-overlap geometry is what the `<ctx>` prompt saw in training, so the GGUF carries them instead of leaving Stage 4 to hardcode magic numbers. |
| Gate B `frontend_config` WARN | **Accepted, both artifacts unchanged.** Two different vocabularies, not a disagreement. See [Preflight Gate B](#preflight-gate-b-one-expected-warn). |
