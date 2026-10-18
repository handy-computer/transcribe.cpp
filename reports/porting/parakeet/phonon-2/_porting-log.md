# Phonon-2 conversion-first implementation

Local uncommitted work based on transcribe.cpp `90cd63d53b3103d137dcf463ec62338a66f56174`.
Scope was authorized by the user: conversion-first Parakeet variant, reference
validation, no commit, push or PR. No remote uploads were performed.

## Inputs and artifact

- Source: `FermionResearch/Phonon-2@7e153e4054c0a10db6d47d6fff5c062d4c77c154`.
- Base configuration/tokenizer: `nvidia/parakeet-tdt-0.6b-v3@541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
- Reference: independently executed, SHA256-checked publisher loader;
  Transformers 5.20.0, Torch 2.10.0, CPU F32. The original 5.4.0 environment
  lacked Parakeet TDT; the version was corrected before validation.
- Model: `models/phonon-2/phonon-2-F32.gguf`, 2,508,435,648 bytes,
  SHA256 `f2b1a5563f0a4a764ca40f4fb72da6d3a4d4ce5653e0f8fbcc874a6a81c5c10e`.
- 697 GGUF tensors / 627,052,166 parameters. All 723 source records match
  publisher reconstruction exactly, including restored convolution dimensions
  and BatchNorm counter handling. LSTM bias collapse and unused-counter
  removal use the existing converter path.
- Public-repository download/extraction conversion and local-bundle conversion
  produced byte-identical GGUFs. Archive and container checksums were verified.
- Conversion manifest: `reports/convert/phonon-2-F32.json` (local artifact;
  explicitly records dirty converter state rather than attributing the change
  to the base commit).

## Native implementation decision

First conversion loaded and produced the correct sample transcript, but
intermediate comparison exposed a boundary error: Kestrel preserves pointwise
bias at invalid subsampler frames; Transformers zeroes it. The initial
subsampler max error was 233.77, concentrated on the last frame. This was fixed
in metadata-controlled graph dispatch, not absorbed into tolerances.

New `encoder.length_masking=transformers` uses existing floor-derived valid
lengths and F32 pointwise compute, but masks after pointwise/ReLU. Since these
convolutions have temporal width 1, this reproduces Transformers' mask after
every convolution. Existing `none` and `kestrel` behavior is preserved. Global
TDT budget handling already exists and is reused. No new ggml type, public ABI,
external inference service or runtime Python dependency was added.

## Completed checks

| Check | Result |
| --- | --- |
| Intake schema | Valid |
| Preflight A | WARN: publisher top-level artifact config has no standalone frontend/dtype; pinned base relationship documented in intake |
| Preflight B | All five checks PASS |
| Independent container comparison | 723/723 exact names, shapes, dtypes and bytes |
| Container unit tests | 7 passed, including signed int6 tails, five-value row padding, scalar counters, malformed records and hash rejection |
| `validate.py all --family parakeet --variant phonon-2 --report` | 18/18 tensor gates and exact JFK transcript PASS |
| Final encoder drift on JFK | max_abs 1.863e-7; mean_abs 1.746e-8 |
| Paired behavior checks | 6/6 exact transcripts: JFK, 1.04 s speech prefix, silence 0.1/1/5 s, 35.332 s dots clip |
| Silence cases | Empty transcripts in both engines |
| Batch transcript checks | Sizes 2/4/8 match serial on four varied-length speech/silence inputs |
| Batch encoder, four identical JFK inputs | Bit-exact |
| Batch encoder, varied lengths | Strict zero-tolerance check fails; max_abs 1.285e-7. Explicit 1e-6 numerical check passes. No tolerance in the serial reference gate was widened. |
| Default CTest suite | 57 passed, 29 skipped because other real-model assets were not supplied; zero failures |
| Phonon real-model smoke | PASS, including metadata/shapes, abort, invalid input, recovery and repeated silence inference on the same session |
| Quant policy sync | 5/5 checks, 68 tensor names |
| Shipped catalog consistency | PASS; candidate records are kept out of the release catalog |
| Formatting and whitespace | Pinned clang-format and `git diff --check` pass |

The varied-length batch discrepancy occurs only on shorter, padded inputs;
the longest input and identical-length batches are exact. The native attention
graph's softmax/value reductions span the padded length, so changing batch
extent changes F32 reduction dimensions. This is consistent with floating-point
accumulation drift, but the strict bit-exact batch gate is **not** claimed green.
It remains an explicit qualification for any upstream acceptance review.

## Capability scope

| Capability | Target and evidence |
| --- | --- |
| English transcription, no hint | PASS: canonical tensor/transcript gate and six paired cases |
| English language hint | Native CLI smoke; same English-only metadata |
| Offline batching | Scoped CPU transcript parity PASS; tiny non-bit-exact encoder drift recorded above. Full test-clean Metal batch-1/batch-8 transcripts are identical. Full CPU batch-8 was skipped at the user's request. |
| Token/word timestamps | Existing Parakeet capability and sample output exercised; timing equivalence to publisher wrapper not claimed |
| Cancellation/session reuse | PASS in real-model smoke |
| Long audio | PASS for whole-clip 35.332 s reference comparison; publisher CLI segmentation and bounded-memory behavior are out of scope |
| Streaming/translation/language detection | Not advertised; out of scope |
| Packed compute/quantization | Out of scope for conversion-first candidate |

## Full test-clean and Metal follow-up (2026-10-16)

The user authorized full acceptance-set WER and Metal validation, retaining
the no-commit/no-push/no-PR boundary. All completed runs cover the same 2,620
LibriSpeech test-clean utterances, with zero inference errors. The new
`scripts/wer/run_reference_parakeet_phonon.py` produces scoreable hypotheses
through the independent publisher loader and stock Transformers generation.

- CPU serial matches all normalized reference transcripts; one raw transcript
  differs only by a final period.
- Metal serial and batch-8 have identical raw transcripts on the full corpus.
- Metal differs from the publisher in three raw transcripts. Two differences
  are punctuation/capitalization. The remaining difference is `Battle Axe`
  (publisher) versus `Battleax` (Metal); the corpus reference is `BATTLEAX`.
  Metal therefore has two fewer word errors across 53,029 reference words.
- The repository scorer and its unchanged English normalizer/bootstrap
  procedure are used for every output. The generated score table and exact
  per-utterance differences live in
  `reports/wer/phonon-2.librispeech-test-clean.summary.md` and
  `reports/wer/phonon-2.librispeech-test-clean.comparison.json`.
- Full CPU batch-8 was explicitly skipped at the user's request. Its queued
  command was prevented from starting. The already-completed Metal batch-8
  result is retained.

Metal ran on an Apple M5 with 16 GB unified memory, macOS 26.6.2 (25G83),
using a separate Release build with validation hooks and real-model tests.
The default suite passed (57 passed, 29 unrelated real-model skips), and the
Phonon real-model lifecycle smoke passed on Metal, including cancellation,
recovery and session reuse.

**Strict Metal tensor parity remains blocked:** only 7/18 JFK tensors pass
the existing CPU-reference budgets in either default or non-flash attention
mode. No tolerances were widened. Default attention final encoder max/mean
absolute errors are 9.484e-4 / 2.773e-5; non-flash errors are
9.366e-4 / 3.018e-5. Subsampler errors are 6.119 / 0.2367 in both modes.
The execution log confirms `kernel_mul_mm_f32_f32`; the vendored specialization
uses half operands with float accumulation. This establishes a reduced-precision
path consistent with the drift, without claiming to isolate every contributor.
Both modes produce the correct JFK transcript. WER acceptance and strict
tensor parity are separate findings; this is not a claim of general Metal
numerical equivalence or publication readiness.

Corpus archive SHA256:
`39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.
Manifest SHA256:
`1d66be0efebe55022362a70fc6f91346742b302bd0ea81ae7ad15c843106ddc1`.
Reference execution used CPU F32 and four Torch threads. Native runs used
default attention and thread settings, English forced, timestamps disabled,
and the existing explicit no-ITN recipe. Batch-8 was length-sorted. The
reference report's logging fields were expanded in the runner after it started;
its inference code and settings were unchanged. Native headers name the base
commit; the implementation under test was the uncommitted worktree described
here, not that clean commit alone.

Runs initially overlapped, then CPU serial was paused to reduce contention
while the reference completed. Its elapsed time and one utterance latency
include the pause. These timings are not controlled performance benchmarks.
No runtime C++ or numerical tolerance changes were made in this follow-up.
Raw logs are retained under
`build/validate/parakeet/phonon-2/evidence/wer-metal/`.

## Not completed or claimed

- Full CPU batch-8 WER (skipped at the user's request).
- Strict Metal tensor parity, Vulkan, CUDA or Windows/Linux execution.
- Publisher CLI segmentation, hotwording, streaming, or MLX/CoreML equivalence.
- Deployment quantization, release performance/memory claims, canonical upload
  or download roundtrip of a published GGUF.
- Human review of the code/tolerances or upstream merge/publication readiness.

The reference-magnitude tolerance budgets were retained without widening;
the pure blank-embedding lookup is exact. Full numerical reports remain under
`build/validate/parakeet/phonon-2/`, not in git. Publication catalog and HF card
drafts are local-only and excluded from this contribution; the release catalog
requires measured publication benchmark cells.

See `docs/models/phonon-2.md` for reproduction commands and license attribution.

## PR review follow-up: extraction and speech smoke coverage

Archive extraction now uses a unique temporary file per caller and atomically
publishes each completed member. Temporary files are removed on copy failure.
Two regression tests exercise overlapping first-time extractions of a small,
hash-verified archive and an injected partial-copy failure; all nine Python
container/extraction tests pass.

The existing decoder smoke now selects a variant-specific JFK golden for
Phonon-2, requiring an exact match to the pinned publisher reference. Parakeet
v2 retains its existing edit-distance budget. The shared test exercises result
accessors and timestamp modes. Its obsolete invalid-parameter assertion was
corrected: the current dispatcher and public API contract preserve the prior
successful result on pre-clear parameter rejection. No runtime behavior or
numerical tolerance was changed for this follow-up.

Validation:

- CPU and Metal builds of `transcribe-cli` and the decoder smoke succeeded.
- CPU: decoder and Parakeet lifecycle smokes passed (2/2).
- Metal: the same smokes passed (2/2); decoder reported `backend=MTL0` and
  transcript edit distance 0.
- Default CTest: 57 passed, 29 model-gated skips, no failures.
- Pinned clang-format 22.1.5 check passed for all 353 files; `git diff --check`
  passed.

The family note now records the completed WER comparison and its limitations.
Neither full WER nor CPU batch-8 was rerun. These short Metal smokes do not
resolve the previously recorded strict tensor-parity failures.
