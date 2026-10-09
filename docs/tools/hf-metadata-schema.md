# HF card metadata: the `transcribe_cpp` block

Every GGUF repo we publish carries a machine-readable `transcribe_cpp` block in
its model-card frontmatter (`README.md`), so an external app can compare our
models on accuracy and speed without scraping prose tables or re-running
benchmarks. It lives in the card, not the GGUF files — weights are untouched.

The block publishes only raw measurements (per-quant WER, per-machine RTF,
or VAD feed-call latency) and capability flags; any
0–100 score is left to the consumer to compute from these.

## Where it comes from

`scripts/hf_cards/generate.py` serializes the block from the catalog record
(`catalog/<variant>.json`): one per-quant map for every accuracy result set the
record holds, realtime factors from the speed rows at the card's default quant,
and capability flags from the record's `capabilities` block. The editorial spec
(`scripts/hf_cards/<variant>.yaml`) contributes nothing numeric to it; its
`default_quant` selects which speed rows are emitted. A non-VAD record with no
speed rows at the default quant emits no block. VAD cards always emit a
role-aware schema 3 block, including when measurements are pending.

## Fields

```yaml
transcribe_cpp:
  schema_version: 2                         # bumped when key names or shapes change
  wer_librispeech_test_clean:               # raw %, per quant — lower is better
    f32: 1.68
    q8_0: 1.69
    q4_k_m: 1.72
  cer_fleurs_zh:                            # one map per result set
    q8_0: 8.10
  cpwer_ami_ihm_test_kernel:                # a decoding mode is its own set
    f32: 19.35
  rtf_ryzen_4750u: { cpu: 8.12, vulkan: 15.4 }   # raw ×realtime — higher is better
  rtf_m4_max:      { cpu: 29.05, metal: 175.2 }
  streaming: false
  diarize: false
  translate: false
  lang_detect: false
  timestamps: token                         # none | segment | word | token
```

| Field | Meaning |
| --- | --- |
| `schema_version` | 2. Version 1 cards (no field) carried one hand-named headline map; a CER or DER set could appear under a `wer_` key there. Consumers should read `.get()` and key on the metric prefix. |
| `<metric>_<dataset>_<split or language>[_<scoring>][_<mode>]` | Error rate (%) per quant on that result set. Lower is better. `wer`, `cer`, `der`, `cpwer` as the row's metric; FLEURS keys carry the language, other datasets the split; a scoring step (`opencc_t2s`) or decoding mode (`kernel`) makes a separate key. |
| `rtf_<machine>` | Speedup-over-realtime (×RT) per backend at the default quant, mean over the published bench samples. Higher is better. |
| `streaming` | Model supports buffered/cache-aware streaming. |
| `diarize` | Model can emit speaker-attributed transcript rows or speaker turns. |
| `translate` | Model can emit a translation (not just transcription). |
| `lang_detect` | Model auto-detects the input language (vs. requiring an explicit hint). |
| `timestamps` | Finest timestamp granularity the model emits (`none`/`segment`/`word`/`token`, mirroring the library's `max_timestamp_kind`). |

The `<machine>` suffix is the catalog machine slug with `-` mapped to `_`.
Standard HF keys (`license`, `language`, `pipeline_tag`, `base_model`, `tags`,
…) are emitted alongside. Empty language lists omit `language`. Non-HF
upstream sources omit `base_model` and `base_model_relation` rather than
misrepresenting a GitHub repo as an HF model ID.

## Schema 3: VAD

Only VAD cards currently use schema 3. Existing ASR, diarization and language-ID
cards retain their schema 2 metadata and rendered content unchanged. Schema 3
identifies the role explicitly and publishes optional latency measurements,
not quality measurements. It does not emit WER, detection accuracy, or a mean
realtime factor across incompatible feed durations. Numerical validation
remains part of the normal model-porting workflow.

The shape below is illustrative; actual values are copied from the catalog:

```yaml
transcribe_cpp:
  schema_version: 3
  role: vad
  params: 309633                            # exact parameter count
  vad_info:                                # optional, from GGUF loader metadata
    sample_rate: 16000
    frame_samples: 512
  latency_benchmarks: []                    # raw speed rows, default quant only
  streaming: true
  timestamps: none
```

| Field | Meaning |
| --- | --- |
| `role` | `vad`: probabilities and speech segments, not transcription. |
| `params` | Exact parameter count, not the rounded display summary. |
| `vad_info` | Optional `sample_rate` (Hz) and `frame_samples` (samples per native inference frame), read from the model artifact. |
| `latency_benchmarks` | Array of catalog speed rows at the card's default quant, preserving machine/backend/sample identity and provenance rather than averaging chunk sizes. An empty array means measurements pending. |

VAD latency rows retain the catalog's existing speed fields and add
`feed_samples`, `frame_samples`, `threads`, `n_calls`, `warmup_calls`,
`median_ms`, and `p95_ms`. For this role:

- `sample` identifies the feed size (`love-loss-32ms`, `love-loss-128ms`, or
  `love-loss-512ms`); `sample_duration_s` is audio **per chunk**, not the full clip.
- `total_ms` is mean native feed-call wall latency, including the Python
  ctypes/native API wall call but excluding model load and audio capture.
- `median_ms` and `p95_ms` measure feed-call wall latency in milliseconds,
  not per-frame latency and not START/END detection delay.
- `n_calls` is the measured call count; `warmup_calls` are excluded from the
  statistics. The audio is processed once with preserved stream state.
- `feed_samples / frame_samples` gives frames/feed. The displayed amortized
  median/frame is `median_ms / frames_per_feed`, not an independently
  measured single-frame statistic. The 32 ms feed is the headline.

The human-readable tables in HF cards and model documentation use shared
catalog rendering helpers and retain sub-millisecond precision. The streaming
table shows chunk duration, frames/feed, median/feed, p95/feed, amortized
median/frame and measured calls; frame samples, threads, warmup calls and
source sample identities appear in generated methodology prose. The mean
(`total_ms`) and all other raw fields remain in machine-readable metadata.
Source provenance links the packaged artifact and its SHA256 separately from the
upstream code commit. A draft with `published_repo: null` has no canonical
HF download URLs; generation alone does not publish a repository.

## Reading it

Use `.get()` throughout — every field is optional, and the whole block is absent
on un-migrated repos.

```python
from huggingface_hub import HfApi

card = HfApi().model_info("handy-computer/parakeet-tdt-0.6b-v2-gguf").card_data.to_dict()
tc = card.get("transcribe_cpp", {})
wer = tc.get("wer_librispeech_test_clean", {}).get("q8_0")   # 1.69  (lower is better)
rtf = tc.get("rtf_ryzen_4750u", {}).get("vulkan")            # 15    (higher is better)
```
