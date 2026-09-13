# HF card metadata: the `transcribe_cpp` block

Every GGUF repo we publish carries a machine-readable `transcribe_cpp` block in
its model-card frontmatter (`README.md`), so an external app can compare our
models on accuracy and speed without scraping prose tables or re-running
benchmarks. It lives in the card, not the GGUF files — weights are untouched.

The block publishes only raw measurements (per-quant WER, per-machine RTF) and
capability flags; any 0–100 score is left to the consumer to compute from these.

## Where it comes from

`scripts/hf_cards/generate.py` serializes the block from the catalog record
(`catalog/<variant>.json`): per-quant error rates from the headline benchmark
rows, realtime factors from the speed rows at the card's default quant, and
capability flags from the record's `capabilities` block. The editorial spec
(`scripts/hf_cards/<variant>.yaml`) adds only optional task-specific raw
measurements (`metrics:`) and secondary per-quant maps under `wer:`. A record
with no speed rows at the default quant emits no block.

## Fields

```yaml
transcribe_cpp:
  wer_librispeech_test_clean:               # raw %, per quant — lower is better
    f32: 1.68
    q8_0: 1.69
    q4_k_m: 1.72
  rtf_ryzen_4750u: { cpu: 8, vulkan: 15 }   # raw ×realtime — higher is better
  rtf_m4_max:      { cpu: 29, metal: 175 }
  cpwer_ami_ihm_test:                    # optional task metric, raw %
    bundle_f32_kernel: 19.35
  streaming: false
  diarize: false
  translate: false
  lang_detect: false
  timestamps: token                         # none | segment | word | token
```

| Field | Meaning |
| --- | --- |
| `wer_<dataset>` | Word error rate (%) per quant, on the named dataset. Lower is better. |
| `rtf_<machine>` | Speedup-over-realtime (×RT) per backend, mean over the published bench samples. Higher is better. |
| Task metrics from `metrics:` | Optional raw measurements emitted verbatim under their spec key, such as `cpwer_ami_ihm_test`. |
| `streaming` | Model supports buffered/cache-aware streaming. |
| `diarize` | Model can emit speaker-attributed transcript rows or speaker turns. |
| `translate` | Model can emit a translation (not just transcription). |
| `lang_detect` | Model auto-detects the input language (vs. requiring an explicit hint). |
| `timestamps` | Finest timestamp granularity the model emits (`none`/`segment`/`word`/`token`, mirroring the library's `max_timestamp_kind`). |

The `<dataset>` suffix is the spec's `wer.metadata_key` (default
`librispeech_test_clean`); the `<machine>` suffix is the `perf:` rig key with `-`
mapped to `_`. A spec can also publish secondary benchmarks inline: any dataset-named key
under `wer:` whose value is a `{quant: wer%}` map (e.g. `librispeech_test_clean:`)
is emitted as its own `wer_<dataset>` block alongside the headline one — so a
model whose per-quant column is e.g. FLEURS can still expose LibriSpeech
machine-readably. Add a dataset by adding a key; no wrapper needed. Other
raw task measurements live under the spec's `metrics:` map and are emitted
verbatim. Standard HF keys (`license`, `language`, `pipeline_tag`, `base_model`,
`tags`, …) are emitted alongside and unchanged by this block.

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
