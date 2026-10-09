# HF card metadata: the `transcribe_cpp` block

Every GGUF repo we publish carries a machine-readable `transcribe_cpp` block in
its model-card frontmatter (`README.md`), so an external app can compare our
models on accuracy and speed without scraping prose tables or re-running
benchmarks. It lives in the card, not the GGUF files — weights are untouched.

The block publishes only raw measurements (per-quant WER, per-machine RTF) and
capability flags; any 0–100 score is left to the consumer to compute from these.

## Where it comes from

`scripts/hf_cards/generate.py` serializes the block from the catalog record
(`catalog/<variant>.json`): one per-quant map for every accuracy result set the
record holds, realtime factors from the speed rows at the card's default quant,
and capability flags from the record's `capabilities` block. The editorial spec
(`scripts/hf_cards/<variant>.yaml`) contributes nothing to it. A record with no
speed rows at the default quant emits no block.

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
…) are emitted alongside and unchanged by this block.

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
