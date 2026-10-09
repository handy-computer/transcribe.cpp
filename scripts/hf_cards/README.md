# HF cards

`<variant>.yaml` is prose only: summary, tags, validation pin, notes. Every
number, repo, licence, language, and capability comes from
`catalog/<variant>.json`, as do the metric column label and the link back to
the model page. `generate.py` accepts only the editorial keys and refuses
everything else, so a catalog-owned field cannot creep back in and a
misspelled key fails instead of silently rendering nothing.
`summary` and `wer.notes` are also the source for `docs/models/<variant>.md`,
rendered into its `catalog:intro` and `catalog:prose` markers by
`scripts/catalog/render.py`. The mechanical WER sentence (dataset, size,
batch, timestamps, build) is generated from the headline rows; `wer.notes`
holds only editorial caveats: state a number there only to compare against
something the catalog does not hold, such as an upstream self-reported
figure. A second download-table column is
`wer.source2` plus `wer.secondary: <metric>_<dataset>_<split|lang>` naming a
result set the catalog holds.

```bash
uv run scripts/hf_cards/check_release.py <variant>       # pin + validation date
uv run scripts/hf_cards/generate.py scripts/hf_cards/<variant>.yaml
                                                          # -> models/<variant>/README.md
hf upload handy-computer/<variant>-gguf models/<variant> . --repo-type model
```

Re-render whenever the catalog record changes (new WER sweep, re-bench,
capability fix). Canonical uploads and publication are separate maintainer
steps; generating a card does not upload it. Repos stay private until a
maintainer flips them.

## VAD and non-HF sources

For `role: vad`, the card renders reference parity and streaming feed-call
latency, not a fabricated WER column. These tables use the same stdlib helpers
in `scripts/catalog/common.py` as the documentation markers
`catalog:reference-parity` and `catalog:stream-perf machine=m4`. Parity is
agreement with a reference segmenter, **not** detection accuracy against
human annotations. The 32 ms feed is the latency headline; larger feeds show
amortization. See `docs/tools/hf-metadata-schema.md` for the role-aware schema 3
metadata; existing ASR, diarization and language-ID cards retain schema 2.

An explicit non-HF `upstream_url` (for example GitHub) is linked directly and
is never fetched through the HF API. `source_artifact` identifies the actual
packaged weights, separately from the upstream code commit. VAD parameter
count and optional `vad_info` sample rate/frame geometry come from the catalog,
not editorial prose. Non-HF cards omit `base_model` / `base_model_relation`;
empty language lists omit `language`. Their default output is
`models/<variant>/README.md`; HF sources retain `models/<upstream-slug>/README.md`.

`published_repo: null` is supported for a local draft: the download table
shows filenames, not nonexistent canonical URLs, and explicitly states that
publication is pending. No original HF model card is claimed for non-HF
sources. The draft can be generated without a network fetch:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/silero-vad-v6.2.yaml
```

Optional editorial fields:

- `compatibility: |` describes alternate loader inputs, not canonical downloads.
  It also renders into documentation with `catalog:prose field=compatibility`.
- `reference_parity: {notes: ...}` and `stream_perf: {notes: ...}` add caveats
  below their generated tables; only `notes` is accepted, never measurements.
- Existing `wer.notes` can hold VAD validation caveats. It is rendered in the
  reference-parity section without printing a WER heading.

Verification (offline):

```bash
uv run --no-project --with pytest --with pyyaml --with jinja2 \
  --with huggingface-hub pytest scripts/catalog/test_vad_render.py
```
