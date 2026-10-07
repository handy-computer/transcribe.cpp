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

Re-render and re-upload whenever the catalog record changes (new WER sweep,
re-bench, capability fix). Repos stay private until a maintainer flips them.
