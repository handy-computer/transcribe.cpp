# HF cards

`<variant>.yaml` is prose only: summary, tags, validation pin, notes. Every
number, repo, licence, language, and capability comes from
`catalog/<variant>.json`; `generate.py` refuses a spec that states one.
`summary` and `wer.notes` are also the source for `docs/models/<variant>.md`,
rendered into its `catalog:intro` and `catalog:prose` markers by
`scripts/catalog/render.py`.

```bash
uv run scripts/hf_cards/check_release.py <variant>       # pin + validation date
uv run scripts/hf_cards/generate.py scripts/hf_cards/<variant>.yaml
                                                          # -> models/<variant>/README.md
hf upload handy-computer/<variant>-gguf models/<variant> . --repo-type model
```

Re-render and re-upload whenever the catalog record changes (new WER sweep,
re-bench, capability fix). Repos stay private until a maintainer flips them.
