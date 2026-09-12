# `catalog/`

One file per shipped variant: `catalog/<variant>.json`. Everything we publish
about a model — what it is, what it supports, what it measured — and nothing
else.

```
catalog/
  _schema.json      the contract; every field has a description
  _format.py        the formatter (four rules, no judgement calls)
  <variant>.json    one record per shipped variant
```

## What belongs in a record

A key earns its place only if it is **published to a user**, **read by a gate**,
or **provenance for one of those**.

- **Bring-up detail stays out.** Dtype distributions, tokenizer summaries,
  forward maps, tolerance rationale, upstream's own benchmark claims, and what
  upstream *advertises* a model can do all live in `intake.json` and the family
  doc. A number published here is one we measured; a capability published here
  is one this port implements.
- **No editorial judgement.** Which quant or preset someone *should* pick is a
  recommendation, and recommendations are made elsewhere.
- **No permanently-null fields.** A field nothing populates is removed until
  something populates it. A field that is null because the harness does not
  stamp it yet — `engine_sha` on accuracy rows — stays, because that is a
  tracked gap rather than a dead column.
- **Full splits only.** A benchmark row is a complete, named, reproducible
  split. Subset runs are bring-up evidence; they can back a capability's
  `verified` flag but never publish a number here.
- **Every capability is listed**, supported or not, so the whole surface is
  visible at a glance and every model renders the same table rows.

Records are machine-assembled from artifacts, never hand-typed. The porting
skills own which stage writes what.

## Formatting

```bash
uv run catalog/_format.py catalog/*.json
```
