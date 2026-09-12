#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""build_db.py — assemble reports/wer/wer.db from the published artifacts.

A DERIVED INDEX, never a store: dropped and rebuilt from scratch on every
run, nothing hand-edited in it. The sources of truth stay where they are:

    reports/wer/*.score.json      scored cells from the FLEURS sweep
    docs/models/*.md              the published speed tables
    scripts/hf_cards/*.yaml       licence, quants, capabilities, family
    scripts/wer/languages.py      BCP-47 -> name + script
    scripts/wer/ingest.py         BCP-47 -> FLEURS config

The point is the joins. Accuracy lives in one place, speed in another,
licence and download size in a third, and no single file lets you ask
"which models cover these languages, under this error rate, above this
speed, under this size". One table does.

WHAT GOES IN. Only the current sweep: FLEURS, the batched timestamps-off
recipe, full splits. No reference or bring-up rows, no subsets, no
LibriSpeech, no numbers from older recipes. The recipe is therefore a
constant of the whole database and lives in `meta` rather than on every
row; when a second recipe matters, that is the moment to add a column.
Models are here because they have a result, and everything else is pruned
to them: a model with no measurement, a language nothing references, and a
quant of a model nobody scored are all absent by construction.

Speed comes from the per-model markdown docs rather than the `perf:` block
in the cards, because the docs keep the two axes the block averages away:
quant and sample. The card number is the mean of the doc's Q8_0 columns
(verified: whisper-large-v3 m4-max metal 23.6 is the mean of jfk 21.6 and
dots 25.7). Three models publish no table of their own and take a
sibling's, which is what `perf.measured_on` records.

Foreign keys are on and the NOT NULLs are meant: a hole is a build failure,
not a row of nulls for the site to work around.
"""
from __future__ import annotations

import ast
import glob
import json
import pathlib
import re
import sqlite3
import sys
from datetime import datetime, timezone

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
WER = REPO / "reports" / "wer"
DOCS = REPO / "docs" / "models"
CARDS = REPO / "scripts" / "hf_cards"
MODELS = REPO / "models"
DB = WER / "wer.db"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from languages import LANGUAGES                    # noqa: E402
from score_matrix import CELL_SCORE_ARGS           # noqa: E402
from gguf_header import identity                   # noqa: E402

SCHEMA = """
PRAGMA user_version = 4;

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE languages(
    lang   TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    script TEXT NOT NULL             -- ISO 15924: Latn, Cyrl, Hans ...
);

CREATE TABLE datasets(
    dataset TEXT PRIMARY KEY,        -- fleurs-de
    source  TEXT NOT NULL,           -- fleurs | librispeech | common-voice ...
    config  TEXT NOT NULL,           -- the source's own selector: de_de
    lang    TEXT NOT NULL REFERENCES languages(lang)
);

CREATE TABLE models(
    model           TEXT PRIMARY KEY,
    family          TEXT NOT NULL,   -- loader family from the GGUF header
    params_m        REAL NOT NULL,   -- millions, summed from tensor shapes
    license         TEXT,
    license_display TEXT,
    upstream_repo   TEXT,
    upstream_commit TEXT,
    streaming       INTEGER,
    translate       INTEGER,
    lang_detect     INTEGER,
    diarize         INTEGER,
    timestamps      TEXT             -- none | segment | word | token
);

CREATE TABLE model_languages(
    model TEXT REFERENCES models(model),
    lang  TEXT REFERENCES languages(lang),   -- the RAW code the card claims;
                                              -- see language_aliases to join
                                              -- it against measured results
    PRIMARY KEY(model, lang)
);

-- FLEURS_LANGS (scripts/wer/ingest.py) is many-to-one: tl/fil both mean
-- fil_ph, no/nb both mean nb_no, zh/zh-cn both mean cmn_hans_cn. The sweep
-- measures and names ONE spelling per config (fleurs_full_matrix.py's
-- canonical_codes()); a model's card can claim any of the others (Whisper
-- says tl/no, Fun-ASR says tl, Nemotron says nb). Without this table a join
-- from model_languages to datasets/results on lang silently drops those
-- models for that language. Derived from FLEURS_LANGS, not hand-maintained:
-- see language_alias_pairs() below. Nynorsk (nn) is deliberately absent --
-- it has no FLEURS config to alias to, and that absence is the honest
-- answer ("no benchmark exists"), not a bug to paper over.
CREATE TABLE language_aliases(
    alias     TEXT PRIMARY KEY REFERENCES languages(lang),
    canonical TEXT NOT NULL REFERENCES languages(lang)
);

-- The join surface most callers actually want: a model's claimed languages,
-- resolved to the spelling datasets/results use. model_languages itself
-- stays raw so the card's original claim is never lost.
CREATE VIEW model_languages_canonical AS
SELECT model, COALESCE(a.canonical, ml.lang) AS lang
FROM model_languages ml
LEFT JOIN language_aliases a ON a.alias = ml.lang;

CREATE TABLE quants(
    model    TEXT REFERENCES models(model),
    quant    TEXT,
    filename TEXT NOT NULL,
    size_gb  REAL NOT NULL,
    PRIMARY KEY(model, quant)
);

CREATE TABLE results(
    dataset TEXT REFERENCES datasets(dataset),
    model   TEXT REFERENCES models(model),
    quant   TEXT NOT NULL,
    metric  TEXT NOT NULL CHECK(metric IN ('wer','cer')),
    err_pct REAL NOT NULL CHECK(err_pct >= 0),
    ci_lo   REAL,
    ci_hi   REAL,
    n_utts  INTEGER NOT NULL CHECK(n_utts > 0),
    note    TEXT,
    PRIMARY KEY(dataset, model, quant)
);

CREATE TABLE rigs(
    rig     TEXT PRIMARY KEY,        -- m4-max, ryzen-4750u, m4
    display TEXT NOT NULL
);

CREATE TABLE perf(
    model       TEXT REFERENCES models(model),
    rig         TEXT REFERENCES rigs(rig),
    backend     TEXT NOT NULL,       -- metal | vulkan | cpu
    quant       TEXT NOT NULL,
    sample      TEXT NOT NULL,       -- jfk | dots | ru
    sample_s    REAL NOT NULL,       -- so an app can weight or average
    xrt         REAL NOT NULL,       -- x realtime over mel+encode+decode
    -- The checkpoint the number was actually measured on. Usually the
    -- model itself; for a fine-tune that publishes no table of its own it
    -- is the sibling, which may not be a row here (Breeze carries
    -- whisper-large-v2's numbers and whisper-large-v2 has no FLEURS
    -- result). Deliberately not a foreign key for that reason.
    measured_on TEXT NOT NULL,
    PRIMARY KEY(model, rig, backend, quant, sample)
);
"""

# --- what the sweep is -------------------------------------------------

SOURCE = "fleurs"
VARIANT = "b8"            # the filename tag for the batched sweep
BATCH_SIZE = 8
TIMESTAMPS = "none"

# Dtype presets that are real shipped files. REF (a framework reference run)
# and CPP (an early bring-up run) occupy the same slot in a filename without
# being shipped quants, and are not sweep rows.
#
# Results are restricted to Q8_0 and Q5_K_M FOR NOW: the 2026-09-02 full
# FLEURS sweep only generated those two, so every other preset (F32, BF16,
# F16, Q6_K, Q4_K_M) is leftover from an earlier, narrower quant probe --
# 8 languages at most, sometimes 1 -- on a different engine build. Mixing
# those into the same results table reads as a quant comparison it isn't:
# voxtral-realtime's Q6_K/Q4_K_M rows were still on the pre-patch offline
# delay while its Q8_0/Q5_K_M rows were the delay-30 re-run, so the "quant
# cost" was actually a decode-setting difference. Widen this once the other
# presets get the same full-matrix treatment.
QUANTS = ("Q8_0", "Q5_K_M")
EXCLUDED_QUANTS = ("REF", "CPP")

NAME = re.compile(
    rf"^(?P<model>.+?)-(?P<quant>{'|'.join(QUANTS + EXCLUDED_QUANTS)})"
    rf"(?P<mode>(?:-[A-Za-z0-9]+)*)\.(?P<rest>.+)\.score\.json$")

# --- the published speed tables ----------------------------------------

# Heading in docs/models/*.md -> rig id. The 4750U is spelled three ways
# across the docs; they are one machine.
RIG_HEADINGS = {
    "apple m4 max": "m4-max",
    "apple m4": "m4",
    "amd ryzen 7 pro 4750u": "ryzen-4750u",
    "amd ryzen 7 4750u pro": "ryzen-4750u",
    "amd ryzen 7 pro 4750u (vega 8 igpu)": "ryzen-4750u",
}
RIG_DISPLAY = {"m4-max": "Apple M4 Max", "m4": "Apple M4",
               "ryzen-4750u": "AMD Ryzen 7 PRO 4750U"}

# Models with no speed table of their own, and the sibling whose numbers
# they carry. Both are fine-tunes of the named checkpoint at identical
# shape, and both already publish exactly its figures in their own card;
# recording it here makes that visible in the data instead of implied.
# (Breeze is handled without an entry: its card's transcribe_docs_url
# already points at whisper-large-v2.md.)
PERF_INHERITS = {
    "parakeet-primeline": "parakeet-tdt-0.6b-v3",
    "cohere-transcribe-arabic-07-2026": "cohere-transcribe-03-2026",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fleurs_map() -> dict[str, str]:
    src = (REPO / "scripts/wer/ingest.py").read_text()
    body = re.search(r"FLEURS_LANGS: dict\[str, str\] = \{(.*?)\n\}", src, re.S).group(1)
    return ast.literal_eval("{" + re.sub(r"#.*", "", body) + "}")


def language_alias_pairs(fleurs: dict[str, str], measured: set[str],
                         claimed: set[str]) -> list[tuple[str, str]]:
    """(alias, canonical) for every code whose FLEURS config was measured
    under a DIFFERENT spelling than a card claims it under.

    FLEURS_LANGS is many-to-one (fil/tl -> fil_ph, nb/no -> nb_no,
    zh/zh-cn -> cmn_hans_cn) and the sweep names one spelling per config, so
    grouping fleurs by config and picking whichever code is actually in
    `measured` gives the same canonical spelling the datasets/results tables
    already use -- no second, hand-typed copy of the tl/no/zh-cn decisions.

    A pair is only emitted when the alias is actually claimed by some
    surviving card and the canonical spelling was actually measured;
    aliasing a spelling nothing references or nothing was measured for
    would add a row with no query it helps."""
    by_cfg: dict[str, list[str]] = {}
    for code, cfg in fleurs.items():
        by_cfg.setdefault(cfg, []).append(code)
    pairs = []
    for codes in by_cfg.values():
        canonical = next((c for c in codes if c in measured), None)
        if canonical is None:
            continue
        pairs += [(c, canonical) for c in codes if c != canonical and c in claimed]
    return pairs


def size_to_gb(s) -> float | None:
    if not isinstance(s, str):
        return None
    m = re.match(r"([\d.]+)\s*(GB|MB)", s.strip(), re.I)
    if not m:
        return None
    v = float(m.group(1))
    return round(v / 1024, 4) if m.group(2).upper() == "MB" else v


def params_m(v) -> float | None:
    if not isinstance(v, str):
        return None
    m = re.match(r"([\d.]+)\s*([MB])", v.strip(), re.I)
    if not m:
        return None
    n = float(m.group(1))
    return round(n * 1000, 1) if m.group(2).upper() == "B" else n


def read_cell(d: dict) -> tuple[str, float, float, float, int] | None:
    """(metric, err_pct, ci_lo, ci_hi, n) from a score.json, either shape.

    score.py grew the metric-neutral keys when CER routing landed; older
    artifacts carry only wer/wer_pct."""
    if d.get("metric") in ("wer", "cer") and d.get("error_rate_pct") is not None:
        return (d["metric"], d["error_rate_pct"],
                round(d.get("error_rate_ci_lo", 0) * 100, 2),
                round(d.get("error_rate_ci_hi", 0) * 100, 2), d["n"])
    for legacy in ("wer", "cer"):
        if d.get(f"{legacy}_pct") is not None:
            return (legacy, d[f"{legacy}_pct"],
                    round(d.get(f"{legacy}_ci_lo", 0) * 100, 2),
                    round(d.get(f"{legacy}_ci_hi", 0) * 100, 2), d["n"])
    return None


def sweep_cells() -> list[dict]:
    """Every score.json that belongs to the sweep, parsed.

    A file qualifies on four counts: the dataset is FLEURS in a language we
    can name, the filename carries the sweep's recipe tag, the quant is a
    shipped preset, and the cell covers the full split. The last is checked
    after the fact, since "full" is the largest n seen for that language."""
    out = []
    for p in sorted(WER.glob("*.score.json")):
        m = NAME.match(p.name)
        if not m or m.group("quant") in EXCLUDED_QUANTS:
            continue
        parts = m.group("rest").split(".")
        dataset, tags = parts[0], parts[1:]
        if tags != [VARIANT] or not dataset.startswith(f"{SOURCE}-"):
            continue
        lang = dataset[len(SOURCE) + 1:]
        if lang not in LANGUAGES:
            continue                      # subset ids like fleurs-ru-508
        cell = read_cell(json.loads(p.read_text()))
        if cell is None:
            continue
        metric, err, lo, hi, n = cell
        out.append({"dataset": dataset, "lang": lang, "model": m.group("model"),
                    "quant": m.group("quant"), "metric": metric, "err": err,
                    "lo": lo, "hi": hi, "n": n})
    full = {}
    for c in out:
        full[c["dataset"]] = max(full.get(c["dataset"], 0), c["n"])
    return [c for c in out if c["n"] == full[c["dataset"]]]


def doc_perf(doc: pathlib.Path) -> list[tuple]:
    """(rig, backend, quant, sample, sample_s, xrt) from a doc's tables.

    Two table shapes appear across docs/models/*.md:
      A) `| Backend | Sample | Q... | Q... |` -- backend is a column, one
         table covers every backend for that rig.
      B) a `**Backend**` line followed by `| Sample | Q... | Q... |` -- the
         AMD 4750U tables use this shape (one table per backend, since the
         quant set sometimes differs, e.g. Vulkan-only wide vs CPU-only).
    Sample cells are `name (11.0s)` or `name (11.0 s)` -- both appear."""
    t = doc.read_text()
    section = re.search(r"^## Performance\s*$(.*?)(?=^## |\Z)", t, re.M | re.S)
    if not section:
        return []
    rows, rig, quants, backend = [], None, [], None
    for line in section.group(1).splitlines():
        head = re.match(r"^### (.+?)\s*$", line)
        if head:
            rig = RIG_HEADINGS.get(head.group(1).strip().lower())
            backend, quants = None, []
            continue
        bold = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        if bold:
            backend, quants = bold.group(1).split()[0].lower(), []
            continue
        cols = re.match(r"^\|\s*Backend\s*\|\s*Sample\s*\|(.+)\|\s*$", line)
        if cols:
            quants, backend = [c.strip() for c in cols.group(1).split("|") if c.strip()], None
            continue
        cols_b = re.match(r"^\|\s*Sample\s*\|(.+)\|\s*$", line)
        if cols_b and backend:
            quants = [c.strip() for c in cols_b.group(1).split("|") if c.strip()]
            continue
        if not rig or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if backend is not None:
            sample_cell, value_cells, row_backend = cells[0], cells[1:], backend
        else:
            if len(cells) < 3:
                continue
            sample_cell, value_cells, row_backend = cells[1], cells[2:], cells[0].lower()
        sm = re.match(r"([\w.-]+)\s*\(([\d.]+)\s*s\)", sample_cell)
        if not sm:
            continue
        for quant, cell in zip(quants, value_cells):
            x = re.search(r"\(([\d.]+)\s*[x×]\)", cell)
            if x:
                rows.append((rig, row_backend, quant, sm.group(1),
                             float(sm.group(2)), float(x.group(1))))
    return rows


def main() -> int:
    fleurs = fleurs_map()
    cards = {norm(pathlib.Path(p).stem): (pathlib.Path(p), yaml.safe_load(open(p).read()) or {})
             for p in glob.glob(str(CARDS / "*.yaml"))}

    cells = sweep_cells()
    if not cells:
        print("no sweep cells found", file=sys.stderr)
        return 1
    models = sorted({c["model"] for c in cells})
    langs = sorted({c["lang"] for c in cells})

    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)

    # Languages: the measured ones plus every code the surviving cards
    # claim, so a coverage question has a name to render.
    wanted = set(langs)
    for m in models:
        _, c = cards.get(norm(m), (None, {}))
        wanted |= {str(l) for l in (c.get("languages") or []) if str(l) in LANGUAGES}
    con.executemany("INSERT INTO languages VALUES (?,?,?)",
                    [(l, LANGUAGES[l][0], LANGUAGES[l][1]) for l in sorted(wanted)])

    alias_pairs = language_alias_pairs(fleurs, measured=set(langs), claimed=wanted)
    con.executemany("INSERT INTO language_aliases VALUES (?,?)", sorted(alias_pairs))

    con.executemany("INSERT INTO datasets VALUES (?,?,?,?)",
                    [(f"{SOURCE}-{l}", SOURCE, fleurs[l], l) for l in langs])

    missing_card = []
    for m in models:
        path, c = cards.get(norm(m), (None, {}))
        family, pm = c.get("family"), params_m(c.get("params"))
        if not family or pm is None:
            # No card, or a card without identity. The GGUF is the source
            # for both either way, so read it directly rather than dropping
            # measured cells over missing paperwork.
            missing_card.append(m)
            d = {norm(p.name): p for p in MODELS.glob("*") if p.is_dir()}.get(norm(m))
            files = sorted(d.glob("*.gguf"), key=lambda p: p.stat().st_size) if d else []
            if not files:
                print(f"error: {m} has no card identity and no local GGUF; "
                      f"run scripts/hf_cards/backfill_identity.py", file=sys.stderr)
                return 1
            info = identity(files[0])
            family, pm = info["family"], round(info["params"] / 1e6, 1)
        caps = c.get("capabilities") or {}
        con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (m, family, pm, c.get("license"), c.get("license_display"),
                     c.get("hf_repo"), c.get("upstream_commit"),
                     int(bool(caps.get("streaming"))), int(bool(caps.get("translate"))),
                     int(bool(caps.get("lang_detect"))), int(bool(caps.get("diarize"))),
                     caps.get("timestamps")))
        for lg in (c.get("languages") or []):
            if str(lg) in wanted:
                con.execute("INSERT OR REPLACE INTO model_languages VALUES (?,?)",
                            (m, str(lg)))
        for q in (c.get("quants") or []):
            gb = size_to_gb(q.get("size"))
            if q.get("name") and q.get("filename") and gb is not None:
                con.execute("INSERT OR REPLACE INTO quants VALUES (?,?,?,?)",
                            (m, q["name"], q["filename"], gb))

    con.executemany("INSERT INTO rigs VALUES (?,?)", sorted(RIG_DISPLAY.items()))

    # Speed, from each model's published doc. A model with no table of its
    # own carries a sibling's numbers, named in measured_on.
    # A doc benches the model it is named after, which is not always the
    # model reading it: breeze-asr-25's card points at whisper-large-v2.md.
    # Built over every card, not just the sweep, so the attribution stays
    # right even when the sibling has no result of its own.
    doc_owner = {}                     # doc filename -> the model it benches
    for key, (path, c) in cards.items():
        doc = (c.get("transcribe_docs_url") or "").rsplit("/", 1)[-1]
        if doc and norm(pathlib.Path(doc).stem) == norm(path.stem):
            doc_owner[doc] = next((m for m in models if norm(m) == key), path.stem)
    n_perf = 0
    for m in models:
        source_model = PERF_INHERITS.get(m, m)
        path, c = cards.get(norm(source_model), (None, {}))
        if not c.get("transcribe_docs_url"):
            continue
        doc = c["transcribe_docs_url"].rsplit("/", 1)[-1]
        rows = doc_perf(DOCS / doc)
        measured_on = doc_owner.get(doc, source_model)
        for rig, backend, quant, sample, sample_s, xrt in rows:
            con.execute("INSERT OR REPLACE INTO perf VALUES (?,?,?,?,?,?,?,?)",
                        (m, rig, backend, quant, sample, sample_s, xrt,
                         measured_on))
            n_perf += 1

    con.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                    [(c["dataset"], c["model"], c["quant"], c["metric"], c["err"],
                      c["lo"], c["hi"], c["n"],
                      CELL_SCORE_ARGS.get((c["model"], c["lang"]), (None, None))[1])
                     for c in cells])

    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("generated", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("rebuild", "uv run scripts/wer/build_db.py (drops and recreates; "
                    "never hand-edit)"),
        ("dataset_scope", "google/fleurs test split, full splits only"),
        ("recipe_batch_size", str(BATCH_SIZE)),
        ("recipe_timestamps", TIMESTAMPS),
        ("perf_metric", "x realtime over mel+encode+decode, from the tables "
                        "in docs/models/*.md"),
    ])
    con.commit()

    for t in ("languages", "language_aliases", "datasets", "models",
              "model_languages", "quants", "results", "rigs", "perf"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:16s} {n:>5}")
    no_perf = [r[0] for r in con.execute(
        "SELECT model FROM models WHERE model NOT IN (SELECT model FROM perf)")]
    inherited = con.execute(
        "SELECT count(DISTINCT model) FROM perf WHERE measured_on <> model").fetchone()[0]
    print(f"\n{inherited} model(s) carry a sibling's speed numbers")
    if missing_card:
        print(f"no hf_card, identity read from the GGUF: {', '.join(missing_card)}")
    if no_perf:
        print(f"no speed numbers: {', '.join(no_perf)}")
    print(f"\n{DB}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
