#!/usr/bin/env python3
"""Canonical formatter for the catalog records. Four rules, no judgement calls:

  1. An element of a ROW array (downloads, accuracy_benchmarks, speed_benchmarks,
     and a streaming capability's presets) is exactly one line. A row is a table
     row; expanding it across eighteen lines is what makes a 164-cell sweep
     unreadable.
  2. A member of a ROW object (capabilities) is one line, unless it carries a
     table of its own -- then it expands and its table follows rule 1.
  3. An array of scalars wraps at 78 columns.
  4. Everything else is fully expanded, one member per line, like `jq --indent 2`.

Rules 1 and 2 key off names that only carry this meaning at the top level of a
record, so they are applied ONLY to the record's own sections -- never inside
_schema.json, where the same words are subschema keys.

    uv run catalog/_format.py catalog/*.json
"""
import json, pathlib, sys

WRAP = 78
ROW_ARRAYS = {"downloads", "accuracy_benchmarks", "speed_benchmarks"}
ROW_OBJECTS = {"capabilities"}
NESTED_ROW_ARRAYS = {"presets"}


def compact(o):
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def has_table(v):
    return any(isinstance(x, list) and any(isinstance(y, dict) for y in x)
               for x in v.values()) if isinstance(v, dict) else False


def fmt(o, ind=0, *, row=False, row_object=False, top=False):
    pad = " " * ind
    if row and not isinstance(o, list):
        return compact(o)
    if isinstance(o, list):
        if not o:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in o):
            c = compact(o)
            if len(c) + ind <= WRAP:
                return c
            lines, cur = [], pad + "  "
            for i, x in enumerate(o):
                add = json.dumps(x, ensure_ascii=False) + ("," if i < len(o) - 1 else "")
                if len(cur) + len(add) + 1 > WRAP and cur.strip():
                    lines.append(cur.rstrip())
                    cur = pad + "  "
                cur += add + " "
            lines.append(cur.rstrip())
            return "[\n" + "\n".join(lines) + "\n" + pad + "]"
        items = [pad + "  " + fmt(x, ind + 2, row=row) for x in o]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = []
        for k, v in o.items():
            key = json.dumps(k, ensure_ascii=False)
            if row_object and isinstance(v, dict) and not has_table(v):
                items.append(f"{pad}  {key}: {compact(v)}")
            else:
                items.append(f"{pad}  {key}: " + fmt(
                    v, ind + 2,
                    row=(top and k in ROW_ARRAYS) or k in NESTED_ROW_ARRAYS,
                    row_object=top and k in ROW_OBJECTS))
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    return compact(o)


for p in map(pathlib.Path, sys.argv[1:]):
    p.write_text(fmt(json.loads(p.read_text()), top=True) + "\n")
    print(f"{p}  {len(p.read_text().splitlines()):>4} lines")
