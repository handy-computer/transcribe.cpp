#!/usr/bin/env python3
"""Audit GGUF metadata for the header-query contract.

For each GGUF it reports the four things that decide whether a published file
is "online-queryable" and self-describing:

  identity   — general.* identity block complete? (name, size_label, author,
               organization, license{,.name,.link}, languages)
  trailer    — are the bulk tokenizer arrays (tokens/scores/token_type/merges/
               chat_template) the LAST KVs, so a header range-read reaches the
               small metadata without pulling them? (the #56 gguf_writer layout)
  file_type  — does general.file_type sit BEFORE the tokenizer trailer? (the
               quantizer used to strand it dead-last)
  streaming  — is stt.capability.streaming present, and does it agree with what
               the slug implies (streaming / unified / nemotron / realtime)?

Usage
  # local files / dirs / globs (default: models/**/*.gguf under the repo)
  uv run --project scripts/envs/moonshine scripts/audit_gguf_metadata.py [PATH ...]

  # a published HF repo — range-fetches only the header of each .gguf
  uv run --project scripts/envs/moonshine scripts/audit_gguf_metadata.py \
      --hf-repo handy-computer/parakeet-unified-en-0.6b-gguf

Exit code is non-zero when any file has an issue, so it can gate a re-export.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import BULK_KV_KEYS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# general.* keys we expect every shipped GGUF to carry. CORE is the subset
# whose absence is a hard failure; the rest are reported but advisory.
IDENTITY_KEYS = [
    "general.name", "general.size_label", "general.author",
    "general.organization", "general.license", "general.license.name",
    "general.license.link", "general.languages",
]
IDENTITY_CORE = {"general.name", "general.size_label", "general.author",
                 "general.license", "general.languages"}

# Slug tokens that imply the model is a streaming variant.
STREAMING_HINTS = ("streaming", "unified", "nemotron", "realtime")


def field_value(field):
    if field is None or not field.types:
        return None
    t = field.types[0]
    if t == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
    if t == GGUFValueType.ARRAY:
        return "<array>"
    return field.parts[field.data[0]][0]


def audit_reader(reader: GGUFReader, label: str) -> dict:
    keys = list(reader.fields.keys())
    kset = set(keys)

    missing = [k for k in IDENTITY_KEYS if k not in kset]
    identity_fail = bool(IDENTITY_CORE - kset)

    present_bulk = [k for k in BULK_KV_KEYS if k in kset]
    if present_bulk:
        first_bulk = min(keys.index(k) for k in present_bulk)
        tail = keys[-len(present_bulk):]
        trailer_ok = tail == present_bulk
    else:
        first_bulk = len(keys)
        trailer_ok = True  # no tokenizer tables → nothing to trailer

    if "general.file_type" not in kset:
        ft_status = "absent"
    elif not present_bulk:
        ft_status = "no-trailer"
    elif keys.index("general.file_type") < first_bulk:
        ft_status = "ok"
    else:
        ft_status = "AFTER-TRAILER"

    ident = (field_value(reader.fields.get("stt.variant"))
             or field_value(reader.fields.get("general.basename"))
             or label)
    looks_streaming = any(h in str(ident).lower() for h in STREAMING_HINTS)
    sval = field_value(reader.fields.get("stt.capability.streaming"))
    streaming = "ABSENT" if sval is None else bool(sval)
    if looks_streaming and streaming is not True:
        streaming_suspect = f"slug implies streaming but flag={streaming}"
    elif (not looks_streaming) and streaming is True:
        streaming_suspect = "flag=true but slug implies offline"
    else:
        streaming_suspect = None

    issues = []
    if identity_fail:
        issues.append("identity")
    if not trailer_ok:
        issues.append("trailer")
    if ft_status == "AFTER-TRAILER":
        issues.append("file_type")
    if streaming_suspect:
        issues.append("streaming")

    return {
        "label": label, "missing": missing, "identity_fail": identity_fail,
        "trailer_ok": trailer_ok, "file_type": ft_status,
        "streaming": streaming, "streaming_suspect": streaming_suspect,
        "issues": issues,
    }


def audit_local(path: Path) -> dict:
    return audit_reader(GGUFReader(str(path)), path.name)


def audit_hf(repo: str, filename: str, prefix_mb: int) -> dict:
    """Range-fetch only the header of an HF-hosted GGUF, then audit it.

    GGUFReader memmaps the whole declared file, so we fetch the first
    `prefix_mb` MB (enough for any of our tokenizer tables) and sparse-pad the
    temp file to the real size; tensor-data views land in the zero region and
    are never read."""
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(repo, filename)
    total = get_hf_file_metadata(url).size
    prefix = min(prefix_mb * 1024 * 1024, total)
    headers = build_hf_headers()
    headers["Range"] = f"bytes=0-{prefix - 1}"
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
        tmp = Path(tf.name)
        tf.write(resp.content)
        tf.truncate(total)  # sparse-pad so memmap views stay in bounds
    try:
        return audit_reader(GGUFReader(str(tmp)), f"{repo}/{filename}")
    finally:
        tmp.unlink(missing_ok=True)


def collect_local(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += [Path(x) for x in glob.glob(str(pp / "**" / "*.gguf"), recursive=True)]
        elif any(c in p for c in "*?["):
            out += [Path(x) for x in glob.glob(p, recursive=True)]
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"warning: no match for {p!r}", file=sys.stderr)
    return sorted(set(out))


def print_report(rows: list[dict]) -> None:
    w = max((len(r["label"]) for r in rows), default=10)
    print(f"\n{'file':<{w}}  ident  trailer  file_type      streaming")
    print("-" * (w + 42))
    for r in rows:
        ident = "MISS" if r["identity_fail"] else ("warn" if r["missing"] else "ok")
        trailer = "ok" if r["trailer_ok"] else "BAD"
        sflag = r["streaming"]
        smark = "  <-- SUSPECT" if r["streaming_suspect"] else ""
        print(f"{r['label']:<{w}}  {ident:<5}  {trailer:<7}  "
              f"{r['file_type']:<13}  {str(sflag):<7}{smark}")
    # Detail on anything with issues.
    flagged = [r for r in rows if r["issues"]]
    if flagged:
        print("\nNeeds attention:")
        for r in flagged:
            bits = []
            if r["identity_fail"] or r["missing"]:
                bits.append("missing " + ", ".join(k.replace("general.", "") for k in r["missing"]))
            if not r["trailer_ok"]:
                bits.append("tokenizer arrays not trailered")
            if r["file_type"] == "AFTER-TRAILER":
                bits.append("file_type stranded after tokenizer trailer")
            if r["streaming_suspect"]:
                bits.append(r["streaming_suspect"])
            print(f"  {r['label']}: {'; '.join(bits)}")
    n_issue = len(flagged)
    print(f"\n{len(rows)} file(s) audited; {n_issue} with issue(s), "
          f"{len(rows) - n_issue} clean.")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="*", help="GGUF files, dirs, or globs")
    p.add_argument("--hf-repo", action="append", default=[],
                   help="audit a published HF repo's .gguf files (range-fetch headers)")
    p.add_argument("--hf-file", action="append", default=[],
                   help="limit --hf-repo to these filenames (default: all .gguf)")
    p.add_argument("--prefix-mb", type=int, default=16,
                   help="header bytes to range-fetch per HF file (default: 16)")
    args = p.parse_args(argv)

    rows: list[dict] = []

    for repo in args.hf_repo:
        from huggingface_hub import HfApi
        files = args.hf_file or [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
        for fn in files:
            try:
                rows.append(audit_hf(repo, fn, args.prefix_mb))
            except Exception as e:  # noqa: BLE001 - report and continue the sweep
                print(f"ERROR {repo}/{fn}: {e}", file=sys.stderr)

    local_paths = args.paths or ([] if args.hf_repo else [str(REPO_ROOT / "models")])
    for path in collect_local(local_paths):
        try:
            rows.append(audit_local(path))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {path}: {e}", file=sys.stderr)

    if not rows:
        print("no GGUFs found to audit", file=sys.stderr)
        return 2

    print_report(rows)
    return 1 if any(r["issues"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
