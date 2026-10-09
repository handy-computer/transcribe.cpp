#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf", "numpy"]
# ///
"""Seed catalog/<variant>.json for a newly ported model.

Everything mechanical comes from artifacts that already exist by the end of
Stage 5: the intake (family, upstream repo and revision, languages), the
local GGUFs under models/<variant>/ (downloads, byte sizes, parameter count,
capability KVs, licence, display name). The few editorial facts the artifacts
cannot supply are flags. Benchmark rows are left empty for Stages 6 and 7.

    uv run scripts/catalog/new_record.py <variant> --long-form soft-window --docs-page <family>.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import sync_capabilities  # noqa: E402

QUANT_ORDER = ("F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
LICENSE_DISPLAY = {"apache-2.0": "Apache-2.0", "mit": "MIT", "cc-by-4.0": "CC-BY-4.0",
                   "cc-by-nc-4.0": "CC-BY-NC-4.0", "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0"}


def quant_of(filename: str) -> str:
    return filename.rsplit("-", 1)[-1].removesuffix(".gguf")


def kv(reader: GGUFReader, key: str, default=None):
    field = reader.fields.get(key)
    return field.contents() if field is not None else default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variant")
    ap.add_argument("--long-form", required=True, choices=("chunked-unbounded", "hard-cap", "soft-window"),
                    help="which bucket in docs/input-limits.md the family falls into")
    ap.add_argument("--docs-page", required=True, help="page under docs/models/ that documents it")
    publication = ap.add_mutually_exclusive_group()
    publication.add_argument("--published-repo", default=None, help="default handy-computer/<variant>-gguf")
    publication.add_argument("--unpublished", action="store_true", help="no canonical upload yet")
    ap.add_argument("--role", choices=("asr", "diarize", "langid", "vad"),
                    help="default: the golden manifest's role, otherwise asr")
    ap.add_argument("--upstream-commit", help="source commit when the intake has no HF revision")
    ap.add_argument("--upstream-url", help="non-HF source repo URL; default: GGUF general.repo_url")
    ap.add_argument("--display-name", default=None, help="default: the GGUF's general.name")
    ap.add_argument("--license", default=None, help="SPDX id; default: the GGUF's general.license")
    ap.add_argument("--license-display", default=None)
    ap.add_argument("--language-tag-form", default="bare-bcp47", choices=("bare-bcp47", "locale", "mixed"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing record")
    args = ap.parse_args()

    out = common.CATALOG_DIR / f"{args.variant}.json"
    if out.exists() and not args.force:
        print(f"{out.relative_to(common.REPO)} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    intakes = list((common.REPO / "reports" / "porting").glob(f"*/{args.variant}/intake.json"))
    if len(intakes) != 1:
        print(f"expected one intake for {args.variant}, found {len(intakes)}", file=sys.stderr)
        return 2
    intake = json.loads(intakes[0].read_text())
    manifest_path = common.REPO / "tests" / "golden" / intake["family"] / f"{args.variant}.manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    role = args.role or manifest.get("role", "asr")
    revision = args.upstream_commit or intake.get("hf_revision")
    if not revision or not re.fullmatch(r"[0-9a-f]{7,40}", str(revision)):
        print("source revision must be a commit SHA; pass --upstream-commit", file=sys.stderr)
        return 2
    model_dir = common.REPO / "models" / args.variant
    files = sorted(model_dir.glob("*.gguf"), key=lambda p: (QUANT_ORDER.index(quant_of(p.name))
                                                           if quant_of(p.name) in QUANT_ORDER else 99))
    if not files:
        print(f"no GGUFs under {model_dir}", file=sys.stderr)
        return 2
    if not (common.DOCS_DIR / args.docs_page).exists():
        print(f"docs/models/{args.docs_page} does not exist", file=sys.stderr)
        return 2

    reference = files[0]
    reader = GGUFReader(str(reference))
    params = sum(int(np.prod(t.shape)) for t in reader.tensors)
    spdx = args.license or str(kv(reader, "general.license", "")).lower()
    if not spdx:
        print("no licence in the GGUF; pass --license", file=sys.stderr)
        return 2
    display = args.license_display or LICENSE_DISPLAY.get(spdx)
    if not display:
        print(f"no display form known for licence {spdx!r}; pass --license-display", file=sys.stderr)
        return 2
    caps = intake.get("capabilities") or {}
    languages = [str(lang) for lang in caps.get("languages", [])]
    acceptance = next((b for b in intake.get("upstream_benchmarks", [])
                       if str(b.get("dataset", "")).lower().startswith("librispeech")), None)

    record = {
        "schema": "transcribe-catalog-v1",
        "variant": args.variant,
        "family": intake["family"],
        "display_name": args.display_name or str(kv(reader, "general.name", args.variant)),
        "params": params,
        "license": {"spdx": spdx, "display": display},
        "upstream_repo": intake["hf_repo"],
        "upstream_commit": str(revision)[:7],
        "published_repo": (None if args.unpublished else
                           args.published_repo or f"handy-computer/{args.variant}-gguf"),
        "docs_page": args.docs_page,
        "languages": languages,
        "language_tag_form": args.language_tag_form,
        "long_form_strategy": args.long_form,
        "capabilities": {},
        "downloads": [{"quant": quant_of(p.name), "filename": p.name, "size_bytes": p.stat().st_size}
                      for p in files],
        "accuracy_benchmarks": [],
        "headline_benchmark": ({"dataset": "librispeech", "split": "test-clean", "language": "en",
                                "metric": "wer", "batch_size": None, "timestamps": "none"}
                               if acceptance else None),
        "speed_benchmarks": [],
    }
    if role != "asr":
        record["role"] = role
    upstream_url = args.upstream_url
    if not intake.get("hf_revision"):
        upstream_url = upstream_url or kv(reader, "general.repo_url")
        if not upstream_url:
            print("non-HF source needs --upstream-url", file=sys.stderr)
            return 2
    if upstream_url:
        record["upstream_url"] = str(upstream_url)
    source = manifest.get("source_model") or {}
    if source.get("package"):
        package, separator, version = source["package"].partition("==")
        if not separator or not version or not source.get("file") or not re.fullmatch(
                r"[0-9a-f]{64}", str(source.get("sha256") or "")):
            print("manifest source_model needs package==version, file and sha256", file=sys.stderr)
            return 2
        gguf_sha = kv(reader, f"stt.{intake['family']}.source_sha256")
        if gguf_sha and gguf_sha != source["sha256"]:
            print("GGUF source checksum disagrees with the manifest", file=sys.stderr)
            return 2
        artifact_url = kv(reader, "general.source.url")
        if not artifact_url:
            print("package source needs general.source.url in the GGUF", file=sys.stderr)
            return 2
        record["source_artifact"] = {"package": package, "version": version,
                                     "filename": source["file"], "sha256": source["sha256"],
                                     "url": str(artifact_url)}
    if role == "vad":
        record.pop("language_tag_form", None)
        sample_rate = kv(reader, "stt.frontend.sample_rate")
        frame_samples = kv(reader, "stt.vad.frame_samples")
        if not sample_rate or not frame_samples:
            print("VAD GGUF needs sample rate and frame geometry", file=sys.stderr)
            return 2
        record["vad_info"] = {"sample_rate": int(sample_rate), "frame_samples": int(frame_samples)}
    # Capabilities come from the file, never from hand: same reader the sweep uses.
    record["capabilities"] = sync_capabilities.build(record, sync_capabilities.read_kvs(reader))
    common.write_record(out, record)
    print(f"wrote {out.relative_to(common.REPO)}: {len(files)} downloads, {params:,} params, "
          f"{len(languages)} language(s), capabilities from {reference.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
