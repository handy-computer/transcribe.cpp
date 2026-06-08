from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpecInfo:
    requested: str
    kind: str
    value: str
    dataset_id: str
    source: str
    split: str
    default_language: str | None
    ingest_args: list[str]


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """'librispeech:test-clean'      -> ('librispeech', 'test-clean')
       'fleurs:zh'                   -> ('fleurs', 'zh')
       'eka-medical-asr:en'          -> ('eka-medical-asr', 'en')
       'librispeech'                 -> ('librispeech', 'test-clean') (default)
    """
    kind, _, val = spec.partition(":")
    kind = kind.strip().lower()
    val = val.strip()
    if kind == "librispeech":
        return "librispeech", val or "test-clean"
    if kind == "fleurs":
        if not val:
            raise ValueError("fleurs: requires a language, e.g. fleurs:zh")
        return "fleurs", val
    if kind == "eka-medical-asr":
        if not val:
            raise ValueError(
                "eka-medical-asr: requires a language, e.g. eka-medical-asr:en"
            )
        return "eka-medical-asr", val
    raise ValueError(f"unknown dataset kind: {spec!r}")


def dataset_spec_info(spec: str) -> DatasetSpecInfo:
    kind, val = parse_dataset_spec(spec)
    ds_id = f"{kind}-{val}"
    if kind == "librispeech":
        return DatasetSpecInfo(
            requested=spec,
            kind=kind,
            value=val,
            dataset_id=ds_id,
            source="openslr/librispeech",
            split=val,
            default_language="en",
            ingest_args=["librispeech", "--split", val],
        )
    if kind == "fleurs":
        return DatasetSpecInfo(
            requested=spec,
            kind=kind,
            value=val,
            dataset_id=ds_id,
            source="google/fleurs",
            split="test",
            default_language=val,
            ingest_args=["fleurs", "--lang", val],
        )
    if kind == "eka-medical-asr":
        return DatasetSpecInfo(
            requested=spec,
            kind=kind,
            value=val,
            dataset_id=ds_id,
            source="ekacare/eka-medical-asr-evaluation-dataset",
            split="test",
            default_language=val,
            ingest_args=["eka-medical-asr", "--lang", val],
        )
    raise ValueError(spec)


def dataset_id(spec: str) -> str:
    """Slug used in filenames + volume paths. 'librispeech-test-clean' / 'fleurs-zh'."""
    return dataset_spec_info(spec).dataset_id


def manifest_path_for(spec: str) -> str:
    return f"/data/wer/{dataset_id(spec)}.manifest.jsonl"


def local_manifest_path_for(repo: Path, spec: str) -> Path:
    return repo / "samples" / "wer" / f"{dataset_id(spec)}.manifest.jsonl"


def ingest_args_for(spec: str) -> list[str]:
    return list(dataset_spec_info(spec).ingest_args)


def default_language_for(spec: str) -> str | None:
    return dataset_spec_info(spec).default_language


def manifest_metadata(manifest: str | Path) -> dict[str, Any]:
    path = Path(manifest)
    sha = hashlib.sha256()
    utterances = 0
    languages: set[str] = set()
    if not path.exists():
        return {
            "exists": False,
            "bytes": 0,
            "utterances": 0,
            "languages": [],
            "manifest_sha256": "",
        }

    with path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            sha.update(raw)
            utterances += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lang = row.get("language")
            if lang:
                languages.add(str(lang))
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "utterances": utterances,
        "languages": sorted(languages),
        "manifest_sha256": sha.hexdigest(),
    }


def dataset_prepare_status(
    spec: str,
    manifest: str | Path,
    cache_status: str,
    elapsed_s: float,
    error: str | None = None,
) -> dict[str, Any]:
    info = dataset_spec_info(spec)
    meta = manifest_metadata(manifest)
    status = {
        **asdict(info),
        **meta,
        "manifest": str(manifest),
        "cache_status": cache_status,
        "elapsed_s": elapsed_s,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    if error:
        status["error"] = error
    return status


def format_dataset_status(status: dict[str, Any], prefix: str = "[data]") -> str:
    langs = status.get("languages") or []
    lang = ",".join(langs) if langs else status.get("default_language") or "unknown"
    sha = (status.get("manifest_sha256") or "")[:12] or "missing"
    token = "yes" if status.get("hf_token_present") else "no"
    return (
        f"{prefix} {status.get('requested')}: {status.get('cache_status')} "
        f"id={status.get('dataset_id')} source={status.get('source')} "
        f"split={status.get('split')} n={status.get('utterances')} "
        f"lang={lang} sha={sha} hf_token={token} "
        f"elapsed={status.get('elapsed_s', 0.0):.1f}s "
        f"manifest={status.get('manifest')}"
    )


def dataset_summary_fields(status: dict[str, Any] | None) -> dict[str, Any]:
    if not status:
        return {}
    return {
        "dataset_id": status.get("dataset_id"),
        "dataset_source": status.get("source"),
        "dataset_split": status.get("split"),
        "dataset_manifest": status.get("manifest"),
        "dataset_manifest_sha256": status.get("manifest_sha256"),
        "dataset_total_utts": status.get("utterances"),
        "dataset_languages": status.get("languages", []),
        "dataset_cache_status": status.get("cache_status"),
    }
