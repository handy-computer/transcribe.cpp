"""Publication-profile loading and benchmark matrix expansion.

The profile is policy: it defines the exact rows the catalog publishes. The
catalog arrays contain measurements only; experiments remain in reports/.
Keep this module stdlib-only so local checks, bench runs, and Modal dispatch
all consume the same expansion logic.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from datetime import date
from typing import Iterable

REPO = pathlib.Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "catalog" / "_benchmark_profiles.json"
WER_DIR = REPO / "scripts" / "wer"
if str(WER_DIR) not in sys.path:
    sys.path.insert(0, str(WER_DIR))

from languages import (  # noqa: E402
    CER_LANGUAGES,
    FLEURS_CANONICAL_BY_CONFIG,
    FLEURS_LANGS,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

MACHINE_ALIASES = {
    "apple-m4": "m4",
    "apple-m4-max": "m4-max",
    "amd-ryzen-7-pro-4750u-with-radeon-graphics": "ryzen-4750u",
    "amd-ryzen-7-4750u-pro": "ryzen-4750u",
}


def load_profiles(path: pathlib.Path = PROFILE_PATH) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data.get("profiles"), dict) or not data["profiles"]:
        raise ValueError(f"{path}: profiles must be a non-empty object")
    default = data.get("default")
    if default not in data["profiles"]:
        raise ValueError(f"{path}: default profile {default!r} is not defined")
    for role, profile_id in (data.get("roles") or {}).items():
        if data["profiles"].get(profile_id, {}).get("role") != role:
            raise ValueError(f"{path}: roles.{role} must name a defined profile "
                             f"whose role is {role!r}")
    return data


def load_profile(profile_id: str | None = None) -> tuple[str, dict]:
    data = load_profiles()
    profile_id = profile_id or data["default"]
    try:
        return profile_id, data["profiles"][profile_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown benchmark profile {profile_id!r}; choose one of "
            f"{sorted(data['profiles'])}"
        ) from exc


def profile_for(record: dict, profile_id: str | None = None) -> tuple[str, dict]:
    """The profile that governs one record: `profile_id` when it is written
    for the record's role, else the profile its role names in `roles`, else
    the default. ASR and diarize records share the default; language ID and VAD
    have their own profiles, since classification accuracy
    and per-chunk streaming latency are not cells of the ASR matrix."""
    data = load_profiles()
    if profile_id:
        named = load_profile(profile_id)[1]
        if governs(named, record):
            return profile_id, named
    chosen = data.get("roles", {}).get(record.get("role", "asr"), data["default"])
    return chosen, data["profiles"][chosen]


def governs(profile: dict, record: dict) -> bool:
    """A profile with a `role` governs only records of that role; a profile
    without one governs every record no role-specific profile claims."""
    role = record.get("role", "asr")
    if profile.get("role"):
        return profile["role"] == role
    return role not in load_profiles().get("roles", {})


def canonical_machine(slug: str) -> str:
    return MACHINE_ALIASES.get(slug, slug)


def _aliases(record: dict) -> dict[str, str]:
    return {str(k).lower(): str(v).lower()
            for k, v in (record.get("language_aliases") or {}).items()}


def canonical_fleurs_language(tag: str, record: dict) -> str | None:
    """Return the canonical FLEURS language represented by a model tag.

    The catalog stores model-advertised spellings, while accuracy rows store
    dataset spellings. Locale tags normally collapse to their primary subtag;
    zh-TW is deliberately excluded because FLEURS has only Simplified
    Mandarin. Explicit language_aliases handle tl/fil, no/nb, and jw/jv.
    """
    value = str(tag).lower()
    value = _aliases(record).get(value, value)
    if value == "zh-tw":
        return None
    candidate = value if value in FLEURS_LANGS else value.split("-", 1)[0]
    candidate = _aliases(record).get(candidate, candidate)
    config = FLEURS_LANGS.get(candidate)
    return FLEURS_CANONICAL_BY_CONFIG.get(config) if config else None


def fleurs_languages(record: dict) -> list[str]:
    out: list[str] = []
    for advertised in record.get("languages", []):
        canonical = canonical_fleurs_language(str(advertised), record)
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def runtime_language(record: dict, canonical: str) -> str:
    """Choose the model spelling used to run one canonical dataset language."""
    for advertised in record.get("languages", []):
        if canonical_fleurs_language(str(advertised), record) == canonical:
            return str(advertised)
    return canonical


def download_quants(record: dict) -> list[str]:
    return [str(item["quant"]) for item in record.get("downloads", [])]


def _quants(spec: str | list[str], record: dict) -> list[str]:
    if spec == "all-downloads":
        return download_quants(record)
    if not isinstance(spec, list):
        raise ValueError(f"invalid quant selector {spec!r}")
    available = set(download_quants(record))
    return [str(quant) for quant in spec if str(quant) in available]


def expected_accuracy(record: dict, profile: dict) -> list[dict]:
    """Expand a profile into publication accuracy cells for one model."""
    cells: list[dict] = []
    if not governs(profile, record):
        return cells
    if not profile.get("role") and not (
            record.get("capabilities", {}).get("transcribe", {}).get("supported")):
        return cells
    for suite in profile.get("accuracy", []):
        selector = suite["languages"]
        if selector == "english-if-supported":
            languages = ["en"] if "en" in fleurs_languages(record) else []
        elif selector == "supported-intersect-fleurs":
            languages = fleurs_languages(record)
        elif selector == "pooled":
            # One result over every evaluated language at once (language ID).
            languages = ["mul"]
        else:
            raise ValueError(f"unknown language selector {selector!r}")
        for language in languages:
            for quant in _quants(suite["quants"], record):
                cells.append({
                    "dataset": suite["dataset"],
                    "split": suite["split"],
                    "language": language,
                    "runtime_language": (None if selector == "pooled"
                                         else runtime_language(record, language)),
                    "quant": quant,
                    "metric": suite.get("metric") or (
                        "cer" if language in CER_LANGUAGES else "wer"),
                    "batch_size": suite["batch_size"],
                    "sort_by_length": suite.get("sort_by_length", False),
                    "timestamps": suite["timestamps"],
                    "gpu": suite.get("gpu"),
                    "backend": suite.get("backend"),
                    # Language ID's crop: which slice of a run.py sweep the
                    # published number is, and which languages its pooled
                    # mean is over.
                    **{key: suite[key] for key in ("crop_s", "pooled_languages")
                       if key in suite},
                })
    return cells


def speed_samples(record: dict, profile: dict) -> list[str]:
    """Which clips a variant is benched on.

    The default pair is English. A variant that supports exactly one other
    language is benched on that language instead, at the same two lengths:
    English audio decodes out of distribution on a single-language fine-tune
    and the figure would not be comparable. The rule replaces what used to be
    one hand-written override per such variant.
    """
    spec = profile["speed"]
    samples = spec.get("samples", [])
    rule = spec.get("single_language_samples")
    languages = record.get("languages") or []
    if rule and len(languages) == 1 and languages[0] != "en":
        return [part.format(lang=languages[0]) for part in rule["pattern"]]
    return samples


def all_speed_samples(profile: dict, records: dict[str, dict]) -> list[str]:
    """Every clip the profile can ask for, across all known variants. The bench
    driver needs this union up front to build its candidate matrix."""
    samples = set(profile["speed"].get("samples", []))
    for record in records.values():
        samples.update(speed_samples(record, profile))
    return sorted(samples)


def expected_speed(record: dict, profile: dict) -> list[dict]:
    """Expand the exact publication speed matrix for one model."""
    if not governs(profile, record):
        return []
    spec = profile["speed"]
    samples = speed_samples(record, profile)
    cells: list[dict] = []
    for target in spec.get("targets", []):
        for backend in target.get("backends", []):
            for quant in _quants(spec["quants"], record):
                for sample in samples:
                    cell = {
                        "machine": target["machine"],
                        "backend": backend,
                        "quant": quant,
                        "sample": sample,
                    }
                    if profile.get("role") == "vad":
                        feed = spec["feed_samples"][sample]
                        cell.update({
                            "feed_samples": feed,
                            "frame_samples": spec["frame_samples"],
                            "threads": spec["threads"],
                            "warmup_calls": spec["warmup_calls"],
                            "n_calls": spec["source_samples"] // feed,
                            "sample_duration_s": feed / spec["sample_rate"],
                        })
                    cells.append(cell)
    return cells


def has_measurement_provenance(row: dict) -> bool:
    return bool(row.get("engine_sha")) or row.get("measurement_provenance") == "legacy-published"


ACCURACY_CORE_KEY = ("dataset", "split", "language", "quant", "metric")
# What a profile cell requires. Batch size is deliberately absent: a cell is
# satisfied at any batch size and the row records the one that was run. The
# profile's batch_size is the recommendation for new runs, not an identity.
PROFILE_KEY = (*ACCURACY_CORE_KEY, "timestamps", "scoring", "mode")
# Published accuracy has one row per profile cell. Batch size is measured
# recipe metadata, not a second publishable identity for the same result.
ACCURACY_KEY = PROFILE_KEY
SPEED_KEY = ("machine", "backend", "quant", "sample")


def accuracy_core_key(cell: dict) -> tuple:
    return tuple(cell.get(field) for field in ACCURACY_CORE_KEY)


def profile_key(cell: dict) -> tuple:
    return tuple(cell.get(field) for field in PROFILE_KEY)


def cell_key(cell: dict, kind: str) -> tuple:
    fields = {"accuracy": ACCURACY_KEY, "speed": SPEED_KEY}[kind]
    return tuple(cell.get(field) for field in fields)


def has_vad_provenance(row: dict, profile_id: str | None = None) -> bool:
    """VAD has no legacy measurements; require an actual build and run date."""
    try:
        measured = row.get("measured_on")
        if not isinstance(measured, str) or date.fromisoformat(measured).isoformat() != measured:
            return False
    except ValueError:
        return False
    return (isinstance(row.get("engine_sha"), str)
            and re.fullmatch(r"[0-9a-f]{7,40}", row["engine_sha"]) is not None
            and bool(row.get("publication_profile"))
            and (profile_id is None or row["publication_profile"] == profile_id)
            and not row.get("measurement_provenance"))


def _positive_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def valid_vad_speed(row: dict, target: dict, profile_id: str) -> bool:
    """Validate a matched speed cell's chunk recipe, provenance and statistics."""
    if not has_vad_provenance(row, profile_id):
        return False
    for key in ("feed_samples", "frame_samples", "threads", "warmup_calls", "n_calls"):
        if type(row.get(key)) is not int or row[key] != target[key]:
            return False
    if not _positive_number(row.get("sample_duration_s")) or not math.isclose(
            row["sample_duration_s"], target["sample_duration_s"], rel_tol=1e-9):
        return False
    if not all(_positive_number(row.get(key))
               for key in ("total_ms", "median_ms", "p95_ms", "xrt_compute")):
        return False
    # Quantiles must be ordered; neither can exceed the sum of all calls.
    if not (row["median_ms"] <= row["p95_ms"] <= row["total_ms"] * row["n_calls"]):
        return False
    return math.isclose(row["xrt_compute"],
                        row["sample_duration_s"] * 1000 / row["total_ms"],
                        rel_tol=1e-6, abs_tol=0.005)


def exception_matches(exception: dict, kind: str, cell: dict) -> bool:
    if exception.get("kind") != kind:
        return False
    match = exception.get("match") or {}
    return all(value == "*" or cell.get(field) == value
               for field, value in match.items())


def apply_exceptions(record: dict, kind: str, cells: Iterable[dict]) -> list[dict]:
    exceptions = record.get("benchmark_exceptions") or []
    return [cell for cell in cells
            if not any(exception_matches(exc, kind, cell) for exc in exceptions)]


def target_for_machine(profile: dict, machine_slug: str) -> dict | None:
    machine = canonical_machine(machine_slug)
    return next((target for target in profile["speed"].get("targets", [])
                 if target["machine"] == machine), None)


def dataset_spec(cell: dict) -> str:
    """The `--dataset` string for one profile cell. See common.dataset_spec."""
    return common.dataset_spec(cell)


def machine_display(profile: dict, machine_slug: str) -> str:
    target = target_for_machine(profile, machine_slug)
    return (target or {}).get("display") or machine_slug
