#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.24",
# ]
# ///
"""intake.py - draft mechanical research packet for a new model family port.

Produces a draft reports/porting/<family>/<variant>/intake.json matching
docs/porting/families/_intake-schema.json.

Mechanical fields are filled by this script. Human-judgment fields
(reference_framework, reference_rationale, architecture_pattern,
known_risks) are left null for the maintainer to complete during intake
review. See docs/porting/1a-intake.md for field semantics.

Usage:
    uv run scripts/intake.py inspect --repo Qwen/Qwen3-ASR-8B \\
        --family qwen3_asr --variant 8B \\
        --out reports/porting/qwen3_asr/8B/intake.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


SCHEMA_VERSION = "transcribe-intake-v1"

# Subset of config keys most STT families surface. Free-form — not
# exhaustive; human adds more via `config.key_fields` at review time.
INTERESTING_CONFIG_KEYS = {
    "model_type", "architectures",
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "d_model", "encoder_layers", "decoder_layers",
    "vocab_size", "max_position_embeddings",
    "activation_function", "hidden_act",
    "layer_norm_eps", "rms_norm_eps",
}


def resolve_revision(repo: str) -> str:
    """Pin the current main-branch commit SHA."""
    return HfApi().model_info(repo).sha


def load_json(repo: str, revision: str, filename: str) -> dict | None:
    try:
        path = hf_hub_download(repo, filename, revision=revision)
    except HfHubHTTPError:
        return None
    with open(path) as f:
        return json.load(f)


def inspect_config(config: dict | None) -> dict[str, Any]:
    if not config:
        return {"architecture_candidates": [], "key_fields": {}}
    return {
        "architecture_candidates": _guess_architecture(config),
        "key_fields": {k: v for k, v in config.items() if k in INTERESTING_CONFIG_KEYS},
    }


def _guess_architecture(config: dict) -> list[str]:
    archs = " ".join(config.get("architectures") or []).lower()
    mt = str(config.get("model_type", "")).lower()
    blob = archs + " " + mt
    candidates = []
    if any(t in blob for t in ["transducer", "rnnt", "tdt", "parakeet"]):
        candidates.append("encoder-transducer")
    if "ctc" in blob and "seq2seq" not in blob:
        candidates.append("encoder-ctc")
    if "speechseq2seq" in blob or ("decoder_layers" in config or "decoder" in config):
        candidates.append("encoder-decoder")
    if any(t in blob for t in ["audiolm", "audio_llm", "qwen2audio", "qwen3asr"]):
        candidates.append("audio-llm")
    return candidates or ["encoder-decoder"]  # default guess; human confirms


def extract_dtype(repo: str, revision: str, config: dict | None) -> dict[str, Any]:
    config_dtype = _config_dtype(config) if config else None
    header_dist = _safetensors_dtype_distribution(repo, revision)
    float_dist = _floating_dtype_distribution(header_dist)
    float_majority = _majority_dtype(float_dist)
    details = {
        "config_declared": config_dtype,
        "header_distribution": header_dist,
    }

    if config_dtype:
        expected = _norm_dtype(config_dtype)
        if float_majority and expected != _norm_dtype(float_majority):
            return {
                "expected": None,
                "source": "unresolved",
                "evidence": (
                    f"config dtype {config_dtype} disagrees with safetensors header "
                    f"{_format_distribution(header_dist)}"
                ),
                "details": details,
            }
        return {
            "expected": expected,
            "source": "config",
            "evidence": f"config dtype {config_dtype}",
            "details": details,
        }

    if float_majority:
        expected = _norm_dtype(float_majority)
        return {
            "expected": expected,
            "source": "weights_header",
            "evidence": (
                f"safetensors header {_format_distribution(header_dist)}; "
                f"{float_majority} selected as dominant floating dtype"
            ),
            "details": details,
        }

    return {
        "expected": None,
        "source": "unresolved",
        "evidence": "No config dtype and no safetensors metadata; fill from reference code",
        "details": details,
    }


# Normalize dtype strings. Safetensors headers use e.g. "BF16"/"F32"/"F16";
# config fields use "bfloat16"/"float32"/"float16". Return a common form.
_DTYPE_ALIASES = {
    "bf16": "bfloat16", "f32": "float32", "f16": "float16",
    "i64": "int64", "i32": "int32", "i16": "int16", "i8": "int8",
    "u8": "uint8", "bool": "bool",
}
_FLOAT_DTYPES = {"bf16", "f32", "f16", "f64", "bfloat16", "float32", "float16", "float64"}


def _norm_dtype(s: str) -> str:
    v = s.replace("torch.", "").strip().lower()
    return _DTYPE_ALIASES.get(v, v)


def _floating_dtype_distribution(dist: dict[str, int]) -> dict[str, int]:
    return {dt: n for dt, n in dist.items() if _norm_dtype(dt) in _FLOAT_DTYPES}


def _majority_dtype(dist: dict[str, int]) -> str | None:
    return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None


def _format_distribution(dist: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "unavailable"


def _config_dtype(config: dict) -> str | None:
    # Top-level first; sub-config is a fallback. See the matching note in
    # scripts/preflight.py:_config_dtype.
    for path in (
        ("dtype",),
        ("torch_dtype",),
        ("text_config", "dtype"),
        ("text_config", "torch_dtype"),
    ):
        node = config
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str):
            return node
    return None


def _safetensors_dtype_distribution(repo: str, revision: str) -> dict[str, int]:
    """Header-only dtype scan. Uses HfApi.get_safetensors_metadata which fetches
    safetensors headers via HTTP Range requests — no tensor bodies downloaded.
    Returns dtype (e.g. 'BF16') → tensor count. Aggregates across all shards."""
    try:
        meta = HfApi().get_safetensors_metadata(repo, revision=revision)
    except Exception:
        return {}
    dist: dict[str, int] = {}
    for file_meta in meta.files_metadata.values():
        for tensor_info in file_meta.tensors.values():
            dt = tensor_info.dtype
            dist[dt] = dist.get(dt, 0) + 1
    return dist


def extract_frontend(pre: dict | None) -> dict[str, Any]:
    if not pre:
        return {}
    return {
        "sample_rate": pre.get("sampling_rate") or pre.get("sample_rate"),
        "n_mels": pre.get("feature_size") or pre.get("num_mel_bins"),
        "hop_length": pre.get("hop_length"),
        "fft_size": pre.get("n_fft") or pre.get("win_length"),
        "window": _window(pre.get("window_function") or pre.get("window")),
        "normalization": "per_feature" if pre.get("do_normalize") else pre.get("normalization"),
        "preemphasis": pre.get("preemphasis"),
        "dither": pre.get("dither"),
        "center": pre.get("center"),
        "padding_mode": pre.get("padding_mode") or pre.get("padding_value_type"),
        "mel_filterbank_norm": pre.get("mel_norm") or pre.get("mel_filterbank_norm"),
    }


def _window(v: str | None) -> str | None:
    if not v:
        return None
    s = str(v).lower()
    if "hann" in s:
        # periodic vs symmetric isn't encoded in the string;
        # default to periodic (torch default). Preflight confirms.
        return "hann_periodic"
    if "hamming" in s:
        return "hamming"
    if "blackman" in s:
        return "blackman"
    return s


def extract_tokenizer(repo: str, revision: str, tok_cfg, tok_json, gen_cfg) -> dict[str, Any]:
    return {
        "type": _tokenizer_type(tok_cfg, tok_json, repo, revision),
        "vocab_size": _vocab_size(tok_json) or 0,
        "special_tokens": _special_token_ids(tok_cfg, gen_cfg),
        "has_language_tokens": _has_language_tokens(tok_cfg),
        "vocab_sha256": _vocab_sha256(tok_json),
    }


def _tokenizer_type(tok_cfg, tok_json, repo, revision) -> str:
    cls = (tok_cfg or {}).get("tokenizer_class", "").lower()
    if "sentencepiece" in cls:
        return "sentencepiece"
    if tok_json:
        mt = (tok_json.get("model") or {}).get("type", "").upper()
        if "BPE" in mt:
            return "bpe"
        if "WORDPIECE" in mt:
            return "wordpiece"
    try:
        hf_hub_download(repo, "tokenizer.model", revision=revision)
        return "sentencepiece"
    except HfHubHTTPError:
        return "other"


def _vocab_size(tok_json) -> int | None:
    if not tok_json:
        return None
    vocab = (tok_json.get("model") or {}).get("vocab")
    if isinstance(vocab, dict):
        base = len(vocab)
    elif isinstance(vocab, list):
        base = len(vocab)
    else:
        return None
    # Whisper-style tokenizers carry their special tokens (lang/task/timestamp)
    # in `added_tokens` rather than the base BPE vocab. The model output dim
    # (and the GGUF vocab_size we ship) is base + added, not base.
    added = tok_json.get("added_tokens") or []
    if isinstance(added, list):
        max_id = base - 1
        for entry in added:
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                max_id = max(max_id, entry["id"])
        return max(base, max_id + 1)
    return base


def extract_capabilities(config, tok_cfg, gen_cfg) -> dict[str, Any]:
    """Capabilities are mostly human-filled at research time. We auto-extract
    languages from config fields when present (common shapes: `languages`,
    `supported_languages`, `language_list`, or text_config variants). Everything
    else — translation, timestamps, streaming, VAD, diarization — requires
    reading the model card.
    """
    languages: list[str] = []
    for path in (
        ("languages",),
        ("supported_languages",),
        ("language_list",),
        ("text_config", "languages"),
        ("text_config", "supported_languages"),
    ):
        node: Any = config or {}
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, list) and node and all(isinstance(x, str) for x in node):
            languages = list(node)
            break

    return {
        "languages": languages,
        "language_detection": None,
        "translation": None,
        "timestamps": [],
        "streaming": None,
        "speaker_diarization": None,
    }


def _special_token_ids(tok_cfg, gen_cfg) -> dict[str, int]:
    src = {**(tok_cfg or {}), **(gen_cfg or {})}
    roles = {
        "bos": "bos_token_id", "eos": "eos_token_id", "pad": "pad_token_id",
        "unk": "unk_token_id", "decoder_start": "decoder_start_token_id",
    }
    return {role: src[k] for role, k in roles.items() if isinstance(src.get(k), int)}


def _has_language_tokens(tok_cfg) -> bool:
    added = (tok_cfg or {}).get("added_tokens_decoder") or {}
    return any(
        "<|" in (t.get("content") or "") and "|>" in (t.get("content") or "")
        for t in added.values() if isinstance(t, dict)
    )


def _vocab_sha256(tok_json) -> str | None:
    if not tok_json:
        return None
    vocab = (tok_json.get("model") or {}).get("vocab")
    if isinstance(vocab, dict):
        tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    elif isinstance(vocab, list):
        tokens = [v[0] if isinstance(v, list) else str(v) for v in vocab]
    else:
        return None
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def build_intake(repo: str, family: str, variant: str) -> dict[str, Any]:
    revision = resolve_revision(repo)
    config = load_json(repo, revision, "config.json")
    preprocessor_config = load_json(repo, revision, "preprocessor_config.json")
    feature_extractor_config = load_json(repo, revision, "feature_extractor_config.json")
    pre = preprocessor_config or feature_extractor_config
    pre_source = (
        "preprocessor_config.json" if preprocessor_config
        else "feature_extractor_config.json" if feature_extractor_config
        else None
    )
    tok_cfg = load_json(repo, revision, "tokenizer_config.json")
    tok_json = load_json(repo, revision, "tokenizer.json")
    gen_cfg = load_json(repo, revision, "generation_config.json")
    dtype = extract_dtype(repo, revision, config)
    capabilities = extract_capabilities(config, tok_cfg, gen_cfg)
    intake: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "hf_repo": repo,
        "hf_revision": revision,
        "sources": _sources(config, pre_source, tok_cfg, tok_json, gen_cfg, dtype),
        "variants": [{"name": variant, "memory_gb": None, "files": []}],
        "config": inspect_config(config),
        "dtype": dtype,
        "frontend": extract_frontend(pre),
        "tokenizer": extract_tokenizer(repo, revision, tok_cfg, tok_json, gen_cfg),
        "capabilities": capabilities,
        "upstream_benchmarks": [],
        "reference_framework": None,
        "reference_rationale": None,
        "architecture_pattern": None,
        "known_risks": [],
        "intake_gaps": [],
    }
    if not config:
        intake["intake_gaps"].append({"field": "config", "reason": "config.json not found"})
    if not pre:
        intake["intake_gaps"].append({
            "field": "frontend",
            "reason": "preprocessor_config.json not found; fill from reference code",
        })
    if intake["dtype"]["source"] == "unresolved":
        intake["intake_gaps"].append({
            "field": "dtype",
            "reason": intake["dtype"]["evidence"],
        })
    if not capabilities["languages"]:
        intake["intake_gaps"].append({
            "field": "capabilities.languages",
            "reason": "no language list in config; fill from model card (BCP-47 codes)",
        })
    intake["intake_gaps"].append({
        "field": "capabilities",
        "reason": "translation, timestamps, streaming, VAD, and diarization flags "
                  "require reading the model card; fill before intake sign-off",
    })
    intake["intake_gaps"].append({
        "field": "upstream_benchmarks",
        "reason": "publisher-reported WER/CER scores are not auto-scraped; "
                  "add entries from the model card during research",
    })
    return intake


def _source_status(path: str, found: bool, *, kind: str = "hf_file", detail: str | None = None) -> dict[str, str]:
    out = {"kind": kind, "path": path, "status": "found" if found else "missing"}
    if detail:
        out["detail"] = detail
    return out


def _sources(config, pre_source, tok_cfg, tok_json, gen_cfg, dtype) -> dict[str, dict[str, str]]:
    return {
        "config": _source_status("config.json", config is not None),
        "preprocessor": _source_status(
            pre_source or "preprocessor_config.json|feature_extractor_config.json",
            pre_source is not None,
        ),
        "tokenizer_config": _source_status("tokenizer_config.json", tok_cfg is not None),
        "tokenizer_json": _source_status("tokenizer.json", tok_json is not None),
        "generation_config": _source_status("generation_config.json", gen_cfg is not None),
        "safetensors_metadata": _source_status(
            "HfApi.get_safetensors_metadata",
            bool(dtype.get("details", {}).get("header_distribution")),
            kind="hf_api",
            detail="header-only floating dtype distribution; no tensor payloads downloaded",
        ),
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    intake = build_intake(args.repo, args.family, args.variant)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(intake, indent=2) + "\n")
    print(f"Wrote draft intake: {out}")
    if intake["intake_gaps"]:
        print("Gaps requiring review:")
        for gap in intake["intake_gaps"]:
            print(f"- {gap['field']}: {gap['reason']}")
    else:
        print("No mechanical gaps detected.")
    print("Human review is still required before converter or C++ work begins.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("inspect", help="Produce an intake.json for a HF repo.")
    insp.add_argument("--repo", required=True, help="HuggingFace repo id (org/name)")
    insp.add_argument("--family", required=True, help="Stable family key")
    insp.add_argument("--variant", required=True, help="Variant name")
    insp.add_argument("--out", required=True, help="Output intake.json path")
    insp.set_defaults(func=cmd_inspect)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
