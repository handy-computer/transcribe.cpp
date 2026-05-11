#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.24",
#   "safetensors>=0.4",
#   "gguf>=0.10",
#   "pyyaml>=6.0",
# ]
# ///
"""preflight.py - cross-source consistency checks for a model port.

Runs before C++ work starts (Gate A) or before the encoder is implemented
(Gate B), validating that intake declarations, GGUF metadata, and reference
framework files all agree. Catches silent STT failure modes (dtype drift,
frontend-config mismatch, tokenizer misalignment) cheaply.

Gates:
    A  Post-intake, no GGUF yet. Declared state (intake) vs reference.
    B  Post-converter. Declared vs GGUF vs reference.

Gate C (post-mel runtime) and Gate D (post-quantization) live in separate
flows; see 4-numerical-validation.md Divergence Classification and the plan.

The "declared state" can come from reports/porting/<family>/<variant>/intake.json
(preferred, captured before the port) or from the golden manifest
(fallback, once the manifest exists). This lets preflight run against
already-ported families without retroactively writing an intake.

Usage:
    uv run scripts/preflight.py --family parakeet --gate A
    uv run scripts/preflight.py --family cohere --gate B --gguf models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf

NOTE: this script duplicates small pieces of reference-side parsing with
scripts/intake.py. Factoring into a shared lib is Stage 3 work, after
Qwen3-ASR tells us what the real shared shape should be.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


# -------- Data types --------

Gate = Literal["A", "B", "C", "D"]
Status = Literal["pass", "fail", "warn"]


@dataclass
class CheckResult:
    name: str
    gate: Gate
    status: Status
    sources: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass
class PreflightReport:
    gate: Gate
    family: str
    variant: str
    overall: Status
    checks: list[CheckResult]
    generated_at: str


# -------- GGUF KV reader --------

def read_gguf_kvs(gguf_path: Path) -> dict[str, Any]:
    """Read scalar/string KV fields from an existing GGUF into a plain dict.

    Array fields (tokens, scores, token_type) are returned as their length only;
    preflight compares summaries, not full vocabularies.
    """
    from gguf.gguf_reader import GGUFReader, GGUFValueType

    reader = GGUFReader(str(gguf_path))
    out: dict[str, Any] = {}
    for name, fld in reader.fields.items():
        out[name] = _decode_field(fld, GGUFValueType)
    return out


def _decode_field(fld, GGUFValueType) -> Any:
    if not fld.types:
        return None
    # Scalar field: types is [scalar_type]; value is in parts[-1].
    # Array field: types is [ARRAY, element_type]; length is len(fld.data).
    if fld.types[0] == GGUFValueType.ARRAY:
        elem = fld.types[1]
        length = len(fld.data)
        # Decode short string arrays in full (language lists, small enums).
        # Large arrays (tokenizer vocab) stay summarized to avoid blowing
        # memory reading every KV block.
        if elem == GGUFValueType.STRING and length <= 200:
            try:
                return [bytes(fld.parts[i]).decode("utf-8") for i in fld.data]
            except (UnicodeDecodeError, TypeError, IndexError):
                pass
        return {"__array__": True, "length": length, "element_type": elem.name}
    last = fld.parts[-1] if fld.parts else None
    if last is None:
        return None
    if fld.types[0] == GGUFValueType.STRING:
        try:
            return bytes(last).decode("utf-8")
        except (UnicodeDecodeError, TypeError):
            return None
    if hasattr(last, "tolist"):
        v = last.tolist()
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        return v
    return last


# -------- Declared state loader --------

def load_declared_state(repo_root: Path, family: str, variant: str | None) -> dict[str, Any]:
    """Prefer intake.json if it exists; otherwise fall back to the golden manifest.

    Returns a normalized dict with: hf_repo, hf_revision, expected_dtype,
    frontend, tokenizer, reference_framework, source ("intake" or "manifest").
    """
    intake = _find_intake(repo_root, family, variant)
    if intake:
        data = json.loads(intake.read_text())
        return {
            "hf_repo": data["hf_repo"],
            "hf_revision": data.get("hf_revision"),
            "expected_dtype": _expected_dtype_from_intake(data),
            "frontend": data.get("frontend", {}),
            "tokenizer": data.get("tokenizer", {}),
            "capabilities": data.get("capabilities", {}),
            "reference_framework": data.get("reference_framework"),
            "source": f"intake:{intake.relative_to(repo_root)}",
        }

    manifest = _find_manifest(repo_root, family, variant)
    if manifest:
        data = json.loads(manifest.read_text())
        source_model = data.get("source_model", {})
        if not isinstance(source_model, dict):
            source_model = {}
        return {
            "hf_repo": source_model.get("hf_repo"),
            "hf_revision": source_model.get("hf_revision"),
            "expected_dtype": data.get("expected_dtype"),
            "frontend": data.get("frontend", {}),
            "tokenizer": data.get("tokenizer_summary", {}),
            "capabilities": data.get("capabilities", {}),
            "reference_framework": data.get("reference", {}).get("kind") if isinstance(data.get("reference"), dict) else None,
            "source": f"manifest:{manifest.relative_to(repo_root)}",
        }

    raise SystemExit(
        f"error: no intake.json or golden manifest found for family={family} "
        f"variant={variant}. Expected one of:\n"
        f"  reports/porting/{family}/<variant>/intake.json\n"
        f"  tests/golden/{family}/<variant>.manifest.json"
    )


def _expected_dtype_from_intake(data: dict[str, Any]) -> str | None:
    dtype = data.get("dtype", {})
    if dtype.get("expected"):
        return dtype.get("expected")
    # Backward-compatible fallback for older draft intakes.
    if dtype.get("source") == "config":
        return dtype.get("config_declared") or dtype.get("details", {}).get("config_declared")
    return None


def _find_intake(repo_root: Path, family: str, variant: str | None) -> Path | None:
    base = repo_root / "reports" / "porting" / family
    if not base.is_dir():
        return None
    if variant:
        p = base / variant / "intake.json"
        return p if p.exists() else None
    matches = sorted(base.glob("*/intake.json"))
    return matches[0] if len(matches) == 1 else None


def _find_manifest(repo_root: Path, family: str, variant: str | None) -> Path | None:
    base = repo_root / "tests" / "golden" / family
    if variant:
        p = base / f"{variant}.manifest.json"
        return p if p.exists() else None
    matches = sorted(base.glob("*.manifest.json"))
    return matches[0] if len(matches) == 1 else None


# -------- Reference state loader --------

def _load_json_from_hf(repo: str, revision: str | None, filename: str) -> dict | None:
    try:
        path = hf_hub_download(repo, filename, revision=revision)
    except HfHubHTTPError:
        return None
    with open(path) as f:
        return json.load(f)


def load_reference_state(
    hf_repo: str,
    hf_revision: str | None,
    reference_framework: str | None = None,
) -> dict[str, Any]:
    """Pull config.json, preprocessor, and tokenizer info from the source HF repo.

    When reference_framework=="nemo" and a .nemo file is in the repo siblings,
    prefer the .nemo's model_config.yaml for the preprocessor block and mark
    the reference as nemo-authoritative so the dtype check skips HF
    config.json's torch_dtype (NeMo training metadata that often disagrees
    with storage). The .nemo's state_dict dtype is the ground truth at
    Stage 3 convert; Gate B compares against the GGUF.
    """
    config = _load_json_from_hf(hf_repo, hf_revision, "config.json")
    pre = (
        _load_json_from_hf(hf_repo, hf_revision, "preprocessor_config.json")
        or _load_json_from_hf(hf_repo, hf_revision, "feature_extractor_config.json")
    )
    tok_cfg = _load_json_from_hf(hf_repo, hf_revision, "tokenizer_config.json")
    gen_cfg = _load_json_from_hf(hf_repo, hf_revision, "generation_config.json")

    nemo_authoritative = False
    nemo_source: str | None = None
    if reference_framework == "nemo":
        nemo_filename = _find_nemo_sibling(hf_repo, hf_revision)
        if nemo_filename:
            nemo_cfg = _load_nemo_model_config(hf_repo, hf_revision, nemo_filename)
            if nemo_cfg is not None:
                nemo_pre = _nemo_preprocessor_to_pre(nemo_cfg)
                if nemo_pre:
                    pre = nemo_pre  # NeMo's model_config.yaml is canonical for the family
                nemo_authoritative = True
                nemo_source = nemo_filename

    return {
        "config": config,
        "preprocessor": pre,
        "tokenizer_config": tok_cfg,
        "generation_config": gen_cfg,
        "nemo_authoritative": nemo_authoritative,
        "nemo_source": nemo_source,
    }


def _find_nemo_sibling(hf_repo: str, hf_revision: str | None) -> str | None:
    """Return the first `.nemo` filename in the repo siblings, or None."""
    try:
        info = HfApi().model_info(hf_repo, revision=hf_revision)
    except HfHubHTTPError:
        return None
    for sib in info.siblings or []:
        name = getattr(sib, "rfilename", None) or ""
        if name.endswith(".nemo"):
            return name
    return None


def _load_nemo_model_config(hf_repo: str, hf_revision: str | None, nemo_filename: str) -> dict | None:
    """Stream the .nemo tar from HF and pull out model_config.yaml without
    downloading the full archive. NeMo .nemo files are uncompressed tars and
    typically place model_config.yaml first, so this is cheap."""
    import yaml  # imported lazily so a missing pyyaml does not break non-NeMo flows

    rev = hf_revision or "main"
    fs_path = f"{hf_repo}@{rev}/{nemo_filename}"
    try:
        fs = HfFileSystem()
        with fs.open(fs_path, "rb") as f:
            # streaming mode: tarfile reads forward only and stops when we break
            with tarfile.open(fileobj=f, mode="r|") as tar:
                for member in tar:
                    base = Path(member.name).name
                    if base == "model_config.yaml":
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            return None
                        return yaml.safe_load(extracted.read())
    except (HfHubHTTPError, tarfile.TarError, OSError):
        return None
    return None


def _nemo_preprocessor_to_pre(model_config: dict) -> dict | None:
    """Translate NeMo's model.cfg.preprocessor block into the same shape that
    `_frontend_from_preprocessor` already understands (the HF preprocessor_config
    shape). Returns None if the preprocessor block is missing."""
    if not isinstance(model_config, dict):
        return None
    block = (model_config.get("preprocessor")
             or model_config.get("model", {}).get("preprocessor"))
    if not isinstance(block, dict):
        return None
    out: dict[str, Any] = {}
    if "sample_rate" in block:
        out["sampling_rate"] = block["sample_rate"]
    # NeMo: `features` is the mel filterbank size.
    if "features" in block:
        out["feature_size"] = block["features"]
    if "n_fft" in block:
        out["n_fft"] = block["n_fft"]
    if "window_size" in block:
        out["window_size"] = block["window_size"]
    if "window_stride" in block:
        out["window_stride"] = block["window_stride"]
    if "window" in block:
        out["window_function"] = block["window"]
    if "normalize" in block:
        out["normalize"] = block["normalize"]
    # NeMo writes "preemph"; HF preprocessors write "preemphasis".
    if "preemph" in block:
        out["preemphasis"] = block["preemph"]
    if "dither" in block:
        out["dither"] = block["dither"]
    return out or None


def _config_dtype(config: dict | None) -> str | None:
    if not config:
        return None
    for path in (("text_config", "dtype"), ("text_config", "torch_dtype"),
                 ("dtype",), ("torch_dtype",)):
        node: Any = config
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str):
            return node
    return None


# -------- GGUF discovery (matches validate.py convention) --------

def find_gguf(
    repo_root: Path,
    family: str,
    variant: str | None,
    hf_repo: str | None = None,
) -> Path | None:
    """Prefer BF16 > F32 > F16 > first match. Discovery, in order:

      1. If `hf_repo` is provided, try `models/<upstream-slug>/` first
         (e.g. `models/SenseVoiceSmall/`). This matches validate.py's
         convention and the converter's output dir for variants whose
         filesystem name preserves upstream HF casing.
      2. If `variant` is provided (and step 1 missed), try
         `models/<variant>/` (kebab-cased fallback for legacy layouts).
      3. Otherwise, scan `models/<family>*/` (legacy family-prefix fallback).
    """
    models_dir = repo_root / "models"
    if not models_dir.is_dir():
        return None
    candidates: list[Path] = []
    if hf_repo:
        upstream_slug = hf_repo.rsplit("/", 1)[-1]
        candidates = list((models_dir / upstream_slug).glob("*.gguf"))
    if not candidates and variant:
        candidates = list((models_dir / variant).glob("*.gguf"))
    if not candidates:
        for d in sorted(models_dir.glob(f"{family}*/")):
            candidates.extend(d.glob("*.gguf"))
    def rank(p: Path) -> int:
        name = p.name.upper()
        if "BF16" in name: return 0
        if "-F32" in name: return 1
        if "-F16" in name: return 2
        return 3
    return sorted(candidates, key=rank)[0] if candidates else None


# -------- Checks --------

def check_dtype(declared, gguf_kvs, reference, gate: Gate) -> CheckResult:
    decl_dtype = declared.get("expected_dtype")
    cfg_dtype = _config_dtype(reference.get("config"))
    nemo_authoritative = bool(reference.get("nemo_authoritative"))
    gguf_ftype = gguf_kvs.get("general.file_type") if gguf_kvs else None
    gguf_dtype = _gguf_file_type_to_str(gguf_ftype) if gguf_ftype is not None else None

    sources: dict[str, Any] = {"declared": decl_dtype, "reference_config": cfg_dtype}
    if nemo_authoritative:
        sources["nemo_authoritative"] = True
        sources["nemo_source"] = reference.get("nemo_source")
    if gguf_kvs:
        sources["gguf_file_type"] = gguf_ftype
        sources["gguf_dtype"] = gguf_dtype

    n_decl = _norm_dtype(decl_dtype) if decl_dtype else None
    n_gguf = _norm_dtype(gguf_dtype) if gguf_dtype else None
    # When NeMo is authoritative for this variant, HF config.json's torch_dtype
    # is training/optimizer metadata, not storage — skip the comparison.
    # Storage dtype is verified at Gate B against the converted GGUF.
    n_ref = None if nemo_authoritative else (_norm_dtype(cfg_dtype) if cfg_dtype else None)

    # Hard fail: declared vs GGUF disagree. The GGUF is what the C++ loader will see.
    if n_decl and n_gguf and n_decl != n_gguf:
        return CheckResult("dtype_consistency", gate, "fail", sources,
                           f"declared dtype '{decl_dtype}' does not match GGUF '{gguf_dtype}'")
    # Hard fail: declared vs reference disagree (declared is the inference contract).
    if n_decl and n_ref and n_decl != n_ref:
        return CheckResult("dtype_consistency", gate, "fail", sources,
                           f"declared dtype '{decl_dtype}' does not match reference config '{cfg_dtype}'")
    # Warn: no cross-source to compare against.
    if not n_ref and not n_gguf:
        detail = "no reference or GGUF dtype to cross-check declared value"
        if nemo_authoritative:
            detail += " (HF config.json torch_dtype skipped: NeMo .nemo is authoritative)"
        return CheckResult("dtype_consistency", gate, "warn", sources, detail)
    return CheckResult("dtype_consistency", gate, "pass", sources)


def _norm_dtype(s: str) -> str:
    return s.replace("torch.", "").lower().strip()


# GGUF general.file_type enum → dtype name. See gguf/constants.py (LlamaFileType).
_GGUF_FTYPE_MAP = {
    0: "float32", 1: "float16", 2: "q4_0", 3: "q4_1",
    7: "q8_0", 8: "q5_0", 9: "q5_1",
    15: "q4_k_m", 17: "q5_k_m", 18: "q6_k", 32: "bfloat16",
}


def _gguf_file_type_to_str(ftype: int) -> str | None:
    if isinstance(ftype, int):
        return _GGUF_FTYPE_MAP.get(ftype)
    return None


def check_frontend(declared, gguf_kvs, reference, gate: Gate) -> CheckResult:
    """Cross-check frontend fields across declared / GGUF / reference preprocessor.

    Fields are stratified by severity:
      HARD  — mismatch is a real bug. sample_rate, n_mels, fft_size, hop_length.
      SOFT  — mismatch is suspicious but often semantic. win_length, window,
              normalization.
      INFO  — shown but not compared. dither and preemphasis often differ
              between training defaults (reference) and inference overrides
              (declared); GGUF stores the inference values.
    """
    decl = declared.get("frontend") or {}
    ref_pre = reference.get("preprocessor") or {}
    gguf = gguf_kvs or {}

    sources: dict[str, Any] = {
        "declared": {k: decl.get(k) for k in _FRONTEND_FIELDS},
    }
    if gguf_kvs:
        sources["gguf"] = _frontend_from_gguf(gguf)
    if ref_pre:
        sources["reference"] = _frontend_from_preprocessor(ref_pre)

    mismatches: list[str] = []
    warnings: list[str] = []

    def cmp(source_name: str, source_dict: dict, fields: tuple, bucket: list):
        for f in fields:
            d, s = decl.get(f), source_dict.get(f)
            if d is not None and s is not None and _not_equal(d, s):
                bucket.append(f"declared.{f}={d} != {source_name}.{f}={s}")

    if gguf_kvs:
        cmp("gguf", sources["gguf"], _FRONTEND_HARD, mismatches)
        cmp("gguf", sources["gguf"], _FRONTEND_SOFT, warnings)
        _compare_window(decl.get("window"), sources["gguf"].get("window"),
                        "gguf", warnings)
    if ref_pre:
        cmp("reference", sources["reference"], _FRONTEND_HARD, mismatches)
        cmp("reference", sources["reference"], _FRONTEND_SOFT, warnings)

    if mismatches:
        status: Status = "fail"
        detail = "; ".join(mismatches)
    elif warnings:
        status = "warn"
        detail = "; ".join(warnings)
    elif not ref_pre and not gguf_kvs:
        status = "warn"
        detail = "no reference preprocessor and no GGUF to cross-check against"
    else:
        status = "pass"
        detail = ""
    return CheckResult("frontend_config", gate, status, sources, detail)


_FRONTEND_FIELDS = ("sample_rate", "n_mels", "hop_length", "fft_size", "win_length",
                    "window", "normalization", "preemphasis", "dither")
_FRONTEND_HARD = ("sample_rate", "n_mels", "fft_size", "hop_length")
_FRONTEND_SOFT = ("win_length", "normalization")
# _FRONTEND_INFO = ("dither", "preemphasis") — displayed but not compared.


def _frontend_from_gguf(gguf: dict) -> dict[str, Any]:
    return {
        "sample_rate": gguf.get("stt.frontend.sample_rate"),
        "n_mels": gguf.get("stt.frontend.num_mels"),
        "hop_length": gguf.get("stt.frontend.hop_length"),
        "fft_size": gguf.get("stt.frontend.n_fft"),
        "win_length": gguf.get("stt.frontend.win_length"),
        "window": gguf.get("stt.frontend.window"),
        "normalization": gguf.get("stt.frontend.normalize"),
        "preemphasis": gguf.get("stt.frontend.pre_emphasis"),
        "dither": gguf.get("stt.frontend.dither"),
    }


def _frontend_from_preprocessor(pre: dict) -> dict[str, Any]:
    sample_rate = pre.get("sampling_rate") or pre.get("sample_rate")
    n_mels = pre.get("feature_size") or pre.get("num_mel_bins")
    win_size = pre.get("window_size")
    if isinstance(win_size, (int, float)) and sample_rate:
        win_length = int(round(float(win_size) * sample_rate))
    else:
        win_length = pre.get("win_length")
    win_stride = pre.get("window_stride")
    if isinstance(win_stride, (int, float)) and sample_rate:
        hop_length = int(round(float(win_stride) * sample_rate))
    else:
        hop_length = pre.get("hop_length")
    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "fft_size": pre.get("n_fft"),
        "win_length": win_length,
        "window": pre.get("window_function") or pre.get("window"),
        "normalization": _canonical_normalize_for_compare(
            pre.get("normalize") or pre.get("normalization")
        ),
        "preemphasis": pre.get("preemphasis"),
        "dither": pre.get("dither"),
    }


def _canonical_normalize_for_compare(raw):
    """Reduce a reference cfg normalize value to the same canonical
    set the converter emits (per_feature / global / per_utterance / none).
    Lets preflight compare the intake's declared value against a
    framework-normalised reference value without spurious string-encoding
    warnings (NeMo "NA" vs schema "none", etc.)."""
    sys_path_added = False
    try:
        if "scripts" not in sys.path[0]:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            sys_path_added = True
        from lib.gguf_common import canonicalize_normalize
        try:
            return canonicalize_normalize(raw)
        except ValueError:
            # Unknown value — leave it raw so the comparison surfaces the
            # mismatch rather than silently masking it.
            return raw
    finally:
        if sys_path_added:
            sys.path.pop(0)


def _compare_window(declared_window, other_window, other_name: str, warnings: list):
    """Window comparison is special: GGUF stores just 'hann'; declared may say
    'hann_periodic' or 'hann_symmetric'. A base-name mismatch is real; a
    periodicity difference is unverifiable from the GGUF KV alone."""
    if not declared_window or not other_window:
        return
    d_base = declared_window.lower().split("_")[0]
    o_base = other_window.lower().split("_")[0]
    if d_base != o_base:
        warnings.append(f"declared.window={declared_window} != {other_name}.window={other_window}")


def _not_equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        # float32 tolerance: catches real differences without false-positives
        # from converter round-tripping (e.g. 0.97 stored as f32 → 0.9700000286).
        return abs(float(a) - float(b)) > 1e-6
    return a != b


def check_tokenizer(declared, gguf_kvs, reference, gate: Gate) -> CheckResult:
    decl = declared.get("tokenizer") or {}
    gguf = gguf_kvs or {}
    tok_cfg = reference.get("tokenizer_config") or {}
    gen_cfg = reference.get("generation_config") or {}

    sources: dict[str, Any] = {
        "declared": {
            "type": decl.get("type"),
            "vocab_size": decl.get("vocab_size"),
            "special_tokens": decl.get("special_tokens", {}),
        },
    }
    if gguf_kvs:
        tokens_field = gguf.get("tokenizer.ggml.tokens")
        vocab_len = tokens_field.get("length") if isinstance(tokens_field, dict) else None
        # RNNT/TDT converters pad the tokenizer table by one for the
        # blank/start-state token that lives in the predictor embed but
        # not in the upstream SPM. Intakes declare the SPM-only vocab,
        # so back the blank out of the count when the converter has
        # written tokenizer.ggml.blank_token_id at the end of the
        # table. This keeps the comparison apples-to-apples.
        blank_id = gguf.get("tokenizer.ggml.blank_token_id")
        if (
            isinstance(blank_id, int)
            and isinstance(vocab_len, int)
            and blank_id == vocab_len - 1
        ):
            vocab_size_for_compare = vocab_len - 1
        else:
            vocab_size_for_compare = vocab_len
        sources["gguf"] = {
            "type": gguf.get("tokenizer.ggml.model"),
            "vocab_size": vocab_size_for_compare,
            "tokens_length": vocab_len,
            "blank_id": blank_id,
            "bos": gguf.get("tokenizer.ggml.bos_token_id"),
            "eos": gguf.get("tokenizer.ggml.eos_token_id"),
            "pad": gguf.get("tokenizer.ggml.padding_token_id")
                or gguf.get("tokenizer.ggml.pad_token_id"),
            "unk": gguf.get("tokenizer.ggml.unknown_token_id"),
        }
    if tok_cfg or gen_cfg:
        src = {**tok_cfg, **gen_cfg}
        sources["reference"] = {
            "type": (tok_cfg.get("tokenizer_class") or "").lower(),
            "bos": src.get("bos_token_id"),
            "eos": src.get("eos_token_id"),
            "pad": src.get("pad_token_id"),
            "unk": src.get("unk_token_id"),
        }

    mismatches: list[str] = []
    if gguf_kvs:
        if decl.get("vocab_size") and sources["gguf"]["vocab_size"]:
            if decl["vocab_size"] != sources["gguf"]["vocab_size"]:
                mismatches.append(
                    f"declared.vocab_size={decl['vocab_size']} != gguf.vocab_size={sources['gguf']['vocab_size']}"
                )
        for role in ("bos", "eos", "pad", "unk"):
            d = decl.get("special_tokens", {}).get(role)
            g = sources["gguf"].get(role)
            if d is not None and g is not None and d != g:
                mismatches.append(f"declared.{role}={d} != gguf.{role}={g}")

    if "reference" in sources:
        for role in ("bos", "eos", "pad", "unk"):
            d = decl.get("special_tokens", {}).get(role)
            r = sources["reference"].get(role)
            if d is not None and r is not None and d != r:
                mismatches.append(f"declared.{role}={d} != reference.{role}={r}")

    status: Status = "fail" if mismatches else "pass"
    return CheckResult("tokenizer_alignment", gate, status, sources,
                       "; ".join(mismatches) if mismatches else "")


def check_architecture_sanity(declared, gguf_kvs, reference, gate: Gate) -> CheckResult:
    if not gguf_kvs:
        return CheckResult("architecture_sanity", gate, "warn",
                           {"detail": "gate A skips architecture check; GGUF does not exist yet"},
                           "skipped at gate A")
    gguf_arch = gguf_kvs.get("general.architecture")
    declared_family = declared.get("family") or ""
    sources = {
        "gguf_architecture": gguf_arch,
        "declared_family": declared_family,
    }
    if gguf_arch and declared_family:
        # family key may differ from upstream architecture (e.g. family=cohere,
        # arch=cohere_asr). Warn on soft mismatch rather than failing.
        if declared_family.lower() not in gguf_arch.lower() and gguf_arch.lower() not in declared_family.lower():
            return CheckResult("architecture_sanity", gate, "warn", sources,
                               f"GGUF architecture '{gguf_arch}' does not contain family key '{declared_family}' — confirm in family note")
    return CheckResult("architecture_sanity", gate, "pass", sources)


def check_capabilities(declared, gguf_kvs, reference, gate: Gate) -> CheckResult:
    """Cross-check declared languages + capability flags against the GGUF.

    Only runs at Gate B (GGUF exists). Gate A returns warn.
    """
    caps = declared.get("capabilities") or {}
    declared_langs = list(caps.get("languages") or [])
    declared_lang_detect = caps.get("language_detection")

    if not gguf_kvs:
        return CheckResult("capabilities", gate, "warn",
                           {"declared_languages": declared_langs,
                            "declared_language_detection": declared_lang_detect},
                           "skipped at gate A; GGUF does not exist yet")

    gguf_langs_raw = gguf_kvs.get("general.languages")
    gguf_langs = gguf_langs_raw if isinstance(gguf_langs_raw, list) else None
    gguf_lang_detect = gguf_kvs.get("stt.capability.lang_detect")

    sources: dict[str, Any] = {
        "declared_languages": declared_langs,
        "gguf_languages": gguf_langs,
        "declared_language_detection": declared_lang_detect,
        "gguf_lang_detect": gguf_lang_detect,
    }

    mismatches: list[str] = []
    warnings: list[str] = []

    if declared_langs and gguf_langs is not None:
        if set(declared_langs) != set(gguf_langs):
            only_decl = sorted(set(declared_langs) - set(gguf_langs))
            only_gguf = sorted(set(gguf_langs) - set(declared_langs))
            diff = []
            if only_decl:
                diff.append(f"only in declared: {only_decl}")
            if only_gguf:
                diff.append(f"only in gguf: {only_gguf}")
            mismatches.append("languages differ — " + "; ".join(diff))
        elif declared_langs != gguf_langs:
            warnings.append("language order differs (contents match)")

    if declared_lang_detect is not None and gguf_lang_detect is not None:
        if bool(declared_lang_detect) != bool(gguf_lang_detect):
            mismatches.append(
                f"language_detection declared={declared_lang_detect} "
                f"but gguf stt.capability.lang_detect={gguf_lang_detect}"
            )

    if mismatches:
        return CheckResult("capabilities", gate, "fail", sources, "; ".join(mismatches))
    if warnings:
        return CheckResult("capabilities", gate, "warn", sources, "; ".join(warnings))
    return CheckResult("capabilities", gate, "pass", sources)


# -------- Orchestration --------

CHECKS_BY_GATE: dict[Gate, list] = {
    "A": [check_dtype, check_frontend, check_tokenizer, check_capabilities],
    "B": [check_dtype, check_frontend, check_tokenizer, check_architecture_sanity, check_capabilities],
}


def run_gate(repo_root: Path, family: str, variant: str | None, gate: Gate,
             gguf_override: Path | None) -> PreflightReport:
    declared = load_declared_state(repo_root, family, variant)
    declared["family"] = family

    reference = load_reference_state(
        declared["hf_repo"],
        declared.get("hf_revision"),
        reference_framework=declared.get("reference_framework"),
    )

    gguf_kvs: dict[str, Any] | None = None
    if gate in ("B", "D"):
        gguf_path = gguf_override or find_gguf(
            repo_root, family, variant, hf_repo=declared.get("hf_repo")
        )
        if not gguf_path:
            raise SystemExit(
                f"error: gate {gate} requires a GGUF but none found under "
                f"models/{family}*/. Pass --gguf to override."
            )
        gguf_kvs = read_gguf_kvs(gguf_path)

    results = [fn(declared, gguf_kvs, reference, gate) for fn in CHECKS_BY_GATE[gate]]
    overall: Status = "pass"
    if any(r.status == "fail" for r in results):
        overall = "fail"
    elif any(r.status == "warn" for r in results):
        overall = "warn"

    resolved_variant = variant or _resolve_variant(repo_root, family)
    return PreflightReport(
        gate=gate,
        family=family,
        variant=resolved_variant,
        overall=overall,
        checks=results,
        generated_at=dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    )


def _resolve_variant(repo_root: Path, family: str) -> str:
    m = _find_manifest(repo_root, family, None)
    if m:
        return json.loads(m.read_text()).get("variant", family)
    return family


# -------- Output --------

def write_report(report: PreflightReport, out_dir: Path, *, render_md: bool) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"preflight-gate-{report.gate}.json"
    json_path.write_text(json.dumps(_report_to_dict(report), indent=2) + "\n")
    md_path: Path | None = None
    if render_md:
        md_path = out_dir / f"preflight-gate-{report.gate}.md"
        md_path.write_text(_render_markdown(report))
    return json_path, md_path


def _report_to_dict(report: PreflightReport) -> dict[str, Any]:
    d = asdict(report)
    d["checks"] = [asdict(c) for c in report.checks]
    return d


def _render_markdown(report: PreflightReport) -> str:
    icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}
    lines = [
        f"# Preflight Gate {report.gate} — {report.family}/{report.variant}",
        "",
        f"**Overall: {icon[report.overall]}**",
        f"Generated at: {report.generated_at}",
        "",
    ]
    for c in report.checks:
        lines.append(f"## {c.name} — {icon[c.status]}")
        lines.append("")
        if c.detail:
            lines.append(f"{c.detail}")
            lines.append("")
        lines.append("```json")
        lines.append(json.dumps(c.sources, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# -------- CLI --------

def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise SystemExit("error: cannot locate transcribe.cpp repo root")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", default=None,
                        help="Variant name; auto-detected if the family has exactly one.")
    parser.add_argument("--gate", required=True, choices=["A", "B"])
    parser.add_argument("--gguf", type=Path, default=None,
                        help="Override GGUF path (Gate B only). Auto-discovered otherwise.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory; defaults to reports/porting/<family>/<variant>/.")
    parser.add_argument("--render-md", action="store_true",
                        help="Also write a human-readable markdown copy alongside the JSON. Default: JSON only.")
    args = parser.parse_args()

    repo_root = _find_repo_root()
    report = run_gate(repo_root, args.family, args.variant, args.gate, args.gguf)
    out_dir = args.out or (repo_root / "reports" / "porting" / args.family / report.variant)
    json_path, md_path = write_report(report, out_dir, render_md=args.render_md)

    print(f"gate {args.gate} {report.overall.upper()}: {args.family}/{report.variant}")
    for c in report.checks:
        print(f"  {c.status.upper()}: {c.name}" + (f" — {c.detail}" if c.detail else ""))
    print(f"wrote {json_path}")
    if md_path is not None:
        print(f"wrote {md_path}")
    return 0 if report.overall != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
