#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
run.py — transcribe-bench driver for perf matrices.

Invokes transcribe-bench across a matrix of (backend, variant, quant,
sample) cells, detects the host machine, and aggregates each (variant,
backend) pair into a single JSON file under reports/perf/<machine-slug>/.

A "variant" is the per-checkpoint slug used as the directory name under
models/, e.g. `Qwen3-ASR-0.6B`, `Qwen3-ASR-1.7B`, `parakeet-tdt-0.6b-v3`.
Each variant is benched independently; different sizes of the same model
family (0.6B vs 1.7B) are never collapsed.

Usage:
    uv run scripts/bench/run.py                                      # all variants
    uv run scripts/bench/run.py --models Qwen3-ASR-0.6B
    uv run scripts/bench/run.py --models Qwen/Qwen3-ASR-0.6B         # HF slug form
    uv run scripts/bench/run.py --models Qwen3-ASR-0.6B,Qwen3-ASR-1.7B
    uv run scripts/bench/run.py --models parakeet-tdt-0.6b-v3 --quants f16
    uv run scripts/bench/run.py --samples jfk --iters 20 --warmup 5
    uv run scripts/bench/run.py --backends metal,cpu,vulkan
    uv run scripts/bench/run.py --backends cpu --name pre-refactor
    uv run scripts/bench/run.py --models models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf --samples jfk

Each --models token resolves one of three ways:
    - `<dir-slug>`            (e.g. Qwen3-ASR-0.6B) — all GGUFs under
                              models/<dir-slug>/, filtered by --quants
    - `<org>/<dir-slug>`      (e.g. Qwen/Qwen3-ASR-0.6B) — HF-style slug;
                              everything before the last `/` is stripped,
                              the remainder is treated as a dir-slug
    - `<path-to-file.gguf>`   — that exact file only; --quants is ignored
                              for this token

When --models is omitted, every variant dir under models/*/ with at least
one matching GGUF is benched.

  Options:
    --models M[,M...]                variant slugs (short or HF-style) or
                                     .gguf paths (default: all variants)
    --quants Q[,Q...]                quants to bench (default: f16,q8_0,q4_k_m)
    --samples S[,S...]               sample stems (default: jfk,dots)
    --iters N                        measured iterations per cell (default: 2)
    --warmup N                       warmup iterations per cell (default: 1)
    --backends B[,B...]|all          backends to bench (default: all, auto-detected)
                                     valid: metal,cpu,cpu_accel,vulkan,all
    --name LABEL                     stable label for named baselines; when set,
                                     output filenames use <name> instead of <ts>
                                     and are overwritten on re-run
    --bench-bin PATH                 legacy override: only valid when exactly one
                                     backend is selected; supplies that backend's
                                     binary path
    --out-dir PATH                   override reports output root
    --dry-run                        print the selected backends + matrix

Backend resolution (each run also passes --backend <name> to the
bench binary so the library selector is exercised end-to-end):
    metal  -> repo/build/bin/transcribe-bench         --backend metal
    cpu    -> repo/build/bin/transcribe-bench         --backend cpu
    vulkan -> repo/build/bin/transcribe-bench         --backend vulkan
              (falls through to repo/build-vulkan/bin/transcribe-bench
              only when build/bin/transcribe-bench does not exist —
              legacy split-build setups)

Auto-detection (when --backends is unset or 'all'):
    metal  available if build/bin/transcribe-bench exists AND sys.platform=='darwin'
    cpu    available if build/bin/transcribe-bench exists
    vulkan available only if build-vulkan/bin/transcribe-bench exists
           (we cannot tell from a path whether build/bin/ was compiled
           with Vulkan support; pass --backends vulkan explicitly to
           bench Vulkan from a unified build)

Output: one aggregated JSON per (variant, backend) at
    reports/perf/<machine-slug>/<timestamp-or-name>_<variant>_<backend>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUANTS = ["f16", "q8_0", "q4_k_m"]
DEFAULT_SAMPLES = ["jfk", "dots"]
KNOWN_BACKENDS = ["metal", "cpu", "cpu_accel", "vulkan"]


@dataclass(slots=True)
class Cell:
    variant: str          # parent dir slug, e.g. "Qwen3-ASR-0.6B"
    quant: str
    model_path: Path
    sample: str           # stem (e.g. "jfk")
    sample_path: Path


@dataclass(slots=True)
class BackendSpec:
    name: str          # canonical id: "metal" | "cpu" | "vulkan"
    binary: Path       # resolved bench binary path
    backend_arg: str   # value passed as `--backend <arg>` to the bench
                       # binary. "metal"|"cpu"|"vulkan"|"auto". We always
                       # pass --backend explicitly so the library's
                       # backend selector is exercised end-to-end, even
                       # for cpu runs (strict-CPU on a BLAS/AMX host is
                       # a real regression target).


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def timestamp_for_filename(iso: str) -> str:
    # "2026-04-11T10:30:00Z" -> "20260411T103000"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", iso)
    if not m:
        return iso.replace("-", "").replace(":", "").replace("Z", "")
    y, mo, d, h, mi, s = m.groups()
    return f"{y}{mo}{d}T{h}{mi}{s}"


def slugify(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return out or "unknown"


def _run_capture(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return (proc.stdout.strip() or None) if proc.returncode == 0 else None


def _linux_file_field(path: str, prefix: str, sep: str, strip_quotes: bool = False) -> str | None:
    try:
        with open(path) as f:
            for line in f:
                if line.lower().startswith(prefix):
                    val = line.split(sep, 1)[1].strip()
                    return val.strip('"') if strip_quotes else val
    except OSError:
        return None
    return None


def detect_machine() -> dict:
    cpu_model: str | None = None
    os_str: str | None = None
    if sys.platform == "darwin":
        cpu_model = _run_capture(["sysctl", "-n", "machdep.cpu.brand_string"])
        product = _run_capture(["sw_vers", "-productName"])
        version = _run_capture(["sw_vers", "-productVersion"])
        os_str = f"{product} {version}" if product and version else product
    elif sys.platform.startswith("linux"):
        cpu_model = _linux_file_field("/proc/cpuinfo", "model name", ":")
        os_str = _linux_file_field("/etc/os-release", "pretty_name=", "=", True)
    slug = slugify(cpu_model) if cpu_model else f"unknown-{platform.machine() or 'arch'}"
    return {
        "slug": slug,
        "cpu_model": cpu_model or "unknown",
        "os": os_str or platform.platform(),
        "hostname": socket.gethostname(),
    }


def get_git_sha(repo: Path) -> str:
    out = _run_capture(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    return out or "unknown"


def _hash_hex(s: str) -> str:
    """SHA256 hex digest of a utf-8 string. Used to add transcript_sha256
    and token_ids_sha256 to each bench run, so a perf change that secretly
    perturbed the decoder output is visible in the diff."""
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_quant_from_filename(path: Path) -> str | None:
    # `<slug>-<QUANT>.gguf` (llama.cpp convention), e.g.
    # "parakeet-tdt-0.6b-v2-Q4_K_M.gguf" -> "q4_k_m"
    # "cohere-transcribe-03-2026-F16.gguf" -> "f16"
    name = path.name
    if not name.endswith(".gguf"):
        return None
    stem = name[: -len(".gguf")]
    if "-" not in stem:
        return None
    return stem.rsplit("-", 1)[1].lower()


def discover_all_variants(repo: Path) -> list[tuple[str, str, Path]]:
    """Every (variant, quant, path) under models/*/*.gguf."""
    model_root = repo / "models"
    if not model_root.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for path in sorted(model_root.glob("*/*.gguf")):
        variant = path.parent.name
        quant = parse_quant_from_filename(path)
        if quant:
            out.append((variant, quant, path))
    return out


def resolve_model_tokens(repo: Path, tokens: list[str] | None,
                         quants_lower: set[str]
                         ) -> list[tuple[str, str, Path]]:
    """Expand --models tokens to (variant, quant, path) triples.

    Semantics:
      - None/empty → all variants under models/, filtered by quants_lower
      - `*.gguf` token → that exact file (quants_lower is ignored for it)
      - anything else → dir slug; HF-style `org/slug` is stripped to `slug`,
        result is looked up at models/<slug>/ and fanned out filtered by
        quants_lower
    """
    if not tokens:
        return [(v, q, p) for v, q, p in discover_all_variants(repo)
                if q in quants_lower]

    out: list[tuple[str, str, Path]] = []
    for tok in tokens:
        if tok.endswith(".gguf"):
            p = Path(tok)
            p = p if p.is_absolute() else (repo / p).resolve()
            if not p.exists():
                print(f"error: model not found: {p}", file=sys.stderr)
                sys.exit(2)
            variant = p.parent.name
            quant = parse_quant_from_filename(p) or "unknown"
            out.append((variant, quant, p))
            continue

        # dir slug; strip any HF-style org prefix (`org/slug` -> `slug`)
        slug = tok.rsplit("/", 1)[-1]
        if not slug:
            print(f"error: empty model token: {tok!r}", file=sys.stderr)
            sys.exit(2)
        # Match case-insensitively against on-disk dirs and canonicalize
        # to the actual dir name. This makes typing / doc examples
        # tolerant on case-sensitive filesystems AND keeps report
        # filenames and variant tags in the canonical casing (e.g.
        # `Qwen3-ASR-0.6B`) regardless of how the token was entered.
        model_root = repo / "models"
        matches = ([d for d in model_root.iterdir()
                    if d.is_dir() and d.name.lower() == slug.lower()]
                   if model_root.is_dir() else [])
        if not matches:
            print(f"error: variant dir not found: "
                  f"{model_root / slug}", file=sys.stderr)
            sys.exit(2)
        dir_path = matches[0]
        slug = dir_path.name

        matched_any = False
        skipped: list[str] = []
        for path in sorted(dir_path.glob("*.gguf")):
            quant = parse_quant_from_filename(path)
            if not quant:
                continue
            if quant not in quants_lower:
                skipped.append(quant)
                continue
            out.append((slug, quant, path))
            matched_any = True
        if not matched_any:
            qs = ",".join(sorted(quants_lower))
            print(f"warning: no [{qs}] GGUFs in {dir_path} "
                  f"(found: {','.join(sorted(set(skipped))) or 'none'})",
                  file=sys.stderr)
    return out


def resolve_samples(repo: Path, stems: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for stem in stems:
        path = repo / "samples" / f"{stem}.wav"
        if not path.exists():
            print(f"error: sample not found: {path}", file=sys.stderr)
            sys.exit(2)
        out.append((stem, path))
    return out


def discover_matrix(repo: Path, model_tokens: list[str] | None,
                    quants: list[str],
                    sample_stems: list[str]) -> list[Cell]:
    quants_lower = {q.lower() for q in quants}
    entries = resolve_model_tokens(repo, model_tokens, quants_lower)
    samples = resolve_samples(repo, sample_stems)
    cells: list[Cell] = []
    for variant, quant, path in entries:
        for stem, sample_path in samples:
            cells.append(Cell(variant=variant, quant=quant,
                              model_path=path, sample=stem,
                              sample_path=sample_path))
    return cells


def group_by_variant(cells: list[Cell]) -> dict[str, list[Cell]]:
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.variant, []).append(cell)
    return groups


def _metal_binary(repo: Path) -> Path:
    return repo / "build/bin/transcribe-bench"


def _vulkan_binary(repo: Path) -> Path:
    """Prefer build/bin/transcribe-bench (modern unified build).

    Fall through to legacy build-vulkan/bin/transcribe-bench only when
    build/bin/transcribe-bench is absent. A modern build with Vulkan
    compiled in always lives at build/bin/. If the user has a stale
    build-vulkan/ directory they should delete it; we do not let it
    shadow the unified build.
    """
    primary = repo / "build/bin/transcribe-bench"
    if primary.exists():
        return primary
    return repo / "build-vulkan/bin/transcribe-bench"


def auto_detect_backends(repo: Path) -> list[str]:
    """Return list of backend ids that appear usable on this host.

    Auto-detect is conservative for vulkan: we cannot tell from a path
    alone whether build/bin/transcribe-bench was compiled with Vulkan
    support, and a --backends all that silently included a non-Vulkan
    binary would hard-fail. So vulkan only auto-detects when the user
    has the legacy build-vulkan/ tree (a strong signal they built
    Vulkan separately). To benchmark Vulkan from a unified build, pass
    --backends vulkan explicitly.
    """
    found: list[str] = []
    metal_bin = _metal_binary(repo)
    legacy_vulkan_bin = repo / "build-vulkan/bin/transcribe-bench"
    if metal_bin.exists() and sys.platform == "darwin":
        found.append("metal")
    if metal_bin.exists():
        found.append("cpu")
    if legacy_vulkan_bin.exists():
        found.append("vulkan")
    return found


def resolve_backends(repo: Path, requested: str | None,
                     bench_bin_override: Path | None) -> list[BackendSpec]:
    """
    Parse --backends and resolve each to a BackendSpec.

    If requested is None or "all", auto-detect.
    Explicit user selection bypasses availability check only for unknown-name
    rejection; missing binaries still fail with a clear error.
    """
    if requested is None or requested.strip().lower() == "all":
        names = auto_detect_backends(repo)
        if not names:
            print("error: no backends auto-detected; build at least one of "
                  "build/ or build-vulkan/", file=sys.stderr)
            sys.exit(2)
    else:
        names = [b.strip().lower() for b in requested.split(",") if b.strip()]
        unknown = [n for n in names if n not in KNOWN_BACKENDS]
        if unknown:
            print(f"error: unknown backend(s): {','.join(unknown)}; "
                  f"valid: {','.join(KNOWN_BACKENDS)}", file=sys.stderr)
            sys.exit(2)
        # dedupe, preserve order
        seen: set[str] = set()
        names = [n for n in names if not (n in seen or seen.add(n))]

    if bench_bin_override is not None and len(names) != 1:
        print("error: --bench-bin requires exactly one backend selected; "
              f"got {len(names)}: {','.join(names)}", file=sys.stderr)
        sys.exit(2)

    specs: list[BackendSpec] = []
    missing: list[str] = []
    for name in names:
        if name == "metal":
            binary = bench_bin_override or _metal_binary(repo)
            backend_arg = "metal"
            if not binary.exists():
                missing.append(f"metal: {binary}")
            elif sys.platform != "darwin" and bench_bin_override is None:
                # Explicit request on non-darwin: surface as unavailable.
                missing.append(f"metal: requires darwin host (got {sys.platform})")
                continue
        elif name == "cpu" or name == "cpu_accel":
            # We reuse the darwin bench binary for CPU runs. --backend cpu
            # forces strict-CPU regardless of what's compiled in;
            # --backend cpu_accel additionally layers any registered
            # accel backends (BLAS/AMX) onto the scheduler. The accel
            # backend has to be compiled in (e.g. GGML_BLAS=ON) for
            # cpu_accel to engage anything; otherwise it degrades to
            # plain cpu semantics.
            binary = bench_bin_override or _metal_binary(repo)
            backend_arg = name
            if not binary.exists():
                missing.append(f"{name}: {binary}")
        elif name == "vulkan":
            binary = bench_bin_override or _vulkan_binary(repo)
            backend_arg = "vulkan"
            if not binary.exists():
                missing.append(f"vulkan: {binary}")
        else:  # pragma: no cover
            continue
        specs.append(BackendSpec(name=name, binary=binary, backend_arg=backend_arg))

    if missing:
        print("error: requested backend(s) unavailable:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("hint: build the missing target, e.g. "
              "`cmake --build build -j --target transcribe-bench`",
              file=sys.stderr)
        sys.exit(2)

    return specs


def run_bench_binary(bench_bin: Path, cell: Cell, iters: int, warmup: int,
                     backend_arg: str) -> dict | None:
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # Always pass --backend explicitly so the library's backend
        # selector is exercised end-to-end on every run (including
        # metal/vulkan, which used to be implicit via binary choice).
        cmd = [
            str(bench_bin),
            "--model", str(cell.model_path),
            "--sample", str(cell.sample_path),
            "--iters", str(iters),
            "--warmup", str(warmup),
            "--json-out", str(tmp_path),
            "--quiet",
            "--backend", backend_arg,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  binary failed (exit {proc.returncode}):", file=sys.stderr)
            if proc.stderr.strip():
                print(f"  stderr: {proc.stderr.strip()}", file=sys.stderr)
            return None
        try:
            data = json.loads(tmp_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  failed to parse json-out: {e}", file=sys.stderr)
            return None
        schema = data.get("schema", "")
        if schema not in ("transcribe-bench-v1", "transcribe-bench-v2"):
            print(f"  bad schema: {schema!r}", file=sys.stderr)
            return None
        # Explicit cell identifiers — required by the Stage 5 standardized
        # schema. Derivable from model_path/sample_path but the review skill
        # wants them as top-level keys so a reader does not have to parse
        # paths.
        data["quant"] = cell.quant
        data["sample"] = cell.sample
        # Structural-regression fingerprints. The bench binary emits the
        # raw transcript and token IDs; we hash here so perf changes that
        # secretly perturbed decoding are visible in a report diff.
        hyp_text = data.get("hyp_text")
        if isinstance(hyp_text, str):
            data["transcript_sha256"] = _hash_hex(hyp_text)
        token_ids_csv = data.get("token_ids_csv")
        if isinstance(token_ids_csv, str):
            data["token_ids_sha256"] = _hash_hex(token_ids_csv)
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def _fmt_mean(summary: dict, field: str) -> str:
    val = (summary.get(field) or {}).get("mean")
    return f"{val:.1f}" if isinstance(val, (int, float)) else "-"


def print_summary_table(variant: str, runs: list[dict], slug: str,
                        iters: int, warmup: int, backend_label: str,
                        name: str | None = None) -> None:
    # All runs in a group share one backend now; still derive a display label
    # from the first run's `backend` field if present, else fall back.
    display_backend = runs[0].get("backend", backend_label) if runs else backend_label

    header = ("quant", "sample", "load_ms", "encode_ms", "decode_ms", "wall_ms", "rtf")
    rows: list[tuple[str, ...]] = []
    for r in runs:
        summary = r.get("summary", {}) or {}
        # Prefer wall-based RTF (v2); fall back to v1's rtf_mean.
        rtf = r.get("rtf_wall_mean") or r.get("rtf_mean")
        rtf_s = f"{rtf:.3f}" if isinstance(rtf, (int, float)) else "-"
        quant = parse_quant_from_filename(Path(r.get("model_path", ""))) or "-"
        sample_name = Path(r.get("sample_path", "")).name or "-"
        # load_ms is a one-shot captured before any measured iteration; it
        # lives at top-level in the cell JSON, not inside summary{}.
        load_ms = r.get("load_ms")
        load_ms_s = f"{load_ms:.1f}" if isinstance(load_ms, (int, float)) else "-"
        rows.append((quant, sample_name, load_ms_s,
                     _fmt_mean(summary, "encode_ms"),
                     _fmt_mean(summary, "decode_ms"),
                     _fmt_mean(summary, "wall_ms"), rtf_s))

    widths = [max(len(header[i]), max((len(row[i]) for row in rows), default=0))
              for i in range(len(header))]
    # Left-align text columns (0,1), right-align numeric (2..6).
    def fmt_row(row: tuple[str, ...]) -> str:
        parts = [row[i].ljust(widths[i]) if i < 2 else row[i].rjust(widths[i])
                 for i in range(len(row))]
        return "  " + "   ".join(parts)

    title = (f"{slug} \u2022 {variant} \u2022 {display_backend} \u2022 "
             f"iters={iters} warmup={warmup}")
    if name:
        title += f" \u2022 name={name}"

    print()
    print(title)
    print()
    print(fmt_row(header))
    for row in rows:
        print(fmt_row(row))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated variant slugs (short form like "
                        "'Qwen3-ASR-0.6B', HF form like 'Qwen/Qwen3-ASR-0.6B', "
                        "or paths to .gguf files); default: all variants "
                        "under models/")
    p.add_argument("--quants", type=str, default=",".join(DEFAULT_QUANTS))
    p.add_argument("--samples", type=str, default=",".join(DEFAULT_SAMPLES))
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--backends", type=str, default=None,
                   help="comma-separated list: metal,cpu,vulkan or 'all' "
                        "(default: auto-detect)")
    p.add_argument("--name", type=str, default=None,
                   help="stable label for named baselines "
                        "(replaces timestamp in output filename)")
    p.add_argument("--bench-bin", type=Path, default=None,
                   help="legacy override for the bench binary "
                        "(only valid when exactly one backend is selected)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output root (default: reports/perf)")
    p.add_argument("--dry-run", action="store_true",
                   help="print selected backends + matrix without running")
    return p.parse_args()


def _run_one_backend(backend: BackendSpec,
                     by_variant: dict[str, list[Cell]],
                     args: argparse.Namespace, repo: Path, machine: dict,
                     out_root: Path, timestamp: str, slug_ts: str,
                     git_sha: str) -> int:
    """Run the full variant matrix against a single backend. Returns exit code."""
    exit_code = 0
    name_slug = slugify(args.name) if args.name else None

    for variant, group in by_variant.items():
        runs: list[dict] = []
        for cell in group:
            print(f"[{backend.name}][{variant}] {cell.quant} \u00d7 {cell.sample} ...",
                  file=sys.stderr, flush=True)
            result = run_bench_binary(backend.binary, cell, args.iters,
                                      args.warmup, backend.backend_arg)
            if result is None:
                print("  FAILED, skipping", file=sys.stderr)
                exit_code = 1
                continue
            runs.append(result)

        if not runs:
            print(f"[{backend.name}][{variant}] no successful runs, "
                  "skipping output", file=sys.stderr)
            continue

        out_dir = out_root / machine["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = name_slug if name_slug else slug_ts
        out_path = out_dir / f"{prefix}_{variant}_{backend.name}.json"

        aggregate = {
            "schema": "transcribe-bench-driver-v1",
            "timestamp": timestamp,
            "name": args.name or "",
            "machine": machine,
            "git_sha": git_sha,
            "variant": variant,
            "backend": backend.name,
            "iters": args.iters,
            "warmup": args.warmup,
            "runs": runs,
        }
        out_path.write_text(json.dumps(aggregate, indent=2) + "\n")
        try:
            rel = out_path.relative_to(repo)
        except ValueError:
            rel = out_path
        print(f"saved: {rel}", file=sys.stderr)

        print_summary_table(variant, runs, machine["slug"], args.iters,
                            args.warmup, backend.name, name=args.name)

    return exit_code


def main() -> int:
    args = parse_args()
    repo = find_repo_root(Path(__file__).parent)
    machine = detect_machine()
    out_root = args.out_dir or (repo / "reports/perf")
    timestamp = now_utc_iso()
    slug_ts = timestamp_for_filename(timestamp)

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    sample_stems = [s.strip() for s in args.samples.split(",") if s.strip()]
    model_tokens: list[str] | None = None
    if args.models:
        model_tokens = [t.strip() for t in args.models.split(",") if t.strip()]

    backends = resolve_backends(repo, args.backends, args.bench_bin)

    cells = discover_matrix(repo, model_tokens, quants, sample_stems)
    by_variant = group_by_variant(cells)

    if args.dry_run:
        print(f"machine: {machine['slug']} ({machine['cpu_model']})")
        print(f"out:     {out_root}")
        print(f"iters={args.iters} warmup={args.warmup}"
              + (f" name={args.name}" if args.name else ""))
        print(f"backends ({len(backends)}):")
        for b in backends:
            print(f"  {b.name:<7} {b.binary} --backend {b.backend_arg}")
        if not by_variant:
            print("(no cells resolved)")
            return 0
        total = sum(len(g) for g in by_variant.values())
        print(f"cells: {total} per backend ({total * len(backends)} total runs)")
        for variant, group in by_variant.items():
            print(f"{variant}: {len(group)} cells")
            for cell in group:
                print(f"  {cell.quant:<8} {cell.sample:<8} "
                      f"{cell.model_path.name}")
        return 0

    if not by_variant:
        print("error: no cells to run", file=sys.stderr)
        return 1

    git_sha = get_git_sha(repo)

    exit_code = 0
    for backend in backends:
        rc = _run_one_backend(backend, by_variant, args, repo, machine,
                              out_root, timestamp, slug_ts, git_sha)
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
