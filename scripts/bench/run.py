#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
run.py — transcribe-bench driver for perf matrices.

Invokes transcribe-bench across a matrix of (backend, family, quant, sample)
cells, detects the host machine, and aggregates each (backend, family) pair
into a single JSON file under reports/perf/<machine-slug>/.

Usage:
    uv run scripts/bench/run.py
    uv run scripts/bench/run.py --family parakeet --quants f16
    uv run scripts/bench/run.py --samples jfk --iters 20 --warmup 5
    uv run scripts/bench/run.py --backends metal,cpu,vulkan
    uv run scripts/bench/run.py --backends cpu --name pre-refactor
    uv run scripts/bench/run.py --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf --samples jfk

  Options:
    --family {parakeet,cohere,all}   model family to bench (default: all)
    --quants Q[,Q...]                quants to bench (default: f16,q8_0,q4_k_m)
    --samples S[,S...]               sample stems (default: jfk,dots)
    --iters N                        measured iterations per cell (default: 2)
    --warmup N                       warmup iterations per cell (default: 1)
    --backends B[,B...]|all          backends to bench (default: all, auto-detected)
                                     valid: metal,cpu,vulkan,all
    --name LABEL                     stable label for named baselines; when set,
                                     output filenames use <name> instead of <ts>
                                     and are overwritten on re-run
    --model PATH                     escape hatch: bypass discovery
    --bench-bin PATH                 legacy override: only valid when exactly one
                                     backend is selected; supplies that backend's
                                     binary path
    --out-dir PATH                   override reports output root
    --dry-run                        print the selected backends + matrix

Backend resolution (each run also passes --backend <name> to the
bench binary so the library selector is exercised end-to-end):
    metal  -> repo/build/bin/transcribe-bench         --backend metal
    cpu    -> repo/build/bin/transcribe-bench         --backend cpu
    vulkan -> repo/build-vulkan/bin/transcribe-bench  --backend vulkan

Auto-detection (when --backends is unset or 'all'):
    metal  available if build/bin/transcribe-bench exists AND sys.platform=='darwin'
    cpu    available if build/bin/transcribe-bench exists
    vulkan available if build-vulkan/bin/transcribe-bench exists

Output: one aggregated JSON per (family, backend) at
    reports/perf/<machine-slug>/<timestamp-or-name>_<family>_<backend>.json
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
KNOWN_FAMILIES = ["parakeet", "cohere"]
KNOWN_BACKENDS = ["metal", "cpu", "vulkan"]


@dataclass(slots=True)
class Cell:
    family: str
    quant: str
    model_path: Path
    sample: str  # stem (e.g. "jfk")
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


def discover_family_models(repo: Path, family: str) -> dict[str, Path]:
    """Return {quant: path} for a given family, picking newest on collision.

    Layout is models/<slug>/<slug>-<QUANT>.gguf; any slug whose filename
    starts with `family` is treated as belonging to that family. Multiple
    variants (e.g. parakeet v2 + v3) collapse into the same {quant: path}
    map; newest-mtime wins per quant.
    """
    model_root = repo / "models"
    if not model_root.is_dir():
        return {}
    by_quant: dict[str, list[Path]] = {}
    for path in model_root.glob("*/*.gguf"):
        if not path.stem.startswith(family):
            continue
        quant = parse_quant_from_filename(path)
        if quant:
            by_quant.setdefault(quant, []).append(path)
    resolved: dict[str, Path] = {}
    for quant, paths in by_quant.items():
        if len(paths) > 1:
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"warning: multiple files for {family}/{quant}, "
                  f"picking newest: {paths[0].name}", file=sys.stderr)
        resolved[quant] = paths[0]
    return resolved


def resolve_samples(repo: Path, stems: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for stem in stems:
        path = repo / "samples" / f"{stem}.wav"
        if not path.exists():
            print(f"error: sample not found: {path}", file=sys.stderr)
            sys.exit(2)
        out.append((stem, path))
    return out


def discover_matrix(repo: Path, family_filter: str, quants: list[str],
                    sample_stems: list[str]) -> list[Cell]:
    families = KNOWN_FAMILIES if family_filter == "all" else [family_filter]
    samples = resolve_samples(repo, sample_stems)
    cells: list[Cell] = []
    for family in families:
        available = discover_family_models(repo, family)
        for quant in quants:
            if quant not in available:
                print(f"skipping {family}/{quant}: no file", file=sys.stderr)
                continue
            for stem, sample_path in samples:
                cells.append(Cell(family=family, quant=quant,
                                  model_path=available[quant],
                                  sample=stem, sample_path=sample_path))
    return cells


def resolve_single_model(repo: Path, model_arg: Path,
                         sample_stems: list[str]) -> list[Cell]:
    model_path = model_arg if model_arg.is_absolute() else (repo / model_arg)
    model_path = model_path.resolve()
    if not model_path.exists():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    family = model_path.parent.name
    quant = parse_quant_from_filename(model_path) or "unknown"
    samples = resolve_samples(repo, sample_stems)
    return [Cell(family=family, quant=quant, model_path=model_path,
                 sample=stem, sample_path=sample_path)
            for stem, sample_path in samples]


def group_by_family(cells: list[Cell]) -> dict[str, list[Cell]]:
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.family, []).append(cell)
    return groups


def _metal_binary(repo: Path) -> Path:
    return repo / "build/bin/transcribe-bench"


def _vulkan_binary(repo: Path) -> Path:
    return repo / "build-vulkan/bin/transcribe-bench"


def auto_detect_backends(repo: Path) -> list[str]:
    """Return list of backend ids that appear usable on this host."""
    found: list[str] = []
    metal_bin = _metal_binary(repo)
    vulkan_bin = _vulkan_binary(repo)
    if metal_bin.exists() and sys.platform == "darwin":
        found.append("metal")
    if metal_bin.exists():
        found.append("cpu")
    if vulkan_bin.exists():
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
        elif name == "cpu":
            # We reuse the darwin bench binary for cpu runs — the
            # --backend cpu flag makes it a strict-CPU run regardless
            # of what's compiled in.
            binary = bench_bin_override or _metal_binary(repo)
            backend_arg = "cpu"
            if not binary.exists():
                missing.append(f"cpu: {binary}")
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
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def _fmt_mean(summary: dict, field: str) -> str:
    val = (summary.get(field) or {}).get("mean")
    return f"{val:.1f}" if isinstance(val, (int, float)) else "-"


def print_summary_table(family: str, runs: list[dict], slug: str,
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

    title = (f"{slug} \u2022 {family} \u2022 {display_backend} \u2022 "
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
    p.add_argument("--family", choices=["parakeet", "cohere", "all"], default="all")
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
    p.add_argument("--model", type=Path, default=None,
                   help="run this specific model file (bypass discovery)")
    p.add_argument("--bench-bin", type=Path, default=None,
                   help="legacy override for the bench binary "
                        "(only valid when exactly one backend is selected)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output root (default: reports/perf)")
    p.add_argument("--dry-run", action="store_true",
                   help="print selected backends + matrix without running")
    return p.parse_args()


def _run_one_backend(backend: BackendSpec, by_family: dict[str, list[Cell]],
                     args: argparse.Namespace, repo: Path, machine: dict,
                     out_root: Path, timestamp: str, slug_ts: str,
                     git_sha: str) -> int:
    """Run the full family matrix against a single backend. Returns exit code."""
    exit_code = 0
    name_slug = slugify(args.name) if args.name else None

    for family, group in by_family.items():
        runs: list[dict] = []
        for cell in group:
            print(f"[{backend.name}][{family}] {cell.quant} \u00d7 {cell.sample} ...",
                  file=sys.stderr, flush=True)
            result = run_bench_binary(backend.binary, cell, args.iters,
                                      args.warmup, backend.backend_arg)
            if result is None:
                print("  FAILED, skipping", file=sys.stderr)
                exit_code = 1
                continue
            runs.append(result)

        if not runs:
            print(f"[{backend.name}][{family}] no successful runs, skipping output",
                  file=sys.stderr)
            continue

        out_dir = out_root / machine["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = name_slug if name_slug else slug_ts
        out_path = out_dir / f"{prefix}_{family}_{backend.name}.json"

        aggregate = {
            "schema": "transcribe-bench-driver-v1",
            "timestamp": timestamp,
            "name": args.name or "",
            "machine": machine,
            "git_sha": git_sha,
            "family": family,
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

        print_summary_table(family, runs, machine["slug"], args.iters,
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

    backends = resolve_backends(repo, args.backends, args.bench_bin)

    if args.model:
        if args.family != "all" or args.quants != ",".join(DEFAULT_QUANTS):
            print("warning: --family/--quants ignored when --model is set",
                  file=sys.stderr)
        cells = resolve_single_model(repo, args.model, sample_stems)
    else:
        cells = discover_matrix(repo, args.family, quants, sample_stems)

    by_family = group_by_family(cells)

    if args.dry_run:
        print(f"machine: {machine['slug']} ({machine['cpu_model']})")
        print(f"out:     {out_root}")
        print(f"iters={args.iters} warmup={args.warmup}"
              + (f" name={args.name}" if args.name else ""))
        print(f"backends ({len(backends)}):")
        for b in backends:
            print(f"  {b.name:<7} {b.binary} --backend {b.backend_arg}")
        if not by_family:
            print("(no cells resolved)")
            return 0
        total = sum(len(g) for g in by_family.values())
        print(f"cells: {total} per backend ({total * len(backends)} total runs)")
        for family, group in by_family.items():
            print(f"{family}: {len(group)} cells")
            for cell in group:
                print(f"  {cell.quant:<8} {cell.sample:<8} "
                      f"{cell.model_path.name}")
        return 0

    if not by_family:
        print("error: no cells to run", file=sys.stderr)
        return 1

    git_sha = get_git_sha(repo)

    exit_code = 0
    for backend in backends:
        rc = _run_one_backend(backend, by_family, args, repo, machine,
                              out_root, timestamp, slug_ts, git_sha)
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
