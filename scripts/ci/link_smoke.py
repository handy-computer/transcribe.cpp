#!/usr/bin/env python3
"""Compile and run a toy C consumer against an INSTALLED transcribe tree.

    python3 scripts/ci/link_smoke.py --prefix <install-prefix> [--cc cc]

The link line is constructed from NOTHING but the installed
lib/transcribe-link.json — the manifest is the artifact under test. If the
manifest's archive order, system-library list, frameworks, or flags are
wrong, this fails at link or run time, which is the point: the manifest is
what the Rust -sys crate's build.rs (and any non-CMake consumer) will trust.

Covers both postures from one entry point: the manifest says whether the
install is static or shared; shared adds an rpath to the installed lib dir
and asserts the binary runs without LD_LIBRARY_PATH/DYLD_* help.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "scripts" / "ci" / "link_smoke.c"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True, help="cmake --install prefix")
    ap.add_argument("--cc", default="cc", help="C compiler driver (default cc)")
    args = ap.parse_args()

    prefix = Path(args.prefix).resolve()
    manifest_path = prefix / "lib" / "transcribe-link.json"
    manifest = json.loads(manifest_path.read_text())
    print(f"manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))

    include_dir = prefix / manifest["include_dir"]
    lib_dir = prefix / manifest["lib_dir"]
    out = Path(tempfile.mkdtemp(prefix="link-smoke-")) / "link_smoke"

    cmd = [args.cc, str(SOURCE), f"-I{include_dir}", "-o", str(out)]
    cmd += manifest["link_flags"]
    cmd += [f"-L{lib_dir}"]

    libs = [f"-l{name}" for name in manifest["libraries"]]
    if manifest["libraries"][1:] and platform.system() == "Linux":
        # Static archive sets are order-sensitive under single-pass GNU ld;
        # group them so the manifest's content (not its luck) is what's
        # being tested. macOS ld64 resolves regardless of order.
        cmd += ["-Wl,--start-group", *libs, *manifest["library_paths"],
                "-Wl,--end-group"]
    else:
        cmd += [*libs, *manifest["library_paths"]]

    cmd += [f"-l{name}" for name in manifest["system_libs"]]
    for framework in manifest["frameworks"]:
        cmd += ["-framework", framework]
    if manifest["shared"]:
        cmd += [f"-Wl,-rpath,{lib_dir}"]

    print("compile:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Run with a clean environment posture: no loader-path help. The rpaths
    # (binary -> lib_dir; installed libs -> $ORIGIN/@loader_path) must carry
    # the shared case on their own. DL installs compile in no backends:
    # hand the toy the installed module directory, the call a real
    # DL-posture consumer makes.
    run_cmd = [str(out)]
    if manifest["backend_dl"]:
        run_cmd.append(str(prefix / manifest["module_dir"]))
    res = subprocess.run(run_cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        print(f"link-smoke FAILED (exit {res.returncode})", file=sys.stderr)
        return 1
    if "link-smoke ok" not in res.stdout:
        print("link-smoke FAILED (missing ok marker)", file=sys.stderr)
        return 1
    posture = "shared" if manifest["shared"] else "static"
    print(f"link-smoke ok ({posture}, backends: {manifest['backends']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
