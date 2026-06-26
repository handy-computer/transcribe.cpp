#!/usr/bin/env python3
"""Extract the canonical native bundle from a repaired provider wheel.

    python3 scripts/ci/extract_native_bundle.py \
        --wheel-dir wheelhouse --tuple linux-x86_64-cpu-vulkan --out bundles

Produces <out>/transcribe-native-<tuple>.tar.gz containing:

    transcribe-native-<tuple>/
        libtranscribe.* + libggml* (+ backend modules)   <- the wheel's
        contract.json   (version, header_hash, backends, lane)  _native/ dir
        licenses/       (the wheel's .dist-info/licenses tree)

The REPAIRED wheel is the most-validated native artifact this project
produces (auditwheel/delvewheel/delocate + cibuildwheel's test phase +
the hardware-truth lanes all ran against it), so the bundle is those exact
bytes re-containered — the decided mechanism (2026-06-13) by which npm
platform packages, prebuilt-Rust, and any future ecosystem share one
canonical build per tuple instead of rebuilding. "Same native bytes" holds
by construction.

Notes:
  - Windows wheels carry no import .lib (the wheel install filters the
    wheel-dev component out); dlopen-style consumers need none, and
    compiled consumers are served by the TRANSCRIBE_INSTALL path. Revisit
    if an ecosystem needs the .lib inside bundles.
  - The tar is deterministic-ish (sorted names, zeroed mtimes) so re-runs
    of the same wheel produce byte-identical bundles.

Stdlib only.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path


def fail(msg: str) -> "NoReturn":  # noqa: F821 - py3.9 compat, comment only
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel-dir", required=True,
                    help="directory containing exactly one repaired *.whl")
    ap.add_argument("--tuple", required=True, dest="tuple_name",
                    help="build tuple, e.g. linux-x86_64-cpu-vulkan")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    wheels = sorted(Path(args.wheel_dir).glob("*.whl"))
    # The native provider wheel is the one we extract from. On the cu12 Modal
    # volume a pure-python API wheel (transcribe_cpp-*-py3-none-any.whl) rides
    # alongside it so the smokes can install the pair from the volume alone
    # (modal_cuda_build.py step 3), so select the native wheel by name rather
    # than assuming the dir holds exactly one. Single-wheel callers (the
    # cibuildwheel lanes, the Windows cu12 path) are unchanged.
    native = [w for w in wheels if w.name.startswith("transcribe_cpp_native")]
    if native:
        if len(native) != 1:
            fail(f"expected exactly one native wheel in {args.wheel_dir}, found "
                 f"{[w.name for w in native]}")
        wheel = native[0]
    elif len(wheels) == 1:
        wheel = wheels[0]
    else:
        fail(f"expected one wheel (or one transcribe_cpp_native* wheel) in "
             f"{args.wheel_dir}, found {[w.name for w in wheels]}")

    bundle_name = f"transcribe-native-{args.tuple_name}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir / f"{bundle_name}.tar.gz"

    # bundle-relative path -> bytes
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            # <pkg>/_native/<file...> -> flattened bundle root
            if len(parts) >= 3 and parts[1] == "_native" and \
                    parts[0].startswith("transcribe_cpp_native"):
                members["/".join(parts[2:])] = zf.read(info)
            # <dist>.dist-info/licenses/<path...> -> licenses/<path...>
            elif len(parts) >= 3 and parts[0].endswith(".dist-info") and \
                    parts[1] == "licenses":
                members["licenses/" + "/".join(parts[2:])] = zf.read(info)

    native_files = [n for n in members if "/" not in n]
    if not any(n.startswith("libtranscribe.") or n == "transcribe.dll"
               for n in native_files):
        fail(f"no libtranscribe in the wheel's _native/ (got {native_files})")
    if "contract.json" not in members:
        fail("contract.json missing from _native/ — wheel built before the "
             "contract stamp, or the install rules regressed")
    if not any(n.startswith("licenses/") for n in members):
        fail("no license files found in the wheel's dist-info")

    contract = json.loads(members["contract.json"])
    for key in ("version", "header_hash", "backends", "lane"):
        if key not in contract:
            fail(f"contract.json missing key {key!r}: {contract}")

    with tarfile.open(out_tar, "w:gz") as tf:
        for rel in sorted(members):
            data = members[rel]
            info = tarfile.TarInfo(name=f"{bundle_name}/{rel}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o755 if "/" not in rel and rel != "contract.json" \
                else 0o644
            tf.addfile(info, io.BytesIO(data))

    total = sum(len(v) for v in members.values())
    print(f"{out_tar}: {len(members)} files, {total / 1e6:.1f} MB uncompressed,"
          f" {out_tar.stat().st_size / 1e6:.1f} MB compressed")
    print(f"contract: version={contract['version']} "
          f"hash={contract['header_hash']} backends={contract['backends']} "
          f"lane={contract['lane']}")
    print(f"native files: {sorted(native_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
