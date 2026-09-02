#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""gen_gpu_denylist.py - regenerate src/transcribe-gpu-denylist-data.h.

The AUTO backend selector skips integrated GPUs that are known to be slower
than the CPU for ASR (see transcribe::gpu_auto_deny_reason). Devices are
identified by PCI vendor:device id, never by name, and the id sets are
generated from the upstream driver tables that define them:

  Intel  Mesa include/pci_ids/crocus_pci_ids.h (Gen 4 .. Gen 7.5) and
         iris_pci_ids.h (Gen 8+), filtered to the pre-Xe platforms.
  AMD    Linux drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c, filtered to the
         GCN 1-3 APU families (entries flagged AMD_IS_APU).

Both sources are MIT-licensed. They are fetched at the pinned commits below
so the output is reproducible; bump the pins deliberately. A platform or
family name that is not in the explicit deny/allow lists aborts generation
instead of guessing, so a table refresh always surfaces new codenames.

Usage:
    uv run scripts/gen_gpu_denylist.py            # fetch + regenerate
    uv run scripts/gen_gpu_denylist.py --check    # exit 1 if the header is stale
    uv run scripts/gen_gpu_denylist.py --src-dir DIR   # use pre-downloaded files
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "transcribe-gpu-denylist-data.h"

MESA_SHA = "d870cef8b7c8a4a11edc669669c9f18ae402314a"
LINUX_SHA = "89a312991dc6e638a36adc43ccb91dbc25504c04"

SOURCES = {
    "crocus_pci_ids.h": (
        f"https://gitlab.freedesktop.org/mesa/mesa/-/raw/{MESA_SHA}/include/pci_ids/crocus_pci_ids.h"
    ),
    "iris_pci_ids.h": (
        f"https://gitlab.freedesktop.org/mesa/mesa/-/raw/{MESA_SHA}/include/pci_ids/iris_pci_ids.h"
    ),
    "amdgpu_drv.c": (
        f"https://raw.githubusercontent.com/torvalds/linux/{LINUX_SHA}/drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c"
    ),
}

# Intel platform prefixes (the token before the first '_' in Mesa's CHIPSET
# name column). crocus covers Gen 4 .. Gen 7.5 and is denied wholesale; iris
# starts at Gen 8 and is split at the Xe boundary.
INTEL_DENY = {
    # crocus: Gen 4 .. Gen 7.5
    "i965", "g4x", "ilk", "snb", "ivb", "byt", "hsw", "chv",
    # iris: Gen 8 .. Gen 11 (Broadwell .. Ice Lake / Elkhart+Jasper Lake)
    "bdw", "bxt", "skl", "kbl", "glk", "cfl", "icl", "ehl",
}
INTEL_ALLOW = {
    # Xe-LP and later
    "tgl", "rkl", "adl", "rpl", "dg1", "sg1", "dg2", "atsm",
    "mtl", "arl", "lnl", "bmg", "ptl", "nvl", "wcl",
}

# AMD GCN 1-3 APU families. Vega (Raven/Picasso/Renoir/Cezanne) and RDNA
# APUs are deliberately not listed; the runtime shader-core-count rule
# handles the tiny RDNA display adapters.
AMD_DENY = {"KABINI", "MULLINS", "KAVERI", "CARRIZO", "STONEY"}

CHIPSET_RE = re.compile(r"CHIPSET\(\s*0x([0-9A-Fa-f]+)\s*,\s*([A-Za-z0-9_]+)")
AMDGPU_RE = re.compile(
    r"PCI_DEVICE\(0x1002,\s*0x([0-9A-Fa-f]+)\)\s*,\s*\.driver_data\s*=\s*([A-Z0-9_|]+)"
)


def fetch(name: str, src_dir: Path | None) -> str:
    if src_dir is not None:
        return (src_dir / name).read_text(encoding="utf-8")
    with urllib.request.urlopen(SOURCES[name], timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def intel_ids(text: str, source: str) -> tuple[set[int], dict[str, int]]:
    ids: set[int] = set()
    seen: dict[str, int] = {}
    for m in CHIPSET_RE.finditer(text):
        dev = int(m.group(1), 16)
        platform = m.group(2).split("_", 1)[0]
        seen[platform] = seen.get(platform, 0) + 1
        if platform in INTEL_DENY:
            ids.add(dev)
        elif platform not in INTEL_ALLOW:
            sys.exit(
                f"{source}: unknown Intel platform '{platform}' (0x{dev:04x}); "
                "add it to INTEL_DENY or INTEL_ALLOW"
            )
    return ids, seen


def amd_ids(text: str) -> tuple[set[int], dict[str, int]]:
    ids: set[int] = set()
    seen: dict[str, int] = {}
    for m in AMDGPU_RE.finditer(text):
        dev = int(m.group(1), 16)
        flags = m.group(2).split("|")
        family = flags[0].removeprefix("CHIP_")
        is_apu = "AMD_IS_APU" in flags
        if family in AMD_DENY:
            if not is_apu:
                sys.exit(f"amdgpu_drv.c: {family} 0x{dev:04x} is not flagged AMD_IS_APU")
            ids.add(dev)
            seen[family] = seen.get(family, 0) + 1
    missing = AMD_DENY - seen.keys()
    if missing:
        sys.exit(f"amdgpu_drv.c: no entries found for {sorted(missing)}")
    return ids, seen


def emit_array(name: str, comment: str, ids: set[int]) -> str:
    rows = []
    vals = sorted(ids)
    for i in range(0, len(vals), 8):
        rows.append("    " + ", ".join(f"0x{v:04x}" for v in vals[i : i + 8]) + ",")
    body = "\n".join(rows)
    return f"{comment}\ninline constexpr uint16_t {name}[] = {{\n{body}\n}};\n"


def render(intel: set[int], intel_seen: dict[str, int], amd: set[int], amd_seen: dict[str, int]) -> str:
    intel_note = ", ".join(f"{p}" for p in sorted(p for p in intel_seen if p in INTEL_DENY))
    amd_note = ", ".join(f"{f.title()}" for f in sorted(amd_seen))
    return f"""// transcribe-gpu-denylist-data.h - PCI device ids of integrated GPUs that
// AUTO backend selection skips. GENERATED by scripts/gen_gpu_denylist.py;
// do not edit by hand.
//
// Sources (both MIT-licensed):
//   Mesa  include/pci_ids/crocus_pci_ids.h, iris_pci_ids.h @ {MESA_SHA}
//   Linux drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c           @ {LINUX_SHA}
//
// These sets are closed: the hardware is out of production, so the ids
// never change and the table needs no refresh to stay correct. Unknown ids
// are always eligible (new hardware is never blocked by omission).

#pragma once

#include <cstdint>

namespace transcribe::gpu_denylist_data {{

{emit_array("kIntelPreXe",
            f"// Intel integrated graphics, Gen 4 .. Gen 11 (pre-Xe): {intel_note}.\n"
            f"// {len(intel)} ids, sorted ascending.",
            intel)}
{emit_array("kAmdGcnApu",
            f"// AMD GCN 1-3 APU graphics: {amd_note}.\n"
            f"// {len(amd)} ids, sorted ascending.",
            amd)}
}}  // namespace transcribe::gpu_denylist_data
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the header is current; do not write")
    ap.add_argument("--src-dir", type=Path, help="directory holding pre-downloaded source files")
    args = ap.parse_args()

    crocus, crocus_seen = intel_ids(fetch("crocus_pci_ids.h", args.src_dir), "crocus_pci_ids.h")
    iris, iris_seen = intel_ids(fetch("iris_pci_ids.h", args.src_dir), "iris_pci_ids.h")
    amd, amd_seen = amd_ids(fetch("amdgpu_drv.c", args.src_dir))

    intel = crocus | iris
    intel_seen = {**crocus_seen, **iris_seen}
    text = render(intel, intel_seen, amd, amd_seen)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(REPO)} is stale; run scripts/gen_gpu_denylist.py", file=sys.stderr)
            return 1
        print("ok")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(intel)} Intel ids, {len(amd)} AMD ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
