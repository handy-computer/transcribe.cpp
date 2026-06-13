#!/usr/bin/env python3
"""Fail if the wheel-* CMake presets and the pyproject wheel-lane overrides
drift apart.

The same official build postures are defined twice, on purpose:

  - CMakePresets.json `wheel-*` presets — how a human hand-builds a
    provider-shaped artifact outside Python packaging (and, soon, how other
    bindings' docs describe the official postures).
  - the repo-root pyproject `[[tool.scikit-build.overrides]]` lanes keyed by
    TRANSCRIBE_WHEEL_LANE — what cibuildwheel actually ships.

Both files say "keep in sync" in comments; the 2026-06-12 lane-override
regex bug (unanchored "cpu" matched "cpu-vulkan"; wheels silently shipped
the wrong posture) is what discipline-by-comment buys. This script makes the
mirror structural:

  1. every lane regex must be anchored (^...$) — the regex bug class itself;
  2. every non-hidden wheel-* preset must be claimed by a lane mapping;
  3. per lane, the EFFECTIVE posture (preset cacheVariables resolved through
     `inherits`, vs lane cmake.args on top of the SKBUILD-implied base from
     CMakeLists.txt) must agree after applying the documented TRANSCRIBE_* ->
     GGML_* derivations.

    uv run --no-project python scripts/ci/check_lane_mirror.py

Exit 0 on agreement; 1 on drift (with a per-key diff).
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRESETS = REPO / "CMakePresets.json"
PYPROJECT = REPO / "pyproject.toml"

# lane value -> the presets that ship that posture. A new preset or lane
# must be entered here (check 2 enforces the preset side).
LANE_PRESETS = {
    "cpu-vulkan": ["wheel-linux-cpu-vulkan", "wheel-windows-cpu-vulkan"],
    "metal": ["wheel-macos-metal"],
    # The macOS x86_64 lane shares the conservative-CPU floor the Linux
    # preset defines; TRANSCRIBE_METAL=OFF is its only (default-restating)
    # addition — see DEFAULT_RESTATEMENTS.
    "cpu": ["wheel-linux-cpu"],
}

# The SKBUILD posture block in CMakeLists.txt: every lane builds on top of
# this implicitly; the presets restate it explicitly (minus the build-shape
# keys stripped by IGNORED_KEYS).
SKBUILD_IMPLIED = {
    "TRANSCRIBE_BUILD_SHARED": "ON",
    "TRANSCRIBE_USE_OPENMP": "OFF",
    "TRANSCRIBE_USE_SYSTEM_BLAS": "OFF",
    "GGML_METAL_EMBED_LIBRARY": "ON",
}

# Not posture: build type and which targets get built.
IGNORED_KEYS = {
    "CMAKE_BUILD_TYPE",
    "TRANSCRIBE_BUILD_TESTS",
    "TRANSCRIBE_BUILD_EXAMPLES",
    "TRANSCRIBE_BUILD_TOOLS",
}

# Keys that one side may pin to a value that is already the platform
# default while the other side omits it (pinned for intent, not effect).
DEFAULT_RESTATEMENTS = {
    ("TRANSCRIBE_METAL", "OFF"),  # default everywhere but Apple Silicon
    ("GGML_METAL", "OFF"),
}


def normalize(value: object) -> str:
    s = str(value).strip().upper()
    return {"TRUE": "ON", "FALSE": "OFF", "1": "ON", "0": "OFF"}.get(s, s)


def derive(posture: dict[str, str]) -> dict[str, str]:
    """Apply the TRANSCRIBE_* -> GGML_* fan-outs CMakeLists.txt performs, so
    a side that states only the TRANSCRIBE_ knob equals a side that also
    restates the GGML_ effect."""
    out = dict(posture)
    fanout = {
        "TRANSCRIBE_METAL": [("GGML_METAL", None)],
        "TRANSCRIBE_VULKAN": [("GGML_VULKAN", None)],
        "TRANSCRIBE_CUDA": [("GGML_CUDA", None)],
        # BACKEND_DL forces GGML_BACKEND_DL and GGML_NATIVE=OFF.
        "TRANSCRIBE_GGML_BACKEND_DL": [("GGML_BACKEND_DL", None),
                                       ("GGML_NATIVE", "OFF")],
        # The conservative floor defaults GGML_NATIVE (and the SIMD tiers,
        # which neither file lists) to OFF.
        "TRANSCRIBE_X86_CONSERVATIVE": [("GGML_NATIVE", "OFF")],
    }
    for knob, effects in fanout.items():
        if out.get(knob) == "ON":
            for key, forced in effects:
                out.setdefault(key, forced if forced else "ON")
    return out


def resolved_preset(presets: dict[str, dict], name: str) -> dict[str, str]:
    node = presets[name]
    base: dict[str, str] = {}
    inherits = node.get("inherits", [])
    for parent in [inherits] if isinstance(inherits, str) else inherits:
        base.update(resolved_preset(presets, parent))
    base.update({k: normalize(v) for k, v in node.get("cacheVariables", {}).items()})
    return base


def main() -> int:
    presets_doc = json.loads(PRESETS.read_text())
    presets = {p["name"]: p for p in presets_doc["configurePresets"]}

    pyproject = tomllib.loads(PYPROJECT.read_text())
    overrides = pyproject["tool"]["scikit-build"].get("overrides", [])

    failures: list[str] = []

    # Lane overrides: anchored regexes, literal lane names, parsed -D args.
    lanes: dict[str, dict[str, str]] = {}
    for ov in overrides:
        pattern = ov.get("if", {}).get("env", {}).get("TRANSCRIBE_WHEEL_LANE")
        if pattern is None:
            continue
        m = re.fullmatch(r"\^([A-Za-z0-9_-]+)\$", pattern)
        if not m:
            failures.append(
                f"lane override pattern {pattern!r} must be an anchored "
                f"literal (^lane$) — unanchored patterns are the 2026-06-12 "
                f"wrong-posture bug")
            continue
        lane = m.group(1)
        flags: dict[str, str] = {}
        for arg in ov.get("cmake", {}).get("args", []):
            dm = re.fullmatch(r"-D([A-Za-z0-9_]+)=(.+)", arg)
            if not dm:
                failures.append(f"lane {lane}: unparseable cmake arg {arg!r}")
                continue
            flags[dm.group(1)] = normalize(dm.group(2))
        lanes[lane] = flags

    if set(lanes) != set(LANE_PRESETS):
        failures.append(
            f"lane set mismatch: pyproject has {sorted(lanes)}, "
            f"LANE_PRESETS maps {sorted(LANE_PRESETS)} — update the mapping")

    claimed = {p for names in LANE_PRESETS.values() for p in names}
    visible_wheel_presets = {
        name for name, node in presets.items()
        if name.startswith("wheel-") and not node.get("hidden", False)
    }
    unclaimed = visible_wheel_presets - claimed
    if unclaimed:
        failures.append(
            f"presets not claimed by any lane: {sorted(unclaimed)} — a new "
            f"wheel preset needs a lane (or an entry here explaining why not)")

    def strip_and_settle(posture: dict[str, str]) -> dict[str, str]:
        eff = derive({k: v for k, v in posture.items() if k not in IGNORED_KEYS})
        # The embed knob is dead without Metal: SKBUILD implies it ON for
        # every lane (a no-op off-macOS) while only the metal preset states
        # it. Compare it only where Metal is actually on.
        if eff.get("GGML_METAL") != "ON" and eff.get("TRANSCRIBE_METAL") != "ON":
            eff.pop("GGML_METAL_EMBED_LIBRARY", None)
        return eff

    for lane, preset_names in LANE_PRESETS.items():
        if lane not in lanes:
            continue
        lane_eff = strip_and_settle({**SKBUILD_IMPLIED, **lanes[lane]})
        for preset_name in preset_names:
            preset_eff = strip_and_settle(resolved_preset(presets, preset_name))
            keys = set(lane_eff) | set(preset_eff)
            for key in sorted(keys):
                lv, pv = lane_eff.get(key), preset_eff.get(key)
                if lv == pv:
                    continue
                missing_side = "lane" if lv is None else "preset"
                present_value = pv if lv is None else lv
                if (key, present_value) in DEFAULT_RESTATEMENTS and None in (lv, pv):
                    continue  # one side pins a platform default for intent
                failures.append(
                    f"{lane} vs {preset_name}: {key}: lane={lv} preset={pv} "
                    f"({missing_side} side missing or different)")

    if failures:
        print("lane/preset mirror drift:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"lane mirror ok: {sorted(lanes)} match "
          f"{sorted(visible_wheel_presets)} (postures settled + compared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
