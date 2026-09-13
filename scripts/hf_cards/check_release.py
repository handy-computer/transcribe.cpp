#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Check the date and validation pin in an HF card before upload."""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
HEX_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def as_date(value, field: str) -> dt.date:
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date, got {value!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--date", default=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                        help="UTC ship date expected in validation.date (default: today)")
    args = parser.parse_args()

    expected_date = as_date(args.date, "--date")
    path = REPO / "scripts" / "hf_cards" / f"{args.variant}.yaml"
    if not path.exists():
        print(f"FAIL: no HF card spec at {path}", file=sys.stderr)
        return 2
    spec = yaml.safe_load(path.read_text()) or {}
    errors = []
    try:
        pin_date = as_date(spec.get("pin_date"), "pin_date")
        if pin_date > expected_date:
            errors.append(f"pin_date {pin_date} is after ship date {expected_date}")
    except ValueError as exc:
        errors.append(str(exc))

    validation = spec.get("validation") or {}
    try:
        validation_date = as_date(validation.get("date"), "validation.date")
        if validation_date != expected_date:
            errors.append(
                f"validation.date is {validation_date}, expected ship date {expected_date}"
            )
    except ValueError as exc:
        errors.append(str(exc))
    commit = str(validation.get("commit") or "")
    if not HEX_SHA.fullmatch(commit):
        errors.append(f"validation.commit is not a commit SHA: {commit!r}")
    elif subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        errors.append(f"validation.commit {commit} is not present in this checkout")

    if errors:
        for error in errors:
            print(f"FAIL {args.variant}: {error}", file=sys.stderr)
        return 1
    print(f"OK {args.variant}: pin_date={pin_date}, validation={commit} on {expected_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
