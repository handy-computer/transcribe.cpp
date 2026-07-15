#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""
der.py — score timed diarization output from a run.py --diarize report.

DER is speaker-label invariant. For each utterance this scorer finds the
one-to-one reference/hypothesis speaker assignment with maximum overlapping
speaker time, then reports the standard components:

    DER = (missed speaker time + false-alarm speaker time + confusion time)
          / reference speaker time

The report contract is deliberately interval-based; inline model text is not
parsed:

    {"id": "...", "duration_ms": 1234,
     "ref_speaker_segments":
       [{"t0_ms": 0, "t1_ms": 600, "speaker_id": "alice"}],
     "hyp_speaker_segments":
       [{"t0_ms": 10, "t1_ms": 590, "speaker_id": 1}]}

Speaker IDs are opaque labels: 0-based, 1-based, strings, and mixed reference /
hypothesis conventions all score identically after optimal assignment.

Reference intervals are trusted annotations and validated strictly. Hypothesis
intervals are model output, so defects that miss/false-alarm accounting
already penalizes are sanitized instead of failing the corpus: rows are
clamped to [0, duration_ms] and rows that are empty after clamping are
dropped. The exception is a hypothesis whose every row is the library's
zero-length [0, 0] "attributed but untimed" sentinel (granite's
speaker-attribution task); that is rejected because DER cannot be recovered
without time intervals.

Usage:
    uv run scripts/wer/der.py reports/wer/<...>.jsonl
    uv run scripts/wer/der.py reports/wer/<...>.jsonl --collar-ms 250
    uv run scripts/wer/der.py reports/wer/<...>.jsonl --ignore-overlap
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable


Speaker = Hashable


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: Speaker


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    reference: frozenset[Speaker]
    hypothesis: frozenset[Speaker]

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DerCounts:
    reference: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    scored: float = 0.0

    @property
    def errors(self) -> float:
        return self.missed + self.false_alarm + self.confusion

    @property
    def der(self) -> float | None:
        return self.errors / self.reference if self.reference > 0 else None

    def __iadd__(self, other: DerCounts) -> DerCounts:
        self.reference += other.reference
        self.missed += other.missed
        self.false_alarm += other.false_alarm
        self.confusion += other.confusion
        self.scored += other.scored
        return self


def _label_key(label: Speaker) -> tuple[str, str]:
    return type(label).__name__, repr(label)


def parse_segments(
    rows: object,
    *,
    field: str,
    utterance_id: str,
    duration_ms: float,
    sanitize: bool = False,
) -> list[Segment]:
    """Parse one side's speaker rows. Structural defects (wrong types, missing
    fields, non-finite times) always fail loud. Geometry is validated strictly
    for the reference; with ``sanitize=True`` (hypothesis) out-of-range model
    rows are clamped to [0, duration_ms] and rows empty after clamping are
    dropped — a defective hypothesis row is a scoring penalty, not a reason to
    make the corpus DER unavailable."""
    if not isinstance(rows, list):
        raise ValueError(f"{utterance_id}: {field} must be a list")

    parsed: list[tuple[float, float, Speaker]] = []
    for index, row in enumerate(rows):
        where = f"{utterance_id}: {field}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where} must be an object")
        if "speaker_id" not in row:
            raise ValueError(f"{where} is missing speaker_id")
        speaker = row["speaker_id"]
        if not isinstance(speaker, (str, int)) or isinstance(speaker, bool):
            raise ValueError(f"{where}.speaker_id must be a string or integer")
        if isinstance(speaker, str) and not speaker:
            raise ValueError(f"{where}.speaker_id must not be empty")

        start = row.get("t0_ms")
        end = row.get("t1_ms")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not math.isfinite(start)
            or not math.isfinite(end)
        ):
            raise ValueError(f"{where} requires finite numeric t0_ms/t1_ms")
        parsed.append((float(start), float(end), speaker))

    # An all-[0, 0] hypothesis is the library's "attributed but untimed"
    # sentinel (granite's speaker-attribution task), not a fixable defect.
    if sanitize and parsed and all(s == 0 and e == 0 for s, e, _ in parsed):
        raise ValueError(
            f"{utterance_id}: {field} is attributed but untimed ([0, 0]); "
            f"DER is unavailable"
        )

    segments: list[Segment] = []
    for index, (start, end, speaker) in enumerate(parsed):
        where = f"{utterance_id}: {field}[{index}]"
        if sanitize:
            start = max(start, 0.0)
            end = min(end, duration_ms)
            if end <= start:
                continue
        else:
            if start == 0 and end == 0:
                raise ValueError(
                    f"{where} is attributed but untimed ([0, 0]); "
                    f"DER is unavailable"
                )
            if start < 0 or end <= start:
                raise ValueError(f"{where} must satisfy 0 <= t0_ms < t1_ms")
            if end > duration_ms:
                raise ValueError(
                    f"{where}.t1_ms={end:g} exceeds duration_ms={duration_ms:g}"
                )
        segments.append(Segment(start, end, speaker))
    return segments


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def atomic_regions(
    reference: list[Segment],
    hypothesis: list[Segment],
    duration_ms: float,
    collar_ms: float,
    ignore_overlap: bool,
) -> list[Region]:
    half_collar = collar_ms / 2
    excluded = merge_intervals(
        (
            max(0.0, boundary - half_collar),
            min(duration_ms, boundary + half_collar),
        )
        for segment in reference
        for boundary in (segment.start, segment.end)
    )

    points = {0.0, duration_ms}
    for segment in (*reference, *hypothesis):
        points.add(segment.start)
        points.add(segment.end)
    for start, end in excluded:
        points.add(start)
        points.add(end)
    ordered = sorted(points)

    regions: list[Region] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        if any(left <= midpoint < right for left, right in excluded):
            continue
        ref_active = frozenset(
            segment.speaker
            for segment in reference
            if segment.start <= midpoint < segment.end
        )
        if ignore_overlap and len(ref_active) > 1:
            continue
        hyp_active = frozenset(
            segment.speaker
            for segment in hypothesis
            if segment.start <= midpoint < segment.end
        )
        regions.append(Region(start, end, ref_active, hyp_active))
    return regions


def maximum_weight_pairs(
    reference_labels: list[Speaker],
    hypothesis_labels: list[Speaker],
    weights: dict[tuple[Speaker, Speaker], float],
) -> list[tuple[Speaker, Speaker]]:
    """Maximum-weight one-to-one assignment using a padded Hungarian solve."""
    if not reference_labels or not hypothesis_labels:
        return []
    size = max(len(reference_labels), len(hypothesis_labels))
    matrix = [[0.0] * size for _ in range(size)]
    for i, ref in enumerate(reference_labels):
        for j, hyp in enumerate(hypothesis_labels):
            matrix[i][j] = weights.get((ref, hyp), 0.0)

    # Hungarian algorithm for square minimum-cost assignment. Negating the
    # overlap weights turns maximum overlap into minimum cost.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cost = -matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cost < minv[j]:
                    minv[j] = cost
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[Speaker, Speaker]] = []
    for hyp_index in range(1, size + 1):
        ref_index = p[hyp_index]
        if ref_index <= len(reference_labels) and hyp_index <= len(hypothesis_labels):
            ref = reference_labels[ref_index - 1]
            hyp = hypothesis_labels[hyp_index - 1]
            if weights.get((ref, hyp), 0.0) > 0:
                pairs.append((ref, hyp))
    return pairs


def score_intervals(
    reference: list[Segment],
    hypothesis: list[Segment],
    *,
    duration_ms: float,
    collar_ms: float = 0.0,
    ignore_overlap: bool = False,
) -> tuple[DerCounts, list[tuple[Speaker, Speaker]]]:
    regions = atomic_regions(
        reference, hypothesis, duration_ms, collar_ms, ignore_overlap
    )
    reference_labels = sorted({s.speaker for s in reference}, key=_label_key)
    hypothesis_labels = sorted({s.speaker for s in hypothesis}, key=_label_key)
    overlap: dict[tuple[Speaker, Speaker], float] = {}
    for region in regions:
        for ref in region.reference:
            for hyp in region.hypothesis:
                overlap[(ref, hyp)] = overlap.get((ref, hyp), 0.0) + region.duration
    pairs = maximum_weight_pairs(reference_labels, hypothesis_labels, overlap)
    paired = set(pairs)

    counts = DerCounts()
    for region in regions:
        duration = region.duration
        n_ref = len(region.reference)
        n_hyp = len(region.hypothesis)
        correct = sum(
            1
            for ref, hyp in paired
            if ref in region.reference and hyp in region.hypothesis
        )
        counts.scored += duration
        counts.reference += n_ref * duration
        counts.missed += max(n_ref - n_hyp, 0) * duration
        counts.false_alarm += max(n_hyp - n_ref, 0) * duration
        counts.confusion += (min(n_ref, n_hyp) - correct) * duration
    return counts, pairs


def _round_ms(value: float) -> float:
    return round(value, 3)


def score_entry(
    entry: dict, *, collar_ms: float, ignore_overlap: bool
) -> tuple[DerCounts, dict]:
    utterance_id = str(entry.get("id", "<unknown>"))
    if entry.get("error"):
        raise ValueError(f"{utterance_id}: transcription failed: {entry['error']}")
    duration = entry.get("duration_ms")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"{utterance_id}: duration_ms must be a positive number")
    duration = float(duration)
    if "ref_speaker_segments" not in entry:
        raise ValueError(f"{utterance_id}: missing ref_speaker_segments")
    if "hyp_speaker_segments" not in entry:
        raise ValueError(f"{utterance_id}: missing hyp_speaker_segments")
    reference = parse_segments(
        entry["ref_speaker_segments"],
        field="ref_speaker_segments",
        utterance_id=utterance_id,
        duration_ms=duration,
    )
    hypothesis = parse_segments(
        entry["hyp_speaker_segments"],
        field="hyp_speaker_segments",
        utterance_id=utterance_id,
        duration_ms=duration,
        sanitize=True,
    )
    counts, pairs = score_intervals(
        reference,
        hypothesis,
        duration_ms=duration,
        collar_ms=collar_ms,
        ignore_overlap=ignore_overlap,
    )
    detail = {
        "id": utterance_id,
        "der": round(counts.der, 6) if counts.der is not None else None,
        "reference_speaker_ms": _round_ms(counts.reference),
        "missed_speaker_ms": _round_ms(counts.missed),
        "false_alarm_speaker_ms": _round_ms(counts.false_alarm),
        "confusion_speaker_ms": _round_ms(counts.confusion),
        "scored_region_ms": _round_ms(counts.scored),
        "speaker_mapping": [
            {"hypothesis": hyp, "reference": ref} for ref, hyp in pairs
        ],
    }
    return counts, detail


def bootstrap_ci(
    per_file: list[DerCounts], n_boot: int, seed: int = 42
) -> tuple[float, float]:
    corpus = DerCounts()
    for counts in per_file:
        corpus += counts
    assert corpus.der is not None
    if n_boot <= 0 or len(per_file) == 1:
        return corpus.der, corpus.der

    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(n_boot):
        sample = DerCounts()
        for _ in per_file:
            sample += per_file[rng.randrange(len(per_file))]
        if sample.der is not None:
            rates.append(sample.der)
    if not rates:
        return corpus.der, corpus.der
    rates.sort()
    lo = rates[int(0.025 * len(rates))]
    hi = rates[min(int(0.975 * len(rates)), len(rates) - 1)]
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="run.py --diarize JSONL report")
    parser.add_argument(
        "--collar-ms",
        type=float,
        default=0.0,
        help="Total width in ms of the collar centered on every reference "
        "boundary (default: 0)",
    )
    parser.add_argument(
        "--ignore-overlap",
        action="store_true",
        help="Exclude regions with more than one active reference speaker",
    )
    parser.add_argument(
        "--n-boot", type=int, default=1000, help="Bootstrap iterations (default: 1000)"
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"error: {args.report} does not exist", file=sys.stderr)
        return 2
    if args.collar_ms < 0 or args.n_boot < 0:
        print("error: --collar-ms and --n-boot must be non-negative", file=sys.stderr)
        return 2

    entries: list[dict] = []
    with open(args.report, encoding="utf-8") as report:
        for line_number, line in enumerate(report, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"error: {args.report}:{line_number}: {error}", file=sys.stderr
                )
                return 2
            if record.get("type") != "batch_header":
                entries.append(record)
    if not entries:
        print("error: report has no utterance records", file=sys.stderr)
        return 2

    per_file_counts: list[DerCounts] = []
    per_utterance: list[dict] = []
    try:
        for entry in entries:
            counts, detail = score_entry(
                entry,
                collar_ms=args.collar_ms,
                ignore_overlap=args.ignore_overlap,
            )
            per_file_counts.append(counts)
            per_utterance.append(detail)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    total = DerCounts()
    for counts in per_file_counts:
        total += counts
    if total.der is None:
        print("error: report contains no reference speaker time", file=sys.stderr)
        return 2
    ci_lo, ci_hi = bootstrap_ci(per_file_counts, args.n_boot)

    print("\n============================================================")
    print(
        f"  DER:   {total.der * 100:.2f}%  "
        f"(95% CI: [{ci_lo * 100:.2f}%, {ci_hi * 100:.2f}%])"
    )
    print(f"  N:     {len(per_file_counts)} utterances")
    print(
        "  speaker time (s): "
        f"ref={total.reference / 1000:.2f}  "
        f"miss={total.missed / 1000:.2f}  "
        f"false alarm={total.false_alarm / 1000:.2f}  "
        f"confusion={total.confusion / 1000:.2f}"
    )
    print(
        f"  collar: {args.collar_ms:g} ms total width; "
        f"overlap: {'ignored' if args.ignore_overlap else 'scored'}"
    )
    print("============================================================\n")

    output_path = args.report.with_suffix(".der.json")
    output = {
        "metric": "der",
        "der": round(total.der, 6),
        "der_pct": round(total.der * 100, 2),
        "der_ci_lo": round(ci_lo, 6),
        "der_ci_hi": round(ci_hi, 6),
        "n": len(per_file_counts),
        "collar_ms": args.collar_ms,
        "ignore_overlap": args.ignore_overlap,
        "reference_speaker_ms": _round_ms(total.reference),
        "missed_speaker_ms": _round_ms(total.missed),
        "false_alarm_speaker_ms": _round_ms(total.false_alarm),
        "confusion_speaker_ms": _round_ms(total.confusion),
        "scored_region_ms": _round_ms(total.scored),
        "report_file": str(args.report),
        "per_utterance": per_utterance,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    print(f"score: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
