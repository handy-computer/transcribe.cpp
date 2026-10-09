"""Strict, dependency-free parser for transcribe-cli's VAD -o output."""
import re

_SEGMENT = re.compile(r"segment: (\d+) start=(\d+) end=(\d+) t0=\S+ t1=\S+")


def parse_segments(text: str) -> list[dict[str, int]]:
    segments = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _SEGMENT.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"unexpected VAD output: {line!r}")
        index, start, end = map(int, match.groups())
        if index != len(segments) or end <= start:
            raise ValueError(f"invalid VAD segment: {line!r}")
        segments.append({"start": start, "end": end})
    return segments
