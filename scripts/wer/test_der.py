#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import unittest

from der import Segment, parse_segments, score_intervals


class DerScorerTest(unittest.TestCase):
    def test_matches_pyannote_tutorial_fixture(self) -> None:
        reference = [
            Segment(0, 10, "A"),
            Segment(12, 20, "B"),
            Segment(24, 27, "A"),
            Segment(30, 40, "C"),
        ]
        hypothesis = [
            Segment(2, 13, "a"),
            Segment(13, 14, "d"),
            Segment(14, 20, "b"),
            Segment(22, 38, "c"),
            Segment(38, 40, "d"),
        ]
        counts, _ = score_intervals(reference, hypothesis, duration_ms=40)
        self.assertAlmostEqual(counts.der or 0, 0.5161290322580645)

    def test_permuted_labels_are_free(self) -> None:
        reference = [Segment(0, 500, "alice"), Segment(500, 1000, "bob")]
        hypothesis = [Segment(0, 500, 7), Segment(500, 1000, 3)]
        counts, pairs = score_intervals(
            reference, hypothesis, duration_ms=1000
        )
        self.assertEqual(counts.der, 0)
        self.assertEqual(set(pairs), {("alice", 7), ("bob", 3)})

    def test_confusion_uses_optimal_one_to_one_mapping(self) -> None:
        reference = [Segment(0, 1000, "a"), Segment(1000, 2000, "b")]
        hypothesis = [Segment(0, 2000, 1)]
        counts, _ = score_intervals(reference, hypothesis, duration_ms=2000)
        self.assertEqual(counts.reference, 2000)
        self.assertEqual(counts.missed, 0)
        self.assertEqual(counts.false_alarm, 0)
        self.assertEqual(counts.confusion, 1000)
        self.assertEqual(counts.der, 0.5)

    def test_miss_and_false_alarm_include_nonspeech(self) -> None:
        reference = [Segment(0, 1000, "a")]
        hypothesis = [Segment(500, 1500, 1)]
        counts, _ = score_intervals(reference, hypothesis, duration_ms=2000)
        self.assertEqual(counts.missed, 500)
        self.assertEqual(counts.false_alarm, 500)
        self.assertEqual(counts.confusion, 0)
        self.assertEqual(counts.der, 1)

    def test_overlap_uses_speaker_time(self) -> None:
        reference = [Segment(0, 1000, "a"), Segment(500, 1500, "b")]
        hypothesis = [Segment(0, 1000, 1), Segment(500, 1500, 2)]
        counts, _ = score_intervals(reference, hypothesis, duration_ms=1500)
        self.assertEqual(counts.reference, 2000)
        self.assertEqual(counts.der, 0)

    def test_collar_excludes_both_sides_of_reference_boundaries(self) -> None:
        reference = [Segment(100, 900, "a")]
        hypothesis = [Segment(120, 880, 1)]
        without_collar, _ = score_intervals(
            reference, hypothesis, duration_ms=1000
        )
        with_collar, _ = score_intervals(
            reference, hypothesis, duration_ms=1000, collar_ms=50
        )
        self.assertEqual(without_collar.missed, 40)
        self.assertEqual(with_collar.der, 0)

    def test_untimed_sentinel_is_rejected(self) -> None:
        for sanitize in (False, True):
            with self.assertRaisesRegex(ValueError, "DER is unavailable"):
                parse_segments(
                    [
                        {"t0_ms": 0, "t1_ms": 0, "speaker_id": 1},
                        {"t0_ms": 0, "t1_ms": 0, "speaker_id": 2},
                    ],
                    field="hyp_speaker_segments",
                    utterance_id="granite",
                    duration_ms=1000,
                    sanitize=sanitize,
                )

    def test_sanitize_clamps_hypothesis_to_duration(self) -> None:
        # Model-emitted timestamps drift past the end of audio (MOSS emergent
        # timestamps are unclamped); the reference contract stays strict.
        segments = parse_segments(
            [{"t0_ms": -20, "t1_ms": 1040, "speaker_id": 1}],
            field="hyp_speaker_segments",
            utterance_id="drift",
            duration_ms=1000,
            sanitize=True,
        )
        self.assertEqual(segments, [Segment(0, 1000, 1)])
        with self.assertRaisesRegex(ValueError, "exceeds duration_ms"):
            parse_segments(
                [{"t0_ms": 0, "t1_ms": 1040, "speaker_id": 1}],
                field="ref_speaker_segments",
                utterance_id="drift",
                duration_ms=1000,
            )

    def test_sanitize_drops_degenerate_hypothesis_rows(self) -> None:
        # Zero-length rows (moss patches out-of-order turns to t1 == t0),
        # fully out-of-range rows, and a stray [0, 0] among timed rows are
        # dropped rather than making the corpus unscoreable.
        segments = parse_segments(
            [
                {"t0_ms": 0, "t1_ms": 0, "speaker_id": 1},
                {"t0_ms": 500, "t1_ms": 500, "speaker_id": 1},
                {"t0_ms": 1200, "t1_ms": 1300, "speaker_id": 1},
                {"t0_ms": 100, "t1_ms": 900, "speaker_id": 2},
            ],
            field="hyp_speaker_segments",
            utterance_id="moss",
            duration_ms=1000,
            sanitize=True,
        )
        self.assertEqual(segments, [Segment(100, 900, 2)])


if __name__ == "__main__":
    unittest.main()
