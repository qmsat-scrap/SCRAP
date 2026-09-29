#!/usr/bin/env python3
"""Behavior checks for the CSV filtering stage."""

from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from star_reject import one_to_one_links, read_frame, run


FIELDS = ["id", "x", "y", "area", "elongation", "segment_flux", "note"]


def write_frame(folder: Path, sequence: int, offset: float, *, star_count: int = 8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"image_20251008_23260{sequence}_{sequence}_149_detections.csv"
    rows = []
    for index in range(star_count):
        rows.append(
            {
                "id": str(1000 * sequence + index),
                "x": str(100.0 + index * 300 + offset),
                "y": str(200.0 + index * 250),
                "area": "15",
                "elongation": "1.1",
                "segment_flux": "250",
                "note": f"star {index}",
            }
        )
    rows.extend(
        [
            {"id": "1", "x": "3800", "y": "100", "area": "5", "elongation": "1", "segment_flux": "30", "note": "fixed sensor pixel"},
            {"id": "2", "x": str(3500 + 40 * sequence), "y": "2800", "area": "12", "elongation": "2", "segment_flux": "90", "note": "moving target"},
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def options(source: Path, destination: Path, *, min_frames: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        input=source,
        output=destination,
        min_frames=min_frames,
        max_shift=20.0,
        registration_tolerance=1.75,
        registration_min_area=10.0,
        min_registration_matches=6,
        min_registration_fraction=0.2,
        match_radius=2.0,
        max_track_spread=3.0,
        static_min_area=10.0,
    )


class StarRejectionTests(unittest.TestCase):
    def test_three_frame_motion_preserves_other_rows_and_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, destination = base / "source", base / "filtered"
            paths = [
                write_frame(source, 1, 0),
                write_frame(source, 2, 9),
                write_frame(source, 3, 13),
            ]
            first = run(options(source, destination))
            self.assertEqual(first["total_removed"], 24)
            self.assertEqual(first["total_remaining"], 6)
            self.assertAlmostEqual(first["frame_pairs"][0]["shift_x"], 9)
            self.assertAlmostEqual(first["frame_pairs"][1]["shift_x"], 4)
            for path in paths:
                with (destination / path.name).open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, FIELDS)
                    rows = list(reader)
                self.assertEqual([row["note"] for row in rows], ["fixed sensor pixel", "moving target"])
            before = {path.name: (destination / path.name).read_bytes() for path in paths}
            second = run(options(source, destination))
            self.assertEqual(second["total_removed"], first["total_removed"])
            self.assertEqual(before, {path.name: (destination / path.name).read_bytes() for path in paths})

    def test_two_frames_require_explicit_lower_evidence_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            write_frame(source, 1, 0)
            write_frame(source, 2, 9)
            self.assertEqual(run(options(source, base / "three"))["total_removed"], 0)
            self.assertEqual(run(options(source, base / "two", min_frames=2))["total_removed"], 16)

    def test_overlapping_three_frame_windows_cover_longer_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            for sequence, offset in ((1, 0), (2, 9), (3, 13), (4, 19)):
                write_frame(source, sequence, offset)
            result = run(options(source, base / "filtered"))
            self.assertEqual(result["total_removed"], 32)
            self.assertEqual([frame["removed_count"] for frame in result["frames"]], [8, 8, 8, 8])

    def test_sequence_gap_prevents_star_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            write_frame(source, 1, 0)
            write_frame(source, 3, 9)
            write_frame(source, 4, 13)
            result = run(options(source, base / "filtered"))
            self.assertEqual(result["total_removed"], 0)
            self.assertEqual(result["frame_pairs"][0]["reason"], "frame_sequence_gap")

    def test_one_detection_cannot_match_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            left = write_frame(source, 1, 0, star_count=0)
            right = write_frame(source, 2, 0, star_count=0)
            left_frame = read_frame(left, source)
            right_frame = read_frame(right, source)
            left_frame.points = np.array([[10.0, 10.0]])
            right_frame.points = np.array([[10.1, 10.0], [10.2, 10.0]])
            links = one_to_one_links(left_frame, right_frame, np.zeros(2), 1.0)
            self.assertEqual(links, {0: 0})


if __name__ == "__main__":
    unittest.main()
