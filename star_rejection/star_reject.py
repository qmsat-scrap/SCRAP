#!/usr/bin/env python3
"""Remove sources that follow the common star-field motion across frames.

Input files are the ``*_detections.csv`` catalogs from SCRAP's
``photutils_detect.py``. The program preserves each CSV's columns and row values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SOBER_NAME = re.compile(
    r"^image_(?P<date>\d{8})_(?P<time>\d{6})_(?P<sequence>\d+)_(?P<channel>\d+)_detections\.csv$"
)


@dataclass
class Frame:
    path: Path
    relative_path: Path
    fields: list[str]
    rows: list[dict[str, str]]
    points: np.ndarray
    areas: np.ndarray
    group: tuple[str, ...]
    order: tuple[object, ...]
    sequence: int | None
    rejected: set[int] = field(default_factory=set)


@dataclass
class PairResult:
    shift: np.ndarray | None
    support: int
    registration_sources: int
    links: dict[int, int]
    reason: str | None = None


def frame_identity(relative_path: Path) -> tuple[tuple[str, ...], tuple[object, ...], int | None]:
    match = SOBER_NAME.fullmatch(relative_path.name)
    if match:
        sequence = int(match["sequence"])
        return (
            (str(relative_path.parent), match["date"], match["channel"]),
            (sequence, match["time"], relative_path.name),
            sequence,
        )
    return (str(relative_path.parent), "generic"), (relative_path.name,), None


def read_frame(path: Path, root: Path) -> Frame:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        required = {"x", "y", "area"}
        if not required.issubset(fields):
            raise ValueError(f"{path}: missing columns {sorted(required - set(fields))}")
        rows = list(reader)

    points = np.full((len(rows), 2), np.nan, dtype=float)
    areas = np.full(len(rows), np.nan, dtype=float)
    for index, row in enumerate(rows):
        try:
            x, y, area = float(row["x"]), float(row["y"]), float(row["area"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(area) and area >= 0:
            points[index] = (x, y)
            areas[index] = area

    relative_path = path.relative_to(root)
    group, order, sequence = frame_identity(relative_path)
    return Frame(path, relative_path, fields, rows, points, areas, group, order, sequence)


def registration_shift(
    left: Frame,
    right: Frame,
    *,
    max_shift: float,
    tolerance: float,
    min_area: float,
    min_support: int,
    min_fraction: float,
) -> tuple[np.ndarray | None, int, int, str | None]:
    left_points = left.points[np.isfinite(left.points).all(axis=1) & (left.areas >= min_area)]
    right_points = right.points[np.isfinite(right.points).all(axis=1) & (right.areas >= min_area)]
    source_count = min(len(left_points), len(right_points))
    if source_count < min_support:
        return None, 0, source_count, "too_few_registration_sources"

    right_tree = cKDTree(right_points)
    displacements = [
        right_points[j] - point
        for point, neighbors in zip(left_points, right_tree.query_ball_point(left_points, max_shift))
        for j in neighbors
    ]
    if len(displacements) < min_support:
        return None, 0, source_count, "too_few_candidate_pairs"

    vectors = np.asarray(displacements, dtype=float)
    neighborhoods = cKDTree(vectors).query_ball_point(vectors, tolerance)
    best = max(range(len(vectors)), key=lambda i: len(neighborhoods[i]))
    shift = np.median(vectors[neighborhoods[best]], axis=0)
    # Count independent source pairs rather than every nearby vector. In a
    # crowded patch, one source may have several candidate neighbors.
    candidate_links: list[tuple[float, int, int]] = []
    for left_index, point in enumerate(left_points):
        target = point + shift
        for right_index in right_tree.query_ball_point(target, tolerance):
            residual = float(np.linalg.norm(right_points[right_index] - target))
            candidate_links.append((residual, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    independent_vectors = []
    for _, left_index, right_index in sorted(candidate_links):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        independent_vectors.append(right_points[right_index] - left_points[left_index])
    support = len(independent_vectors)
    if support < min_support or support / source_count < min_fraction:
        return None, support, source_count, "no_dominant_motion"
    return np.median(independent_vectors, axis=0), support, source_count, None


def one_to_one_links(
    left: Frame, right: Frame, shift: np.ndarray, radius: float
) -> dict[int, int]:
    left_indices = np.flatnonzero(np.isfinite(left.points).all(axis=1))
    right_indices = np.flatnonzero(np.isfinite(right.points).all(axis=1))
    if not len(left_indices) or not len(right_indices):
        return {}

    right_points = right.points[right_indices]
    tree = cKDTree(right_points)
    candidates: list[tuple[float, int, int]] = []
    for left_index in left_indices:
        target = left.points[left_index] + shift
        for local_right_index in tree.query_ball_point(target, radius):
            right_index = int(right_indices[local_right_index])
            distance = float(np.linalg.norm(right.points[right_index] - target))
            candidates.append((distance, int(left_index), right_index))

    # Only reciprocal best matches survive. This leaves crowded, ambiguous
    # detections unmatched instead of silently linking the wrong objects.
    left_best: dict[int, tuple[float, int]] = {}
    right_best: dict[int, tuple[float, int]] = {}
    for distance, left_index, right_index in sorted(candidates):
        left_best.setdefault(left_index, (distance, right_index))
        right_best.setdefault(right_index, (distance, left_index))
    return {
        left_index: right_index
        for left_index, (_, right_index) in left_best.items()
        if right_best[right_index][1] == left_index
    }


def pair_frames(left: Frame, right: Frame, args: argparse.Namespace) -> PairResult:
    if (
        left.sequence is not None
        and right.sequence is not None
        and right.sequence - left.sequence != 1
    ):
        return PairResult(None, 0, 0, {}, "frame_sequence_gap")

    shift, support, source_count, reason = registration_shift(
        left,
        right,
        max_shift=args.max_shift,
        tolerance=args.registration_tolerance,
        min_area=args.registration_min_area,
        min_support=args.min_registration_matches,
        min_fraction=args.min_registration_fraction,
    )
    if shift is None:
        return PairResult(None, support, source_count, {}, reason)
    return PairResult(
        shift,
        support,
        source_count,
        one_to_one_links(left, right, shift, args.match_radius),
    )


def is_consistent_track(
    detections: list[tuple[Frame, int]],
    shifts: list[np.ndarray],
    *,
    max_spread: float,
    static_min_area: float,
) -> bool:
    positions = [detections[0][0].points[detections[0][1]]]
    accumulated_shift = np.zeros(2)
    for (frame, index), shift in zip(detections[1:], shifts):
        accumulated_shift += shift
        positions.append(frame.points[index] - accumulated_shift)

    for i, point in enumerate(positions):
        for other in positions[i + 1 :]:
            if np.linalg.norm(point - other) > max_spread:
                return False

    # Exact sensor-coordinate repeats in the SOBER sample are mostly tiny
    # detector defects. Demand a larger source if the field itself did not move.
    if all(np.linalg.norm(shift) < 1.0 for shift in shifts):
        if any(frame.areas[index] < static_min_area for frame, index in detections):
            return False
    return True


def pair_summary(left: Frame, right: Frame, result: PairResult) -> dict[str, object]:
    return {
        "left": str(left.relative_path),
        "right": str(right.relative_path),
        "shift_x": float(result.shift[0]) if result.shift is not None else None,
        "shift_y": float(result.shift[1]) if result.shift is not None else None,
        "registration_support": result.support,
        "registration_sources": result.registration_sources,
        "linked_detections": len(result.links),
        "reason": result.reason,
    }


def classify_window(
    frames: list[Frame], pairs: list[PairResult], args: argparse.Namespace
) -> None:
    if any(result.shift is None for result in pairs):
        return
    for first_index in pairs[0].links:
        detections = [(frames[0], first_index)]
        current_index = first_index
        for offset, result in enumerate(pairs):
            next_index = result.links.get(current_index)
            if next_index is None:
                break
            detections.append((frames[offset + 1], next_index))
            current_index = next_index
        if len(detections) != args.min_frames:
            continue
        if is_consistent_track(
            detections,
            [result.shift for result in pairs],
            max_spread=args.max_track_spread,
            static_min_area=args.static_min_area,
        ):
            for frame, index in detections:
                frame.rejected.add(index)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise ValueError(f"Input folder does not exist: {source_root}")
    if (
        output_root == source_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ValueError("Output folder must be separate from the input folder")
    if args.min_frames not in (2, 3):
        raise ValueError("--min-frames must be 2 or 3")
    for name in (
        "max_shift",
        "registration_tolerance",
        "match_radius",
        "max_track_spread",
        "registration_min_area",
        "static_min_area",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_registration_matches < 1:
        raise ValueError("--min-registration-matches must be positive")
    if not 0 < args.min_registration_fraction <= 1:
        raise ValueError("--min-registration-fraction must be in (0, 1]")

    paths = sorted(source_root.rglob("*_detections.csv"))
    if not paths:
        raise ValueError(f"No *_detections.csv files found under {source_root}")
    groups: dict[tuple[str, ...], list[tuple[tuple[object, ...], Path]]] = {}
    for path in paths:
        group, order, _ = frame_identity(path.relative_to(source_root))
        groups.setdefault(group, []).append((order, path))

    expected_paths = {output_root / path.relative_to(source_root) for path in paths}
    existing_paths = set(output_root.rglob("*_detections.csv")) if output_root.exists() else set()
    stale_paths = existing_paths - expected_paths
    if stale_paths:
        raise ValueError(f"Output folder contains {len(stale_paths)} stale detection CSVs")

    frame_report: list[dict[str, object]] = []
    pair_report: list[dict[str, object]] = []
    total_input = total_removed = total_remaining = 0

    def finish_frame(frame: Frame, audit_writer: csv.DictWriter) -> None:
        nonlocal total_input, total_removed, total_remaining
        kept = [row for index, row in enumerate(frame.rows) if index not in frame.rejected]
        atomic_csv(output_root / frame.relative_path, frame.fields, kept)
        frame_report.append(
            {
                "source_file": str(frame.relative_path),
                "input_count": len(frame.rows),
                "removed_count": len(frame.rejected),
                "remaining_count": len(kept),
                "invalid_coordinate_count": int(np.count_nonzero(~np.isfinite(frame.points).all(axis=1))),
            }
        )
        total_input += len(frame.rows)
        total_removed += len(frame.rejected)
        total_remaining += len(kept)
        for index in sorted(frame.rejected):
            row = frame.rows[index]
            audit_writer.writerow(
                {
                    "source_file": str(frame.relative_path),
                    "source_row": index + 2,
                    "id": row.get("id", ""),
                    "x": row["x"],
                    "y": row["y"],
                    "area": row["area"],
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output_root, delete=False
    ) as audit_stream:
        audit_temporary = Path(audit_stream.name)
        audit_writer = csv.DictWriter(
            audit_stream,
            fieldnames=["source_file", "source_row", "id", "x", "y", "area"],
        )
        audit_writer.writeheader()
        for group_paths in groups.values():
            window: list[Frame] = []
            pairs: list[PairResult] = []
            for _, path in sorted(group_paths):
                frame = read_frame(path, source_root)
                if window:
                    result = pair_frames(window[-1], frame, args)
                    pair_report.append(pair_summary(window[-1], frame, result))
                    pairs.append(result)
                window.append(frame)
                if len(window) == args.min_frames:
                    classify_window(window, pairs, args)
                    finish_frame(window.pop(0), audit_writer)
                    pairs.pop(0)
            for frame in window:
                finish_frame(frame, audit_writer)
    os.replace(audit_temporary, output_root / "removed_star_candidates.csv")
    summary: dict[str, object] = {
        "input_folder": str(source_root),
        "output_folder": str(output_root),
        "parameters": {
            "min_frames": args.min_frames,
            "max_shift": args.max_shift,
            "registration_tolerance": args.registration_tolerance,
            "registration_min_area": args.registration_min_area,
            "min_registration_matches": args.min_registration_matches,
            "min_registration_fraction": args.min_registration_fraction,
            "match_radius": args.match_radius,
            "max_track_spread": args.max_track_spread,
            "static_min_area": args.static_min_area,
        },
        "frame_count": len(paths),
        "total_input": total_input,
        "total_removed": total_removed,
        "total_remaining": total_remaining,
        "frames": frame_report,
        "frame_pairs": pair_report,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder with *_detections.csv files")
    parser.add_argument("--output", type=Path, required=True, help="Separate folder for filtered CSVs")
    parser.add_argument("--min-frames", type=int, default=3, choices=(2, 3))
    parser.add_argument("--max-shift", type=float, default=20.0, help="Largest star-field shift between frames, pixels")
    parser.add_argument("--registration-tolerance", type=float, default=1.75, help="Displacement consensus radius, pixels")
    parser.add_argument("--registration-min-area", type=float, default=10.0, help="Minimum source area for measuring field motion")
    parser.add_argument("--min-registration-matches", type=int, default=6)
    parser.add_argument("--min-registration-fraction", type=float, default=0.2)
    parser.add_argument("--match-radius", type=float, default=2.0, help="Maximum residual after field registration, pixels")
    parser.add_argument("--max-track-spread", type=float, default=3.0, help="Maximum spread across a multi-frame candidate, pixels")
    parser.add_argument("--static-min-area", type=float, default=10.0, help="Minimum source area when field motion is below 1 pixel")
    args = parser.parse_args()
    try:
        summary = run(args)
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Processed {summary['frame_count']} frames: "
        f"removed {summary['total_removed']} star-likely observations; "
        f"retained {summary['total_remaining']}."
    )
    print(f"Summary: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
