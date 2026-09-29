#!/usr/bin/env python3
"""Run SCRAP's existing Photutils detector over a folder of SOBER RAW frames.

This wrapper writes only the CSV catalogs needed by star_reject.py. It does
not change the detector's background, segmentation, or measurement logic.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qmsat-matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import photutils_detect as detector  # noqa: E402


DETECTION_FIELDS = [
    "id", "x", "y", "x_min", "x_max", "y_min", "y_max", "area",
    "semimajor_axis", "semiminor_axis", "elongation", "orientation_deg",
    "eccentricity", "segment_flux",
]


def process(raw_path: Path, csv_path: Path) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image = detector.load_raw(raw_path)
    _, catalog = detector.detect_objects(image)
    with tempfile.NamedTemporaryFile(dir=csv_path.parent, suffix=".csv", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if catalog is None:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=DETECTION_FIELDS).writeheader()
            count = 0
        else:
            detector.save_catalog(catalog, temporary)
            count = len(catalog)
        os.replace(temporary, csv_path)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder containing .raw frames")
    parser.add_argument("--output", type=Path, required=True, help="Folder for *_detections.csv files")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted frames")
    parser.add_argument("--force", action="store_true", help="Recompute completed CSVs")
    args = parser.parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        parser.error(f"Input folder does not exist: {source_root}")
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        parser.error("Input and output folders must be separate")
    raw_paths = sorted(source_root.rglob("*.raw"))
    if not raw_paths:
        parser.error(f"No .raw frames found under {source_root}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        raw_paths = raw_paths[: args.limit]

    completed = 0
    skipped = 0
    for raw_path in raw_paths:
        relative = raw_path.relative_to(source_root)
        csv_path = output_root / relative.parent / f"{raw_path.stem}_detections.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0 and not args.force:
            skipped += 1
            continue
        count = process(raw_path, csv_path)
        completed += 1
        print(f"{relative}: {count} detections -> {csv_path}", flush=True)
    print(f"Frames: {len(raw_paths)}; processed: {completed}; already complete: {skipped}")


if __name__ == "__main__":
    main()
