#!/usr/bin/env python3
"""Check that filtered CSVs equal source CSVs minus the audited removals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    input_total = output_total = audit_total = 0
    seen: set[str] = set()
    with (destination / "removed_star_candidates.csv").open(newline="", encoding="utf-8-sig") as audit_stream:
        audit_reader = iter(csv.DictReader(audit_stream))
        current = next(audit_reader, None)
        for frame in summary["frames"]:
            name = frame["source_file"]
            if name in seen:
                raise AssertionError(f"Duplicate source in summary: {name}")
            seen.add(name)
            source_fields, source_rows = read_csv(source / name)
            result_fields, result_rows = read_csv(destination / name)
            removed_rows: set[int] = set()
            while current is not None and current["source_file"] == name:
                row_number = int(current["source_row"])
                if row_number in removed_rows or not 2 <= row_number < len(source_rows) + 2:
                    raise AssertionError(f"Invalid or duplicate removal: {name}, row {row_number}")
                source_row = source_rows[row_number - 2]
                for column in ("id", "x", "y", "area"):
                    if current[column] != source_row.get(column, ""):
                        raise AssertionError(f"Audit value differs from source: {name}, row {row_number}")
                removed_rows.add(row_number)
                audit_total += 1
                current = next(audit_reader, None)
            expected = [
                row for row_number, row in enumerate(source_rows, start=2)
                if row_number not in removed_rows
            ]
            if source_fields != result_fields or result_rows != expected:
                raise AssertionError(f"Filtered content differs from audited source rows: {name}")
            if frame["input_count"] != len(source_rows) or frame["remaining_count"] != len(result_rows):
                raise AssertionError(f"Incorrect frame counts in summary: {name}")
            if frame["removed_count"] != len(removed_rows):
                raise AssertionError(f"Incorrect removal count in summary: {name}")
            input_total += len(source_rows)
            output_total += len(result_rows)
        if current is not None:
            raise AssertionError("Audit references a frame missing from the summary or is out of order")

    if (input_total, audit_total, output_total) != (
        summary["total_input"], summary["total_removed"], summary["total_remaining"]
    ):
        raise AssertionError("Aggregate counts do not agree")
    print(
        f"Verified {len(seen)} filtered CSVs: {input_total} original rows, "
        f"{audit_total} audited removals, {output_total} unchanged remaining rows."
    )


if __name__ == "__main__":
    main()
