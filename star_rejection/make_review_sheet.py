#!/usr/bin/env python3
"""Explain star-rejection decisions with real, unaltered RAW-frame crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qmsat-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def crop(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    center_x, center_y = int(round(x)), int(round(y))
    return image[
        max(0, center_y - radius) : min(image.shape[0], center_y + radius + 1),
        max(0, center_x - radius) : min(image.shape[1], center_x + radius + 1),
    ]


def closest(items: list[dict[str, str]], x: float, y: float, allowed: set[int]) -> tuple[int, dict[str, str]] | None:
    candidates = [
        (np.hypot(float(row["x"]) - x, float(row["y"]) - y), index, row)
        for index, row in enumerate(items)
        if index in allowed
    ]
    if not candidates:
        return None
    distance, index, row = min(candidates, key=lambda value: value[0])
    return (index, row) if distance < 3 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.filtered / "summary.json").read_text())
    names = [frame["source_file"] for frame in summary["frames"]]
    if len(names) != 3:
        parser.error("This sample review sheet expects exactly three frames")
    catalogs = [rows(args.detections / name) for name in names]
    images = [
        np.fromfile(args.raw / name.replace("_detections.csv", ".raw"), dtype=np.uint8).reshape(3000, 4096)
        for name in names
    ]
    rejected = [set() for _ in names]
    name_to_index = {name: index for index, name in enumerate(names)}
    for item in rows(args.filtered / "removed_star_candidates.csv"):
        rejected[name_to_index[item["source_file"]]].add(int(item["source_row"]) - 2)

    first = catalogs[0]
    star_indices = sorted(rejected[0], key=lambda index: float(first[index]["segment_flux"]), reverse=True)[:4]
    fixed_indices = []
    for index, row in enumerate(first):
        if index in rejected[0] or float(row["area"]) > 5:
            continue
        x, y = float(row["x"]), float(row["y"])
        if all(
            closest(catalogs[frame_index], x, y, set(range(len(catalogs[frame_index]))) - rejected[frame_index])
            for frame_index in (1, 2)
        ):
            fixed_indices.append(index)
    fixed_indices = sorted(fixed_indices, key=lambda index: float(first[index]["segment_flux"]), reverse=True)[:2]

    selections = [("removed", index) for index in star_indices] + [("kept", index) for index in fixed_indices]
    cumulative = [(0.0, 0.0)]
    for pair in summary["frame_pairs"]:
        old_x, old_y = cumulative[-1]
        cumulative.append((old_x + pair["shift_x"], old_y + pair["shift_y"]))
    figure = plt.figure(figsize=(12, 21), facecolor="white")
    grid = GridSpec(
        10, 4, figure=figure,
        height_ratios=[1.65, 0.68, 2.5, 2.5, 2.5, 2.5, 0.68, 2.5, 2.5, 1.18],
        width_ratios=[1.85, 3, 3, 3],
        left=0.035, right=0.985, top=0.985, bottom=0.02,
        hspace=0.32, wspace=0.10,
    )

    intro = figure.add_subplot(grid[0, :])
    intro.axis("off")
    intro.text(0, 0.95, "Star rejection: what was removed and what was kept",
               ha="left", va="top", fontsize=18, fontweight="bold", transform=intro.transAxes)
    intro.text(0, 0.65, "Each row follows one detection through three consecutive RAW frames.",
               ha="left", va="center", fontsize=11.5, transform=intro.transAxes)
    intro.scatter([0.018], [0.35], s=190, facecolors="none", edgecolors="#dc2626",
                  linewidths=1.8, transform=intro.transAxes)
    intro.text(0.045, 0.35, "Red circle = centre of the 31 × 31 pixel crop, near the detection. It is not a detection boundary.",
               ha="left", va="center", fontsize=10.5, transform=intro.transAxes)
    intro.text(0, 0.08, "The crops are recentered. Read the full-frame x, y coordinates above them to see motion.",
               ha="left", va="bottom", fontsize=10.5, transform=intro.transAxes)

    def section(grid_row: int, title: str, explanation: str, background: str, foreground: str) -> None:
        axis = figure.add_subplot(grid[grid_row, :])
        axis.set_facecolor(background)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(0.015, 0.67, title, ha="left", va="center", fontsize=13,
                  fontweight="bold", color=foreground, transform=axis.transAxes)
        axis.text(0.015, 0.23, explanation, ha="left", va="center", fontsize=10.5,
                  color="#20262e", transform=axis.transAxes)

    shifts = summary["frame_pairs"]
    section(
        1, "REMOVED  |  follows the common star-field shift",
        f"Field shift: x +{shifts[0]['shift_x']:.1f} px, then +{shifts[1]['shift_x']:.1f} px. These are star candidates, not confirmed stars.",
        "#ffebe8", "#9f281e",
    )
    section(
        6, "KEPT  |  stays at the same sensor coordinates",
        "These tiny bright spots are likely sensor defects. 'Kept' does not mean spacecraft or a useful target.",
        "#e7f3fd", "#155381",
    )

    for row_index, (kind, source_index) in enumerate(selections):
        source = first[source_index]
        x0, y0 = float(source["x"]), float(source["y"])
        chosen = []
        for frame_index in range(3):
            delta_x, delta_y = cumulative[frame_index] if kind == "removed" else (0, 0)
            allowed = rejected[frame_index] if kind == "removed" else set(range(len(catalogs[frame_index]))) - rejected[frame_index]
            match = closest(catalogs[frame_index], x0 + delta_x, y0 + delta_y, allowed)
            if match is None:
                raise ValueError(f"Cannot find {kind} in frame {frame_index}")
            chosen.append(match[1])
        patches = [crop(image, float(row["x"]), float(row["y"]), 15) for image, row in zip(images, chosen)]
        vmax = max(10, max(int(patch.max()) for patch in patches))
        grid_row = row_index + 2 if kind == "removed" else row_index + 3
        label = figure.add_subplot(grid[grid_row, 0])
        label.set_facecolor("#fff4f2" if kind == "removed" else "#f0f8fe")
        label.set_xticks([])
        label.set_yticks([])
        for spine in label.spines.values():
            spine.set_visible(False)
        label.axvline(0, color="#b7392f" if kind == "removed" else "#2474a9", linewidth=6)
        # Match the deltas to the one-decimal coordinates printed above the crops.
        x_values = [round(float(row["x"]), 1) for row in chosen]
        x_steps = [x_values[1] - x_values[0], x_values[2] - x_values[1]]
        label.text(
            0.09, 0.57,
            f"{'Removed' if kind == 'removed' else 'Kept'} example {row_index + 1 if kind == 'removed' else row_index - len(star_indices) + 1}\n"
            f"x shift: {x_steps[0]:+.1f}, then {x_steps[1]:+.1f} px\n"
            f"Frame-1 flux: {float(source['segment_flux']):,.0f}",
            ha="left", va="center", fontsize=10.5, linespacing=1.7, transform=label.transAxes,
        )
        for frame_index, (patch, row) in enumerate(zip(patches, chosen)):
            axis = figure.add_subplot(grid[grid_row, frame_index + 1])
            axis.imshow(patch, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            axis.scatter([(patch.shape[1] - 1) / 2], [(patch.shape[0] - 1) / 2],
                         s=180, facecolors="none", edgecolors="#dc2626", linewidths=1.8)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"Frame {frame_index + 1}   x={float(row['x']):.1f}, y={float(row['y']):.1f}",
                fontsize=9.7, pad=6,
            )

    footer = figure.add_subplot(grid[9, :])
    footer.axis("off")
    footer.text(0, 0.84,
                "Check the coordinate pattern, not whether the spot stays centred inside each crop.",
                ha="left", va="top", fontsize=11, fontweight="bold", transform=footer.transAxes)
    footer.text(0, 0.54,
                "Flux is detector-measured brightness in frame 1, not a confidence score. Brightness scales match within a row, not between rows.",
                ha="left", va="top", fontsize=10, transform=footer.transAxes)
    footer.text(0, 0.22,
                "Illustrative selection: the four brightest removed sources and two bright fixed pixels. This is not a representative accuracy test.",
                ha="left", va="top", fontsize=10, color="#4c5864", transform=footer.transAxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
