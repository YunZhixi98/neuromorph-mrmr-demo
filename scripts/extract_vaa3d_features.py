"""Extract the 22 global morphology features reported by Vaa3D.

This is a small preparation script for the teaching notebook.  Vaa3D's
global_neuron_feature plugin prints one labeled value per feature, so we call
it once per SWC and collect those values in a simple CSV table.
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FEATURE_COLUMNS = [
    "number_of_nodes",
    "soma_surface",
    "number_of_stems",
    "number_of_bifurcations",
    "number_of_branches",
    "number_of_tips",
    "overall_width",
    "overall_height",
    "overall_depth",
    "average_diameter",
    "total_length",
    "total_surface",
    "total_volume",
    "max_euclidean_distance",
    "max_path_distance",
    "max_branch_order",
    "average_contraction",
    "average_fragmentation",
    "average_parent_daughter_ratio",
    "average_bifurcation_angle_local",
    "average_bifurcation_angle_remote",
    "hausdorff_dimension",
]


# The installed plugin spells "Bifurcations" as "Bifurcatons" in its output.
# Accept both spellings so the CSV stays stable across Vaa3D builds.
LABEL_TO_COLUMN = {
    "number of nodes": "number_of_nodes",
    "soma surface": "soma_surface",
    "number of stems": "number_of_stems",
    "number of bifurcatons": "number_of_bifurcations",
    "number of bifurcations": "number_of_bifurcations",
    "number of branches": "number_of_branches",
    "number of tips": "number_of_tips",
    "overall width": "overall_width",
    "overall height": "overall_height",
    "overall depth": "overall_depth",
    "average diameter": "average_diameter",
    "total length": "total_length",
    "total surface": "total_surface",
    "total volume": "total_volume",
    "max euclidean distance": "max_euclidean_distance",
    "max path distance": "max_path_distance",
    "max branch order": "max_branch_order",
    "average contraction": "average_contraction",
    "average fragmentation": "average_fragmentation",
    "average parent-daughter ratio": "average_parent_daughter_ratio",
    "average parent daughter ratio": "average_parent_daughter_ratio",
    "average bifurcation angle local": "average_bifurcation_angle_local",
    "average bifurcation angle remote": "average_bifurcation_angle_remote",
    "hausdorff dimension": "hausdorff_dimension",
}

NUMBER = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"|[-+]?(?:nan|inf|1\.#(?:IND|INF|QNAN|SNAN))"
)
VALUE_PATTERN = re.compile(rf"^\s*({NUMBER})\s*$", re.IGNORECASE)


def vaa3d_command(vaa3d: Path, plugin: str, swc_path: Path) -> list[str]:
    """Build the platform-specific Vaa3D plugin command."""

    # Vaa3D documents slash-prefixed switches for its Windows executable and
    # hyphen-prefixed switches for Linux/macOS builds.
    prefix = "/" if platform.system() == "Windows" or vaa3d.suffix.lower() == ".exe" else "-"
    return [
        str(vaa3d),
        f"{prefix}x",
        plugin,
        f"{prefix}f",
        "compute_feature",
        f"{prefix}i",
        str(swc_path),
    ]


def parse_feature_output(output: str) -> dict[str, float]:
    """Read the 22 labeled feature values printed by Vaa3D."""

    values: dict[str, float] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        label, text_value = line.split(":", 1)
        column = LABEL_TO_COLUMN.get(label.strip().lower())
        if column is None:
            continue
        match = VALUE_PATTERN.match(text_value)
        if match is None:
            continue
        parsed = match.group(1)
        values[column] = float("nan") if "#" in parsed else float(parsed)

    missing = [column for column in FEATURE_COLUMNS if column not in values]
    if missing:
        raise RuntimeError(
            "Vaa3D output did not contain all 22 expected features; "
            f"missing: {', '.join(missing)}"
        )
    return {column: values[column] for column in FEATURE_COLUMNS}


def extract_one(vaa3d: Path, plugin: str, swc_path: Path) -> dict[str, float]:
    """Run Vaa3D for one SWC and parse its feature values."""

    command = vaa3d_command(vaa3d, plugin, swc_path)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = result.stdout + "\n" + result.stderr
    try:
        return parse_feature_output(output)
    except RuntimeError as error:
        raise RuntimeError(f"Could not extract {swc_path.name}: {error}") from error


def swc_files(swc_dir: Path) -> list[Path]:
    """Return SWC/ESWC files in a directory in stable teaching order."""

    return sorted(
        (
            path
            for path in swc_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".swc", ".eswc"}
        ),
        key=lambda path: path.relative_to(swc_dir).as_posix().lower(),
    )


def write_features(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Write filenames and the 22 features to a CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["swc_file", *FEATURE_COLUMNS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Vaa3D global_neuron_feature values from a SWC directory."
    )
    parser.add_argument("swc_dir", type=Path, help="directory containing .swc/.eswc files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output CSV path")
    parser.add_argument("--vaa3d", default="vaa3d", type=Path, help="Vaa3D executable")
    parser.add_argument(
        "--plugin",
        default="global_neuron_feature",
        help="plugin name or unique plugin path accepted by Vaa3D",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel Vaa3D processes")
    args = parser.parse_args()

    files = swc_files(args.swc_dir)
    if not files:
        raise SystemExit(f"No .swc or .eswc files found in {args.swc_dir}")

    def extract_row(swc_path: Path) -> dict[str, object]:
        features = extract_one(args.vaa3d, args.plugin, swc_path)
        relative_name = swc_path.relative_to(args.swc_dir).as_posix()
        return {"swc_file": relative_name, **features}

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(extract_row, files), start=1):
            rows.append(row)
            print(f"[{index}/{len(files)}] {row['swc_file']}")

    write_features(args.output, rows)
    print(f"Wrote {len(rows)} rows and 22 features to {args.output}")


if __name__ == "__main__":
    main()
