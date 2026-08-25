"""Split SWCs into dendrite/axon trees and resample dendrites with Vaa3D."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


COMPARTMENT_TYPES = {
    "dendrite": {1, 3, 4},  # soma, basal dendrite, apical dendrite
    "axon": {1, 2},         # soma, axon
}


def read_swc(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read comments and standard seven-column SWC rows."""

    comments = []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped)
                continue
            fields = stripped.split()
            if len(fields) < 7:
                raise ValueError(f"{path}: expected at least 7 SWC columns")
            rows.append(fields[:7])
    return comments, rows


def compartment_rows(rows: list[list[str]], retained_types: set[int]) -> tuple[list[list[str]], int]:
    """Keep one compartment and turn parents outside it into new roots."""

    kept = [row.copy() for row in rows if int(float(row[1])) in retained_types]
    kept_ids = {int(float(row[0])) for row in kept}
    rerooted = 0
    for row in kept:
        parent = int(float(row[6]))
        if parent != -1 and parent not in kept_ids:
            row[6] = "-1"
            rerooted += 1
    return kept, rerooted


def write_swc(path: Path, comments: list[str], rows: list[list[str]], label: str) -> None:
    """Write a compact, valid SWC while preserving source comments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# compartment: {label}", *comments]
    lines.extend(" ".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def option_prefix(vaa3d: Path) -> str:
    """Return Vaa3D's Windows slash or Unix hyphen switch prefix."""

    return "/" if platform.system() == "Windows" or vaa3d.suffix.lower() == ".exe" else "-"


def resample_command(vaa3d: Path, input_path: Path, output_path: Path, step_um: float) -> list[str]:
    """Build a platform-adaptive Vaa3D resample_swc command."""

    prefix = option_prefix(vaa3d)
    return [
        str(vaa3d),
        f"{prefix}x",
        "resample_swc",
        f"{prefix}f",
        "resample_swc",
        f"{prefix}i",
        str(input_path),
        f"{prefix}o",
        str(output_path),
        f"{prefix}p",
        str(step_um),
    ]


def resample_one(vaa3d: Path, input_path: Path, output_path: Path, step_um: float) -> None:
    """Resample one dendrite tree and report Vaa3D failures plainly."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        resample_command(vaa3d, input_path, output_path, step_um),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not output_path.exists():
        message = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"Vaa3D could not resample {input_path.name}:\n{message}")


def swc_files(input_dir: Path) -> list[Path]:
    """Return input SWCs in a stable order."""

    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".swc", ".eswc"}
    )


def copy_examples(metadata_path: Path, resampled_dir: Path, example_dir: Path) -> None:
    """Copy small examples for every supported region into the teaching data."""

    metadata = pd.read_csv(metadata_path)
    examples = pd.concat([
        metadata[metadata["broad_region"] == "Cortex"].head(3),
        metadata[metadata["broad_region"] == "Thalamus"].head(3),
        metadata[metadata["broad_region"] == "CP"].head(2),
    ])
    example_dir.mkdir(parents=True, exist_ok=True)
    for row in examples.itertuples(index=False):
        shutil.copy2(resampled_dir / row.swc_file, example_dir / row.swc_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split SWCs and resample dendrites at a fixed step with Vaa3D."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--vaa3d", required=True, type=Path)
    parser.add_argument("--step-um", type=float, default=2.0)
    parser.add_argument("--dendrite-dir", type=Path, default=Path("downloaded_data/subset_dendrite_swc"))
    parser.add_argument("--axon-dir", type=Path, default=Path("downloaded_data/subset_axon_swc"))
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        default=Path("downloaded_data/subset_dendrite_resampled_2um"),
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--example-dir", type=Path, default=Path("data/swc"))
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="parallel Vaa3D processes")
    args = parser.parse_args()

    files = swc_files(args.input_dir)
    if not files:
        raise SystemExit(f"No SWCs found in {args.input_dir}")

    rerooted_totals = {"dendrite": 0, "axon": 0}
    resample_jobs = []
    for source in files:
        comments, rows = read_swc(source)
        outputs = {}
        for label, retained_types in COMPARTMENT_TYPES.items():
            kept, rerooted = compartment_rows(rows, retained_types)
            output_dir = args.dendrite_dir if label == "dendrite" else args.axon_dir
            target = output_dir / f"{source.stem}.swc"
            write_swc(target, comments, kept, label)
            rerooted_totals[label] += rerooted
            outputs[label] = target

        resampled = args.resampled_dir / f"{source.stem}.swc"
        resample_jobs.append((outputs["dendrite"], resampled))

    def run_job(job: tuple[Path, Path]) -> None:
        resample_one(args.vaa3d, job[0], job[1], args.step_um)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(run_job, resample_jobs), start=1):
            print(f"[{index}/{len(files)}] resampled")

    if not args.skip_examples:
        copy_examples(args.metadata, args.resampled_dir, args.example_dir)
    print(f"Processed {len(files)} SWCs; resampled dendrites at {args.step_um:g} um")
    print(f"Parents outside compartment rerooted: {rerooted_totals}")
    if not args.skip_examples:
        print(f"Copied 8 resampled dendrite examples to {args.example_dir}")


if __name__ == "__main__":
    main()
