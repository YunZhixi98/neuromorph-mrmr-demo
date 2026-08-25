"""Prepare the 300-neuron teaching subset from SEU-A1876 metadata and SWCs."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd


GROUP_PREFIXES = {"Cortex": "CTX_", "Thalamus": "TH_", "CP": "CP_"}


def stable_key(text: str) -> int:
    """Small cross-platform key used only for reproducible subset ordering."""

    value = 2166136261
    for character in str(text):
        value ^= ord(character)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def load_metadata(path: Path) -> pd.DataFrame:
    """Read the supplied Excel metadata."""

    return pd.read_excel(path)


def choose_subset(metadata: pd.DataFrame, n_per_group: int = 100) -> pd.DataFrame:
    """Choose manually checked CTX, TH, and CP neurons reproducibly."""

    chosen_groups = []
    for group, prefix in GROUP_PREFIXES.items():
        candidates = metadata[
            (metadata["isManuallyChecked"] == 1)
            & metadata["Projection class"].fillna("").str.startswith(prefix)
        ].copy()
        candidates["_order"] = candidates["Morphology Name"].map(stable_key)
        chosen = candidates.sort_values("_order").head(n_per_group)
        chosen["broad_region"] = group
        chosen_groups.append(chosen)

    subset = pd.concat(chosen_groups, ignore_index=True)
    output = pd.DataFrame(
        {
            "neuron_id": subset["Morphology Name"],
            "swc_file": subset["Morphology Name"] + ".swc",
            "broad_region": subset["broad_region"],
            "soma_region": subset["Soma region"],
            "projection_class": subset["Projection class"],
            "cortical_layer": subset["Cortical Lamination of soma"],
            "manually_checked": subset["isManuallyChecked"],
            "brain_id": subset["fMOST Brain ID"],
            "soma_x_ccf_um": subset["Soma_X(CCFv3_1𝜇𝑚)"],
            "soma_y_ccf_um": subset["Soma_Y(CCFv3_1𝜇𝑚)"],
            "soma_z_ccf_um": subset["Soma_Z(CCFv3_1𝜇𝑚)"],
        }
    )
    return output


def archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map each SWC stem to its member name in the Zenodo archive."""

    members = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() in {".swc", ".eswc"}:
            members[path.stem] = name
    return members


def extract_swcs(archive_path: Path, metadata: pd.DataFrame, output_dir: Path) -> None:
    """Extract selected SWCs directly, without unpacking the full archive."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive)
        for neuron_id in metadata["neuron_id"]:
            member = members.get(neuron_id)
            if member is None:
                raise FileNotFoundError(f"{neuron_id} was not found in {archive_path}")
            target = output_dir / f"{neuron_id}{PurePosixPath(member).suffix.lower()}"
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())


def copy_classroom_examples(all_swc_dir: Path, metadata: pd.DataFrame, output_dir: Path) -> None:
    """Copy three Cortex and three Thalamus SWCs for notebook visualization."""

    output_dir.mkdir(parents=True, exist_ok=True)
    examples = metadata[metadata["broad_region"].isin(["Cortex", "Thalamus"])].groupby(
        "broad_region", sort=False
    ).head(3)
    for row in examples.itertuples(index=False):
        source = all_swc_dir / row.swc_file
        (output_dir / row.swc_file).write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the SEU-A1876 teaching subset.")
    parser.add_argument("metadata_xlsx", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--swc-archive", type=Path)
    parser.add_argument("--all-swc-dir", type=Path, default=Path("downloaded_data/subset_swc"))
    parser.add_argument("--example-swc-dir", type=Path, default=Path("data/swc"))
    args = parser.parse_args()

    subset = choose_subset(load_metadata(args.metadata_xlsx))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)
    print(subset["broad_region"].value_counts().sort_index())
    print(f"Wrote {len(subset)} metadata rows to {args.output}")

    if args.swc_archive:
        extract_swcs(args.swc_archive, subset, args.all_swc_dir)
        copy_classroom_examples(args.all_swc_dir, subset, args.example_swc_dir)
        print(f"Extracted {len(subset)} SWCs to {args.all_swc_dir}")
        print(f"Copied 6 classroom examples to {args.example_swc_dir}")


if __name__ == "__main__":
    main()
