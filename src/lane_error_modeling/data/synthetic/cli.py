"""Command-line entry point for complete synthetic dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from lane_error_modeling.data.synthetic.config import SyntheticDatasetConfig
from lane_error_modeling.data.synthetic.generator import generate_dataset
from lane_error_modeling.data.synthetic.io import save_dataset, write_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible lane-estimation error datasets."
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated dataset that contains a manifest",
    )
    return parser.parse_args()


def generate_from_config(
    config_path: Path,
    output_root: Path,
    overwrite: bool = False,
) -> Path:
    """Generate every configured scenario/split and return the manifest path."""

    config = SyntheticDatasetConfig.from_json(config_path)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory {output_root} is not empty; use --overwrite"
            )
        if not (output_root / "manifest.json").is_file():
            raise FileExistsError(
                "refusing to overwrite a non-empty directory without manifest.json"
            )
        shutil.rmtree(output_root)

    records: list[dict[str, object]] = []
    for scenario in config.scenarios:
        for split in ("train", "validation", "test"):
            dataset = generate_dataset(config, scenario, split)
            destination = output_root / scenario / f"{split}.npz"
            record = save_dataset(destination, dataset)
            record["scenario"] = scenario
            record["split"] = split
            records.append(record)
    return write_manifest(output_root, config, records)


def main() -> None:
    args = _parse_args()
    manifest_path = generate_from_config(args.config, args.output, args.overwrite)
    print(manifest_path)


if __name__ == "__main__":
    main()

