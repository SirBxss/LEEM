"""Command-line entry point for leakage-safe AIOHMM synthetic experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from lane_error_modeling.evaluation import run_aiohmm_experiment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select AIOHMM on validation and evaluate held-out test once"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only an output directory previously created by this runner",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    manifest = run_aiohmm_experiment(
        project_root=project_root,
        config_path=arguments.config,
        output_root=arguments.output,
        overwrite=arguments.overwrite,
    )
    print(f"AIOHMM experiment passed: {manifest}")


if __name__ == "__main__":
    main()
