"""Create deterministic cross-model tables from existing evaluation files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lane_error_modeling.evaluation.comparison import (  # noqa: E402
    compare_experiment_results,
    save_comparison_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two LEEM experiment result directories"
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = compare_experiment_results(
        baseline_root=arguments.baseline,
        candidate_root=arguments.candidate,
    )
    paths = save_comparison_report(report, arguments.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
