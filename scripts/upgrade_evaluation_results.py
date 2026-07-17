"""Add finite-ensemble metadata to existing evaluation JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lane_error_modeling.evaluation.migration import (  # noqa: E402
    upgrade_evaluation_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or upgrade persisted LEEM evaluation files"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write upgraded JSON; without this flag the command is read-only",
    )
    arguments = parser.parse_args()
    paths = upgrade_evaluation_tree(arguments.root, write=arguments.write)
    action = "updated" if arguments.write else "validated"
    for path in paths:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()
