"""Small, deterministic migrations for persisted evaluation results."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .finite_sample import finite_ensemble_interval_metadata
from .io import atomic_write_json
from .result import EVALUATION_RESULT_SCHEMA_VERSION


def add_finite_ensemble_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Upgrade one evaluation payload without model samples or training data."""

    upgraded = deepcopy(dict(payload))
    sample_count = upgraded.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("evaluation sample_count must be an integer")
    interval_metrics = upgraded.get("interval_metrics")
    if not isinstance(interval_metrics, dict) or not interval_metrics:
        raise ValueError("evaluation interval_metrics must be a non-empty object")
    for key, raw_details in interval_metrics.items():
        if not isinstance(raw_details, dict):
            raise ValueError(f"interval {key!r} must be an object")
        nominal = raw_details.get("nominal_coverage")
        empirical = raw_details.get("global_empirical_coverage")
        if isinstance(nominal, bool) or not isinstance(nominal, (int, float)):
            raise ValueError(f"interval {key!r} nominal coverage is invalid")
        if isinstance(empirical, bool) or not isinstance(empirical, (int, float)):
            raise ValueError(f"interval {key!r} empirical coverage is invalid")
        raw_details["finite_ensemble"] = finite_ensemble_interval_metadata(
            sample_count=sample_count,
            nominal_coverage=float(nominal),
            empirical_coverage=float(empirical),
        )
    approximation = upgraded.setdefault("approximation_metadata", {})
    if not isinstance(approximation, dict):
        raise ValueError("evaluation approximation_metadata must be an object")
    approximation["interval_quantile_method"] = "linear"
    approximation[
        "interval_finite_sample_reference"
    ] = "uniform_order_statistic_expectation"
    upgraded["schema_version"] = EVALUATION_RESULT_SCHEMA_VERSION
    return upgraded


def upgrade_evaluation_tree(
    root: str | Path,
    *,
    write: bool = False,
) -> tuple[Path, ...]:
    """Validate and optionally update every evaluation.json below ``root``."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"evaluation root does not exist: {root_path}")
    paths = tuple(sorted(root_path.rglob("evaluation.json")))
    if not paths:
        raise ValueError("evaluation root contains no evaluation.json files")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        upgraded = add_finite_ensemble_metadata(payload)
        if write:
            atomic_write_json(path, upgraded)
    return paths
