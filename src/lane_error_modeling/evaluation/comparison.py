"""Deterministic cross-model comparison for common LEEM evaluations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .io import atomic_write_json, sha256_file


COMPARISON_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ComparisonMetric:
    """One scalar metric used in the thesis model comparison."""

    key: str
    label: str
    unit: str
    lower_is_better: bool = True


DEFAULT_COMPARISON_METRICS = (
    ComparisonMetric("predictive_mean_rmse_m", "Predictive-mean RMSE", "m"),
    ComparisonMetric("crps_m", "CRPS", "m"),
    ComparisonMetric("dimension_normalized_energy_score_m", "Energy score", "m"),
    ComparisonMetric("error_jensen_shannon_distance", "Marginal JS distance", "1"),
    ComparisonMetric(
        "first_difference_jensen_shannon_distance",
        "First-difference JS distance",
        "1",
    ),
    ComparisonMetric(
        "residual_lag_one_correlation_mae",
        "Lag-one correlation MAE",
        "1",
    ),
    ComparisonMetric(
        "residual_spatial_correlation_rmse",
        "Spatial-correlation RMSE",
        "1",
    ),
    ComparisonMetric("abs_quantile_error_q99_m", "Absolute q99 error", "m"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scenario_directories(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "evaluation.json").is_file()
    }


def _finite_metric(
    metrics: Mapping[str, Any], metric: ComparisonMetric, source: Path
) -> float:
    value = metrics.get(metric.key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} is missing scalar metric {metric.key!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source} metric {metric.key!r} is not finite")
    return result


def _assert_comparable(
    *,
    scenario: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_directory: Path,
    candidate_directory: Path,
) -> str:
    for payload, name in ((baseline, "baseline"), (candidate, "candidate")):
        if payload.get("scenario") != scenario:
            raise ValueError(f"{name} evaluation scenario does not match {scenario!r}")
    for field in ("sequence_count", "observed_value_count", "s_grid_m"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(
                f"scenario {scenario!r} has incompatible {field!r} values"
            )
    baseline_reference = baseline_directory / "evaluation_reference.json"
    candidate_reference = candidate_directory / "evaluation_reference.json"
    if not baseline_reference.is_file() or not candidate_reference.is_file():
        raise ValueError(
            f"scenario {scenario!r} requires both evaluation_reference.json files"
        )
    baseline_hash = sha256_file(baseline_reference)
    candidate_hash = sha256_file(candidate_reference)
    if baseline_hash != candidate_hash:
        raise ValueError(
            f"scenario {scenario!r} was not evaluated with the same reference"
        )
    return baseline_hash


def compare_experiment_results(
    *,
    baseline_root: str | Path,
    candidate_root: str | Path,
    metrics: tuple[ComparisonMetric, ...] = DEFAULT_COMPARISON_METRICS,
) -> dict[str, object]:
    """Compare two experiment directories without reading training data."""

    if not metrics or len({metric.key for metric in metrics}) != len(metrics):
        raise ValueError("comparison metrics must be non-empty and unique")
    baseline_path = Path(baseline_root)
    candidate_path = Path(candidate_root)
    baseline_scenarios = _scenario_directories(baseline_path)
    candidate_scenarios = _scenario_directories(candidate_path)
    if set(baseline_scenarios) != set(candidate_scenarios):
        raise ValueError("baseline and candidate scenario sets must match exactly")
    if not baseline_scenarios:
        raise ValueError("experiment directories contain no scenario evaluations")

    rows: list[dict[str, object]] = []
    reference_hashes: dict[str, str] = {}
    baseline_model: str | None = None
    candidate_model: str | None = None
    for scenario in sorted(baseline_scenarios):
        baseline_file = baseline_scenarios[scenario] / "evaluation.json"
        candidate_file = candidate_scenarios[scenario] / "evaluation.json"
        baseline = _load_json(baseline_file)
        candidate = _load_json(candidate_file)
        reference_hashes[scenario] = _assert_comparable(
            scenario=scenario,
            baseline=baseline,
            candidate=candidate,
            baseline_directory=baseline_scenarios[scenario],
            candidate_directory=candidate_scenarios[scenario],
        )
        current_baseline_model = str(baseline.get("model_name", ""))
        current_candidate_model = str(candidate.get("model_name", ""))
        if not current_baseline_model or not current_candidate_model:
            raise ValueError("evaluation model_name must not be empty")
        if baseline_model is None:
            baseline_model = current_baseline_model
            candidate_model = current_candidate_model
        elif (
            baseline_model != current_baseline_model
            or candidate_model != current_candidate_model
        ):
            raise ValueError("model names must be consistent across scenarios")

        baseline_metrics = baseline.get("global_metrics")
        candidate_metrics = candidate.get("global_metrics")
        if not isinstance(baseline_metrics, Mapping) or not isinstance(
            candidate_metrics, Mapping
        ):
            raise ValueError("evaluation global_metrics must be objects")
        for metric in metrics:
            baseline_value = _finite_metric(baseline_metrics, metric, baseline_file)
            candidate_value = _finite_metric(candidate_metrics, metric, candidate_file)
            if baseline_value == 0.0:
                improvement: float | None = None
            else:
                direction = 1.0 if metric.lower_is_better else -1.0
                improvement = float(
                    direction
                    * (baseline_value - candidate_value)
                    / abs(baseline_value)
                    * 100.0
                )
            if candidate_value == baseline_value:
                better_model = "tie"
            elif (candidate_value < baseline_value) == metric.lower_is_better:
                better_model = current_candidate_model
            else:
                better_model = current_baseline_model
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "unit": metric.unit,
                    "lower_is_better": metric.lower_is_better,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "candidate_improvement_percent": improvement,
                    "better_model": better_model,
                }
            )

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "scenario_count": len(baseline_scenarios),
        "reference_sha256_by_scenario": reference_hashes,
        "interpretation": (
            "positive candidate_improvement_percent means the candidate is better; "
            "synthetic results are capability checks, not final real-world rankings"
        ),
        "rows": rows,
    }


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.stem}-",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def save_comparison_report(
    report: Mapping[str, object], output_directory: str | Path
) -> tuple[Path, Path, Path]:
    """Write deterministic JSON, CSV, and Markdown comparison artifacts."""

    destination = Path(output_directory)
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison report rows must be a non-empty list")
    json_path = atomic_write_json(destination / "model_comparison.json", report)

    fieldnames = (
        "scenario",
        "metric",
        "metric_label",
        "unit",
        "lower_is_better",
        "baseline_value",
        "candidate_value",
        "candidate_improvement_percent",
        "better_model",
    )
    csv_lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("comparison rows must be objects")
            writer.writerow({name: row.get(name) for name in fieldnames})
        handle.seek(0)
        csv_lines.append(handle.read())
    csv_path = _atomic_write_text(
        destination / "model_comparison.csv", "".join(csv_lines)
    )

    baseline_model = str(report.get("baseline_model"))
    candidate_model = str(report.get("candidate_model"))
    markdown = [
        "# Synthetic model comparison\n",
        "Positive improvement means the candidate is better. These synthetic "
        "results are capability checks, not final BMW-data rankings.\n",
        f"Baseline: `{baseline_model}`  \nCandidate: `{candidate_model}`\n",
        "| Scenario | Metric | Baseline | Candidate | Improvement [%] | Better |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("comparison rows must be objects")
        markdown.append(
            "| {scenario} | {metric_label} | {baseline} | {candidate} | "
            "{improvement} | {better} |".format(
                scenario=row["scenario"],
                metric_label=row["metric_label"],
                baseline=_format_value(row["baseline_value"]),
                candidate=_format_value(row["candidate_value"]),
                improvement=_format_value(row["candidate_improvement_percent"]),
                better=row["better_model"],
            )
        )
    markdown_path = _atomic_write_text(
        destination / "model_comparison.md", "\n".join(markdown) + "\n"
    )
    return json_path, csv_path, markdown_path
