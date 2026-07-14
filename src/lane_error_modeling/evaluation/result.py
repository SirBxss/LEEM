"""Serializable result schema for one model/scenario/test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json


EVALUATION_RESULT_SCHEMA_VERSION = "1.0.0"


def _validate_finite_json(value: Any, path: str = "result") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            _validate_finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationResult:
    """Common sample-based metrics plus dependence diagnostics."""

    model_name: str
    scenario: str
    sample_count: int
    sequence_count: int
    observed_value_count: int
    s_grid_m: tuple[float, ...]
    global_metrics: dict[str, float]
    station_metrics: dict[str, list[float | int | None]]
    interval_metrics: dict[str, dict[str, object]]
    dependence_diagnostics: dict[str, object]
    approximation_metadata: dict[str, int | str]

    def validate(self) -> None:
        if not self.model_name or not self.scenario:
            raise ValueError("model_name and scenario must not be empty")
        if self.sample_count < 2 or self.sequence_count <= 0:
            raise ValueError("sample and sequence counts are invalid")
        if self.observed_value_count <= 0:
            raise ValueError("observed_value_count must be positive")
        if not self.s_grid_m:
            raise ValueError("s_grid_m must not be empty")
        if not all(math.isfinite(value) for value in self.s_grid_m) or not all(
            following > previous
            for previous, following in zip(self.s_grid_m, self.s_grid_m[1:])
        ):
            raise ValueError("s_grid_m must be finite and strictly increasing")
        if not self.global_metrics:
            raise ValueError("global_metrics must not be empty")
        station_count = len(self.s_grid_m)
        for name, values in self.station_metrics.items():
            if len(values) != station_count:
                raise ValueError(f"station metric {name!r} has an invalid length")
        required_interval_fields = {
            "nominal_coverage",
            "global_empirical_coverage",
            "global_mean_width_m",
            "station_empirical_coverage",
            "station_mean_width_m",
        }
        if not self.interval_metrics:
            raise ValueError("interval_metrics must not be empty")
        for level, details in self.interval_metrics.items():
            missing = required_interval_fields - set(details)
            if missing:
                raise ValueError(
                    f"interval {level!r} is missing fields: {sorted(missing)}"
                )
            for name in (
                "station_empirical_coverage",
                "station_mean_width_m",
            ):
                values = details[name]
                if not isinstance(values, (list, tuple)) or len(values) != station_count:
                    raise ValueError(
                        f"interval {level!r} field {name!r} has an invalid length"
                    )
        _validate_finite_json(self.to_dict(), "evaluation_result")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
            "model_name": self.model_name,
            "scenario": self.scenario,
            "sample_count": self.sample_count,
            "sequence_count": self.sequence_count,
            "observed_value_count": self.observed_value_count,
            "s_grid_m": list(self.s_grid_m),
            "global_metrics": self.global_metrics,
            "station_metrics": self.station_metrics,
            "interval_metrics": self.interval_metrics,
            "dependence_diagnostics": self.dependence_diagnostics,
            "approximation_metadata": self.approximation_metadata,
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        return atomic_write_json(path, self.to_dict())
