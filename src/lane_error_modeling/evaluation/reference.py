"""Training-derived metric references that prevent test-adaptive evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lane_error_modeling.data.preprocessing import SequenceDataset

from .config import EvaluationConfig
from .io import atomic_write_json


EVALUATION_REFERENCE_SCHEMA_VERSION = "1.0.0"


def valid_first_differences(dataset: SequenceDataset) -> NDArray[np.float64]:
    """Flatten within-sequence first differences with both endpoints observed."""

    dataset.validate()
    pair_mask = dataset.valid_mask[:, 1:, :] & dataset.valid_mask[:, :-1, :]
    differences = (
        dataset.errors[:, 1:, :].astype(np.float64)
        - dataset.errors[:, :-1, :].astype(np.float64)
    )
    return differences[pair_mask]


def _stable_histogram_edges(
    values: NDArray[np.float64],
    quantiles: tuple[float, float],
    bin_count: int,
) -> tuple[float, ...]:
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("at least two finite training values are required")
    lower, upper = np.quantile(values, quantiles)
    scale = max(float(np.std(values)), 1e-6)
    if float(upper - lower) < scale * 1e-6:
        center = 0.5 * float(lower + upper)
        lower, upper = center - scale, center + scale
    return tuple(
        float(value)
        for value in np.linspace(float(lower), float(upper), bin_count + 1)
    )


@dataclass(frozen=True)
class EvaluationReference:
    """Persisted training-only histogram ranges and tail thresholds."""

    schema_version: str
    fitted_split: str
    s_grid_m: tuple[float, ...]
    histogram_clip_quantiles: tuple[float, float]
    error_histogram_edges_m: tuple[float, ...]
    first_difference_histogram_edges_m: tuple[float, ...]
    absolute_error_thresholds_m: dict[str, float]
    training_observed_value_count: int
    training_first_difference_count: int

    @property
    def n_stations(self) -> int:
        return len(self.s_grid_m)

    def validate(self) -> None:
        if self.schema_version != EVALUATION_REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation reference {self.schema_version!r}")
        if self.fitted_split != "train":
            raise ValueError("evaluation reference must be fitted on training data")
        if not self.s_grid_m or not all(
            math.isfinite(value) for value in self.s_grid_m
        ):
            raise ValueError("evaluation reference grid must be finite and non-empty")
        if not all(
            second > first
            for first, second in zip(self.s_grid_m, self.s_grid_m[1:])
        ):
            raise ValueError("evaluation reference grid must be increasing")
        for name, edges in (
            ("error", self.error_histogram_edges_m),
            ("first-difference", self.first_difference_histogram_edges_m),
        ):
            if len(edges) < 11 or not all(math.isfinite(value) for value in edges):
                raise ValueError(f"{name} histogram edges are invalid")
            if not all(second > first for first, second in zip(edges, edges[1:])):
                raise ValueError(f"{name} histogram edges must be increasing")
        if not self.absolute_error_thresholds_m:
            raise ValueError("absolute error thresholds must not be empty")
        if any(
            value < 0 or not math.isfinite(value)
            for value in self.absolute_error_thresholds_m.values()
        ):
            raise ValueError("absolute error thresholds must be finite and non-negative")
        if self.training_observed_value_count < 2:
            raise ValueError("training observed value count is too small")
        if self.training_first_difference_count < 2:
            raise ValueError("training first-difference count is too small")

    @classmethod
    def fit(
        cls,
        dataset: SequenceDataset,
        config: EvaluationConfig,
        *,
        split_name: str,
    ) -> "EvaluationReference":
        if split_name != "train":
            raise ValueError("evaluation reference may only be fitted on train")
        dataset.validate()
        if dataset.standardized:
            raise ValueError("evaluation reference requires physical-unit data")
        config.validate()
        errors = dataset.errors[dataset.valid_mask].astype(np.float64)
        first_differences = valid_first_differences(dataset)
        state = cls(
            schema_version=EVALUATION_REFERENCE_SCHEMA_VERSION,
            fitted_split="train",
            s_grid_m=tuple(float(value) for value in dataset.s_grid_m),
            histogram_clip_quantiles=config.histogram_clip_quantiles,
            error_histogram_edges_m=_stable_histogram_edges(
                errors,
                config.histogram_clip_quantiles,
                config.histogram_bin_count,
            ),
            first_difference_histogram_edges_m=_stable_histogram_edges(
                first_differences,
                config.histogram_clip_quantiles,
                config.histogram_bin_count,
            ),
            absolute_error_thresholds_m={
                f"q{int(round(probability * 100)):02d}": float(
                    np.quantile(np.abs(errors), probability)
                )
                for probability in config.tail_probabilities
            },
            training_observed_value_count=int(errors.size),
            training_first_difference_count=int(first_differences.size),
        )
        state.validate()
        return state

    def assert_compatible(self, dataset: SequenceDataset) -> None:
        dataset.validate()
        if dataset.s_grid_m.shape != (self.n_stations,) or not np.allclose(
            dataset.s_grid_m,
            np.asarray(self.s_grid_m),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("evaluation data grid differs from training reference")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReference":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("evaluation reference root must be an object")
        for name in (
            "s_grid_m",
            "histogram_clip_quantiles",
            "error_histogram_edges_m",
            "first_difference_histogram_edges_m",
        ):
            raw[name] = tuple(raw[name])
        state = cls(**raw)
        state.validate()
        return state
