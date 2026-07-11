"""Configuration objects for reproducible synthetic datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_SCENARIOS = (
    "conditional_gaussian",
    "latent_autoregressive",
    "nonlinear_heavy_tailed",
)


@dataclass(frozen=True)
class SplitSizes:
    """Number of independent sequences in every split."""

    train: int
    validation: int
    test: int

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"split size {name!r} must be positive, got {value}")


@dataclass(frozen=True)
class ConditionRanges:
    """Physical bounds used by the reference-scene generator."""

    speed_min_mps: float = 3.0
    speed_max_mps: float = 38.0
    lane_width_min_m: float = 2.8
    lane_width_max_m: float = 4.2
    curvature_abs_max_inv_m: float = 0.03
    curvature_gradient_abs_max_inv_m2: float = 0.0002

    def validate(self) -> None:
        if not 0 <= self.speed_min_mps < self.speed_max_mps:
            raise ValueError("invalid speed interval")
        if not 0 < self.lane_width_min_m < self.lane_width_max_m:
            raise ValueError("invalid lane-width interval")
        if self.curvature_abs_max_inv_m <= 0:
            raise ValueError("curvature_abs_max_inv_m must be positive")
        if self.curvature_gradient_abs_max_inv_m2 <= 0:
            raise ValueError("curvature_gradient_abs_max_inv_m2 must be positive")


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Complete, serializable definition of one synthetic experiment dataset."""

    schema_version: str
    master_seed: int
    sample_rate_hz: float
    min_sequence_frames: int
    max_sequence_frames: int
    s_grid_m: tuple[float, ...]
    splits: SplitSizes
    scenarios: tuple[str, ...]
    condition_ranges: ConditionRanges = ConditionRanges()
    max_plausible_abs_error_m: float = 5.0

    @property
    def n_stations(self) -> int:
        return len(self.s_grid_m)

    @property
    def n_features(self) -> int:
        return 6

    def validate(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.min_sequence_frames < 2:
            raise ValueError("min_sequence_frames must be at least 2")
        if self.max_sequence_frames < self.min_sequence_frames:
            raise ValueError("max_sequence_frames must be >= min_sequence_frames")
        if len(self.s_grid_m) < 2:
            raise ValueError("s_grid_m must contain at least two stations")
        if self.s_grid_m[0] != 0.0:
            raise ValueError("s_grid_m must start at 0 m")
        if any(b <= a for a, b in zip(self.s_grid_m, self.s_grid_m[1:])):
            raise ValueError("s_grid_m must be strictly increasing")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        unknown = sorted(set(self.scenarios) - set(SUPPORTED_SCENARIOS))
        if unknown:
            raise ValueError(f"unsupported scenarios: {unknown}")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must not contain duplicates")
        if self.max_plausible_abs_error_m <= 0:
            raise ValueError("max_plausible_abs_error_m must be positive")
        self.splits.validate()
        self.condition_ranges.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SyntheticDatasetConfig":
        data = dict(raw)
        data["s_grid_m"] = tuple(float(value) for value in data["s_grid_m"])
        data["scenarios"] = tuple(str(value) for value in data["scenarios"])
        data["splits"] = SplitSizes(**data["splits"])
        data["condition_ranges"] = ConditionRanges(**data.get("condition_ranges", {}))
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "SyntheticDatasetConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_dict(raw)
