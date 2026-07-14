"""Configuration for the conditional multivariate Gaussian baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GaussianConfig:
    """Numerical choices for masked regression and spatial covariance fitting.

    The intercept is never penalized. ``covariance_shrinkage`` mixes the
    pairwise residual covariance with its diagonal before the symmetric matrix
    is projected onto the positive-definite cone.
    """

    ridge_penalty: float = 1e-3
    covariance_shrinkage: float = 0.10
    minimum_eigenvalue: float = 1e-6
    minimum_station_observations: int = 32
    minimum_pair_observations: int = 32

    def validate(self) -> None:
        for name, value in (
            ("ridge_penalty", self.ridge_penalty),
            ("covariance_shrinkage", self.covariance_shrinkage),
            ("minimum_eigenvalue", self.minimum_eigenvalue),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.ridge_penalty < 0:
            raise ValueError("ridge_penalty must be non-negative")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        if self.minimum_eigenvalue <= 0:
            raise ValueError("minimum_eigenvalue must be positive")
        for name, value in (
            ("minimum_station_observations", self.minimum_station_observations),
            ("minimum_pair_observations", self.minimum_pair_observations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.minimum_station_observations < 2:
            raise ValueError("minimum_station_observations must be at least two")
        if self.minimum_pair_observations < 2:
            raise ValueError("minimum_pair_observations must be at least two")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GaussianConfig":
        known_fields = set(cls.__dataclass_fields__)
        unknown_fields = set(raw) - known_fields
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"unknown Gaussian configuration fields: {unknown}")
        config = cls(**dict(raw))
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "GaussianConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Gaussian configuration root must be an object")
        return cls.from_dict(raw)
