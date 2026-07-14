"""Configuration for the multivariate autoregressive input-output HMM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AIOHMMConfig:
    """Numerical and structural choices for one deterministic EM run.

    The model uses state-specific conditional linear means, diagonal
    station-wise autoregression, full state-specific spatial covariance, and
    input-dependent multinomial-logistic state transitions.
    """

    n_states: int = 4
    max_em_iterations: int = 40
    min_em_iterations: int = 4
    convergence_tolerance: float = 1e-4
    ridge_penalty: float = 1e-3
    covariance_shrinkage: float = 0.10
    minimum_eigenvalue: float = 1e-5
    minimum_effective_station_observations: float = 20.0
    minimum_effective_pair_observations: float = 20.0
    maximum_absolute_autoregression: float = 0.98
    transition_l2_penalty: float = 1e-3
    transition_learning_rate: float = 0.03
    transition_adam_steps: int = 40
    initial_probability_smoothing: float = 1e-2
    minimum_state_occupancy_fraction: float = 0.005
    initialization_seed: int = 20260715
    input_dependent_transitions: bool = True

    def validate(self) -> None:
        for name in (
            "n_states",
            "max_em_iterations",
            "min_em_iterations",
            "transition_adam_steps",
            "initialization_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if self.max_em_iterations < 1:
            raise ValueError("max_em_iterations must be positive")
        if not 1 <= self.min_em_iterations <= self.max_em_iterations:
            raise ValueError(
                "min_em_iterations must lie between one and max_em_iterations"
            )
        if self.transition_adam_steps < 1:
            raise ValueError("transition_adam_steps must be positive")
        if self.initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative")

        for name in (
            "convergence_tolerance",
            "ridge_penalty",
            "covariance_shrinkage",
            "minimum_eigenvalue",
            "minimum_effective_station_observations",
            "minimum_effective_pair_observations",
            "maximum_absolute_autoregression",
            "transition_l2_penalty",
            "transition_learning_rate",
            "initial_probability_smoothing",
            "minimum_state_occupancy_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")

        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")
        if self.ridge_penalty < 0 or self.transition_l2_penalty < 0:
            raise ValueError("regularization penalties must be non-negative")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        if self.minimum_eigenvalue <= 0:
            raise ValueError("minimum_eigenvalue must be positive")
        if self.minimum_effective_station_observations < 2:
            raise ValueError(
                "minimum_effective_station_observations must be at least two"
            )
        if self.minimum_effective_pair_observations < 2:
            raise ValueError(
                "minimum_effective_pair_observations must be at least two"
            )
        if not 0.0 < self.maximum_absolute_autoregression < 1.0:
            raise ValueError(
                "maximum_absolute_autoregression must lie strictly inside (0, 1)"
            )
        if self.transition_learning_rate <= 0:
            raise ValueError("transition_learning_rate must be positive")
        if self.initial_probability_smoothing <= 0:
            raise ValueError("initial_probability_smoothing must be positive")
        if not 0.0 < self.minimum_state_occupancy_fraction < 1.0:
            raise ValueError(
                "minimum_state_occupancy_fraction must lie strictly inside (0, 1)"
            )
        if not isinstance(self.input_dependent_transitions, bool):
            raise ValueError("input_dependent_transitions must be boolean")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AIOHMMConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown AIOHMM configuration fields: {', '.join(sorted(unknown))}"
            )
        config = cls(**dict(raw))
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "AIOHMMConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("AIOHMM configuration root must be an object")
        return cls.from_dict(raw)
