"""Strict schemas for common evaluation and all three thesis models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from lane_error_modeling.models.aiohmm import AIOHMMConfig
from lane_error_modeling.models.gaussian import GaussianConfig
from lane_error_modeling.models.rcgan.config import RCGANConfig


EXPERIMENT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SYNTHETIC_SCENARIOS = (
    "conditional_gaussian",
    "latent_autoregressive",
    "nonlinear_heavy_tailed",
)


def _reject_unknown(raw: Mapping[str, Any], known: set[str], name: str) -> None:
    unknown = set(raw) - known
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {name} fields: {fields}")


@dataclass(frozen=True)
class EvaluationConfig:
    """Computational and statistical choices shared by all model families."""

    interval_levels: tuple[float, ...] = (0.50, 0.90, 0.95)
    histogram_bin_count: int = 80
    histogram_clip_quantiles: tuple[float, float] = (0.001, 0.999)
    tail_probabilities: tuple[float, ...] = (0.95, 0.99)
    crps_chunk_size: int = 4096
    max_distribution_values: int = 200_000
    max_energy_frames: int = 2_000
    energy_pair_count: int = 256
    max_dependence_frames: int = 20_000
    max_dependence_samples: int = 16
    metric_seed: int = 20260714

    def validate(self) -> None:
        if not self.interval_levels or any(
            not 0.0 < value < 1.0 for value in self.interval_levels
        ):
            raise ValueError("interval_levels must lie strictly inside (0, 1)")
        if tuple(sorted(set(self.interval_levels))) != self.interval_levels:
            raise ValueError("interval_levels must be unique and increasing")
        if self.histogram_bin_count < 10:
            raise ValueError("histogram_bin_count must be at least 10")
        if (
            len(self.histogram_clip_quantiles) != 2
            or not 0.0 <= self.histogram_clip_quantiles[0]
            < self.histogram_clip_quantiles[1]
            <= 1.0
        ):
            raise ValueError("histogram_clip_quantiles must be increasing in [0, 1]")
        if not self.tail_probabilities or any(
            not 0.5 < value < 1.0 for value in self.tail_probabilities
        ):
            raise ValueError("tail_probabilities must lie strictly inside (0.5, 1)")
        if tuple(sorted(set(self.tail_probabilities))) != self.tail_probabilities:
            raise ValueError("tail_probabilities must be unique and increasing")
        for name in (
            "crps_chunk_size",
            "max_distribution_values",
            "max_energy_frames",
            "energy_pair_count",
            "max_dependence_frames",
            "max_dependence_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.metric_seed, bool) or not isinstance(self.metric_seed, int):
            raise ValueError("metric_seed must be an integer")
        if self.metric_seed < 0:
            raise ValueError("metric_seed must be non-negative")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationConfig":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "evaluation configuration")
        data = dict(raw)
        for name in (
            "interval_levels",
            "histogram_clip_quantiles",
            "tail_probabilities",
        ):
            if name in data:
                data[name] = tuple(data[name])
        config = cls(**data)
        config.validate()
        return config


@dataclass(frozen=True)
class GaussianSearchSpace:
    """Validation-only Cartesian search space for the Gaussian baseline."""

    ridge_penalties: tuple[float, ...] = (0.0, 1e-3, 1e-2)
    covariance_shrinkages: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    minimum_eigenvalue: float = 1e-6
    minimum_station_observations: int = 32
    minimum_pair_observations: int = 32

    def validate(self) -> None:
        if not self.ridge_penalties or not self.covariance_shrinkages:
            raise ValueError("Gaussian search dimensions must not be empty")
        for name, values in (
            ("ridge_penalties", self.ridge_penalties),
            ("covariance_shrinkages", self.covariance_shrinkages),
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"{name} must contain finite numbers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if tuple(sorted(self.ridge_penalties)) != self.ridge_penalties:
            raise ValueError("ridge_penalties must be increasing")
        if tuple(sorted(self.covariance_shrinkages)) != self.covariance_shrinkages:
            raise ValueError("covariance_shrinkages must be increasing")
        for ridge_penalty in self.ridge_penalties:
            if ridge_penalty < 0:
                raise ValueError("ridge penalties must be non-negative")
        for shrinkage in self.covariance_shrinkages:
            if not 0.0 <= shrinkage <= 1.0:
                raise ValueError("covariance shrinkages must lie in [0, 1]")
        GaussianConfig(
            ridge_penalty=float(self.ridge_penalties[0]),
            covariance_shrinkage=float(self.covariance_shrinkages[0]),
            minimum_eigenvalue=self.minimum_eigenvalue,
            minimum_station_observations=self.minimum_station_observations,
            minimum_pair_observations=self.minimum_pair_observations,
        ).validate()

    def candidates(self) -> tuple[GaussianConfig, ...]:
        self.validate()
        return tuple(
            GaussianConfig(
                ridge_penalty=float(ridge_penalty),
                covariance_shrinkage=float(shrinkage),
                minimum_eigenvalue=self.minimum_eigenvalue,
                minimum_station_observations=self.minimum_station_observations,
                minimum_pair_observations=self.minimum_pair_observations,
            )
            for ridge_penalty in self.ridge_penalties
            for shrinkage in self.covariance_shrinkages
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GaussianSearchSpace":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "Gaussian search")
        data = dict(raw)
        for name in ("ridge_penalties", "covariance_shrinkages"):
            if name in data:
                data[name] = tuple(data[name])
        search = cls(**data)
        search.validate()
        return search


@dataclass(frozen=True)
class GaussianExperimentConfig:
    """Complete reproducible configuration for one multi-scenario experiment."""

    schema_version: str
    experiment_name: str
    dataset_root: str
    scenarios: tuple[str, ...]
    sample_count: int
    sample_seed: int
    evaluation: EvaluationConfig
    gaussian_search: GaussianSearchSpace
    create_plots: bool = True

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment schema {self.schema_version!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.experiment_name):
            raise ValueError("experiment_name must be a lowercase filesystem-safe slug")
        dataset_path = Path(self.dataset_root)
        if dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise ValueError("dataset_root must be a project-relative path")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be non-empty and unique")
        unsupported = set(self.scenarios) - set(SUPPORTED_SYNTHETIC_SCENARIOS)
        if unsupported:
            raise ValueError(f"unsupported scenarios: {sorted(unsupported)}")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise ValueError("sample_count must be an integer")
        if self.sample_count < 8:
            raise ValueError("sample_count must be at least eight")
        if isinstance(self.sample_seed, bool) or not isinstance(self.sample_seed, int):
            raise ValueError("sample_seed must be an integer")
        if self.sample_seed < 0:
            raise ValueError("sample_seed must be non-negative")
        if not isinstance(self.create_plots, bool):
            raise ValueError("create_plots must be boolean")
        self.evaluation.validate()
        self.gaussian_search.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "dataset_root": self.dataset_root,
            "scenarios": list(self.scenarios),
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "evaluation": self.evaluation.to_dict(),
            "gaussian_search": self.gaussian_search.to_dict(),
            "create_plots": self.create_plots,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GaussianExperimentConfig":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "experiment configuration")
        data = dict(raw)
        if "scenarios" in data:
            data["scenarios"] = tuple(data["scenarios"])
        evaluation = data.get("evaluation", {})
        search = data.get("gaussian_search", {})
        if not isinstance(evaluation, Mapping) or not isinstance(search, Mapping):
            raise ValueError("evaluation and gaussian_search must be objects")
        data["evaluation"] = EvaluationConfig.from_dict(evaluation)
        data["gaussian_search"] = GaussianSearchSpace.from_dict(search)
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "GaussianExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("experiment configuration root must be an object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class AIOHMMSearchSpace:
    """Validation-only state-count and deterministic-restart search."""

    state_counts: tuple[int, ...] = (3, 4, 5)
    initialization_seeds: tuple[int, ...] = (20260715, 20260716)
    max_em_iterations: int = 25
    min_em_iterations: int = 5
    convergence_tolerance: float = 5e-4
    ridge_penalty: float = 1e-3
    covariance_shrinkage: float = 0.10
    minimum_eigenvalue: float = 1e-5
    minimum_effective_station_observations: float = 20.0
    minimum_effective_pair_observations: float = 20.0
    maximum_absolute_autoregression: float = 0.98
    transition_l2_penalty: float = 1e-3
    transition_learning_rate: float = 0.03
    transition_adam_steps: int = 25
    initial_probability_smoothing: float = 1e-2
    minimum_state_occupancy_fraction: float = 0.005
    input_dependent_transitions: bool = True

    def validate(self) -> None:
        if not self.state_counts or not self.initialization_seeds:
            raise ValueError("AIOHMM search dimensions must not be empty")
        if tuple(sorted(set(self.state_counts))) != self.state_counts:
            raise ValueError("state_counts must be unique and increasing")
        if len(set(self.initialization_seeds)) != len(self.initialization_seeds):
            raise ValueError("initialization_seeds must not contain duplicates")
        for state_count in self.state_counts:
            if isinstance(state_count, bool) or not isinstance(state_count, int):
                raise ValueError("state_counts must contain integers")
        for seed in self.initialization_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    "initialization_seeds must contain non-negative integers"
                )
        self._candidate(self.state_counts[0], self.initialization_seeds[0]).validate()

    def _candidate(self, state_count: int, seed: int) -> AIOHMMConfig:
        return AIOHMMConfig(
            n_states=state_count,
            max_em_iterations=self.max_em_iterations,
            min_em_iterations=self.min_em_iterations,
            convergence_tolerance=self.convergence_tolerance,
            ridge_penalty=self.ridge_penalty,
            covariance_shrinkage=self.covariance_shrinkage,
            minimum_eigenvalue=self.minimum_eigenvalue,
            minimum_effective_station_observations=(
                self.minimum_effective_station_observations
            ),
            minimum_effective_pair_observations=(
                self.minimum_effective_pair_observations
            ),
            maximum_absolute_autoregression=(
                self.maximum_absolute_autoregression
            ),
            transition_l2_penalty=self.transition_l2_penalty,
            transition_learning_rate=self.transition_learning_rate,
            transition_adam_steps=self.transition_adam_steps,
            initial_probability_smoothing=self.initial_probability_smoothing,
            minimum_state_occupancy_fraction=(
                self.minimum_state_occupancy_fraction
            ),
            initialization_seed=seed,
            input_dependent_transitions=self.input_dependent_transitions,
        )

    def candidates(self) -> tuple[AIOHMMConfig, ...]:
        self.validate()
        return tuple(
            self._candidate(state_count, seed)
            for state_count in self.state_counts
            for seed in self.initialization_seeds
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AIOHMMSearchSpace":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "AIOHMM search")
        data = dict(raw)
        for name in ("state_counts", "initialization_seeds"):
            if name in data:
                data[name] = tuple(data[name])
        search = cls(**data)
        search.validate()
        return search


@dataclass(frozen=True)
class AIOHMMExperimentConfig:
    """Complete reproducible configuration for an AIOHMM experiment."""

    schema_version: str
    experiment_name: str
    dataset_root: str
    scenarios: tuple[str, ...]
    sample_count: int
    sample_seed: int
    evaluation: EvaluationConfig
    aiohmm_search: AIOHMMSearchSpace
    create_plots: bool = True

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment schema {self.schema_version!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.experiment_name):
            raise ValueError("experiment_name must be a lowercase filesystem-safe slug")
        dataset_path = Path(self.dataset_root)
        if dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise ValueError("dataset_root must be a project-relative path")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be non-empty and unique")
        unsupported = set(self.scenarios) - set(SUPPORTED_SYNTHETIC_SCENARIOS)
        if unsupported:
            raise ValueError(f"unsupported scenarios: {sorted(unsupported)}")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 8
        ):
            raise ValueError("sample_count must be an integer of at least eight")
        if (
            isinstance(self.sample_seed, bool)
            or not isinstance(self.sample_seed, int)
            or self.sample_seed < 0
        ):
            raise ValueError("sample_seed must be a non-negative integer")
        if not isinstance(self.create_plots, bool):
            raise ValueError("create_plots must be boolean")
        self.evaluation.validate()
        self.aiohmm_search.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "dataset_root": self.dataset_root,
            "scenarios": list(self.scenarios),
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "evaluation": self.evaluation.to_dict(),
            "aiohmm_search": self.aiohmm_search.to_dict(),
            "create_plots": self.create_plots,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AIOHMMExperimentConfig":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "experiment configuration")
        data = dict(raw)
        if "scenarios" in data:
            data["scenarios"] = tuple(data["scenarios"])
        evaluation = data.get("evaluation", {})
        search = data.get("aiohmm_search", {})
        if not isinstance(evaluation, Mapping) or not isinstance(search, Mapping):
            raise ValueError("evaluation and aiohmm_search must be objects")
        data["evaluation"] = EvaluationConfig.from_dict(evaluation)
        data["aiohmm_search"] = AIOHMMSearchSpace.from_dict(search)
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "AIOHMMExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("experiment configuration root must be an object")
        return cls.from_dict(raw)


@dataclass(frozen=True)
class RCGANSearchSpace:
    """Paper architecture with declared learning-rate and seed candidates."""

    initialization_seeds: tuple[int, ...] = (20260717, 20260718)
    latent_size: int = 32
    noise_hidden_size: int = 64
    context_hidden_size: int = 64
    context_layers: int = 2
    discriminator_hidden_size: int = 64
    discriminator_layers: int = 2
    dense_hidden_size: int = 64
    discriminator_dropout: float = 0.05
    leaky_relu_slope: float = 0.2
    epochs: int = 4
    batch_size: int = 1
    learning_rate: float = 1e-5
    learning_rate_candidates: tuple[float, ...] = ()
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    gradient_clip_norm: float = 1.0
    discriminator_steps: int = 1
    generator_steps: int = 1
    sample_batch_size: int = 16
    diagnostic_sample_count: int = 8
    device: str = "cpu"

    def validate(self) -> None:
        if not self.initialization_seeds:
            raise ValueError("initialization_seeds must not be empty")
        if len(set(self.initialization_seeds)) != len(self.initialization_seeds):
            raise ValueError("initialization_seeds must not contain duplicates")
        for seed in self.initialization_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    "initialization_seeds must contain non-negative integers"
                )
        if len(set(self.learning_rate_candidates)) != len(
            self.learning_rate_candidates
        ):
            raise ValueError("learning_rate_candidates must not contain duplicates")
        rates = self._effective_learning_rates()
        for rate in rates:
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or rate <= 0.0
            ):
                raise ValueError(
                    "learning_rate_candidates must contain positive finite numbers"
                )
        self._candidate(self.initialization_seeds[0], rates[0]).validate()

    def _effective_learning_rates(self) -> tuple[float, ...]:
        if self.learning_rate_candidates:
            return tuple(float(value) for value in self.learning_rate_candidates)
        return (float(self.learning_rate),)

    def _candidate(self, seed: int, learning_rate: float) -> RCGANConfig:
        values = asdict(self)
        values.pop("initialization_seeds")
        values.pop("learning_rate_candidates")
        values["learning_rate"] = learning_rate
        return RCGANConfig(initialization_seed=seed, **values)

    def candidates(self) -> tuple[RCGANConfig, ...]:
        self.validate()
        return tuple(
            self._candidate(seed, learning_rate)
            for learning_rate in self._effective_learning_rates()
            for seed in self.initialization_seeds
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RCGANSearchSpace":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "RC-GAN search")
        data = dict(raw)
        if "initialization_seeds" in data:
            data["initialization_seeds"] = tuple(data["initialization_seeds"])
        if "learning_rate_candidates" in data:
            data["learning_rate_candidates"] = tuple(
                data["learning_rate_candidates"]
            )
        search = cls(**data)
        search.validate()
        return search


@dataclass(frozen=True)
class RCGANExperimentConfig:
    """Leakage-safe validation selection and held-out RC-GAN evaluation."""

    schema_version: str
    experiment_name: str
    dataset_root: str
    scenarios: tuple[str, ...]
    selection_sample_count: int
    selection_sample_seed: int
    sample_count: int
    sample_seed: int
    evaluation: EvaluationConfig
    rcgan_search: RCGANSearchSpace
    minimum_validation_diversity_ratio: float = 0.0
    create_plots: bool = True

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment schema {self.schema_version!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.experiment_name):
            raise ValueError("experiment_name must be a lowercase filesystem-safe slug")
        dataset_path = Path(self.dataset_root)
        if dataset_path.is_absolute() or ".." in dataset_path.parts:
            raise ValueError("dataset_root must be a project-relative path")
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be non-empty and unique")
        unsupported = set(self.scenarios) - set(SUPPORTED_SYNTHETIC_SCENARIOS)
        if unsupported:
            raise ValueError(f"unsupported scenarios: {sorted(unsupported)}")
        for name in ("selection_sample_count", "sample_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 8:
                raise ValueError(f"{name} must be an integer of at least eight")
        for name in ("selection_sample_seed", "sample_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.create_plots, bool):
            raise ValueError("create_plots must be boolean")
        if (
            isinstance(self.minimum_validation_diversity_ratio, bool)
            or not isinstance(
                self.minimum_validation_diversity_ratio, (int, float)
            )
            or not math.isfinite(float(self.minimum_validation_diversity_ratio))
            or not 0.0 <= self.minimum_validation_diversity_ratio <= 1.0
        ):
            raise ValueError(
                "minimum_validation_diversity_ratio must lie in [0, 1]"
            )
        self.evaluation.validate()
        self.rcgan_search.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "dataset_root": self.dataset_root,
            "scenarios": list(self.scenarios),
            "selection_sample_count": self.selection_sample_count,
            "selection_sample_seed": self.selection_sample_seed,
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "evaluation": self.evaluation.to_dict(),
            "rcgan_search": self.rcgan_search.to_dict(),
            "minimum_validation_diversity_ratio": (
                self.minimum_validation_diversity_ratio
            ),
            "create_plots": self.create_plots,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RCGANExperimentConfig":
        _reject_unknown(raw, set(cls.__dataclass_fields__), "experiment configuration")
        data = dict(raw)
        if "scenarios" in data:
            data["scenarios"] = tuple(data["scenarios"])
        evaluation = data.get("evaluation", {})
        search = data.get("rcgan_search", {})
        if not isinstance(evaluation, Mapping) or not isinstance(search, Mapping):
            raise ValueError("evaluation and rcgan_search must be objects")
        data["evaluation"] = EvaluationConfig.from_dict(evaluation)
        data["rcgan_search"] = RCGANSearchSpace.from_dict(search)
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "RCGANExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("experiment configuration root must be an object")
        return cls.from_dict(raw)
