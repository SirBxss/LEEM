"""Common contract for Gaussian, AIOHMM, and RC-GAN sequence models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lane_error_modeling.data.preprocessing.batching import SequenceDataset


@dataclass(frozen=True)
class ModelCapabilities:
    """Declare capabilities that differ across the three model families."""

    supports_log_probability: bool
    supports_missing_targets: bool = True
    supports_variable_length: bool = True


@dataclass(frozen=True)
class FitReport:
    """Small serializable summary returned by every model's fitting procedure."""

    model_name: str
    train_sequence_count: int
    validation_sequence_count: int
    metrics: Mapping[str, float]
    warnings: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.train_sequence_count <= 0:
            raise ValueError("train_sequence_count must be positive")
        if self.validation_sequence_count < 0:
            raise ValueError("validation_sequence_count must be non-negative")
        if any(not key for key in self.metrics):
            raise ValueError("metric names must not be empty")
        if any(not np.isfinite(value) for value in self.metrics.values()):
            raise ValueError("fit metrics must be finite")


@dataclass(frozen=True)
class SampleResult:
    """Generated standardized or physical error sequences.

    ``values`` has shape ``[S,B,T,K]`` for samples, sequences, time steps, and
    look-ahead stations. Padding frames must be zero. ``valid_mask`` describes
    evaluation availability and is not required to hide generated full profiles.
    """

    values: NDArray[np.float32]
    lengths: NDArray[np.int32]
    s_grid_m: NDArray[np.float32]
    standardized: bool
    valid_mask: NDArray[np.bool_] | None = None

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    def validate(self) -> None:
        if self.values.ndim != 4:
            raise ValueError("sample values must have shape [S, B, T, K]")
        sample_count, sequence_count, max_length, station_count = self.values.shape
        if sample_count <= 0 or sequence_count <= 0:
            raise ValueError("sample and sequence dimensions must be positive")
        if self.lengths.shape != (sequence_count,):
            raise ValueError("lengths must have shape [B]")
        if np.any(self.lengths <= 0) or np.any(self.lengths > max_length):
            raise ValueError("lengths contain invalid values")
        if self.s_grid_m.shape != (station_count,):
            raise ValueError("s_grid_m does not match generated station count")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("generated values contain non-finite entries")
        time_mask = np.arange(max_length)[None, :] < self.lengths[:, None]
        if np.any(self.values * (~time_mask)[None, :, :, None] != 0.0):
            raise ValueError("generated padding frames must be zero")
        if self.valid_mask is not None:
            if self.valid_mask.shape != (sequence_count, max_length, station_count):
                raise ValueError("valid_mask must have shape [B, T, K]")
            if np.any(self.valid_mask & ~time_mask[:, :, None]):
                raise ValueError("valid_mask marks padding frames as valid")


class ProbabilisticSequenceModel(ABC):
    """Shared lifecycle and tensor contract for all three thesis models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable name used in experiment metadata."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Capabilities used to select fair common metrics."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether model parameters are available for sampling/scoring."""

    @abstractmethod
    def fit(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None = None,
    ) -> FitReport:
        """Fit model parameters using standardized sequence data."""

    @abstractmethod
    def sample(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None = None,
    ) -> SampleResult:
        """Generate error sequences conditional on a standardized input sequence."""

    def log_probability(self, dataset: SequenceDataset) -> NDArray[np.float64]:
        """Return per-sequence log probability when the model has a density."""

        raise NotImplementedError(
            f"{self.model_name} does not implement a tractable log probability"
        )

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist fitted parameters and model configuration."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a fitted model from a persisted artifact."""

    @staticmethod
    def validate_fit_datasets(
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None,
    ) -> None:
        """Enforce common preprocessing and schema rules before fitting."""

        train_data.validate()
        if not train_data.standardized:
            raise ValueError("train_data must be standardized before model fitting")
        if validation_data is None:
            return
        validation_data.validate()
        if not validation_data.standardized:
            raise ValueError("validation_data must be standardized")
        if validation_data.feature_names != train_data.feature_names:
            raise ValueError("validation feature names/order differ from training")
        if not np.array_equal(validation_data.s_grid_m, train_data.s_grid_m):
            raise ValueError("validation look-ahead grid differs from training")

    @staticmethod
    def validate_sample_request(
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        expected_feature_count: int,
        expected_station_count: int,
        valid_mask: ArrayLike | None = None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
    ]:
        """Validate shared conditional-sampling arguments for subclasses."""

        condition_array = np.asarray(conditions, dtype=np.float32)
        length_array = np.asarray(lengths, dtype=np.int32)
        if condition_array.ndim != 3 or condition_array.shape[2] != expected_feature_count:
            raise ValueError("conditions have an incompatible shape")
        if length_array.shape != (condition_array.shape[0],):
            raise ValueError("lengths must have shape [B]")
        if np.any(length_array <= 0) or np.any(length_array > condition_array.shape[1]):
            raise ValueError("lengths contain invalid values")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        time_mask = np.arange(condition_array.shape[1])[None, :] < length_array[:, None]
        if not np.all(np.isfinite(condition_array[time_mask])):
            raise ValueError("active conditions contain non-finite values")
        if np.any(condition_array[~time_mask] != 0.0):
            raise ValueError("condition padding must be zero")

        mask_array: NDArray[np.bool_] | None = None
        if valid_mask is not None:
            mask_array = np.asarray(valid_mask, dtype=np.bool_)
            expected_shape = (
                condition_array.shape[0],
                condition_array.shape[1],
                expected_station_count,
            )
            if mask_array.shape != expected_shape:
                raise ValueError("valid_mask has an incompatible shape")
            if np.any(mask_array & ~time_mask[:, :, None]):
                raise ValueError("valid_mask marks padding frames as valid")
        return condition_array, length_array, mask_array

