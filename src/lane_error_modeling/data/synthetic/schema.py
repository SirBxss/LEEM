"""In-memory schemas and strict validation for synthetic sequences and splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FEATURE_NAMES = (
    "ego_speed_mps",
    "mean_reference_curvature_inv_m",
    "reference_curvature_range_inv_m",
    "lane_width_m",
    "marking_quality",
    "environment_quality",
)


@dataclass(frozen=True)
class SequenceSample:
    """One continuous, independent synthetic driving segment."""

    sequence_id: str
    sequence_seed: int
    scenario: str
    conditions: NDArray[np.float64]
    errors: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    conditional_mean: NDArray[np.float64]
    latent_state: NDArray[np.int8]
    reference_curvature: NDArray[np.float64]
    reference_heading: NDArray[np.float64]
    reference_xy: NDArray[np.float64]

    @property
    def length(self) -> int:
        return len(self.conditions)

    def validate(self, n_stations: int, max_plausible_abs_error_m: float) -> None:
        expected_matrix = (self.length, n_stations)
        if self.conditions.shape != (self.length, len(FEATURE_NAMES)):
            raise ValueError(f"invalid conditions shape {self.conditions.shape}")
        for name, value in (
            ("errors", self.errors),
            ("valid_mask", self.valid_mask),
            ("conditional_mean", self.conditional_mean),
            ("reference_curvature", self.reference_curvature),
            ("reference_heading", self.reference_heading),
        ):
            if value.shape != expected_matrix:
                raise ValueError(f"invalid {name} shape {value.shape}")
        if self.reference_xy.shape != (self.length, n_stations, 2):
            raise ValueError(f"invalid reference_xy shape {self.reference_xy.shape}")
        if self.latent_state.shape != (self.length,):
            raise ValueError(f"invalid latent_state shape {self.latent_state.shape}")
        if not np.all(np.isfinite(self.conditions)):
            raise ValueError("conditions contain non-finite values")
        if not np.all(np.isfinite(self.errors)):
            raise ValueError("errors contain non-finite values")
        if not np.all(np.isfinite(self.conditional_mean)):
            raise ValueError("conditional means contain non-finite values")
        valid_errors = self.errors[self.valid_mask]
        if len(valid_errors) and np.max(np.abs(valid_errors)) > max_plausible_abs_error_m:
            raise ValueError("generated errors exceed the configured plausibility bound")
        if not np.all((self.conditions[:, 4:6] >= 0.0) & (self.conditions[:, 4:6] <= 1.0)):
            raise ValueError("quality features must lie in [0, 1]")


@dataclass(frozen=True)
class PaddedDataset:
    """One split represented by model-ready padded arrays."""

    sequence_ids: NDArray[np.str_]
    sequence_seeds: NDArray[np.uint64]
    lengths: NDArray[np.int32]
    conditions: NDArray[np.float32]
    errors: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    conditional_mean: NDArray[np.float32]
    latent_state: NDArray[np.int8]
    reference_curvature: NDArray[np.float32]
    reference_heading: NDArray[np.float32]
    reference_xy: NDArray[np.float32]
    s_grid_m: NDArray[np.float32]

    def validate(self) -> None:
        sequence_count, max_length, feature_count = self.conditions.shape
        station_count = len(self.s_grid_m)
        if feature_count != len(FEATURE_NAMES):
            raise ValueError("unexpected feature count")
        if self.errors.shape != (sequence_count, max_length, station_count):
            raise ValueError("errors have an incompatible shape")
        if self.valid_mask.shape != self.errors.shape:
            raise ValueError("valid_mask must match errors")
        if self.conditional_mean.shape != self.errors.shape:
            raise ValueError("conditional_mean must match errors")
        if self.reference_curvature.shape != self.errors.shape:
            raise ValueError("reference_curvature must match errors")
        if self.reference_heading.shape != self.errors.shape:
            raise ValueError("reference_heading must match errors")
        if self.reference_xy.shape != (sequence_count, max_length, station_count, 2):
            raise ValueError("reference_xy has an incompatible shape")
        if self.latent_state.shape != (sequence_count, max_length):
            raise ValueError("latent_state has an incompatible shape")
        if self.lengths.shape != (sequence_count,):
            raise ValueError("lengths has an incompatible shape")
        if np.any(self.lengths <= 0) or np.any(self.lengths > max_length):
            raise ValueError("invalid sequence lengths")
        if not np.all(np.diff(self.s_grid_m) > 0):
            raise ValueError("s_grid_m must be strictly increasing")
        for sequence_index, length in enumerate(self.lengths):
            if np.any(self.valid_mask[sequence_index, int(length) :]):
                raise ValueError("padding frames must be invalid")
            if np.any(self.errors[sequence_index, int(length) :] != 0.0):
                raise ValueError("padded errors must be zero")
        for array_name, array in (
            ("conditions", self.conditions),
            ("errors", self.errors),
            ("conditional_mean", self.conditional_mean),
            ("reference_curvature", self.reference_curvature),
            ("reference_heading", self.reference_heading),
            ("reference_xy", self.reference_xy),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{array_name} contains non-finite values")


def pad_samples(
    samples: list[SequenceSample],
    s_grid_m: NDArray[np.float64],
) -> PaddedDataset:
    """Pad independent variable-length samples without losing validity information."""

    if not samples:
        raise ValueError("at least one sample is required")
    sequence_count = len(samples)
    max_length = max(sample.length for sample in samples)
    station_count = len(s_grid_m)

    conditions = np.zeros((sequence_count, max_length, len(FEATURE_NAMES)), dtype=np.float32)
    errors = np.zeros((sequence_count, max_length, station_count), dtype=np.float32)
    valid_mask = np.zeros_like(errors, dtype=np.bool_)
    conditional_mean = np.zeros_like(errors)
    latent_state = np.full((sequence_count, max_length), -1, dtype=np.int8)
    reference_curvature = np.zeros_like(errors)
    reference_heading = np.zeros_like(errors)
    reference_xy = np.zeros((sequence_count, max_length, station_count, 2), dtype=np.float32)

    for sequence_index, sample in enumerate(samples):
        length = sample.length
        conditions[sequence_index, :length] = sample.conditions
        errors[sequence_index, :length] = sample.errors
        valid_mask[sequence_index, :length] = sample.valid_mask
        conditional_mean[sequence_index, :length] = sample.conditional_mean
        latent_state[sequence_index, :length] = sample.latent_state
        reference_curvature[sequence_index, :length] = sample.reference_curvature
        reference_heading[sequence_index, :length] = sample.reference_heading
        reference_xy[sequence_index, :length] = sample.reference_xy

    dataset = PaddedDataset(
        sequence_ids=np.asarray([sample.sequence_id for sample in samples]),
        sequence_seeds=np.asarray([sample.sequence_seed for sample in samples], dtype=np.uint64),
        lengths=np.asarray([sample.length for sample in samples], dtype=np.int32),
        conditions=conditions,
        errors=errors,
        valid_mask=valid_mask,
        conditional_mean=conditional_mean,
        latent_state=latent_state,
        reference_curvature=reference_curvature,
        reference_heading=reference_heading,
        reference_xy=reference_xy,
        s_grid_m=np.asarray(s_grid_m, dtype=np.float32),
    )
    dataset.validate()
    return dataset

