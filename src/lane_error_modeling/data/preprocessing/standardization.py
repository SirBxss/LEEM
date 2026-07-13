"""Train-only, mask-aware standardization for conditional sequence models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


STANDARDIZATION_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class StandardizationState:
    """Persistable statistics fitted exclusively on the training split."""

    schema_version: str
    fitted_split: str
    feature_names: tuple[str, ...]
    s_grid_m: tuple[float, ...]
    condition_mean: tuple[float, ...]
    condition_scale: tuple[float, ...]
    condition_count: int
    error_mean: tuple[float, ...]
    error_scale: tuple[float, ...]
    error_count: tuple[int, ...]
    constant_condition_indices: tuple[int, ...]
    constant_error_station_indices: tuple[int, ...]
    minimum_scale: float

    @property
    def n_features(self) -> int:
        return len(self.condition_mean)

    @property
    def n_stations(self) -> int:
        return len(self.error_mean)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "StandardizationState":
        tuple_fields = (
            "feature_names",
            "s_grid_m",
            "condition_mean",
            "condition_scale",
            "error_mean",
            "error_scale",
            "error_count",
            "constant_condition_indices",
            "constant_error_station_indices",
        )
        data = dict(raw)
        for field_name in tuple_fields:
            data[field_name] = tuple(data[field_name])  # type: ignore[arg-type]
        state = cls(**data)  # type: ignore[arg-type]
        state.validate()
        return state

    def validate(self) -> None:
        if self.schema_version != STANDARDIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported standardization schema {self.schema_version!r}"
            )
        if self.fitted_split != "train":
            raise ValueError("standardization state must be fitted on the train split")
        if not self.feature_names or len(self.feature_names) != self.n_features:
            raise ValueError("feature_names and condition statistics are incompatible")
        if len(self.condition_scale) != self.n_features:
            raise ValueError("condition mean and scale dimensions differ")
        if not self.s_grid_m or len(self.s_grid_m) != self.n_stations:
            raise ValueError("s_grid_m and error statistics are incompatible")
        if len(self.error_scale) != self.n_stations:
            raise ValueError("error mean and scale dimensions differ")
        if len(self.error_count) != self.n_stations:
            raise ValueError("error counts and statistics dimensions differ")
        if self.condition_count < 2:
            raise ValueError("at least two active training frames are required")
        if any(count < 2 for count in self.error_count):
            raise ValueError("each station needs at least two valid training errors")
        if self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        if any(scale <= 0 or not np.isfinite(scale) for scale in self.condition_scale):
            raise ValueError("condition scales must be finite and positive")
        if any(scale <= 0 or not np.isfinite(scale) for scale in self.error_scale):
            raise ValueError("error scales must be finite and positive")


def _validate_core_arrays(
    conditions: ArrayLike,
    errors: ArrayLike,
    valid_mask: ArrayLike,
    lengths: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.bool_],
]:
    condition_array = np.asarray(conditions, dtype=np.float64)
    error_array = np.asarray(errors, dtype=np.float64)
    mask_array = np.asarray(valid_mask, dtype=np.bool_)
    length_array = np.asarray(lengths, dtype=np.int64)

    if condition_array.ndim != 3:
        raise ValueError("conditions must have shape [B, T, F]")
    sequence_count, max_length, _ = condition_array.shape
    if error_array.ndim != 3 or error_array.shape[:2] != (sequence_count, max_length):
        raise ValueError("errors must have shape [B, T, K]")
    if mask_array.shape != error_array.shape:
        raise ValueError("valid_mask must have the same shape as errors")
    if length_array.shape != (sequence_count,):
        raise ValueError("lengths must have shape [B]")
    if np.any(length_array <= 0) or np.any(length_array > max_length):
        raise ValueError("lengths contain invalid values")

    time_mask = np.arange(max_length)[None, :] < length_array[:, None]
    if np.any(mask_array & ~time_mask[:, :, None]):
        raise ValueError("valid_mask marks padding frames as valid")
    if not np.all(np.isfinite(condition_array[time_mask])):
        raise ValueError("active conditions contain non-finite values")
    if not np.all(np.isfinite(error_array[mask_array])):
        raise ValueError("valid errors contain non-finite values")
    return condition_array, error_array, mask_array, length_array, time_mask


class SequenceStandardizer:
    """Standardize conditions feature-wise and errors station-wise.

    The object cannot be fitted on a split name other than ``train``. Population
    statistics (``ddof=0``) are used because the fitted values parameterize the
    preprocessing transform rather than estimate an unbiased sample variance.
    Padding and unavailable targets remain exactly zero after transformation.
    """

    def __init__(
        self,
        state: StandardizationState | None = None,
        minimum_scale: float = 1e-8,
    ) -> None:
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        self._state = state
        self._minimum_scale = float(minimum_scale)
        if state is not None:
            state.validate()

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> StandardizationState:
        if self._state is None:
            raise RuntimeError("standardizer has not been fitted")
        return self._state

    def fit(
        self,
        conditions: ArrayLike,
        errors: ArrayLike,
        valid_mask: ArrayLike,
        lengths: ArrayLike,
        *,
        split_name: str,
        feature_names: Sequence[str],
        s_grid_m: ArrayLike,
    ) -> "SequenceStandardizer":
        """Fit statistics, rejecting validation/test data by construction."""

        if split_name != "train":
            raise ValueError(
                "standardization may only be fitted on split_name='train'"
            )
        condition_array, error_array, mask_array, _, time_mask = _validate_core_arrays(
            conditions, errors, valid_mask, lengths
        )
        names = tuple(str(name) for name in feature_names)
        grid = tuple(float(value) for value in np.asarray(s_grid_m, dtype=np.float64))
        if len(names) != condition_array.shape[2]:
            raise ValueError("feature_names do not match the conditions dimension")
        if len(grid) != error_array.shape[2]:
            raise ValueError("s_grid_m does not match the error dimension")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")

        active_conditions = condition_array[time_mask]
        condition_mean = np.mean(active_conditions, axis=0)
        raw_condition_scale = np.std(active_conditions, axis=0, ddof=0)
        constant_conditions = np.flatnonzero(
            raw_condition_scale < self._minimum_scale
        )
        condition_scale = np.where(
            raw_condition_scale < self._minimum_scale, 1.0, raw_condition_scale
        )

        station_count = error_array.shape[2]
        error_mean = np.empty(station_count, dtype=np.float64)
        error_scale = np.empty(station_count, dtype=np.float64)
        error_count = np.empty(station_count, dtype=np.int64)
        constant_error_stations: list[int] = []
        for station_index in range(station_count):
            station_values = error_array[:, :, station_index][
                mask_array[:, :, station_index]
            ]
            if len(station_values) < 2:
                raise ValueError(
                    f"station {station_index} has fewer than two valid training errors"
                )
            error_count[station_index] = len(station_values)
            error_mean[station_index] = np.mean(station_values)
            raw_scale = float(np.std(station_values, ddof=0))
            if raw_scale < self._minimum_scale:
                constant_error_stations.append(station_index)
                error_scale[station_index] = 1.0
            else:
                error_scale[station_index] = raw_scale

        state = StandardizationState(
            schema_version=STANDARDIZATION_SCHEMA_VERSION,
            fitted_split="train",
            feature_names=names,
            s_grid_m=grid,
            condition_mean=tuple(condition_mean.tolist()),
            condition_scale=tuple(condition_scale.tolist()),
            condition_count=int(len(active_conditions)),
            error_mean=tuple(error_mean.tolist()),
            error_scale=tuple(error_scale.tolist()),
            error_count=tuple(int(value) for value in error_count),
            constant_condition_indices=tuple(int(value) for value in constant_conditions),
            constant_error_station_indices=tuple(constant_error_stations),
            minimum_scale=self._minimum_scale,
        )
        state.validate()
        self._state = state
        return self

    def assert_compatible(
        self,
        feature_names: Sequence[str],
        s_grid_m: ArrayLike,
    ) -> None:
        """Reject silent feature reordering or a changed look-ahead grid."""

        state = self.state
        if tuple(feature_names) != state.feature_names:
            raise ValueError("feature names/order differ from the fitted standardizer")
        grid = np.asarray(s_grid_m, dtype=np.float64)
        if grid.shape != (state.n_stations,) or not np.allclose(
            grid, np.asarray(state.s_grid_m), rtol=0.0, atol=1e-10
        ):
            raise ValueError("s_grid_m differs from the fitted standardizer")

    def transform_conditions(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
    ) -> NDArray[np.float32]:
        """Transform active frames and keep padding exactly zero."""

        state = self.state
        array = np.asarray(conditions, dtype=np.float64)
        length_array = np.asarray(lengths, dtype=np.int64)
        if array.ndim != 3 or array.shape[2] != state.n_features:
            raise ValueError("conditions are incompatible with fitted statistics")
        if length_array.shape != (array.shape[0],):
            raise ValueError("lengths must have shape [B]")
        if np.any(length_array <= 0) or np.any(length_array > array.shape[1]):
            raise ValueError("lengths contain invalid values")
        time_mask = np.arange(array.shape[1])[None, :] < length_array[:, None]
        if not np.all(np.isfinite(array[time_mask])):
            raise ValueError("active conditions contain non-finite values")
        transformed = np.zeros_like(array, dtype=np.float64)
        transformed[time_mask] = (
            array[time_mask] - np.asarray(state.condition_mean)
        ) / np.asarray(state.condition_scale)
        return transformed.astype(np.float32)

    def transform_errors(
        self,
        errors: ArrayLike,
        valid_mask: ArrayLike,
    ) -> NDArray[np.float32]:
        """Transform only observed errors and keep invalid entries zero."""

        state = self.state
        array = np.asarray(errors, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=np.bool_)
        if array.ndim != 3 or array.shape[2] != state.n_stations:
            raise ValueError("errors are incompatible with fitted statistics")
        if mask.shape != array.shape:
            raise ValueError("valid_mask must have the same shape as errors")
        if not np.all(np.isfinite(array[mask])):
            raise ValueError("valid errors contain non-finite values")
        transformed = np.zeros_like(array, dtype=np.float64)
        centered = array - np.asarray(state.error_mean)[None, None, :]
        scaled = centered / np.asarray(state.error_scale)[None, None, :]
        transformed[mask] = scaled[mask]
        return transformed.astype(np.float32)

    def inverse_transform_errors(
        self,
        standardized_errors: ArrayLike,
        valid_mask: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        """Return errors in metres for data or generated samples.

        Arbitrary leading dimensions are accepted, for example ``[S,B,T,K]``
        for multiple generated samples. When a mask is provided, invalid entries
        are reset to zero after inversion.
        """

        state = self.state
        array = np.asarray(standardized_errors, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != state.n_stations:
            raise ValueError("last dimension must match the fitted station count")
        restored = (
            array * np.asarray(state.error_scale)
            + np.asarray(state.error_mean)
        )
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=np.bool_)
            try:
                broadcast_mask = np.broadcast_to(mask, restored.shape)
            except ValueError as error:
                raise ValueError("valid_mask cannot be broadcast to error shape") from error
            restored = np.where(broadcast_mask, restored, 0.0)
        return restored.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        """Atomically persist the fitted transform as readable JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.stem}-",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(asdict(self.state), temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SequenceStandardizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("standardization JSON root must be an object")
        state = StandardizationState.from_dict(raw)
        return cls(state=state, minimum_scale=state.minimum_scale)

