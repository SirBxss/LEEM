"""Model-independent sequence datasets and deterministic complete-sequence batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .standardization import SequenceStandardizer, _validate_core_arrays


@dataclass(frozen=True)
class SequenceDataset:
    """Minimal model-facing representation shared by all three model families."""

    sequence_ids: NDArray[np.str_]
    conditions: NDArray[np.float32]
    errors: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    lengths: NDArray[np.int32]
    feature_names: tuple[str, ...]
    s_grid_m: NDArray[np.float32]
    standardized: bool = False

    @classmethod
    def from_arrays(
        cls,
        *,
        sequence_ids: ArrayLike,
        conditions: ArrayLike,
        errors: ArrayLike,
        valid_mask: ArrayLike,
        lengths: ArrayLike,
        feature_names: Sequence[str],
        s_grid_m: ArrayLike,
        standardized: bool = False,
    ) -> "SequenceDataset":
        dataset = cls(
            sequence_ids=np.asarray(sequence_ids, dtype=np.str_),
            conditions=np.asarray(conditions, dtype=np.float32),
            errors=np.asarray(errors, dtype=np.float32),
            valid_mask=np.asarray(valid_mask, dtype=np.bool_),
            lengths=np.asarray(lengths, dtype=np.int32),
            feature_names=tuple(str(name) for name in feature_names),
            s_grid_m=np.asarray(s_grid_m, dtype=np.float32),
            standardized=bool(standardized),
        )
        dataset.validate()
        return dataset

    @property
    def n_sequences(self) -> int:
        return len(self.lengths)

    @property
    def max_length(self) -> int:
        return self.conditions.shape[1]

    @property
    def n_features(self) -> int:
        return self.conditions.shape[2]

    @property
    def n_stations(self) -> int:
        return self.errors.shape[2]

    @property
    def time_mask(self) -> NDArray[np.bool_]:
        return np.arange(self.max_length)[None, :] < self.lengths[:, None]

    def validate(self) -> None:
        _, _, _, _, time_mask = _validate_core_arrays(
            self.conditions, self.errors, self.valid_mask, self.lengths
        )
        if self.sequence_ids.shape != (self.conditions.shape[0],):
            raise ValueError("sequence_ids must have shape [B]")
        if len(set(self.sequence_ids.tolist())) != len(self.sequence_ids):
            raise ValueError("sequence_ids must be unique")
        if len(self.feature_names) != self.n_features:
            raise ValueError("feature_names do not match conditions")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if self.s_grid_m.shape != (self.n_stations,):
            raise ValueError("s_grid_m does not match errors")
        if not np.all(np.diff(self.s_grid_m) > 0):
            raise ValueError("s_grid_m must be strictly increasing")
        if np.any(self.conditions[~time_mask] != 0.0):
            raise ValueError("condition padding must be zero")
        if np.any(self.errors[~self.valid_mask] != 0.0):
            raise ValueError("invalid and padded errors must be zero")

    def standardized_copy(
        self,
        standardizer: SequenceStandardizer,
    ) -> "SequenceDataset":
        """Apply an already fitted training transform without mutating raw data."""

        if self.standardized:
            raise ValueError("dataset is already standardized")
        standardizer.assert_compatible(self.feature_names, self.s_grid_m)
        return SequenceDataset.from_arrays(
            sequence_ids=self.sequence_ids,
            conditions=standardizer.transform_conditions(
                self.conditions, self.lengths
            ),
            errors=standardizer.transform_errors(self.errors, self.valid_mask),
            valid_mask=self.valid_mask,
            lengths=self.lengths,
            feature_names=self.feature_names,
            s_grid_m=self.s_grid_m,
            standardized=True,
        )


@dataclass(frozen=True)
class SequenceBatch:
    """A time-trimmed batch containing complete, unmixed sequences."""

    sequence_indices: NDArray[np.int64]
    sequence_ids: NDArray[np.str_]
    conditions: NDArray[np.float32]
    errors: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    lengths: NDArray[np.int32]
    s_grid_m: NDArray[np.float32]
    standardized: bool

    @property
    def time_mask(self) -> NDArray[np.bool_]:
        return np.arange(self.conditions.shape[1])[None, :] < self.lengths[:, None]

    def validate(self) -> None:
        _validate_core_arrays(
            self.conditions, self.errors, self.valid_mask, self.lengths
        )
        if self.sequence_indices.shape != self.lengths.shape:
            raise ValueError("sequence_indices must have shape [B]")
        if self.sequence_ids.shape != self.lengths.shape:
            raise ValueError("sequence_ids must have shape [B]")
        if self.s_grid_m.shape != (self.errors.shape[2],):
            raise ValueError("s_grid_m does not match errors")


def iter_sequence_batches(
    dataset: SequenceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[SequenceBatch]:
    """Yield deterministic sequence-level batches with no frame-level leakage.

    A seed is mandatory whenever shuffling is requested. The caller should derive
    a distinct deterministic seed per training epoch. Every batch is trimmed to
    its longest sequence, reducing padding without changing sequence contents.
    """

    dataset.validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle and seed is None:
        raise ValueError("a seed is required for reproducible shuffling")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative")

    indices = np.arange(dataset.n_sequences, dtype=np.int64)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    for start in range(0, dataset.n_sequences, batch_size):
        selected = indices[start : start + batch_size]
        if len(selected) < batch_size and drop_last:
            break
        batch_max_length = int(np.max(dataset.lengths[selected]))
        batch = SequenceBatch(
            sequence_indices=selected,
            sequence_ids=dataset.sequence_ids[selected],
            conditions=dataset.conditions[selected, :batch_max_length],
            errors=dataset.errors[selected, :batch_max_length],
            valid_mask=dataset.valid_mask[selected, :batch_max_length],
            lengths=dataset.lengths[selected],
            s_grid_m=dataset.s_grid_m,
            standardized=dataset.standardized,
        )
        batch.validate()
        yield batch

