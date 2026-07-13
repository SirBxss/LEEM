from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.preprocessing import (
    SequenceDataset,
    SequenceStandardizer,
    iter_sequence_batches,
)
from lane_error_modeling.data.synthetic.generator import generate_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES

from test_config import minimal_config_dict
from lane_error_modeling.data.synthetic.config import SyntheticDatasetConfig


def _raw_dataset() -> SequenceDataset:
    config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
    padded = generate_dataset(config, "conditional_gaussian", "train")
    return SequenceDataset.from_arrays(
        sequence_ids=padded.sequence_ids,
        conditions=padded.conditions,
        errors=padded.errors,
        valid_mask=padded.valid_mask,
        lengths=padded.lengths,
        feature_names=FEATURE_NAMES,
        s_grid_m=padded.s_grid_m,
    )


class StandardizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = _raw_dataset()
        self.standardizer = SequenceStandardizer().fit(
            self.raw.conditions,
            self.raw.errors,
            self.raw.valid_mask,
            self.raw.lengths,
            split_name="train",
            feature_names=self.raw.feature_names,
            s_grid_m=self.raw.s_grid_m,
        )

    def test_active_conditions_and_valid_errors_are_standardized(self) -> None:
        normalized = self.raw.standardized_copy(self.standardizer)
        active_conditions = normalized.conditions[normalized.time_mask]
        np.testing.assert_allclose(np.mean(active_conditions, axis=0), 0.0, atol=2e-6)
        np.testing.assert_allclose(np.std(active_conditions, axis=0), 1.0, atol=2e-6)
        for station_index in range(normalized.n_stations):
            values = normalized.errors[:, :, station_index][
                normalized.valid_mask[:, :, station_index]
            ]
            self.assertAlmostEqual(float(np.mean(values)), 0.0, places=5)
            self.assertAlmostEqual(float(np.std(values)), 1.0, places=5)

    def test_padding_and_invalid_targets_remain_zero(self) -> None:
        normalized = self.raw.standardized_copy(self.standardizer)
        self.assertTrue(np.all(normalized.conditions[~normalized.time_mask] == 0.0))
        self.assertTrue(np.all(normalized.errors[~normalized.valid_mask] == 0.0))

    def test_error_round_trip_recovers_physical_metres(self) -> None:
        normalized = self.raw.standardized_copy(self.standardizer)
        recovered = self.standardizer.inverse_transform_errors(
            normalized.errors, normalized.valid_mask
        )
        np.testing.assert_allclose(
            recovered[self.raw.valid_mask],
            self.raw.errors[self.raw.valid_mask],
            rtol=2e-6,
            atol=2e-7,
        )
        self.assertTrue(np.all(recovered[~self.raw.valid_mask] == 0.0))

    def test_non_training_fit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only be fitted"):
            SequenceStandardizer().fit(
                self.raw.conditions,
                self.raw.errors,
                self.raw.valid_mask,
                self.raw.lengths,
                split_name="validation",
                feature_names=self.raw.feature_names,
                s_grid_m=self.raw.s_grid_m,
            )

    def test_saved_state_round_trip_and_feature_order_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "standardization.json"
            self.standardizer.save(path)
            loaded = SequenceStandardizer.load(path)
            self.assertEqual(loaded.state, self.standardizer.state)
            with self.assertRaisesRegex(ValueError, "feature names/order"):
                loaded.assert_compatible(
                    tuple(reversed(self.raw.feature_names)), self.raw.s_grid_m
                )


class BatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _raw_dataset()

    def test_batches_preserve_complete_sequence_identity(self) -> None:
        batches = list(
            iter_sequence_batches(
                self.dataset, batch_size=2, shuffle=True, seed=99
            )
        )
        visited = np.concatenate([batch.sequence_indices for batch in batches])
        self.assertEqual(set(visited.tolist()), set(range(self.dataset.n_sequences)))
        self.assertEqual(len(visited), self.dataset.n_sequences)
        for batch in batches:
            self.assertEqual(batch.conditions.shape[1], int(np.max(batch.lengths)))
            for local_index, source_index in enumerate(batch.sequence_indices):
                length = int(batch.lengths[local_index])
                np.testing.assert_array_equal(
                    batch.conditions[local_index, :length],
                    self.dataset.conditions[source_index, :length],
                )

    def test_shuffle_is_reproducible_and_requires_seed(self) -> None:
        first = np.concatenate(
            [
                batch.sequence_indices
                for batch in iter_sequence_batches(
                    self.dataset, batch_size=2, shuffle=True, seed=123
                )
            ]
        )
        repeated = np.concatenate(
            [
                batch.sequence_indices
                for batch in iter_sequence_batches(
                    self.dataset, batch_size=2, shuffle=True, seed=123
                )
            ]
        )
        np.testing.assert_array_equal(first, repeated)
        with self.assertRaisesRegex(ValueError, "seed is required"):
            list(
                iter_sequence_batches(
                    self.dataset, batch_size=2, shuffle=True
                )
            )

    def test_drop_last_drops_only_incomplete_sequence_batch(self) -> None:
        kept = list(
            iter_sequence_batches(
                self.dataset,
                batch_size=2,
                shuffle=False,
                drop_last=True,
            )
        )
        self.assertEqual(sum(len(batch.lengths) for batch in kept), 2)

