from __future__ import annotations

import numpy as np
from pathlib import Path
import unittest

from lane_error_modeling.data.synthetic.config import SyntheticDatasetConfig
from lane_error_modeling.data.synthetic.generator import generate_dataset, generate_sequence

from test_config import minimal_config_dict


class GenerationTest(unittest.TestCase):
    def test_sequence_generation_is_bitwise_reproducible(self) -> None:
        config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
        first = generate_sequence(config, "latent_autoregressive", "train", 0)
        second = generate_sequence(config, "latent_autoregressive", "train", 0)
        self.assertEqual(first.sequence_seed, second.sequence_seed)
        np.testing.assert_array_equal(first.conditions, second.conditions)
        np.testing.assert_array_equal(first.errors, second.errors)
        np.testing.assert_array_equal(first.latent_state, second.latent_state)

    def test_splits_have_distinct_random_streams(self) -> None:
        config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
        train = generate_sequence(config, "conditional_gaussian", "train", 0)
        test = generate_sequence(config, "conditional_gaussian", "test", 0)
        self.assertNotEqual(train.sequence_seed, test.sequence_seed)
        self.assertFalse(np.array_equal(train.conditions, test.conditions))

    def test_padded_dataset_preserves_lengths_and_masks(self) -> None:
        config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
        dataset = generate_dataset(config, "nonlinear_heavy_tailed", "train")
        dataset.validate()
        self.assertEqual(dataset.conditions.shape[0], config.splits.train)
        self.assertEqual(dataset.conditions.shape[2], 6)
        self.assertEqual(dataset.errors.shape[2], config.n_stations)
        for sequence_index, length in enumerate(dataset.lengths):
            self.assertFalse(dataset.valid_mask[sequence_index, int(length) :].any())
            self.assertTrue(np.all(dataset.errors[sequence_index, int(length) :] == 0.0))

    def test_smoke_training_has_observations_at_every_station(self) -> None:
        config = SyntheticDatasetConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "synthetic_smoke.json"
        )
        dataset = generate_dataset(config, "conditional_gaussian", "train")
        station_counts = np.sum(dataset.valid_mask, axis=(0, 1))
        self.assertTrue(np.all(station_counts >= 2), station_counts)