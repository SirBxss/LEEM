from __future__ import annotations

import unittest

from lane_error_modeling.data.synthetic.config import SyntheticDatasetConfig


def minimal_config_dict() -> dict[str, object]:
    return {
        "schema_version": "test",
        "master_seed": 42,
        "sample_rate_hz": 10.0,
        "min_sequence_frames": 8,
        "max_sequence_frames": 12,
        "s_grid_m": [0.0, 5.0, 10.0],
        "splits": {"train": 3, "validation": 2, "test": 2},
        "scenarios": [
            "conditional_gaussian",
            "latent_autoregressive",
            "nonlinear_heavy_tailed",
        ],
    }


class ConfigTest(unittest.TestCase):
    def test_config_round_trip_and_dimensions(self) -> None:
        config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
        self.assertEqual(config.n_stations, 3)
        self.assertEqual(config.n_features, 6)
        self.assertEqual(config.to_dict()["master_seed"], 42)

    def test_config_rejects_non_increasing_grid(self) -> None:
        raw = minimal_config_dict()
        raw["s_grid_m"] = [0.0, 5.0, 5.0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SyntheticDatasetConfig.from_dict(raw)

    def test_config_rejects_unknown_scenario(self) -> None:
        raw = minimal_config_dict()
        raw["scenarios"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "unsupported"):
            SyntheticDatasetConfig.from_dict(raw)
