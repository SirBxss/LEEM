from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.synthetic.cli import generate_from_config
from lane_error_modeling.data.synthetic.io import load_dataset


class IoTest(unittest.TestCase):
    def test_cli_generation_writes_valid_manifest_and_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = temporary_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "test",
                        "master_seed": 7,
                        "sample_rate_hz": 10.0,
                        "min_sequence_frames": 6,
                        "max_sequence_frames": 8,
                        "s_grid_m": [0.0, 5.0, 10.0],
                        "splits": {"train": 2, "validation": 1, "test": 1},
                        "scenarios": ["conditional_gaussian"],
                    }
                ),
                encoding="utf-8",
            )
            output = temporary_path / "generated"
            manifest_path = generate_from_config(config_path, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 3)
            self.assertTrue(all(len(record["sha256"]) == 64 for record in manifest["files"]))
            self.assertIn("absolute_q99", manifest["files"][0]["error_summary_m"])

            dataset = load_dataset(output / "conditional_gaussian" / "train.npz")
            self.assertEqual(dataset.conditions.shape[0], 2)
            self.assertTrue(np.isfinite(dataset.errors).all())
