from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lane_error_modeling.evaluation import (
    AIOHMMExperimentConfig,
    run_aiohmm_experiment,
)


class AIOHMMExperimentTest(unittest.TestCase):
    def test_configuration_candidate_counts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        smoke = AIOHMMExperimentConfig.load(
            project_root / "configs" / "aiohmm_experiment_smoke.json"
        )
        prototype = AIOHMMExperimentConfig.load(
            project_root / "configs" / "aiohmm_experiment_prototype.json"
        )
        self.assertEqual(len(smoke.aiohmm_search.candidates()), 4)
        self.assertEqual(len(prototype.aiohmm_search.candidates()), 6)

    def test_runner_selects_before_test_and_persists_model(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = {
            "schema_version": "1.0.0",
            "experiment_name": "unit_aiohmm_experiment",
            "dataset_root": "outputs/synthetic_smoke",
            "scenarios": ["latent_autoregressive"],
            "sample_count": 8,
            "sample_seed": 17,
            "evaluation": {
                "interval_levels": [0.5, 0.9],
                "histogram_bin_count": 20,
                "histogram_clip_quantiles": [0.01, 0.99],
                "tail_probabilities": [0.9, 0.95],
                "crps_chunk_size": 1024,
                "max_distribution_values": 5000,
                "max_energy_frames": 30,
                "energy_pair_count": 64,
                "max_dependence_frames": 100,
                "max_dependence_samples": 4,
                "metric_seed": 18,
            },
            "aiohmm_search": {
                "state_counts": [2],
                "initialization_seeds": [19],
                "max_em_iterations": 2,
                "min_em_iterations": 2,
                "convergence_tolerance": 0.001,
                "ridge_penalty": 0.001,
                "covariance_shrinkage": 0.1,
                "minimum_eigenvalue": 0.00001,
                "minimum_effective_station_observations": 5.0,
                "minimum_effective_pair_observations": 5.0,
                "maximum_absolute_autoregression": 0.98,
                "transition_l2_penalty": 0.001,
                "transition_learning_rate": 0.03,
                "transition_adam_steps": 2,
                "initial_probability_smoothing": 0.01,
                "minimum_state_occupancy_fraction": 0.005,
                "input_dependent_transitions": True,
            },
            "create_plots": False,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            manifest_path = run_aiohmm_experiment(
                project_root=project_root,
                config_path=config_path,
                output_root=output,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scenario_output = output / "latent_autoregressive"
            selection = json.loads(
                (scenario_output / "model_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            result = json.loads(
                (scenario_output / "scenario_result.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(manifest["status"], "passed")
            self.assertFalse(selection["test_data_accessed_during_selection"])
            self.assertEqual(selection["selection_split"], "validation")
            self.assertIn("state_diagnostics", result)
            self.assertTrue((scenario_output / "aiohmm_model.npz").is_file())


if __name__ == "__main__":
    unittest.main()
