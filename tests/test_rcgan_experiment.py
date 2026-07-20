from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lane_error_modeling.evaluation import RCGANExperimentConfig


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from lane_error_modeling.evaluation import run_rcgan_experiment


class RCGANExperimentConfigurationTest(unittest.TestCase):
    def test_smoke_pilot_and_prototype_candidate_counts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        smoke = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_smoke.json"
        )
        prototype = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_prototype.json"
        )
        pilot = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_pilot.json"
        )
        pilot_v2 = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_pilot_v2.json"
        )
        self.assertEqual(len(smoke.rcgan_search.candidates()), 1)
        self.assertEqual(len(pilot.rcgan_search.candidates()), 3)
        self.assertEqual(len(pilot_v2.rcgan_search.candidates()), 3)
        self.assertEqual(len(prototype.rcgan_search.candidates()), 2)
        self.assertEqual(
            [candidate.learning_rate for candidate in pilot.rcgan_search.candidates()],
            [1e-5, 1e-4, 3e-4],
        )
        self.assertEqual(pilot.scenarios, ("conditional_gaussian",))
        self.assertEqual(pilot.minimum_validation_diversity_ratio, 0.05)
        self.assertEqual(
            [
                candidate.learning_rate
                for candidate in pilot_v2.rcgan_search.candidates()
            ],
            [1e-5, 3e-5, 5e-5],
        )
        self.assertEqual(pilot_v2.rcgan_search.batch_size, 1)
        self.assertEqual(pilot_v2.minimum_validation_diversity_ratio, 0.10)
        self.assertEqual(
            pilot_v2.stability_checks.maximum_generator_clipped_fraction,
            0.50,
        )
        self.assertEqual(
            pilot_v2.stability_checks.minimum_tail_exceedance_fraction_of_observed,
            0.20,
        )
        self.assertEqual(prototype.rcgan_search.candidates()[0].latent_size, 32)
        self.assertEqual(prototype.rcgan_search.candidates()[0].context_layers, 2)

    def test_invalid_diversity_gate_is_rejected(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_pilot.json"
        ).to_dict()
        source["minimum_validation_diversity_ratio"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            RCGANExperimentConfig.from_dict(source)

    def test_invalid_stability_checks_are_rejected(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_pilot_v2.json"
        ).to_dict()
        source["stability_checks"][
            "maximum_generator_clipped_fraction"
        ] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            RCGANExperimentConfig.from_dict(source)

    def test_enabled_stability_level_must_be_evaluated(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_pilot_v2.json"
        ).to_dict()
        source["stability_checks"]["interval_level"] = 0.80
        with self.assertRaisesRegex(ValueError, "interval_level"):
            RCGANExperimentConfig.from_dict(source)


@unittest.skipUnless(TORCH_AVAILABLE, "RC-GAN tests require the optional PyTorch extra")
class RCGANExperimentRunnerTest(unittest.TestCase):
    def test_runner_selects_before_test_and_persists_safe_model(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_smoke.json"
        ).to_dict()
        source["experiment_name"] = "unit_rcgan_experiment"
        source["scenarios"] = ["latent_autoregressive"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            output = root / "output"
            manifest_path = run_rcgan_experiment(
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
            self.assertEqual(
                selection["selection_metric"],
                "dimension_normalized_energy_score_m",
            )
            self.assertFalse(selection["stability_gate"]["enabled"])
            self.assertTrue(selection["stability_gate"]["passed"])
            self.assertIn(
                "generated_to_observed_std_ratio",
                selection["candidates"][0]["fit_report"]["metrics"],
            )
            self.assertFalse(result["density_metrics"]["available"])
            self.assertEqual(result["architecture"]["noise_layers"], 1)
            self.assertTrue((scenario_output / "rcgan_model.npz").is_file())

    def test_all_rejected_candidates_persist_failure_without_opening_test(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = RCGANExperimentConfig.load(
            project_root / "configs" / "rcgan_experiment_smoke.json"
        ).to_dict()
        source["experiment_name"] = "unit_rcgan_stability_failure"
        source["scenarios"] = ["latent_autoregressive"]
        source["minimum_validation_diversity_ratio"] = 1.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            output = root / "output"
            manifest_path = run_rcgan_experiment(
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

            self.assertEqual(manifest["status"], "stability_failed")
            self.assertEqual(result["status"], "stability_failed")
            self.assertIsNone(selection["selected_candidate_index"])
            self.assertIsNone(selection["selected_config"])
            self.assertFalse(selection["test_data_accessed"])
            self.assertFalse(result["test_data_accessed"])
            self.assertNotIn("test", result["data_provenance"])
            self.assertEqual(selection["candidates"][0]["status"], "rejected")
            self.assertGreater(
                len(selection["candidates"][0]["training_history"]),
                0,
            )
            self.assertFalse((scenario_output / "rcgan_model.npz").exists())
            self.assertFalse((scenario_output / "evaluation.json").exists())
            self.assertTrue((scenario_output / "standardizer.json").is_file())
            self.assertTrue(
                (scenario_output / "evaluation_reference.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
