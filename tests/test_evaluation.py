from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset
from lane_error_modeling.evaluation import (
    EvaluationConfig,
    EvaluationReference,
    GaussianExperimentConfig,
    GaussianSearchSpace,
    evaluate_probabilistic_samples,
    run_gaussian_experiment,
)


def _physical_dataset(seed: int, *, sequence_count: int = 4) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    max_length = 18
    station_count = 3
    lengths = np.asarray(
        [max_length - (index % 3) * 3 for index in range(sequence_count)],
        dtype=np.int32,
    )
    conditions = np.zeros((sequence_count, max_length, 1), dtype=np.float32)
    errors = np.zeros((sequence_count, max_length, station_count), dtype=np.float32)
    valid_mask = np.zeros_like(errors, dtype=np.bool_)
    covariance = np.array(
        [[0.010, 0.005, 0.002], [0.005, 0.016, 0.006], [0.002, 0.006, 0.025]],
        dtype=np.float64,
    )
    cholesky = np.linalg.cholesky(covariance)
    for sequence_index, length in enumerate(lengths):
        length = int(length)
        condition = rng.normal(size=length)
        mean = condition[:, None] * np.array([0.02, -0.03, 0.05])[None, :]
        innovations = rng.normal(size=(length, station_count)) @ cholesky.T
        active_errors = mean + innovations
        active_mask = np.ones((length, station_count), dtype=np.bool_)
        active_mask[::7, 2] = False
        conditions[sequence_index, :length, 0] = condition
        valid_mask[sequence_index, :length] = active_mask
        errors[sequence_index, :length] = np.where(active_mask, active_errors, 0.0)
    return SequenceDataset.from_arrays(
        sequence_ids=[f"physical-{seed}-{index}" for index in range(sequence_count)],
        conditions=conditions,
        errors=errors,
        valid_mask=valid_mask,
        lengths=lengths,
        feature_names=("condition",),
        s_grid_m=np.array([0.0, 5.0, 10.0], dtype=np.float32),
        standardized=False,
    )


def _physical_samples(dataset: SequenceDataset, seed: int, count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.zeros((count, *dataset.errors.shape), dtype=np.float32)
    station_scale = np.array([0.09, 0.12, 0.16], dtype=np.float32)
    conditional_mean = (
        dataset.conditions[:, :, 0, None]
        * np.array([0.02, -0.03, 0.05], dtype=np.float32)[None, None, :]
    )
    samples[:, dataset.time_mask, :] = (
        conditional_mean[dataset.time_mask][None, :, :]
        + rng.normal(
            size=(count, int(np.count_nonzero(dataset.time_mask)), dataset.n_stations)
        )
        * station_scale[None, None, :]
    )
    return samples


def _evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        interval_levels=(0.5, 0.9),
        histogram_bin_count=20,
        histogram_clip_quantiles=(0.01, 0.99),
        tail_probabilities=(0.9, 0.95),
        crps_chunk_size=17,
        max_distribution_values=500,
        max_energy_frames=30,
        energy_pair_count=64,
        max_dependence_frames=50,
        max_dependence_samples=6,
        metric_seed=77,
    )


class EvaluationTest(unittest.TestCase):
    def test_reference_is_training_only_and_round_trips(self) -> None:
        train = _physical_dataset(1)
        config = _evaluation_config()
        reference = EvaluationReference.fit(train, config, split_name="train")
        self.assertEqual(reference.fitted_split, "train")
        self.assertEqual(len(reference.error_histogram_edges_m), 21)
        with self.assertRaisesRegex(ValueError, "only be fitted on train"):
            EvaluationReference.fit(train, config, split_name="validation")

        standardized = SequenceDataset.from_arrays(
            sequence_ids=train.sequence_ids,
            conditions=train.conditions,
            errors=train.errors,
            valid_mask=train.valid_mask,
            lengths=train.lengths,
            feature_names=train.feature_names,
            s_grid_m=train.s_grid_m,
            standardized=True,
        )
        with self.assertRaisesRegex(ValueError, "physical-unit"):
            EvaluationReference.fit(standardized, config, split_name="train")

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = reference.save(Path(temporary_directory) / "reference.json")
            restored = EvaluationReference.load(path)
        self.assertEqual(reference, restored)

    def test_common_metrics_match_direct_crps_and_are_deterministic(self) -> None:
        train = _physical_dataset(2)
        test = _physical_dataset(3, sequence_count=3)
        samples = _physical_samples(test, 4)
        config = _evaluation_config()
        reference = EvaluationReference.fit(train, config, split_name="train")
        first = evaluate_probabilistic_samples(
            model_name="test_model",
            scenario="test_scenario",
            dataset=test,
            physical_samples=samples,
            reference=reference,
            config=config,
        )
        second = evaluate_probabilistic_samples(
            model_name="test_model",
            scenario="test_scenario",
            dataset=test,
            physical_samples=samples,
            reference=reference,
            config=config,
        )
        self.assertEqual(first.to_dict(), second.to_dict())

        observations = test.errors[test.valid_mask].astype(np.float64)
        draws = samples[:, test.valid_mask].astype(np.float64)
        direct_crps = np.mean(
            np.mean(np.abs(draws - observations[None, :]), axis=0)
            - 0.5
            * np.mean(
                np.abs(draws[:, None, :] - draws[None, :, :]), axis=(0, 1)
            )
        )
        self.assertAlmostEqual(first.global_metrics["crps_m"], direct_crps, places=12)
        predictive_mean = np.mean(samples, axis=0)
        direct_rmse = np.sqrt(
            np.mean(
                (predictive_mean[test.valid_mask] - test.errors[test.valid_mask]) ** 2
            )
        )
        self.assertAlmostEqual(
            first.global_metrics["predictive_mean_rmse_m"], direct_rmse, places=7
        )
        self.assertGreaterEqual(
            first.global_metrics["error_jensen_shannon_distance"], 0.0
        )
        self.assertLessEqual(
            first.global_metrics["error_jensen_shannon_distance"], 1.0
        )
        finite_ensemble = first.interval_metrics["0.90"]["finite_ensemble"]
        self.assertEqual(finite_ensemble["ensemble_sample_count"], 12)
        self.assertEqual(finite_ensemble["quantile_method"], "linear")
        first.validate()

    def test_evaluation_rejects_nonzero_physical_padding(self) -> None:
        train = _physical_dataset(5)
        test = _physical_dataset(6)
        samples = _physical_samples(test, 7)
        samples[0, 1, -1, 0] = 1.0
        reference = EvaluationReference.fit(
            train, _evaluation_config(), split_name="train"
        )
        with self.assertRaisesRegex(ValueError, "padding"):
            evaluate_probabilistic_samples(
                model_name="test_model",
                scenario="test_scenario",
                dataset=test,
                physical_samples=samples,
                reference=reference,
                config=_evaluation_config(),
            )

    def test_experiment_configuration_and_candidate_order(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        smoke = GaussianExperimentConfig.load(
            project_root / "configs" / "gaussian_experiment_smoke.json"
        )
        prototype = GaussianExperimentConfig.load(
            project_root / "configs" / "gaussian_experiment_prototype.json"
        )
        self.assertEqual(len(smoke.gaussian_search.candidates()), 6)
        self.assertEqual(len(prototype.gaussian_search.candidates()), 20)

        search = GaussianSearchSpace(
            ridge_penalties=(0.0, 0.01),
            covariance_shrinkages=(0.0, 0.2),
        )
        candidates = search.candidates()
        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0].ridge_penalty, 0.0)
        self.assertEqual(candidates[-1].covariance_shrinkage, 0.2)
        with self.assertRaisesRegex(ValueError, "increasing"):
            GaussianSearchSpace(ridge_penalties=(0.1, 0.0)).validate()

    def test_smoke_experiment_writes_provenance_and_never_selects_on_test(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = {
            "schema_version": "1.0.0",
            "experiment_name": "unit_gaussian_experiment",
            "dataset_root": "outputs/synthetic_smoke",
            "scenarios": ["conditional_gaussian"],
            "sample_count": 8,
            "sample_seed": 12,
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
                "metric_seed": 13
            },
            "gaussian_search": {
                "ridge_penalties": [0.001],
                "covariance_shrinkages": [0.1],
                "minimum_eigenvalue": 0.000001,
                "minimum_station_observations": 32,
                "minimum_pair_observations": 32
            },
            "create_plots": False
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            manifest_path = run_gaussian_experiment(
                project_root=project_root,
                config_path=config_path,
                output_root=output,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selection = json.loads(
                (output / "conditional_gaussian" / "model_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "passed")
            self.assertFalse(selection["test_data_accessed_during_selection"])
            self.assertEqual(selection["selection_split"], "validation")
            self.assertTrue(
                (output / "conditional_gaussian" / "gaussian_model.npz").is_file()
            )


if __name__ == "__main__":
    unittest.main()
