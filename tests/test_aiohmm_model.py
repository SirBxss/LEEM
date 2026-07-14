from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset
from lane_error_modeling.models import (
    AIOHMMConfig,
    AutoregressiveInputOutputHMM,
)


def _aiohmm_dataset(
    seed: int,
    *,
    sequence_count: int = 18,
    max_length: int = 55,
) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    station_count = 2
    lengths = rng.integers(
        max_length - 8, max_length + 1, size=sequence_count, dtype=np.int32
    )
    conditions = np.zeros((sequence_count, max_length, 1), dtype=np.float32)
    errors = np.zeros(
        (sequence_count, max_length, station_count), dtype=np.float32
    )
    valid_mask = np.zeros_like(errors, dtype=np.bool_)
    covariances = np.array(
        [
            [[0.10, 0.035], [0.035, 0.16]],
            [[0.35, 0.12], [0.12, 0.45]],
        ],
        dtype=np.float64,
    )
    cholesky = np.linalg.cholesky(covariances)
    intercepts = np.array([[-0.25, 0.10], [0.35, -0.15]])
    slopes = np.array([[[0.18, -0.12]], [[-0.10, 0.20]]])
    autoregression = np.array([[0.70, 0.55], [0.88, 0.78]])
    for sequence_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        active_conditions = rng.normal(size=length)
        conditions[sequence_index, :length, 0] = active_conditions
        state = int(rng.random() > 0.55)
        previous = np.zeros(station_count, dtype=np.float64)
        for time_index, condition in enumerate(active_conditions):
            if time_index > 0:
                switch_probability = 1.0 / (
                    1.0 + np.exp(-(0.9 * condition - 1.5))
                )
                if rng.random() < switch_probability:
                    state = 1 - state
            mean = (
                intercepts[state]
                + condition * slopes[state, 0]
                + autoregression[state] * previous
            )
            value = mean + rng.normal(size=station_count) @ cholesky[state].T
            active_mask = rng.random(station_count) > np.array([0.03, 0.12])
            if time_index % 19 == 0:
                active_mask[1] = False
            valid_mask[sequence_index, time_index] = active_mask
            errors[sequence_index, time_index] = np.where(
                active_mask, value, 0.0
            )
            previous = value
    return SequenceDataset.from_arrays(
        sequence_ids=[f"aiohmm-{seed}-{index}" for index in range(sequence_count)],
        conditions=conditions,
        errors=errors,
        valid_mask=valid_mask,
        lengths=lengths,
        feature_names=("condition",),
        s_grid_m=np.array([0.0, 5.0], dtype=np.float32),
        standardized=True,
    )


def _config(**overrides: object) -> AIOHMMConfig:
    values: dict[str, object] = {
        "n_states": 2,
        "max_em_iterations": 8,
        "min_em_iterations": 3,
        "convergence_tolerance": 1e-5,
        "ridge_penalty": 1e-3,
        "covariance_shrinkage": 0.05,
        "minimum_eigenvalue": 1e-5,
        "minimum_effective_station_observations": 5.0,
        "minimum_effective_pair_observations": 5.0,
        "maximum_absolute_autoregression": 0.95,
        "transition_l2_penalty": 1e-3,
        "transition_learning_rate": 0.03,
        "transition_adam_steps": 15,
        "initial_probability_smoothing": 0.01,
        "minimum_state_occupancy_fraction": 0.005,
        "initialization_seed": 123,
        "input_dependent_transitions": True,
    }
    values.update(overrides)
    return AIOHMMConfig.from_dict(values)


class AIOHMMModelTest(unittest.TestCase):
    def test_configuration_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            AIOHMMConfig(n_states=1).validate()
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            AIOHMMConfig(maximum_absolute_autoregression=1.0).validate()
        with self.assertRaisesRegex(ValueError, "boolean"):
            AIOHMMConfig(input_dependent_transitions=1).validate()  # type: ignore[arg-type]

    def test_fit_improves_likelihood_and_handles_missing_targets(self) -> None:
        train = _aiohmm_dataset(10)
        validation = _aiohmm_dataset(11, sequence_count=5)
        model = AutoregressiveInputOutputHMM(_config())
        report = model.fit(train, validation)

        self.assertTrue(model.is_fitted)
        self.assertTrue(model.capabilities.supports_log_probability)
        self.assertGreater(
            model.log_likelihood_history[-1], model.log_likelihood_history[0]
        )
        self.assertGreater(np.min(model.state_occupancies), 0.005)
        self.assertLessEqual(
            np.max(np.abs(model.autoregressive_coefficients)), 0.95
        )
        self.assertIn(
            "validation_nll_per_observed_value_standardized", report.metrics
        )
        self.assertTrue(np.all(np.isfinite(model.log_probability(validation))))
        for posterior, length in zip(
            model.posterior_state_probabilities(validation), validation.lengths
        ):
            self.assertEqual(posterior.shape, (int(length), 2))
            np.testing.assert_allclose(np.sum(posterior, axis=1), 1.0, atol=1e-8)

    def test_sampling_is_deterministic_and_padding_is_zero(self) -> None:
        train = _aiohmm_dataset(20)
        model = AutoregressiveInputOutputHMM(
            _config(max_em_iterations=5, min_em_iterations=2)
        )
        model.fit(train)
        first = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=90,
            valid_mask=train.valid_mask,
        )
        second = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=90,
            valid_mask=train.valid_mask,
        )
        different = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=91,
            valid_mask=train.valid_mask,
        )
        np.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(np.array_equal(first.values, different.values))
        self.assertTrue(np.all(first.values[:, ~train.time_mask, :] == 0.0))
        first.validate()

    def test_persistence_preserves_density_and_samples(self) -> None:
        train = _aiohmm_dataset(30)
        validation = _aiohmm_dataset(31, sequence_count=4)
        model = AutoregressiveInputOutputHMM(
            _config(max_em_iterations=5, min_em_iterations=2)
        )
        model.fit(train)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = model.save(Path(temporary_directory) / "aiohmm.npz")
            restored = AutoregressiveInputOutputHMM.load(path)

        np.testing.assert_array_equal(
            model.autoregressive_coefficients,
            restored.autoregressive_coefficients,
        )
        np.testing.assert_array_equal(model.covariances, restored.covariances)
        np.testing.assert_allclose(
            model.log_probability(validation),
            restored.log_probability(validation),
            rtol=0.0,
            atol=0.0,
        )
        original_samples = model.sample(
            validation.conditions, validation.lengths, n_samples=3, seed=8
        )
        restored_samples = restored.sample(
            validation.conditions, validation.lengths, n_samples=3, seed=8
        )
        np.testing.assert_array_equal(
            original_samples.values, restored_samples.values
        )


if __name__ == "__main__":
    unittest.main()
