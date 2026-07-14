from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset
from lane_error_modeling.models import ConditionalMultivariateGaussian, GaussianConfig


TRUE_COEFFICIENTS = np.array(
    [
        [0.20, -0.10, 0.05],
        [0.50, 0.10, -0.20],
        [-0.30, 0.40, 0.20],
    ],
    dtype=np.float64,
)
TRUE_COVARIANCE = np.array(
    [
        [0.40, 0.15, 0.05],
        [0.15, 0.30, -0.02],
        [0.05, -0.02, 0.20],
    ],
    dtype=np.float64,
)


def _gaussian_dataset(
    seed: int,
    *,
    missing: bool,
    sequence_count: int = 10,
    max_length: int = 140,
) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(
        max_length - 20, max_length + 1, size=sequence_count, dtype=np.int32
    )
    conditions = np.zeros((sequence_count, max_length, 2), dtype=np.float32)
    errors = np.zeros((sequence_count, max_length, 3), dtype=np.float32)
    valid_mask = np.zeros_like(errors, dtype=np.bool_)
    cholesky = np.linalg.cholesky(TRUE_COVARIANCE)
    for sequence_index, length in enumerate(lengths):
        length = int(length)
        active_conditions = rng.normal(size=(length, 2))
        design = np.column_stack((np.ones(length), active_conditions))
        active_errors = (
            design @ TRUE_COEFFICIENTS
            + rng.normal(size=(length, 3)) @ cholesky.T
        )
        active_mask = np.ones((length, 3), dtype=np.bool_)
        if missing:
            active_mask[:, 1] = rng.random(length) > 0.18
            active_mask[:, 2] = rng.random(length) > 0.35
            active_mask[::11, :] = False
        conditions[sequence_index, :length] = active_conditions
        valid_mask[sequence_index, :length] = active_mask
        errors[sequence_index, :length] = np.where(active_mask, active_errors, 0.0)
    return SequenceDataset.from_arrays(
        sequence_ids=[f"sequence-{index}" for index in range(sequence_count)],
        conditions=conditions,
        errors=errors,
        valid_mask=valid_mask,
        lengths=lengths,
        feature_names=("condition_a", "condition_b"),
        s_grid_m=np.array([0.0, 5.0, 10.0], dtype=np.float32),
        standardized=True,
    )


def _config(**overrides: object) -> GaussianConfig:
    values: dict[str, object] = {
        "ridge_penalty": 1e-8,
        "covariance_shrinkage": 0.0,
        "minimum_eigenvalue": 1e-8,
        "minimum_station_observations": 20,
        "minimum_pair_observations": 20,
    }
    values.update(overrides)
    return GaussianConfig.from_dict(values)


class ConditionalMultivariateGaussianTest(unittest.TestCase):
    def test_configuration_rejects_non_finite_and_non_integer_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite number"):
            GaussianConfig(ridge_penalty=float("nan")).validate()
        with self.assertRaisesRegex(ValueError, "finite number"):
            GaussianConfig(minimum_eigenvalue=float("inf")).validate()
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            GaussianConfig(minimum_pair_observations=2.5).validate()  # type: ignore[arg-type]

    def test_recovers_linear_mean_and_spatial_covariance(self) -> None:
        train = _gaussian_dataset(10, missing=False)
        validation = _gaussian_dataset(11, missing=False, sequence_count=4)
        model = ConditionalMultivariateGaussian(_config())
        report = model.fit(train, validation)

        self.assertTrue(model.is_fitted)
        self.assertTrue(model.capabilities.supports_log_probability)
        self.assertLess(np.max(np.abs(model.coefficients - TRUE_COEFFICIENTS)), 0.06)
        self.assertLess(np.max(np.abs(model.covariance - TRUE_COVARIANCE)), 0.06)
        self.assertGreater(np.min(np.linalg.eigvalsh(model.covariance)), 0.0)
        self.assertIn("validation_nll_per_observed_value_standardized", report.metrics)
        self.assertEqual(report.warnings, ())

    def test_masked_likelihood_matches_direct_marginal_calculation(self) -> None:
        train = _gaussian_dataset(20, missing=True)
        evaluation = _gaussian_dataset(21, missing=True, sequence_count=3)
        model = ConditionalMultivariateGaussian(
            _config(covariance_shrinkage=0.15, minimum_eigenvalue=1e-6)
        )
        model.fit(train)
        actual = model.log_probability(evaluation)
        predicted_mean = model.predict_mean(
            evaluation.conditions, evaluation.lengths
        ).astype(np.float64)

        expected = np.zeros(evaluation.n_sequences, dtype=np.float64)
        for sequence_index, length in enumerate(evaluation.lengths):
            for time_index in range(int(length)):
                indices = np.flatnonzero(
                    evaluation.valid_mask[sequence_index, time_index]
                )
                if len(indices) == 0:
                    continue
                residual = (
                    evaluation.errors[sequence_index, time_index, indices]
                    - predicted_mean[sequence_index, time_index, indices]
                )
                covariance = model.covariance[np.ix_(indices, indices)]
                sign, log_determinant = np.linalg.slogdet(covariance)
                self.assertEqual(sign, 1.0)
                expected[sequence_index] += -0.5 * (
                    len(indices) * np.log(2.0 * np.pi)
                    + log_determinant
                    + residual @ np.linalg.solve(covariance, residual)
                )

        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.min(model.pair_observation_counts), 20)

    def test_sampling_is_seeded_and_padding_is_zero(self) -> None:
        train = _gaussian_dataset(30, missing=True)
        model = ConditionalMultivariateGaussian(
            _config(covariance_shrinkage=0.10, minimum_eigenvalue=1e-6)
        )
        model.fit(train)
        first = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=20260713,
            valid_mask=train.valid_mask,
        )
        second = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=20260713,
            valid_mask=train.valid_mask,
        )
        third = model.sample(
            train.conditions,
            train.lengths,
            n_samples=4,
            seed=20260714,
            valid_mask=train.valid_mask,
        )

        np.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(np.array_equal(first.values, third.values))
        time_mask = train.time_mask
        self.assertTrue(np.all(first.values[:, ~time_mask, :] == 0.0))
        unavailable_active = (~train.valid_mask) & time_mask[:, :, None]
        self.assertTrue(np.any(first.values[0][unavailable_active] != 0.0))
        first.validate()

    def test_persistence_round_trip_preserves_scores_and_samples(self) -> None:
        train = _gaussian_dataset(40, missing=True)
        evaluation = _gaussian_dataset(41, missing=True, sequence_count=3)
        model = ConditionalMultivariateGaussian(
            _config(covariance_shrinkage=0.20, minimum_eigenvalue=1e-6)
        )
        model.fit(train)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = model.save(Path(temporary_directory) / "gaussian-model.npz")
            restored = ConditionalMultivariateGaussian.load(path)

        np.testing.assert_array_equal(model.coefficients, restored.coefficients)
        np.testing.assert_array_equal(model.covariance, restored.covariance)
        np.testing.assert_allclose(
            model.log_probability(evaluation),
            restored.log_probability(evaluation),
            rtol=0.0,
            atol=0.0,
        )
        original_samples = model.sample(
            evaluation.conditions,
            evaluation.lengths,
            n_samples=2,
            seed=91,
        )
        restored_samples = restored.sample(
            evaluation.conditions,
            evaluation.lengths,
            n_samples=2,
            seed=91,
        )
        np.testing.assert_array_equal(original_samples.values, restored_samples.values)

    def test_rejects_station_with_insufficient_training_observations(self) -> None:
        dataset = _gaussian_dataset(50, missing=False, sequence_count=2, max_length=30)
        mask = dataset.valid_mask.copy()
        errors = dataset.errors.copy()
        mask[:, :, 2] = False
        mask[0, :3, 2] = True
        errors[:, :, 2] = 0.0
        errors[0, :3, 2] = dataset.errors[0, :3, 2]
        sparse = SequenceDataset.from_arrays(
            sequence_ids=dataset.sequence_ids,
            conditions=dataset.conditions,
            errors=errors,
            valid_mask=mask,
            lengths=dataset.lengths,
            feature_names=dataset.feature_names,
            s_grid_m=dataset.s_grid_m,
            standardized=True,
        )
        model = ConditionalMultivariateGaussian(
            _config(minimum_station_observations=10)
        )
        with self.assertRaisesRegex(ValueError, "station 2 has 3 observations"):
            model.fit(sparse)


if __name__ == "__main__":
    unittest.main()
