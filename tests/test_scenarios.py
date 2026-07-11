from __future__ import annotations

import numpy as np
import unittest

from lane_error_modeling.data.synthetic.conditions import generate_condition_sequence
from lane_error_modeling.data.synthetic.config import ConditionRanges
from lane_error_modeling.data.synthetic.scenarios import generate_errors


def _conditions(length: int = 2500):
    rng = np.random.default_rng(91)
    s_grid = np.arange(0.0, 105.0, 5.0)
    sequence = generate_condition_sequence(
        rng, length, s_grid, ConditionRanges(), sample_rate_hz=10.0
    )
    return sequence.features, s_grid


def _excess_kurtosis(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    variance = np.mean(centered**2)
    return float(np.mean(centered**4) / variance**2 - 3.0)


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conditions, cls.s_grid = _conditions()

    def test_gaussian_oracle_residual_has_little_lag_one_dependence(self) -> None:
        output = generate_errors(
            "conditional_gaussian",
            np.random.default_rng(123),
            self.conditions,
            self.s_grid,
        )
        residual = output.errors[:, -1] - output.conditional_mean[:, -1]
        lag_one = np.corrcoef(residual[:-1], residual[1:])[0, 1]
        self.assertLess(abs(lag_one), 0.08)

    def test_latent_process_visits_multiple_persistent_states(self) -> None:
        output = generate_errors(
            "latent_autoregressive",
            np.random.default_rng(456),
            self.conditions,
            self.s_grid,
        )
        self.assertEqual(len(np.unique(output.latent_state)), 3)
        persistence = np.mean(output.latent_state[1:] == output.latent_state[:-1])
        self.assertGreater(persistence, 0.70)

    def test_nonlinear_process_is_heavy_tailed_and_contains_bursts(self) -> None:
        output = generate_errors(
            "nonlinear_heavy_tailed",
            np.random.default_rng(789),
            self.conditions,
            self.s_grid,
        )
        residual = output.errors[:, -1] - output.conditional_mean[:, -1]
        self.assertGreater(_excess_kurtosis(residual), 0.5)
        self.assertTrue(np.any(output.latent_state == 1))
