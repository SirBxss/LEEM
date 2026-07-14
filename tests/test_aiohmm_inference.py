from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from lane_error_modeling.models.aiohmm.inference import (
    forward_backward,
    transition_log_probabilities,
)


class AIOHMMInferenceTest(unittest.TestCase):
    def test_forward_backward_matches_exact_state_enumeration(self) -> None:
        initial = np.array([0.65, 0.35], dtype=np.float64)
        transitions = np.array(
            [
                [[0.80, 0.20], [0.25, 0.75]],
                [[0.55, 0.45], [0.10, 0.90]],
                [[0.70, 0.30], [0.40, 0.60]],
            ],
            dtype=np.float64,
        )
        emissions = np.array(
            [
                [0.30, 0.75],
                [0.60, 0.20],
                [0.45, 0.80],
                [0.70, 0.35],
            ],
            dtype=np.float64,
        )
        actual = forward_backward(
            np.log(initial), np.log(transitions), np.log(emissions)
        )

        paths = list(product(range(2), repeat=4))
        weights = np.empty(len(paths), dtype=np.float64)
        for path_index, path in enumerate(paths):
            weight = initial[path[0]] * emissions[0, path[0]]
            for time_index in range(1, 4):
                weight *= transitions[time_index - 1, path[time_index - 1], path[time_index]]
                weight *= emissions[time_index, path[time_index]]
            weights[path_index] = weight
        normalizer = float(np.sum(weights))
        expected_gamma = np.zeros((4, 2), dtype=np.float64)
        expected_xi = np.zeros((3, 2, 2), dtype=np.float64)
        for path, weight in zip(paths, weights / normalizer):
            for time_index, state in enumerate(path):
                expected_gamma[time_index, state] += weight
            for time_index in range(3):
                expected_xi[time_index, path[time_index], path[time_index + 1]] += weight

        self.assertAlmostEqual(actual.log_probability, np.log(normalizer), places=13)
        np.testing.assert_allclose(
            actual.state_probabilities, expected_gamma, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            actual.transition_probabilities, expected_xi, rtol=1e-12, atol=1e-12
        )

    def test_single_frame_sequence_has_no_transition_posterior(self) -> None:
        result = forward_backward(
            np.log(np.array([0.4, 0.6])),
            np.empty((0, 2, 2), dtype=np.float64),
            np.log(np.array([[0.2, 0.8]])),
        )
        self.assertEqual(result.transition_probabilities.shape, (0, 2, 2))
        np.testing.assert_allclose(
            result.state_probabilities[0], np.array([1.0 / 7.0, 6.0 / 7.0])
        )

    def test_transition_probabilities_use_current_condition(self) -> None:
        conditions = np.array([[-2.0], [-1.0], [1.0]], dtype=np.float64)
        weights = np.zeros((2, 2, 2), dtype=np.float64)
        weights[:, 1, 1] = 2.0

        dependent = np.exp(
            transition_log_probabilities(
                conditions, weights, input_dependent=True
            )
        )
        independent = np.exp(
            transition_log_probabilities(
                conditions, weights, input_dependent=False
            )
        )

        self.assertLess(dependent[0, 0, 1], 0.5)
        self.assertGreater(dependent[1, 0, 1], 0.5)
        np.testing.assert_allclose(independent, 0.5)


if __name__ == "__main__":
    unittest.main()
