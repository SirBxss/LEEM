from __future__ import annotations

import numpy as np
import unittest

from lane_error_modeling.data.synthetic.geometry import (
    integrate_curvature,
    perturb_path_along_normal,
    signed_normal_error,
)


class GeometryTest(unittest.TestCase):
    def test_straight_path_follows_x_axis(self) -> None:
        s = np.arange(0.0, 25.0, 5.0)
        xy, heading = integrate_curvature(s, np.zeros_like(s))
        np.testing.assert_allclose(xy[:, 0], s, atol=1e-12)
        np.testing.assert_allclose(xy[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(heading, 0.0, atol=1e-12)

    def test_normal_perturbation_recovers_signed_error_on_curve(self) -> None:
        s = np.arange(0.0, 105.0, 5.0)
        curvature = np.full_like(s, 0.01)
        reference_xy, heading = integrate_curvature(s, curvature)
        requested_error = np.linspace(-0.2, 0.45, len(s))
        estimated_xy = perturb_path_along_normal(reference_xy, heading, requested_error)
        recovered = signed_normal_error(reference_xy, heading, estimated_xy)
        np.testing.assert_allclose(recovered, requested_error, atol=1e-12)
