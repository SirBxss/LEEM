"""Reference-path construction and geometrically consistent error injection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def integrate_curvature(
    s_grid_m: FloatArray,
    curvature_inv_m: FloatArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a planar path from curvature sampled over arc length.

    The path begins at the ego origin with zero heading. Trapezoidal curvature
    integration yields heading, and midpoint heading integration yields position.
    """

    s = np.asarray(s_grid_m, dtype=np.float64)
    curvature = np.asarray(curvature_inv_m, dtype=np.float64)
    if s.ndim != 1 or curvature.shape != s.shape:
        raise ValueError("s_grid_m and curvature_inv_m must be equal-length vectors")
    if len(s) < 2 or not np.all(np.diff(s) > 0):
        raise ValueError("s_grid_m must contain at least two increasing values")

    delta_s = np.diff(s)
    heading = np.zeros_like(s)
    heading[1:] = np.cumsum(
        0.5 * (curvature[:-1] + curvature[1:]) * delta_s
    )

    midpoint_heading = 0.5 * (heading[:-1] + heading[1:])
    xy = np.zeros((len(s), 2), dtype=np.float64)
    xy[1:, 0] = np.cumsum(delta_s * np.cos(midpoint_heading))
    xy[1:, 1] = np.cumsum(delta_s * np.sin(midpoint_heading))
    return xy, heading


def reference_normals(reference_heading_rad: FloatArray) -> NDArray[np.float64]:
    """Return left-pointing unit normals for reference headings."""

    heading = np.asarray(reference_heading_rad, dtype=np.float64)
    if heading.ndim != 1:
        raise ValueError("reference_heading_rad must be a vector")
    return np.column_stack((-np.sin(heading), np.cos(heading)))


def perturb_path_along_normal(
    reference_xy_m: FloatArray,
    reference_heading_rad: FloatArray,
    lateral_error_m: FloatArray,
) -> NDArray[np.float64]:
    """Create an estimated path by shifting reference points along their normals."""

    xy = np.asarray(reference_xy_m, dtype=np.float64)
    error = np.asarray(lateral_error_m, dtype=np.float64)
    heading = np.asarray(reference_heading_rad, dtype=np.float64)
    if xy.shape != (len(error), 2) or heading.shape != error.shape:
        raise ValueError("incompatible path, heading, and error shapes")
    return xy + error[:, None] * reference_normals(heading)


def signed_normal_error(
    reference_xy_m: FloatArray,
    reference_heading_rad: FloatArray,
    estimated_xy_m: FloatArray,
) -> NDArray[np.float64]:
    """Measure estimated-minus-reference displacement along the reference normal."""

    reference = np.asarray(reference_xy_m, dtype=np.float64)
    estimated = np.asarray(estimated_xy_m, dtype=np.float64)
    heading = np.asarray(reference_heading_rad, dtype=np.float64)
    if reference.shape != estimated.shape or reference.shape != (len(heading), 2):
        raise ValueError("incompatible reference, estimate, and heading shapes")
    return np.sum((estimated - reference) * reference_normals(heading), axis=1)

