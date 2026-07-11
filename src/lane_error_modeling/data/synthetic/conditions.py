"""Generate temporally coherent road, ego-motion, and visibility conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lane_error_modeling.data.synthetic.config import ConditionRanges
from lane_error_modeling.data.synthetic.geometry import integrate_curvature


@dataclass(frozen=True)
class ConditionSequence:
    """Scene conditions and reference geometry for one sequence."""

    features: NDArray[np.float64]
    reference_curvature: NDArray[np.float64]
    reference_heading: NDArray[np.float64]
    reference_xy: NDArray[np.float64]


def _bounded_quality_process(
    rng: np.random.Generator,
    length: int,
    onset_probability: float,
    recovery_probability: float,
) -> NDArray[np.float64]:
    quality = np.empty(length, dtype=np.float64)
    quality[0] = rng.uniform(0.75, 0.98)
    degraded = False
    for time_index in range(1, length):
        if degraded:
            degraded = rng.random() >= recovery_probability
        else:
            degraded = rng.random() < onset_probability
        target = rng.uniform(0.15, 0.50) if degraded else rng.uniform(0.78, 0.98)
        quality[time_index] = np.clip(
            0.92 * quality[time_index - 1]
            + 0.08 * target
            + rng.normal(0.0, 0.018),
            0.02,
            1.0,
        )
    return quality


def generate_condition_sequence(
    rng: np.random.Generator,
    length: int,
    s_grid_m: NDArray[np.float64],
    ranges: ConditionRanges,
    sample_rate_hz: float,
) -> ConditionSequence:
    """Generate the six fixed conditions and their underlying reference paths."""

    if length < 2:
        raise ValueError("length must be at least 2")
    s_grid = np.asarray(s_grid_m, dtype=np.float64)
    if s_grid.ndim != 1:
        raise ValueError("s_grid_m must be a vector")

    speed = np.empty(length, dtype=np.float64)
    acceleration = np.empty(length, dtype=np.float64)
    speed[0] = rng.uniform(ranges.speed_min_mps, ranges.speed_max_mps)
    acceleration[0] = rng.normal(0.0, 0.35)
    dt_s = 1.0 / sample_rate_hz
    for time_index in range(1, length):
        acceleration[time_index] = np.clip(
            0.94 * acceleration[time_index - 1] + rng.normal(0.0, 0.12),
            -2.5,
            2.0,
        )
        speed[time_index] = np.clip(
            speed[time_index - 1] + acceleration[time_index] * dt_s,
            ranges.speed_min_mps,
            ranges.speed_max_mps,
        )

    center_curvature = np.empty(length, dtype=np.float64)
    curvature_gradient = np.empty(length, dtype=np.float64)
    center_curvature[0] = rng.uniform(-0.008, 0.008)
    curvature_gradient[0] = rng.uniform(-0.00005, 0.00005)
    for time_index in range(1, length):
        center_curvature[time_index] = np.clip(
            0.992 * center_curvature[time_index - 1]
            + rng.normal(0.0, 0.00045),
            -ranges.curvature_abs_max_inv_m,
            ranges.curvature_abs_max_inv_m,
        )
        curvature_gradient[time_index] = np.clip(
            0.985 * curvature_gradient[time_index - 1]
            + rng.normal(0.0, 0.000008),
            -ranges.curvature_gradient_abs_max_inv_m2,
            ranges.curvature_gradient_abs_max_inv_m2,
        )

    centered_s = s_grid - np.mean(s_grid)
    reference_curvature = np.clip(
        center_curvature[:, None] + curvature_gradient[:, None] * centered_s[None, :],
        -ranges.curvature_abs_max_inv_m,
        ranges.curvature_abs_max_inv_m,
    )

    lane_width = np.empty(length, dtype=np.float64)
    lane_width[0] = rng.uniform(3.2, 3.9)
    for time_index in range(1, length):
        lane_width[time_index] = np.clip(
            0.985 * lane_width[time_index - 1]
            + 0.015 * rng.uniform(3.3, 3.8)
            + rng.normal(0.0, 0.008),
            ranges.lane_width_min_m,
            ranges.lane_width_max_m,
        )

    marking_quality = _bounded_quality_process(
        rng, length, onset_probability=0.009, recovery_probability=0.035
    )
    environment_quality = _bounded_quality_process(
        rng, length, onset_probability=0.006, recovery_probability=0.025
    )

    mean_curvature = np.mean(reference_curvature, axis=1)
    curvature_range = np.ptp(reference_curvature, axis=1)
    features = np.column_stack(
        (
            speed,
            mean_curvature,
            curvature_range,
            lane_width,
            marking_quality,
            environment_quality,
        )
    )

    reference_heading = np.empty_like(reference_curvature)
    reference_xy = np.empty((length, len(s_grid), 2), dtype=np.float64)
    for time_index in range(length):
        xy, heading = integrate_curvature(s_grid, reference_curvature[time_index])
        reference_xy[time_index] = xy
        reference_heading[time_index] = heading

    return ConditionSequence(
        features=features,
        reference_curvature=reference_curvature,
        reference_heading=reference_heading,
        reference_xy=reference_xy,
    )

