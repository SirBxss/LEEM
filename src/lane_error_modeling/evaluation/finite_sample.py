"""Finite-ensemble reference calculations for predictive intervals."""

from __future__ import annotations

import math


def linear_quantile_uniform_reference_coverage(
    sample_count: int,
    nominal_coverage: float,
) -> float:
    """Return the finite-sample reference coverage for linear quantiles.

    NumPy's ``method="linear"`` interpolates between adjacent sample order
    statistics.  For draws from a continuous uniform distribution, the
    expected lower and upper endpoints have CDF positions ``(h + 1)/(n + 1)``,
    where ``h = (n - 1) * p``.  Their difference is a transparent diagnostic
    reference on probability scale, but it is not a calibration
    correction for an arbitrary non-uniform predictive distribution.
    """

    _validate_inputs(sample_count, nominal_coverage)
    lower_probability = 0.5 * (1.0 - nominal_coverage)
    upper_probability = 1.0 - lower_probability
    lower_position = (sample_count - 1) * lower_probability
    upper_position = (sample_count - 1) * upper_probability
    return float((upper_position - lower_position) / (sample_count + 1))


def central_order_statistic_reference(
    sample_count: int,
    nominal_coverage: float,
) -> dict[str, int | float | bool]:
    """Describe the closest conservative central order-statistic interval.

    For an independent observation and ``n`` exchangeable predictive draws,
    the interval ``[X_(lower), X_(upper)]`` has reference coverage
    ``(upper - lower)/(n + 1)`` for a continuous distribution.  Coverage is
    quantized in increments of ``1/(n + 1)`` and very high nominal levels may
    be unattainable with a small ensemble.
    """

    _validate_inputs(sample_count, nominal_coverage)
    maximum_rank_gap = sample_count - 1
    requested_rank_gap = int(math.ceil(nominal_coverage * (sample_count + 1)))
    rank_gap = min(requested_rank_gap, maximum_rank_gap)
    lower_rank = max(1, (sample_count - rank_gap + 1) // 2)
    upper_rank = lower_rank + rank_gap
    reference_coverage = rank_gap / (sample_count + 1)
    return {
        "lower_rank_1_based": lower_rank,
        "upper_rank_1_based": upper_rank,
        "reference_coverage": float(reference_coverage),
        "meets_nominal_coverage": bool(reference_coverage >= nominal_coverage),
    }


def finite_ensemble_interval_metadata(
    *,
    sample_count: int,
    nominal_coverage: float,
    empirical_coverage: float,
) -> dict[str, int | float | bool | str]:
    """Build auditable interval metadata for one nominal coverage level."""

    _validate_inputs(sample_count, nominal_coverage)
    if not 0.0 <= empirical_coverage <= 1.0:
        raise ValueError("empirical_coverage must lie in [0, 1]")
    linear_reference = linear_quantile_uniform_reference_coverage(
        sample_count, nominal_coverage
    )
    rank_reference = central_order_statistic_reference(
        sample_count, nominal_coverage
    )
    return {
        "ensemble_sample_count": sample_count,
        "quantile_method": "linear",
        "linear_uniform_reference_coverage": linear_reference,
        "empirical_minus_linear_uniform_reference": float(
            empirical_coverage - linear_reference
        ),
        "central_rank_lower_1_based": int(rank_reference["lower_rank_1_based"]),
        "central_rank_upper_1_based": int(rank_reference["upper_rank_1_based"]),
        "central_rank_reference_coverage": float(
            rank_reference["reference_coverage"]
        ),
        "central_rank_meets_nominal_coverage": bool(
            rank_reference["meets_nominal_coverage"]
        ),
        "interpretation": (
            "finite-ensemble diagnostic only; final calibration conclusions "
            "require a larger predictive ensemble or a separately specified "
            "rank-based interval"
        ),
    }


def _validate_inputs(sample_count: int, nominal_coverage: float) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie strictly inside (0, 1)")
