"""Common sample-based probabilistic metrics evaluated in physical units."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lane_error_modeling.data.preprocessing import SequenceDataset

from .config import EvaluationConfig
from .reference import EvaluationReference
from .result import EvaluationResult


def _optional_station_values(
    sums: NDArray[np.float64], counts: NDArray[np.int64]
) -> list[float | None]:
    return [
        float(total / count) if count > 0 else None
        for total, count in zip(sums, counts)
    ]


def _root_mean_square_station_values(
    squared_sums: NDArray[np.float64], counts: NDArray[np.int64]
) -> list[float | None]:
    return [
        float(np.sqrt(total / count)) if count > 0 else None
        for total, count in zip(squared_sums, counts)
    ]


def _jensen_shannon_distance(
    observed_values: NDArray[np.float64],
    generated_values: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> float:
    observed_clipped = np.clip(observed_values, edges[0], edges[-1])
    generated_clipped = np.clip(generated_values, edges[0], edges[-1])
    observed_counts = np.histogram(observed_clipped, bins=edges)[0].astype(np.float64)
    generated_counts = np.histogram(generated_clipped, bins=edges)[0].astype(np.float64)
    observed_probability = observed_counts / np.sum(observed_counts)
    generated_probability = generated_counts / np.sum(generated_counts)
    midpoint = 0.5 * (observed_probability + generated_probability)

    def divergence(probability: NDArray[np.float64]) -> float:
        positive = probability > 0.0
        return float(
            np.sum(
                probability[positive]
                * np.log2(probability[positive] / midpoint[positive])
            )
        )

    return float(
        np.sqrt(
            0.5 * divergence(observed_probability)
            + 0.5 * divergence(generated_probability)
        )
    )


def _sample_marginal_values(
    observations: NDArray[np.float64],
    samples: NDArray[np.float64],
    valid_positions: NDArray[np.int64],
    maximum_count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed_count = min(len(valid_positions), maximum_count)
    if observed_count == len(valid_positions):
        selected_observed = valid_positions
    else:
        selected_observed = rng.choice(
            valid_positions, size=observed_count, replace=False
        )
    observed_values = observations[selected_observed]

    sample_count = samples.shape[0]
    total_generated = sample_count * len(valid_positions)
    generated_count = min(total_generated, maximum_count)
    if total_generated <= maximum_count:
        generated_values = samples[:, valid_positions].reshape(-1)
    else:
        selected = rng.choice(total_generated, size=generated_count, replace=False)
        sample_indices = selected // len(valid_positions)
        position_indices = valid_positions[selected % len(valid_positions)]
        generated_values = samples[sample_indices, position_indices]
    return observed_values.astype(np.float64), generated_values.astype(np.float64)


def _sample_first_difference_values(
    dataset: SequenceDataset,
    samples: NDArray[np.float64],
    maximum_count: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sequence_count, max_length, station_count = dataset.errors.shape
    pair_mask = dataset.valid_mask[:, 1:, :] & dataset.valid_mask[:, :-1, :]
    pair_positions = np.flatnonzero(pair_mask.reshape(-1)).astype(np.int64)
    if len(pair_positions) == 0:
        raise ValueError("evaluation data contain no valid first differences")
    observed_count = min(len(pair_positions), maximum_count)
    selected_observed = (
        pair_positions
        if observed_count == len(pair_positions)
        else rng.choice(pair_positions, size=observed_count, replace=False)
    )

    def endpoints(positions: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        station = positions % station_count
        frame_position = positions // station_count
        time_index = frame_position % (max_length - 1)
        sequence_index = frame_position // (max_length - 1)
        previous = (
            sequence_index * max_length * station_count
            + time_index * station_count
            + station
        )
        following = previous + station_count
        return previous.astype(np.int64), following.astype(np.int64)

    observed_flat = dataset.errors.astype(np.float64).reshape(-1)
    previous, following = endpoints(selected_observed)
    observed_values = observed_flat[following] - observed_flat[previous]

    sample_count = samples.shape[0]
    total_generated = sample_count * len(pair_positions)
    generated_count = min(total_generated, maximum_count)
    if total_generated <= maximum_count:
        selected = np.arange(total_generated, dtype=np.int64)
    else:
        selected = rng.choice(total_generated, size=generated_count, replace=False)
    sample_indices = selected // len(pair_positions)
    selected_pairs = pair_positions[selected % len(pair_positions)]
    previous, following = endpoints(selected_pairs)
    generated_values = samples[sample_indices, following] - samples[
        sample_indices, previous
    ]
    return observed_values, generated_values


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float | None:
    if len(first) < 2:
        return None
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = float(
        np.sqrt(np.sum(centered_first**2) * np.sum(centered_second**2))
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.sum(centered_first * centered_second) / denominator)


def _pairwise_correlations(
    values: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    station_count = values.shape[1]
    correlations = np.full((station_count, station_count), np.nan, dtype=np.float64)
    for first in range(station_count):
        for second in range(first, station_count):
            jointly_valid = valid_mask[:, first] & valid_mask[:, second]
            value = _correlation(
                values[jointly_valid, first], values[jointly_valid, second]
            )
            if value is not None:
                correlations[first, second] = correlations[second, first] = value
    return correlations


def _matrix_with_none(matrix: NDArray[np.float64]) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def _lag_one_residual_correlations(
    observed_residuals: NDArray[np.float64],
    generated_samples: NDArray[np.float64],
    predictive_mean: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    sample_indices: NDArray[np.int64],
) -> tuple[list[float | None], list[float | None], float]:
    station_count = valid_mask.shape[2]
    observed_values: list[float | None] = []
    generated_values: list[float | None] = []
    absolute_differences: list[float] = []
    for station_index in range(station_count):
        pairs = (
            valid_mask[:, 1:, station_index]
            & valid_mask[:, :-1, station_index]
        )
        observed = _correlation(
            observed_residuals[:, :-1, station_index][pairs],
            observed_residuals[:, 1:, station_index][pairs],
        )
        per_sample: list[float] = []
        for sample_index in sample_indices:
            residual = (
                generated_samples[sample_index, :, :, station_index]
                - predictive_mean[:, :, station_index]
            )
            correlation = _correlation(residual[:, :-1][pairs], residual[:, 1:][pairs])
            if correlation is not None:
                per_sample.append(correlation)
        generated = float(np.mean(per_sample)) if per_sample else None
        observed_values.append(observed)
        generated_values.append(generated)
        if observed is not None and generated is not None:
            absolute_differences.append(abs(observed - generated))
    if not absolute_differences:
        raise ValueError("no station has sufficient temporal pairs for correlation")
    return observed_values, generated_values, float(np.mean(absolute_differences))


def _energy_scores(
    dataset: SequenceDataset,
    samples: NDArray[np.float64],
    config: EvaluationConfig,
    rng: np.random.Generator,
) -> tuple[float, float, int, int]:
    sample_count = samples.shape[0]
    frame_mask = np.any(dataset.valid_mask, axis=2)
    frame_positions = np.flatnonzero(frame_mask.reshape(-1)).astype(np.int64)
    selected_frame_count = min(len(frame_positions), config.max_energy_frames)
    if selected_frame_count < len(frame_positions):
        frame_positions = rng.choice(
            frame_positions, size=selected_frame_count, replace=False
        )
    energy_sample_count = min(
        sample_count,
        max(2, int(np.sqrt(config.energy_pair_count))),
    )
    if energy_sample_count == sample_count:
        selected_samples = np.arange(sample_count, dtype=np.int64)
    else:
        selected_samples = np.sort(
            rng.choice(sample_count, size=energy_sample_count, replace=False)
        )

    observed_frames = dataset.errors.astype(np.float64).reshape(
        -1, dataset.n_stations
    )
    frame_validity = dataset.valid_mask.reshape(-1, dataset.n_stations)
    generated_frames = samples.reshape(
        sample_count, -1, dataset.n_stations
    )
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    for frame_position in frame_positions:
        observed_indices = frame_validity[frame_position]
        dimension = int(np.count_nonzero(observed_indices))
        observation = observed_frames[frame_position, observed_indices]
        draws = generated_frames[
            selected_samples, frame_position, :
        ][:, observed_indices]
        first_term = float(np.mean(np.linalg.norm(draws - observation, axis=1)))
        pairwise = draws[:, None, :] - draws[None, :, :]
        second_term = 0.5 * float(
            np.mean(np.linalg.norm(pairwise, axis=2))
        )
        score = first_term - second_term
        raw_scores.append(score)
        normalized_scores.append(score / np.sqrt(dimension))
    return (
        float(np.mean(raw_scores)),
        float(np.mean(normalized_scores)),
        selected_frame_count,
        energy_sample_count,
    )


def _point_interval_and_crps_metrics(
    dataset: SequenceDataset,
    samples: NDArray[np.float64],
    config: EvaluationConfig,
) -> tuple[
    dict[str, float],
    dict[str, list[float | int | None]],
    dict[str, dict[str, object]],
    NDArray[np.float64],
]:
    sample_count = samples.shape[0]
    station_count = dataset.n_stations
    observed_flat = dataset.errors.astype(np.float64).reshape(-1)
    sample_flat = samples.reshape(sample_count, -1)
    valid_positions = np.flatnonzero(dataset.valid_mask.reshape(-1)).astype(np.int64)
    station_indices = valid_positions % station_count
    predictive_mean = np.mean(samples, axis=0, dtype=np.float64)
    prediction_error = predictive_mean.reshape(-1)[valid_positions] - observed_flat[
        valid_positions
    ]
    station_counts = np.bincount(
        station_indices, minlength=station_count
    ).astype(np.int64)
    absolute_sums = np.bincount(
        station_indices,
        weights=np.abs(prediction_error),
        minlength=station_count,
    )
    squared_sums = np.bincount(
        station_indices,
        weights=prediction_error**2,
        minlength=station_count,
    )
    crps_sums = np.zeros(station_count, dtype=np.float64)
    interval_coverage_sums = {
        level: np.zeros(station_count, dtype=np.float64)
        for level in config.interval_levels
    }
    interval_width_sums = {
        level: np.zeros(station_count, dtype=np.float64)
        for level in config.interval_levels
    }
    order_weights = (
        2.0 * np.arange(1, sample_count + 1, dtype=np.float64)
        - sample_count
        - 1.0
    )

    for start in range(0, len(valid_positions), config.crps_chunk_size):
        positions = valid_positions[start : start + config.crps_chunk_size]
        stations = positions % station_count
        observations = observed_flat[positions]
        draws = sample_flat[:, positions].astype(np.float64)
        first_term = np.mean(np.abs(draws - observations[None, :]), axis=0)
        sorted_draws = np.sort(draws, axis=0)
        second_term = np.sum(
            order_weights[:, None] * sorted_draws, axis=0
        ) / (sample_count**2)
        crps = first_term - second_term
        crps_sums += np.bincount(
            stations, weights=crps, minlength=station_count
        )

        for level in config.interval_levels:
            lower_probability = 0.5 * (1.0 - level)
            lower, upper = np.quantile(
                draws,
                (lower_probability, 1.0 - lower_probability),
                axis=0,
                method="linear",
            )
            covered = (observations >= lower) & (observations <= upper)
            interval_coverage_sums[level] += np.bincount(
                stations, weights=covered.astype(np.float64), minlength=station_count
            )
            interval_width_sums[level] += np.bincount(
                stations, weights=upper - lower, minlength=station_count
            )

    global_metrics = {
        "predictive_mean_mae_m": float(np.mean(np.abs(prediction_error))),
        "predictive_mean_rmse_m": float(np.sqrt(np.mean(prediction_error**2))),
        "crps_m": float(np.sum(crps_sums) / np.sum(station_counts)),
    }
    station_metrics: dict[str, list[float | int | None]] = {
        "observation_count": [int(value) for value in station_counts],
        "predictive_mean_mae_m": _optional_station_values(
            absolute_sums, station_counts
        ),
        "predictive_mean_rmse_m": _root_mean_square_station_values(
            squared_sums, station_counts
        ),
        "crps_m": _optional_station_values(crps_sums, station_counts),
    }
    interval_metrics: dict[str, dict[str, object]] = {}
    total_count = int(np.sum(station_counts))
    for level in config.interval_levels:
        key = f"{level:.2f}"
        interval_metrics[key] = {
            "nominal_coverage": level,
            "global_empirical_coverage": float(
                np.sum(interval_coverage_sums[level]) / total_count
            ),
            "global_mean_width_m": float(
                np.sum(interval_width_sums[level]) / total_count
            ),
            "station_empirical_coverage": _optional_station_values(
                interval_coverage_sums[level], station_counts
            ),
            "station_mean_width_m": _optional_station_values(
                interval_width_sums[level], station_counts
            ),
        }
    return global_metrics, station_metrics, interval_metrics, predictive_mean


def evaluate_probabilistic_samples(
    *,
    model_name: str,
    scenario: str,
    dataset: SequenceDataset,
    physical_samples: ArrayLike,
    reference: EvaluationReference,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate model draws against physical-metre observations on a frozen split."""

    dataset.validate()
    if dataset.standardized:
        raise ValueError("evaluation requires physical-unit observations")
    reference.assert_compatible(dataset)
    config.validate()
    samples = np.asarray(physical_samples)
    if not np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float64)
    expected_shape = (
        dataset.n_sequences,
        dataset.max_length,
        dataset.n_stations,
    )
    if samples.ndim != 4 or samples.shape[1:] != expected_shape:
        raise ValueError("physical_samples must have shape [S, B, T, K]")
    if samples.shape[0] < 2:
        raise ValueError("at least two generated samples are required")
    if not np.all(np.isfinite(samples[:, dataset.time_mask, :])):
        raise ValueError("active generated samples contain non-finite values")
    if np.any(samples[:, ~dataset.time_mask, :] != 0.0):
        raise ValueError("physical generated padding must remain zero")
    if not np.any(dataset.valid_mask):
        raise ValueError("evaluation dataset contains no observed targets")

    rng = np.random.default_rng(config.metric_seed)
    global_metrics, station_metrics, interval_metrics, predictive_mean = (
        _point_interval_and_crps_metrics(dataset, samples, config)
    )
    observed_flat = dataset.errors.astype(np.float64).reshape(-1)
    sample_flat = samples.reshape(samples.shape[0], -1)
    valid_positions = np.flatnonzero(dataset.valid_mask.reshape(-1)).astype(np.int64)
    observed_marginal, generated_marginal = _sample_marginal_values(
        observed_flat,
        sample_flat,
        valid_positions,
        config.max_distribution_values,
        rng,
    )
    observed_differences, generated_differences = _sample_first_difference_values(
        dataset,
        sample_flat,
        config.max_distribution_values,
        rng,
    )
    global_metrics["error_jensen_shannon_distance"] = _jensen_shannon_distance(
        observed_marginal,
        generated_marginal,
        np.asarray(reference.error_histogram_edges_m),
    )
    global_metrics[
        "first_difference_jensen_shannon_distance"
    ] = _jensen_shannon_distance(
        observed_differences,
        generated_differences,
        np.asarray(reference.first_difference_histogram_edges_m),
    )

    for label, threshold in reference.absolute_error_thresholds_m.items():
        observed_rate = float(np.mean(np.abs(observed_marginal) > threshold))
        generated_rate = float(np.mean(np.abs(generated_marginal) > threshold))
        global_metrics[f"observed_abs_exceedance_{label}"] = observed_rate
        global_metrics[f"generated_abs_exceedance_{label}"] = generated_rate
        global_metrics[f"abs_exceedance_error_{label}"] = abs(
            observed_rate - generated_rate
        )
    for probability in config.tail_probabilities:
        label = f"q{int(round(probability * 100)):02d}"
        observed_quantile = float(np.quantile(np.abs(observed_marginal), probability))
        generated_quantile = float(np.quantile(np.abs(generated_marginal), probability))
        global_metrics[f"observed_abs_{label}_m"] = observed_quantile
        global_metrics[f"generated_abs_{label}_m"] = generated_quantile
        global_metrics[f"abs_quantile_error_{label}_m"] = abs(
            observed_quantile - generated_quantile
        )

    (
        energy_score,
        normalized_energy_score,
        energy_frame_count,
        energy_sample_count,
    ) = _energy_scores(dataset, samples, config, rng)
    global_metrics["energy_score_m"] = energy_score
    global_metrics["dimension_normalized_energy_score_m"] = normalized_energy_score

    frame_mask = np.any(dataset.valid_mask, axis=2)
    frame_positions = np.flatnonzero(frame_mask.reshape(-1)).astype(np.int64)
    dependence_frame_count = min(
        len(frame_positions), config.max_dependence_frames
    )
    if dependence_frame_count < len(frame_positions):
        frame_positions = rng.choice(
            frame_positions, size=dependence_frame_count, replace=False
        )
    dependence_sample_count = min(
        samples.shape[0], config.max_dependence_samples
    )
    if dependence_sample_count == samples.shape[0]:
        sample_indices = np.arange(samples.shape[0], dtype=np.int64)
    else:
        sample_indices = np.sort(
            rng.choice(
                samples.shape[0], size=dependence_sample_count, replace=False
            )
        )

    observed_residuals = dataset.errors.astype(np.float64) - predictive_mean
    observed_frames = observed_residuals.reshape(-1, dataset.n_stations)[
        frame_positions
    ]
    selected_validity = dataset.valid_mask.reshape(-1, dataset.n_stations)[
        frame_positions
    ]
    generated_frames = (
        samples.reshape(samples.shape[0], -1, dataset.n_stations)[
            sample_indices
        ][:, frame_positions, :]
        - predictive_mean.reshape(-1, dataset.n_stations)[frame_positions][
            None, :, :
        ]
    ).reshape(-1, dataset.n_stations)
    generated_validity = np.tile(selected_validity, (dependence_sample_count, 1))
    observed_spatial = _pairwise_correlations(observed_frames, selected_validity)
    generated_spatial = _pairwise_correlations(
        generated_frames, generated_validity
    )
    upper_triangle = np.triu(
        np.ones_like(observed_spatial, dtype=np.bool_), k=1
    )
    comparable = (
        upper_triangle
        & np.isfinite(observed_spatial)
        & np.isfinite(generated_spatial)
    )
    if not np.any(comparable):
        raise ValueError("no spatial station pairs have sufficient observations")
    spatial_rmse = float(
        np.sqrt(
            np.mean(
                (observed_spatial[comparable] - generated_spatial[comparable]) ** 2
            )
        )
    )
    global_metrics["residual_spatial_correlation_rmse"] = spatial_rmse

    observed_temporal, generated_temporal, temporal_mae = (
        _lag_one_residual_correlations(
            observed_residuals,
            samples,
            predictive_mean,
            dataset.valid_mask,
            sample_indices,
        )
    )
    global_metrics["residual_lag_one_correlation_mae"] = temporal_mae
    dependence_diagnostics: dict[str, object] = {
        "observed_residual_spatial_correlation": _matrix_with_none(
            observed_spatial
        ),
        "generated_residual_spatial_correlation": _matrix_with_none(
            generated_spatial
        ),
        "observed_residual_lag_one_correlation_by_station": observed_temporal,
        "generated_residual_lag_one_correlation_by_station": generated_temporal,
    }

    result = EvaluationResult(
        model_name=model_name,
        scenario=scenario,
        sample_count=int(samples.shape[0]),
        sequence_count=dataset.n_sequences,
        observed_value_count=int(len(valid_positions)),
        s_grid_m=tuple(float(value) for value in dataset.s_grid_m),
        global_metrics=global_metrics,
        station_metrics=station_metrics,
        interval_metrics=interval_metrics,
        dependence_diagnostics=dependence_diagnostics,
        approximation_metadata={
            "metric_seed": config.metric_seed,
            "marginal_observed_value_count": int(len(observed_marginal)),
            "marginal_generated_value_count": int(len(generated_marginal)),
            "first_difference_observed_value_count": int(
                len(observed_differences)
            ),
            "first_difference_generated_value_count": int(
                len(generated_differences)
            ),
            "energy_frame_count": energy_frame_count,
            "energy_sample_count": energy_sample_count,
            "dependence_frame_count": dependence_frame_count,
            "dependence_sample_count": dependence_sample_count,
            "jensen_shannon_logarithm_base": "2",
        },
    )
    result.validate()
    return result
