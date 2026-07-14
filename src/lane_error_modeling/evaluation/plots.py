"""Deterministic diagnostic plots for thesis experiments."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray

from lane_error_modeling.data.preprocessing import SequenceDataset

from .reference import EvaluationReference
from .result import EvaluationResult


def _finite_matrix(values: object) -> NDArray[np.float64]:
    rows = values if isinstance(values, list) else []
    return np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in rows
        ],
        dtype=np.float64,
    )


def create_evaluation_plots(
    *,
    output_directory: str | Path,
    dataset: SequenceDataset,
    physical_samples: NDArray[np.float32] | NDArray[np.float64],
    reference: EvaluationReference,
    result: EvaluationResult,
    seed: int,
) -> tuple[Path, ...]:
    """Write a compact deterministic set of PNG diagnostics."""

    try:
        matplotlib_cache = Path(tempfile.gettempdir()) / "leem-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot creation requires the optional 'evaluation' dependencies"
        ) from error

    result.validate()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(physical_samples)
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("physical_samples must use a floating dtype")
    predictive_mean = np.mean(samples, axis=0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []

    station_path = destination / "station_metrics.png"
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    grid = np.asarray(result.s_grid_m)
    for metric, label in (
        ("predictive_mean_rmse_m", "RMSE"),
        ("crps_m", "CRPS"),
    ):
        values = np.asarray(
            [
                np.nan if value is None else value
                for value in result.station_metrics[metric]
            ],
            dtype=np.float64,
        )
        axes[0].plot(grid, values, marker="o", label=label)
    axes[0].set_ylabel("Error metric [m]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for level, details in result.interval_metrics.items():
        values = np.asarray(
            [
                np.nan if value is None else value
                for value in details["station_empirical_coverage"]
            ],
            dtype=np.float64,
        )
        axes[1].plot(grid, values, marker="o", label=f"Empirical {level}")
        axes[1].axhline(
            float(details["nominal_coverage"]),
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
    axes[1].set_xlabel("Look-ahead station [m]")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(station_path, dpi=180)
    plt.close(figure)
    paths.append(station_path)

    distribution_path = destination / "marginal_error_distribution.png"
    valid_positions = np.flatnonzero(dataset.valid_mask.reshape(-1))
    maximum_values = 200_000
    observed_positions = (
        valid_positions
        if len(valid_positions) <= maximum_values
        else rng.choice(valid_positions, size=maximum_values, replace=False)
    )
    observed = dataset.errors.astype(np.float64).reshape(-1)[observed_positions]
    sample_flat = samples.reshape(samples.shape[0], -1)
    total_generated = samples.shape[0] * len(valid_positions)
    generated_count = min(total_generated, maximum_values)
    selected = rng.choice(total_generated, size=generated_count, replace=False)
    generated = sample_flat[
        selected // len(valid_positions),
        valid_positions[selected % len(valid_positions)],
    ]
    edges = np.asarray(reference.error_histogram_edges_m)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.hist(
        np.clip(observed, edges[0], edges[-1]),
        bins=edges,
        density=True,
        alpha=0.55,
        label="Observed",
    )
    axis.hist(
        np.clip(generated, edges[0], edges[-1]),
        bins=edges,
        density=True,
        alpha=0.55,
        label="Generated",
    )
    axis.set_xlabel("Signed lateral error [m]")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=180)
    plt.close(figure)
    paths.append(distribution_path)

    correlation_path = destination / "residual_spatial_correlation.png"
    observed_correlation = _finite_matrix(
        result.dependence_diagnostics["observed_residual_spatial_correlation"]
    )
    generated_correlation = _finite_matrix(
        result.dependence_diagnostics["generated_residual_spatial_correlation"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    for axis, matrix, title in (
        (axes[0], observed_correlation, "Observed residual"),
        (axes[1], generated_correlation, "Generated residual"),
    ):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("Station index")
    axes[0].set_ylabel("Station index")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Correlation")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.88, wspace=0.18)
    figure.savefig(correlation_path, dpi=180)
    plt.close(figure)
    paths.append(correlation_path)

    example_path = destination / "example_temporal_sequence.png"
    sequence_index = int(np.argmax(dataset.lengths))
    station_index = int(np.argmin(np.abs(dataset.s_grid_m - 50.0)))
    length = int(dataset.lengths[sequence_index])
    time = np.arange(length)
    observed_mask = dataset.valid_mask[sequence_index, :length, station_index]
    draws = samples[:, sequence_index, :length, station_index]
    lower, upper = np.quantile(draws, (0.05, 0.95), axis=0)
    figure, axis = plt.subplots(figsize=(9.0, 4.5))
    axis.fill_between(time, lower, upper, alpha=0.25, label="90% interval")
    axis.plot(
        time,
        predictive_mean[sequence_index, :length, station_index],
        linewidth=1.5,
        label="Predictive mean",
    )
    axis.plot(
        time[observed_mask],
        dataset.errors[sequence_index, :length, station_index][observed_mask],
        linewidth=1.0,
        color="black",
        label="Observed",
    )
    axis.set_xlabel("Frame")
    axis.set_ylabel(f"Error at {dataset.s_grid_m[station_index]:g} m [m]")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(example_path, dpi=180)
    plt.close(figure)
    paths.append(example_path)

    profile_path = destination / "example_spatial_profile.png"
    valid_frames = np.flatnonzero(
        np.any(dataset.valid_mask[sequence_index, :length], axis=1)
    )
    frame_index = int(valid_frames[len(valid_frames) // 2])
    frame_mask = dataset.valid_mask[sequence_index, frame_index]
    profile_draws = samples[:, sequence_index, frame_index]
    lower, upper = np.quantile(profile_draws, (0.05, 0.95), axis=0)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.fill_between(grid, lower, upper, alpha=0.25, label="90% interval")
    axis.plot(
        grid,
        predictive_mean[sequence_index, frame_index],
        marker="o",
        label="Predictive mean",
    )
    axis.plot(
        grid[frame_mask],
        dataset.errors[sequence_index, frame_index, frame_mask],
        marker="o",
        color="black",
        label="Observed",
    )
    axis.set_xlabel("Look-ahead station [m]")
    axis.set_ylabel("Signed lateral error [m]")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(profile_path, dpi=180)
    plt.close(figure)
    paths.append(profile_path)
    return tuple(paths)
