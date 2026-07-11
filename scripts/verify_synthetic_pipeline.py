"""Dependency-light end-to-end verification for the synthetic data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from lane_error_modeling.data.synthetic.cli import generate_from_config
from lane_error_modeling.data.synthetic.conditions import generate_condition_sequence
from lane_error_modeling.data.synthetic.config import ConditionRanges, SyntheticDatasetConfig
from lane_error_modeling.data.synthetic.generator import generate_sequence
from lane_error_modeling.data.synthetic.geometry import (
    integrate_curvature,
    perturb_path_along_normal,
    signed_normal_error,
)
from lane_error_modeling.data.synthetic.io import load_dataset
from lane_error_modeling.data.synthetic.scenarios import generate_errors


def _lag_one(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _excess_kurtosis(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    variance = np.mean(centered**2)
    return float(np.mean(centered**4) / variance**2 - 3.0)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "synthetic_smoke.json"
    config = SyntheticDatasetConfig.from_json(config_path)

    s_grid = np.asarray(config.s_grid_m, dtype=np.float64)
    reference_xy, heading = integrate_curvature(s_grid, np.full_like(s_grid, 0.01))
    injected_error = np.linspace(-0.25, 0.40, len(s_grid))
    estimated_xy = perturb_path_along_normal(reference_xy, heading, injected_error)
    recovered_error = signed_normal_error(reference_xy, heading, estimated_xy)
    np.testing.assert_allclose(recovered_error, injected_error, atol=1e-12)

    first = generate_sequence(config, "latent_autoregressive", "train", 0)
    repeated = generate_sequence(config, "latent_autoregressive", "train", 0)
    np.testing.assert_array_equal(first.conditions, repeated.conditions)
    np.testing.assert_array_equal(first.errors, repeated.errors)
    assert first.sequence_seed == repeated.sequence_seed

    long_rng = np.random.default_rng(91)
    long_conditions = generate_condition_sequence(
        long_rng,
        length=2500,
        s_grid_m=s_grid,
        ranges=ConditionRanges(),
        sample_rate_hz=10.0,
    ).features
    gaussian = generate_errors(
        "conditional_gaussian", np.random.default_rng(123), long_conditions, s_grid
    )
    gaussian_residual = gaussian.errors[:, -1] - gaussian.conditional_mean[:, -1]
    gaussian_lag_one = _lag_one(gaussian_residual)
    assert abs(gaussian_lag_one) < 0.08

    latent = generate_errors(
        "latent_autoregressive", np.random.default_rng(456), long_conditions, s_grid
    )
    latent_persistence = float(np.mean(latent.latent_state[1:] == latent.latent_state[:-1]))
    assert len(np.unique(latent.latent_state)) == 3
    assert latent_persistence > 0.70

    nonlinear = generate_errors(
        "nonlinear_heavy_tailed", np.random.default_rng(789), long_conditions, s_grid
    )
    nonlinear_residual = nonlinear.errors[:, -1] - nonlinear.conditional_mean[:, -1]
    nonlinear_excess_kurtosis = _excess_kurtosis(nonlinear_residual)
    assert nonlinear_excess_kurtosis > 0.5
    assert np.any(nonlinear.latent_state == 1)

    with tempfile.TemporaryDirectory(prefix="lane-error-verification-") as temporary:
        output_root = Path(temporary) / "generated"
        manifest_path = generate_from_config(config_path, output_root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["files"]) == 9
        for record in manifest["files"]:
            assert len(record["sha256"]) == 64
            assert record["valid_error_values"] > 0
        loaded = load_dataset(output_root / "conditional_gaussian" / "train.npz")
        loaded.validate()
        loaded_shape = list(loaded.errors.shape)

    summary = {
        "status": "passed",
        "signed_error_max_abs_recovery_difference_m": float(
            np.max(np.abs(recovered_error - injected_error))
        ),
        "gaussian_oracle_residual_lag_one": gaussian_lag_one,
        "latent_state_persistence": latent_persistence,
        "nonlinear_residual_excess_kurtosis": nonlinear_excess_kurtosis,
        "smoke_train_error_shape": loaded_shape,
        "verified_archive_count": 9,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

