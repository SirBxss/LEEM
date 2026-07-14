"""Fit and verify the Gaussian baseline independently on every smoke scenario."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset, SequenceStandardizer
from lane_error_modeling.data.synthetic.io import load_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES
from lane_error_modeling.models import ConditionalMultivariateGaussian, GaussianConfig


SCENARIOS = (
    "conditional_gaussian",
    "latent_autoregressive",
    "nonlinear_heavy_tailed",
)


def _load_model_dataset(path: Path) -> SequenceDataset:
    padded = load_dataset(path)
    return SequenceDataset.from_arrays(
        sequence_ids=padded.sequence_ids,
        conditions=padded.conditions,
        errors=padded.errors,
        valid_mask=padded.valid_mask,
        lengths=padded.lengths,
        feature_names=FEATURE_NAMES,
        s_grid_m=padded.s_grid_m,
    )


def _nll_per_observed_value(
    model: ConditionalMultivariateGaussian,
    dataset: SequenceDataset,
) -> float:
    observed_count = int(np.count_nonzero(dataset.valid_mask))
    if observed_count == 0:
        raise AssertionError("verification split contains no observed targets")
    return float(-np.sum(model.log_probability(dataset)) / observed_count)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    smoke_root = project_root / "outputs" / "synthetic_smoke"
    if not (smoke_root / "manifest.json").is_file():
        raise FileNotFoundError(
            "generate outputs/synthetic_smoke before verifying the Gaussian model"
        )
    config = GaussianConfig.load(project_root / "configs" / "gaussian_baseline.json")

    summary: dict[str, object] = {}
    for scenario in SCENARIOS:
        raw_datasets = {
            split: _load_model_dataset(smoke_root / scenario / f"{split}.npz")
            for split in ("train", "validation", "test")
        }
        raw_train = raw_datasets["train"]
        standardizer = SequenceStandardizer().fit(
            raw_train.conditions,
            raw_train.errors,
            raw_train.valid_mask,
            raw_train.lengths,
            split_name="train",
            feature_names=raw_train.feature_names,
            s_grid_m=raw_train.s_grid_m,
        )
        datasets = {
            split: dataset.standardized_copy(standardizer)
            for split, dataset in raw_datasets.items()
        }

        model = ConditionalMultivariateGaussian(config)
        report = model.fit(datasets["train"], datasets["validation"])
        test_log_probability = model.log_probability(datasets["test"])
        if test_log_probability.shape != (datasets["test"].n_sequences,):
            raise AssertionError(f"{scenario} returned an invalid likelihood shape")
        if not np.all(np.isfinite(test_log_probability)):
            raise AssertionError(f"{scenario} returned non-finite likelihoods")

        samples = model.sample(
            datasets["test"].conditions,
            datasets["test"].lengths,
            n_samples=8,
            seed=20260713,
            valid_mask=datasets["test"].valid_mask,
        )
        repeated_samples = model.sample(
            datasets["test"].conditions,
            datasets["test"].lengths,
            n_samples=8,
            seed=20260713,
            valid_mask=datasets["test"].valid_mask,
        )
        if not np.array_equal(samples.values, repeated_samples.values):
            raise AssertionError(f"{scenario} sampling is not seed-reproducible")

        physical_samples = standardizer.inverse_transform_errors(
            samples.values,
            datasets["test"].time_mask[None, :, :, None],
        )
        observed_sample_values = physical_samples[:, datasets["test"].valid_mask]
        if not np.all(np.isfinite(observed_sample_values)):
            raise AssertionError(f"{scenario} produced non-finite physical samples")

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = model.save(
                Path(temporary_directory) / f"{scenario}-gaussian.npz"
            )
            restored = ConditionalMultivariateGaussian.load(artifact)
            restored_log_probability = restored.log_probability(datasets["test"])
        if not np.array_equal(test_log_probability, restored_log_probability):
            raise AssertionError(f"{scenario} persistence changed likelihood values")

        eigenvalues = np.linalg.eigvalsh(model.covariance)
        if float(np.min(eigenvalues)) < config.minimum_eigenvalue * (1.0 - 1e-7):
            raise AssertionError(f"{scenario} covariance is not positive definite")
        summary[scenario] = {
            "fit_metrics": dict(report.metrics),
            "fit_warnings": list(report.warnings),
            "test_nll_per_observed_value_standardized": _nll_per_observed_value(
                model, datasets["test"]
            ),
            "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
            "maximum_covariance_eigenvalue": float(np.max(eigenvalues)),
            "minimum_pair_observation_count": int(
                np.min(model.pair_observation_counts)
            ),
            "physical_sample_abs_q95_m": float(
                np.quantile(np.abs(observed_sample_values), 0.95)
            ),
            "physical_sample_abs_q99_m": float(
                np.quantile(np.abs(observed_sample_values), 0.99)
            ),
            "sample_shape": list(samples.values.shape),
        }

    print(
        json.dumps(
            {
                "status": "passed",
                "interpretation": (
                    "implementation verification only; synthetic scenarios are not "
                    "a final model ranking"
                ),
                "scenarios": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
