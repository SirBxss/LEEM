"""Leakage-safe Gaussian selection and synthetic experiment orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset, SequenceStandardizer
from lane_error_modeling.data.synthetic.io import load_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES, PaddedDataset
from lane_error_modeling.models import ConditionalMultivariateGaussian, GaussianConfig

from .config import GaussianExperimentConfig
from .io import atomic_write_json, sha256_file
from .metrics import evaluate_probabilistic_samples
from .plots import create_evaluation_plots
from .reference import EvaluationReference


def _model_dataset(padded: PaddedDataset) -> SequenceDataset:
    return SequenceDataset.from_arrays(
        sequence_ids=padded.sequence_ids,
        conditions=padded.conditions,
        errors=padded.errors,
        valid_mask=padded.valid_mask,
        lengths=padded.lengths,
        feature_names=FEATURE_NAMES,
        s_grid_m=padded.s_grid_m,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("synthetic manifest has an invalid structure")
    if tuple(manifest.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("synthetic manifest feature names differ from the model contract")
    return manifest


def _manifest_record(
    manifest: Mapping[str, Any], scenario: str, split: str
) -> Mapping[str, Any]:
    records = [
        record
        for record in manifest["files"]
        if isinstance(record, Mapping)
        and record.get("scenario") == scenario
        and record.get("split") == split
    ]
    if len(records) != 1:
        raise ValueError(f"manifest must contain one {scenario}/{split} record")
    return records[0]


def _load_verified_split(
    dataset_root: Path,
    manifest: Mapping[str, Any],
    scenario: str,
    split: str,
) -> tuple[PaddedDataset, dict[str, object]]:
    path = dataset_root / scenario / f"{split}.npz"
    record = _manifest_record(manifest, scenario, split)
    expected_checksum = record.get("sha256")
    actual_checksum = sha256_file(path)
    if actual_checksum != expected_checksum:
        raise ValueError(f"checksum mismatch for {scenario}/{split}")
    padded = load_dataset(path)
    return padded, {
        "path": path.as_posix(),
        "sha256": actual_checksum,
        "size_bytes": path.stat().st_size,
        "sequence_count": padded.lengths.size,
    }


def _validation_nll_per_observed_value(
    model: ConditionalMultivariateGaussian,
    dataset: SequenceDataset,
) -> float:
    observed_count = int(np.count_nonzero(dataset.valid_mask))
    if observed_count == 0:
        raise ValueError("validation/test dataset contains no observed targets")
    return float(-np.sum(model.log_probability(dataset)) / observed_count)


def _select_gaussian(
    train: SequenceDataset,
    validation: SequenceDataset,
    candidates: tuple[GaussianConfig, ...],
) -> tuple[GaussianConfig, list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    selection_metric = "validation_nll_per_observed_value_standardized"
    for candidate_index, candidate in enumerate(candidates):
        model = ConditionalMultivariateGaussian(candidate)
        report = model.fit(train, validation)
        validation_nll = float(report.metrics[selection_metric])
        records.append(
            {
                "candidate_index": candidate_index,
                "config": candidate.to_dict(),
                "selection_metric": selection_metric,
                "selection_metric_value": validation_nll,
                "fit_metrics": dict(report.metrics),
                "fit_warnings": list(report.warnings),
            }
        )
    selected_index = min(
        range(len(records)),
        key=lambda index: (
            float(records[index]["selection_metric_value"]),
            candidates[index].ridge_penalty,
            candidates[index].covariance_shrinkage,
        ),
    )
    return candidates[selected_index], records, selected_index


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(np.float64).tiny))
    return covariance / (scale[:, None] * scale[None, :])


def _inverse_errors_memory_efficient(
    standardizer: SequenceStandardizer,
    standardized_values: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Invert station scaling with one float32 output allocation.

    Prototype sample tensors can contain hundreds of millions of values. The
    general standardizer method intentionally promotes to float64, whereas this
    experiment helper keeps generated float32 draws in float32 to avoid several
    simultaneous gigabyte-sized intermediate arrays.
    """

    values = np.asarray(standardized_values, dtype=np.float32)
    if values.shape[-1] != standardizer.state.n_stations:
        raise ValueError("standardized sample station dimension is incompatible")
    try:
        broadcast_mask = np.broadcast_to(np.asarray(active_mask, dtype=np.bool_), values.shape)
    except ValueError as error:
        raise ValueError("active_mask cannot be broadcast to generated values") from error
    scale = np.asarray(standardizer.state.error_scale, dtype=np.float32)
    mean = np.asarray(standardizer.state.error_mean, dtype=np.float32)
    physical = np.empty_like(values, dtype=np.float32)
    np.multiply(values, scale, out=physical)
    np.add(physical, mean, out=physical)
    np.multiply(physical, broadcast_mask, out=physical)
    return physical


def _synthetic_oracle_diagnostics(
    *,
    scenario: str,
    oracle_conditional_mean: np.ndarray,
    test_data: SequenceDataset,
    model: ConditionalMultivariateGaussian,
    standardizer: SequenceStandardizer,
) -> dict[str, float | str]:
    standardized_mean = model.predict_mean(test_data.conditions, test_data.lengths)
    physical_mean = _inverse_errors_memory_efficient(
        standardizer,
        standardized_mean,
        test_data.time_mask[:, :, None],
    )
    oracle_residual = (
        physical_mean[test_data.valid_mask]
        - oracle_conditional_mean[test_data.valid_mask]
    ).astype(np.float64)
    diagnostics: dict[str, float | str] = {
        "interpretation": (
            "oracle synthetic diagnostic only; unavailable for final BMW data"
        ),
        "conditional_mean_oracle_rmse_m": float(
            np.sqrt(np.mean(oracle_residual**2))
        ),
        "conditional_mean_oracle_mae_m": float(np.mean(np.abs(oracle_residual))),
    }
    if scenario == "conditional_gaussian":
        s_grid_m = test_data.s_grid_m.astype(np.float64)
        normalized_distance = s_grid_m / s_grid_m[-1]
        sigma = 0.012 + 0.095 * normalized_distance**1.35
        distance = np.abs(s_grid_m[:, None] - s_grid_m[None, :])
        true_correlation = np.exp(-distance / 25.0)
        true_correlation.flat[:: len(s_grid_m) + 1] += 1e-8
        true_covariance = sigma[:, None] * true_correlation * sigma[None, :]
        error_scale = np.asarray(standardizer.state.error_scale, dtype=np.float64)
        fitted_covariance = (
            error_scale[:, None] * model.covariance * error_scale[None, :]
        )
        diagnostics["covariance_relative_frobenius_error"] = float(
            np.linalg.norm(fitted_covariance - true_covariance, ord="fro")
            / np.linalg.norm(true_covariance, ord="fro")
        )
        true_correlation = _correlation_from_covariance(true_covariance)
        fitted_correlation = _correlation_from_covariance(fitted_covariance)
        upper = np.triu(np.ones_like(true_correlation, dtype=np.bool_), k=1)
        diagnostics["spatial_correlation_oracle_rmse"] = float(
            np.sqrt(
                np.mean(
                    (fitted_correlation[upper] - true_correlation[upper]) ** 2
                )
            )
        )
    return diagnostics


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    contents = list(path.iterdir())
    if not contents:
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory {path} is not empty; pass overwrite explicitly"
        )
    marker = path / "experiment_config.json"
    completed_marker = path / "experiment_manifest.json"
    if not marker.is_file() and not completed_marker.is_file():
        raise ValueError("refusing to overwrite a directory not created by this runner")
    for child in contents:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _artifact_record(path: Path, output_root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_gaussian_experiment(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Run validation-only selection and one frozen test evaluation per scenario."""

    root = Path(project_root).resolve()
    source_config_path = Path(config_path).resolve()
    config = GaussianExperimentConfig.load(source_config_path)
    destination = Path(output_root).resolve()
    _prepare_output(destination, overwrite=overwrite)
    config_artifact = atomic_write_json(
        destination / "experiment_config.json", config.to_dict()
    )

    dataset_root = root / config.dataset_root
    source_manifest_path = dataset_root / "manifest.json"
    source_manifest = _read_manifest(source_manifest_path)
    source_manifest_checksum = sha256_file(source_manifest_path)
    scenario_summaries: dict[str, object] = {}
    all_artifacts: list[dict[str, object]] = [_artifact_record(config_artifact, destination)]

    for scenario in config.scenarios:
        scenario_output = destination / scenario
        scenario_output.mkdir(parents=True, exist_ok=True)

        # Test data are intentionally not loaded until selection is complete.
        train_padded, train_provenance = _load_verified_split(
            dataset_root, source_manifest, scenario, "train"
        )
        validation_padded, validation_provenance = _load_verified_split(
            dataset_root, source_manifest, scenario, "validation"
        )
        raw_train = _model_dataset(train_padded)
        raw_validation = _model_dataset(validation_padded)
        del train_padded, validation_padded
        standardizer = SequenceStandardizer().fit(
            raw_train.conditions,
            raw_train.errors,
            raw_train.valid_mask,
            raw_train.lengths,
            split_name="train",
            feature_names=raw_train.feature_names,
            s_grid_m=raw_train.s_grid_m,
        )
        train = raw_train.standardized_copy(standardizer)
        validation = raw_validation.standardized_copy(standardizer)
        reference = EvaluationReference.fit(
            raw_train, config.evaluation, split_name="train"
        )

        selected_config, candidate_records, selected_index = _select_gaussian(
            train,
            validation,
            config.gaussian_search.candidates(),
        )
        model = ConditionalMultivariateGaussian(selected_config)
        fit_report = model.fit(train, validation)
        del train, validation, raw_train, raw_validation

        # This is the first point at which the held-out test archive is opened.
        test_padded, test_provenance = _load_verified_split(
            dataset_root, source_manifest, scenario, "test"
        )
        raw_test = _model_dataset(test_padded)
        oracle_conditional_mean = test_padded.conditional_mean
        del test_padded
        test = raw_test.standardized_copy(standardizer)
        samples = model.sample(
            test.conditions,
            test.lengths,
            n_samples=config.sample_count,
            seed=config.sample_seed,
            valid_mask=test.valid_mask,
        )
        physical_samples = _inverse_errors_memory_efficient(
            standardizer,
            samples.values,
            test.time_mask[None, :, :, None],
        )
        del samples
        evaluation = evaluate_probabilistic_samples(
            model_name=model.model_name,
            scenario=scenario,
            dataset=raw_test,
            physical_samples=physical_samples,
            reference=reference,
            config=config.evaluation,
        )
        density_metrics = {
            "test_nll_per_observed_value_standardized": (
                _validation_nll_per_observed_value(model, test)
            ),
            "test_log_probability_sum_standardized": float(
                np.sum(model.log_probability(test))
            ),
            "observed_test_value_count": int(np.count_nonzero(test.valid_mask)),
            "interpretation": (
                "secondary density metric; unavailable for RC-GAN comparison"
            ),
        }
        oracle_diagnostics = _synthetic_oracle_diagnostics(
            scenario=scenario,
            oracle_conditional_mean=oracle_conditional_mean,
            test_data=test,
            model=model,
            standardizer=standardizer,
        )
        del test, oracle_conditional_mean

        standardizer_path = standardizer.save(scenario_output / "standardizer.json")
        model_path = model.save(scenario_output / "gaussian_model.npz")
        reference_path = reference.save(
            scenario_output / "evaluation_reference.json"
        )
        evaluation_path = evaluation.save(scenario_output / "evaluation.json")
        selection_path = atomic_write_json(
            scenario_output / "model_selection.json",
            {
                "selection_split": "validation",
                "selection_metric": (
                    "validation_nll_per_observed_value_standardized"
                ),
                "lower_is_better": True,
                "selected_candidate_index": selected_index,
                "selected_config": selected_config.to_dict(),
                "candidates": candidate_records,
                "test_data_accessed_during_selection": False,
            },
        )
        scenario_result = {
            "status": "passed",
            "model_name": model.model_name,
            "scenario": scenario,
            "interpretation": (
                "synthetic capability experiment; not evidence of BMW sensor behaviour"
            ),
            "data_provenance": {
                "train": train_provenance,
                "validation": validation_provenance,
                "test": test_provenance,
            },
            "selected_config": selected_config.to_dict(),
            "fit_report": {
                "model_name": fit_report.model_name,
                "train_sequence_count": fit_report.train_sequence_count,
                "validation_sequence_count": fit_report.validation_sequence_count,
                "metrics": dict(fit_report.metrics),
                "warnings": list(fit_report.warnings),
            },
            "density_metrics": density_metrics,
            "oracle_diagnostics": oracle_diagnostics,
            "common_evaluation": evaluation.to_dict(),
        }
        scenario_result_path = atomic_write_json(
            scenario_output / "scenario_result.json", scenario_result
        )

        plot_paths: tuple[Path, ...] = ()
        if config.create_plots:
            plot_paths = create_evaluation_plots(
                output_directory=scenario_output / "plots",
                dataset=raw_test,
                physical_samples=physical_samples,
                reference=reference,
                result=evaluation,
                seed=config.evaluation.metric_seed,
            )
        artifact_paths = (
            standardizer_path,
            model_path,
            reference_path,
            evaluation_path,
            selection_path,
            scenario_result_path,
            *plot_paths,
        )
        records = [_artifact_record(path, destination) for path in artifact_paths]
        all_artifacts.extend(records)
        scenario_summaries[scenario] = {
            "status": "passed",
            "selected_candidate_index": selected_index,
            "selected_config": selected_config.to_dict(),
            "validation_selection_metric": candidate_records[selected_index][
                "selection_metric_value"
            ],
            "test_common_metrics": evaluation.global_metrics,
            "test_density_metrics": density_metrics,
            "oracle_diagnostics": oracle_diagnostics,
            "artifacts": records,
        }
        del raw_test, physical_samples

    manifest = {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "conditional_multivariate_gaussian",
        "source_configuration": {
            "path": source_config_path.as_posix(),
            "sha256": sha256_file(source_config_path),
        },
        "source_dataset_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": source_manifest_checksum,
        },
        "scientific_scope": (
            "synthetic capability evaluation only; scenarios are not pooled into a "
            "single final leaderboard"
        ),
        "scenarios": scenario_summaries,
        "artifacts": all_artifacts,
    }
    return atomic_write_json(destination / "experiment_manifest.json", manifest)
