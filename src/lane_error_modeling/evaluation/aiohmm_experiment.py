"""Leakage-safe AIOHMM selection and synthetic experiment orchestration."""

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
from lane_error_modeling.models import (
    AIOHMMConfig,
    AutoregressiveInputOutputHMM,
)
from lane_error_modeling.models.aiohmm.inference import (
    transition_log_probabilities,
)

from .config import AIOHMMExperimentConfig
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
        raise ValueError("synthetic manifest feature names differ from model contract")
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
    actual_checksum = sha256_file(path)
    if actual_checksum != record.get("sha256"):
        raise ValueError(f"checksum mismatch for {scenario}/{split}")
    padded = load_dataset(path)
    return padded, {
        "path": path.as_posix(),
        "sha256": actual_checksum,
        "size_bytes": path.stat().st_size,
        "sequence_count": padded.lengths.size,
    }


def _inverse_errors_memory_efficient(
    standardizer: SequenceStandardizer,
    standardized_values: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(standardized_values, dtype=np.float32)
    if values.shape[-1] != standardizer.state.n_stations:
        raise ValueError("standardized sample station dimension is incompatible")
    try:
        mask = np.broadcast_to(
            np.asarray(active_mask, dtype=np.bool_), values.shape
        )
    except ValueError as error:
        raise ValueError("active_mask cannot be broadcast to generated values") from error
    physical = np.empty_like(values, dtype=np.float32)
    np.multiply(
        values,
        np.asarray(standardizer.state.error_scale, dtype=np.float32),
        out=physical,
    )
    np.add(
        physical,
        np.asarray(standardizer.state.error_mean, dtype=np.float32),
        out=physical,
    )
    np.multiply(physical, mask, out=physical)
    return physical


def _select_aiohmm(
    train: SequenceDataset,
    validation: SequenceDataset,
    candidates: tuple[AIOHMMConfig, ...],
) -> tuple[AIOHMMConfig, list[dict[str, object]], int]:
    metric_name = "validation_nll_per_observed_value_standardized"
    records: list[dict[str, object]] = []
    successful_indices: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        model = AutoregressiveInputOutputHMM(candidate)
        try:
            report = model.fit(train, validation)
        except (ValueError, np.linalg.LinAlgError) as error:
            records.append(
                {
                    "candidate_index": candidate_index,
                    "config": candidate.to_dict(),
                    "status": "failed",
                    "failure": str(error),
                    "selection_metric": metric_name,
                    "selection_metric_value": None,
                }
            )
            continue
        records.append(
            {
                "candidate_index": candidate_index,
                "config": candidate.to_dict(),
                "status": "passed",
                "selection_metric": metric_name,
                "selection_metric_value": float(report.metrics[metric_name]),
                "fit_metrics": dict(report.metrics),
                "fit_warnings": list(report.warnings),
            }
        )
        successful_indices.append(candidate_index)
    if not successful_indices:
        raise ValueError("every AIOHMM candidate failed during validation selection")
    selected_index = min(
        successful_indices,
        key=lambda index: (
            float(records[index]["selection_metric_value"]),
            candidates[index].n_states,
            candidates[index].initialization_seed,
        ),
    )
    return candidates[selected_index], records, selected_index


def _normalized_mutual_information(
    predicted: np.ndarray, oracle: np.ndarray
) -> float | None:
    predicted_values, predicted_inverse = np.unique(
        predicted, return_inverse=True
    )
    oracle_values, oracle_inverse = np.unique(oracle, return_inverse=True)
    if len(predicted_values) < 2 or len(oracle_values) < 2:
        return None
    contingency = np.zeros(
        (len(predicted_values), len(oracle_values)), dtype=np.float64
    )
    np.add.at(contingency, (predicted_inverse, oracle_inverse), 1.0)
    probability = contingency / np.sum(contingency)
    predicted_probability = np.sum(probability, axis=1)
    oracle_probability = np.sum(probability, axis=0)
    expected = predicted_probability[:, None] * oracle_probability[None, :]
    positive = probability > 0.0
    mutual_information = float(
        np.sum(probability[positive] * np.log(probability[positive] / expected[positive]))
    )
    predicted_entropy = float(
        -np.sum(
            predicted_probability[predicted_probability > 0.0]
            * np.log(predicted_probability[predicted_probability > 0.0])
        )
    )
    oracle_entropy = float(
        -np.sum(
            oracle_probability[oracle_probability > 0.0]
            * np.log(oracle_probability[oracle_probability > 0.0])
        )
    )
    denominator = np.sqrt(predicted_entropy * oracle_entropy)
    if denominator <= np.finfo(np.float64).eps:
        return None
    return mutual_information / denominator


def _state_diagnostics(
    model: AutoregressiveInputOutputHMM,
    test: SequenceDataset,
    oracle_latent_state: np.ndarray,
) -> dict[str, object]:
    posteriors = model.posterior_state_probabilities(test)
    concatenated = np.concatenate(posteriors, axis=0)
    occupancy = np.mean(concatenated, axis=0)
    entropy = -np.sum(
        concatenated * np.log(np.maximum(concatenated, 1e-300)), axis=1
    )
    predicted_states = np.argmax(concatenated, axis=1)
    active_oracle = np.concatenate(
        [
            oracle_latent_state[index, : int(length)]
            for index, length in enumerate(test.lengths)
        ]
    )
    nmi = _normalized_mutual_information(predicted_states, active_oracle)

    transition_probabilities: list[np.ndarray] = []
    for sequence_index, raw_length in enumerate(test.lengths):
        length = int(raw_length)
        if length <= 1:
            continue
        log_transition = transition_log_probabilities(
            test.conditions[sequence_index, :length].astype(np.float64),
            model.transition_weights,
            input_dependent=model.config.input_dependent_transitions,
        )
        transition_probabilities.append(np.exp(log_transition))
    transitions = np.concatenate(transition_probabilities, axis=0)
    diagonal = np.diagonal(transitions, axis1=1, axis2=2)
    diagnostics: dict[str, object] = {
        "posterior_state_occupancy": [float(value) for value in occupancy],
        "posterior_mean_entropy_nats": float(np.mean(entropy)),
        "mean_self_transition_probability": float(np.mean(diagonal)),
        "mean_transition_probability_standard_deviation": float(
            np.mean(np.std(transitions, axis=0))
        ),
        "minimum_autoregressive_coefficient": float(
            np.min(model.autoregressive_coefficients)
        ),
        "maximum_autoregressive_coefficient": float(
            np.max(model.autoregressive_coefficients)
        ),
        "minimum_covariance_eigenvalue": float(
            min(
                np.min(np.linalg.eigvalsh(covariance))
                for covariance in model.covariances
            )
        ),
        "oracle_interpretation": (
            "oracle latent states are synthetic post-test diagnostics only"
        ),
    }
    if nmi is not None:
        diagnostics["oracle_latent_state_normalized_mutual_information"] = nmi
    return diagnostics


def _oracle_mean_diagnostics(
    physical_samples: np.ndarray,
    test: SequenceDataset,
    oracle_conditional_mean: np.ndarray,
) -> dict[str, float | str]:
    predictive_mean = np.mean(physical_samples, axis=0, dtype=np.float64)
    residual = (
        predictive_mean[test.valid_mask]
        - oracle_conditional_mean[test.valid_mask]
    )
    return {
        "interpretation": (
            "oracle synthetic diagnostic only; unavailable for final BMW data"
        ),
        "conditional_mean_oracle_rmse_m": float(
            np.sqrt(np.mean(residual**2))
        ),
        "conditional_mean_oracle_mae_m": float(np.mean(np.abs(residual))),
    }


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
    if not (
        (path / "experiment_config.json").is_file()
        or (path / "experiment_manifest.json").is_file()
    ):
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


def run_aiohmm_experiment(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Select AIOHMM candidates on validation and evaluate held-out test once."""

    root = Path(project_root).resolve()
    source_config_path = Path(config_path).resolve()
    config = AIOHMMExperimentConfig.load(source_config_path)
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
    all_artifacts = [_artifact_record(config_artifact, destination)]

    for scenario in config.scenarios:
        scenario_output = destination / scenario
        scenario_output.mkdir(parents=True, exist_ok=True)
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
        selected_config, candidate_records, selected_index = _select_aiohmm(
            train, validation, config.aiohmm_search.candidates()
        )
        model = AutoregressiveInputOutputHMM(selected_config)
        fit_report = model.fit(train, validation)
        del train, validation, raw_train, raw_validation

        # Held-out test data are first opened after candidate selection is frozen.
        test_padded, test_provenance = _load_verified_split(
            dataset_root, source_manifest, scenario, "test"
        )
        raw_test = _model_dataset(test_padded)
        oracle_conditional_mean = test_padded.conditional_mean
        oracle_latent_state = test_padded.latent_state
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
        observed_count = int(np.count_nonzero(test.valid_mask))
        test_log_probability = model.log_probability(test)
        density_metrics = {
            "test_nll_per_observed_value_standardized": float(
                -np.sum(test_log_probability) / observed_count
            ),
            "test_log_probability_sum_standardized": float(
                np.sum(test_log_probability)
            ),
            "observed_test_value_count": observed_count,
            "interpretation": (
                "secondary density metric; unavailable for RC-GAN comparison"
            ),
        }
        state_diagnostics = _state_diagnostics(
            model, test, oracle_latent_state
        )
        oracle_diagnostics = _oracle_mean_diagnostics(
            physical_samples, test, oracle_conditional_mean
        )
        del test, oracle_conditional_mean, oracle_latent_state

        standardizer_path = standardizer.save(
            scenario_output / "standardizer.json"
        )
        model_path = model.save(scenario_output / "aiohmm_model.npz")
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
            "state_diagnostics": state_diagnostics,
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
        records = [
            _artifact_record(path, destination) for path in artifact_paths
        ]
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
            "state_diagnostics": state_diagnostics,
            "oracle_diagnostics": oracle_diagnostics,
            "artifacts": records,
        }
        del raw_test, physical_samples

    manifest = {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "autoregressive_input_output_hmm",
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
