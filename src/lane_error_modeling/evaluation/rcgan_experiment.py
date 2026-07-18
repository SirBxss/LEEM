"""Leakage-safe RC-GAN restart selection and synthetic evaluation."""

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
from lane_error_modeling.models.rcgan import RCGANConfig, RecurrentConditionalGAN

from .config import RCGANExperimentConfig
from .io import atomic_write_json, sha256_file
from .metrics import evaluate_probabilistic_samples
from .plots import create_evaluation_plots
from .reference import EvaluationReference


SELECTION_METRIC = "dimension_normalized_energy_score_m"
DIVERSITY_RATIO_METRIC = "generated_to_observed_std_ratio"


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


def _inverse_errors(
    standardizer: SequenceStandardizer,
    standardized_values: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(standardized_values, dtype=np.float32)
    physical = values * np.asarray(
        standardizer.state.error_scale, dtype=np.float32
    )
    physical += np.asarray(standardizer.state.error_mean, dtype=np.float32)
    physical *= np.broadcast_to(np.asarray(active_mask, dtype=np.bool_), values.shape)
    return physical.astype(np.float32, copy=False)


def _fit_report_dict(report) -> dict[str, object]:
    return {
        "model_name": report.model_name,
        "train_sequence_count": report.train_sequence_count,
        "validation_sequence_count": report.validation_sequence_count,
        "metrics": dict(report.metrics),
        "warnings": list(report.warnings),
    }


def _stability_gate(report, *, minimum_ratio: float) -> dict[str, object]:
    """Return the predeclared validation diversity guard for one candidate."""

    if DIVERSITY_RATIO_METRIC not in report.metrics:
        raise ValueError(
            f"RC-GAN fit report is missing {DIVERSITY_RATIO_METRIC!r}"
        )
    value = float(report.metrics[DIVERSITY_RATIO_METRIC])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("RC-GAN validation diversity ratio is invalid")
    enabled = minimum_ratio > 0.0
    return {
        "enabled": enabled,
        "metric": DIVERSITY_RATIO_METRIC,
        "value": value,
        "minimum": float(minimum_ratio),
        "passed": (not enabled) or value >= minimum_ratio,
        "interpretation": (
            "engineering guard against severe under-dispersion; candidates are "
            "still ranked only by validation Energy Score"
        ),
    }


def _select_rcgan(
    *,
    train: SequenceDataset,
    validation: SequenceDataset,
    raw_validation: SequenceDataset,
    standardizer: SequenceStandardizer,
    reference: EvaluationReference,
    candidates: tuple[RCGANConfig, ...],
    scenario: str,
    experiment: RCGANExperimentConfig,
) -> tuple[RecurrentConditionalGAN, object, list[dict[str, object]], int]:
    """Fit restarts and select only by validation physical-unit Energy Score."""

    records: list[dict[str, object]] = []
    best_model: RecurrentConditionalGAN | None = None
    best_report = None
    best_index: int | None = None
    best_key: tuple[float, int] | None = None
    for candidate_index, candidate in enumerate(candidates):
        model = RecurrentConditionalGAN(candidate)
        try:
            report = model.fit(train, validation)
            samples = model.sample(
                validation.conditions,
                validation.lengths,
                n_samples=experiment.selection_sample_count,
                seed=experiment.selection_sample_seed,
                valid_mask=validation.valid_mask,
            )
            physical = _inverse_errors(
                standardizer,
                samples.values,
                validation.time_mask[None, :, :, None],
            )
            validation_evaluation = evaluate_probabilistic_samples(
                model_name=model.model_name,
                scenario=scenario,
                dataset=raw_validation,
                physical_samples=physical,
                reference=reference,
                config=experiment.evaluation,
            )
            value = float(validation_evaluation.global_metrics[SELECTION_METRIC])
            stability_gate = _stability_gate(
                report,
                minimum_ratio=experiment.minimum_validation_diversity_ratio,
            )
        except (RuntimeError, ValueError) as error:
            records.append(
                {
                    "candidate_index": candidate_index,
                    "config": candidate.to_dict(),
                    "status": "failed",
                    "failure": str(error),
                    "selection_metric": SELECTION_METRIC,
                    "selection_metric_value": None,
                }
            )
            continue
        records.append(
            {
                "candidate_index": candidate_index,
                "config": candidate.to_dict(),
                "status": (
                    "passed" if stability_gate["passed"] else "rejected"
                ),
                "selection_metric": SELECTION_METRIC,
                "selection_metric_value": value,
                "selection_sample_count": experiment.selection_sample_count,
                "selection_sample_seed": experiment.selection_sample_seed,
                "fit_report": _fit_report_dict(report),
                "stability_gate": stability_gate,
            }
        )
        if not stability_gate["passed"]:
            continue
        key = (value, candidate.initialization_seed)
        if best_key is None or key < best_key:
            best_key = key
            best_model = model
            best_report = report
            best_index = candidate_index
    if best_model is None or best_report is None or best_index is None:
        raise ValueError(
            "no RC-GAN candidate passed validation selection and stability checks"
        )
    return best_model, best_report, records, best_index


def _oracle_mean_diagnostics(
    physical_samples: np.ndarray,
    test: SequenceDataset,
    oracle_conditional_mean: np.ndarray,
) -> dict[str, float | str]:
    predictive_mean = np.mean(physical_samples, axis=0, dtype=np.float64)
    residual = predictive_mean[test.valid_mask] - oracle_conditional_mean[test.valid_mask]
    return {
        "interpretation": "oracle synthetic diagnostic only; unavailable for BMW data",
        "conditional_mean_oracle_rmse_m": float(np.sqrt(np.mean(residual**2))),
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


def run_rcgan_experiment(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Select an RC-GAN restart on validation and evaluate test exactly once."""

    root = Path(project_root).resolve()
    source_config_path = Path(config_path).resolve()
    config = RCGANExperimentConfig.load(source_config_path)
    destination = Path(output_root).resolve()
    _prepare_output(destination, overwrite=overwrite)
    config_artifact = atomic_write_json(
        destination / "experiment_config.json", config.to_dict()
    )
    dataset_root = root / config.dataset_root
    source_manifest_path = dataset_root / "manifest.json"
    source_manifest = _read_manifest(source_manifest_path)
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
        model, fit_report, candidate_records, selected_index = _select_rcgan(
            train=train,
            validation=validation,
            raw_validation=raw_validation,
            standardizer=standardizer,
            reference=reference,
            candidates=config.rcgan_search.candidates(),
            scenario=scenario,
            experiment=config,
        )
        selected_config = model.config
        del train, validation, raw_train, raw_validation

        # Test remains unopened until the candidate index is irrevocably frozen.
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
        physical_samples = _inverse_errors(
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
        oracle_diagnostics = _oracle_mean_diagnostics(
            physical_samples, test, oracle_conditional_mean
        )
        del test, oracle_conditional_mean

        standardizer_path = standardizer.save(scenario_output / "standardizer.json")
        model_path = model.save(scenario_output / "rcgan_model.npz")
        reference_path = reference.save(
            scenario_output / "evaluation_reference.json"
        )
        evaluation_path = evaluation.save(scenario_output / "evaluation.json")
        selection_path = atomic_write_json(
            scenario_output / "model_selection.json",
            {
                "selection_split": "validation",
                "selection_metric": SELECTION_METRIC,
                "selection_metric_units": "metres per square-root observed dimension",
                "lower_is_better": True,
                "selected_candidate_index": selected_index,
                "selected_config": selected_config.to_dict(),
                "candidates": candidate_records,
                "stability_gate": candidate_records[selected_index][
                    "stability_gate"
                ],
                "test_data_accessed_during_selection": False,
                "interpretation": (
                    "sample-based proper score used because RC-GAN has no tractable "
                    "likelihood; declared learning-rate/seed candidates are filtered "
                    "by the validation-only diversity guard before ranking"
                ),
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
            "architecture": model.architecture_summary(),
            "fit_report": _fit_report_dict(fit_report),
            "validation_stability_gate": candidate_records[selected_index][
                "stability_gate"
            ],
            "density_metrics": {
                "available": False,
                "interpretation": (
                    "RC-GAN is an implicit generative model without tractable NLL"
                ),
            },
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
            "validation_stability_gate": candidate_records[selected_index][
                "stability_gate"
            ],
            "test_common_metrics": evaluation.global_metrics,
            "oracle_diagnostics": oracle_diagnostics,
            "artifacts": records,
        }
        del raw_test, physical_samples

    manifest = {
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "recurrent_conditional_gan",
        "source_configuration": {
            "path": source_config_path.as_posix(),
            "sha256": sha256_file(source_config_path),
        },
        "source_dataset_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "scientific_scope": (
            "synthetic capability evaluation only; scenarios are not pooled into a "
            "single final leaderboard"
        ),
        "stability_protocol": {
            "metric": DIVERSITY_RATIO_METRIC,
            "minimum_validation_ratio": (
                config.minimum_validation_diversity_ratio
            ),
            "test_data_used": False,
            "interpretation": (
                "engineering guard against severe under-dispersion; not a common "
                "comparison metric and not a claim of calibration"
            ),
        },
        "paper_basis": {
            "citation": (
                "Arnelid, L. et al. (2019), Recurrent Conditional Generative "
                "Adversarial Networks for Autonomous Driving Sensor Modelling"
            ),
            "doi": "10.1109/ITSC.2019.8916999",
        },
        "scenarios": scenario_summaries,
        "artifacts": all_artifacts,
    }
    return atomic_write_json(destination / "experiment_manifest.json", manifest)
