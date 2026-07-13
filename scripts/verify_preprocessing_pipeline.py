"""Verify train-only preprocessing on every generated smoke scenario."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lane_error_modeling.data.preprocessing import (
    SequenceDataset,
    SequenceStandardizer,
    iter_sequence_batches,
)
from lane_error_modeling.data.synthetic.io import load_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES


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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    smoke_root = project_root / "outputs" / "synthetic_smoke"
    if not (smoke_root / "manifest.json").is_file():
        raise FileNotFoundError(
            "generate outputs/synthetic_smoke before verifying preprocessing"
        )

    summary: dict[str, object] = {}
    for scenario in SCENARIOS:
        datasets = {
            split: _load_model_dataset(smoke_root / scenario / f"{split}.npz")
            for split in ("train", "validation", "test")
        }
        train = datasets["train"]
        standardizer = SequenceStandardizer().fit(
            train.conditions,
            train.errors,
            train.valid_mask,
            train.lengths,
            split_name="train",
            feature_names=train.feature_names,
            s_grid_m=train.s_grid_m,
        )
        standardized = {
            split: dataset.standardized_copy(standardizer)
            for split, dataset in datasets.items()
        }
        active_conditions = standardized["train"].conditions[
            standardized["train"].time_mask
        ]
        batches = list(
            iter_sequence_batches(
                standardized["train"],
                batch_size=5,
                shuffle=True,
                seed=20260710,
            )
        )
        scenario_summary = {
            "minimum_valid_training_count_per_station": min(
                standardizer.state.error_count
            ),
            "train_active_condition_abs_mean_max": float(
                np.max(np.abs(np.mean(active_conditions, axis=0)))
            ),
            "train_active_condition_std_deviation_error_max": float(
                np.max(np.abs(np.std(active_conditions, axis=0) - 1.0))
            ),
            "validation_finite": bool(
                np.isfinite(standardized["validation"].conditions).all()
                and np.isfinite(standardized["validation"].errors).all()
            ),
            "test_finite": bool(
                np.isfinite(standardized["test"].conditions).all()
                and np.isfinite(standardized["test"].errors).all()
            ),
            "batch_count": len(batches),
            "batched_sequence_count": sum(len(batch.lengths) for batch in batches),
        }
        if scenario_summary["minimum_valid_training_count_per_station"] < 2:
            raise AssertionError(f"{scenario} has insufficient station coverage")
        if scenario_summary["train_active_condition_abs_mean_max"] > 1e-5:
            raise AssertionError(f"{scenario} condition means are not standardized")
        if scenario_summary["train_active_condition_std_deviation_error_max"] > 1e-5:
            raise AssertionError(f"{scenario} condition scales are not standardized")
        if not scenario_summary["validation_finite"] or not scenario_summary["test_finite"]:
            raise AssertionError(f"{scenario} produced non-finite transformed data")
        if scenario_summary["batched_sequence_count"] != train.n_sequences:
            raise AssertionError(f"{scenario} batching lost or duplicated sequences")
        summary[scenario] = scenario_summary

    print(json.dumps({"status": "passed", "scenarios": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

