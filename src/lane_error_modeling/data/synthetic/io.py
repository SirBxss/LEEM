"""Safe serialization and integrity metadata for generated datasets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .config import SyntheticDatasetConfig
from .schema import FEATURE_NAMES, PaddedDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dataset(path: str | Path, dataset: PaddedDataset) -> dict[str, Any]:
    """Atomically write one compressed split and return its integrity record."""

    dataset.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=f".{destination.stem}-", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary,
            sequence_ids=dataset.sequence_ids,
            sequence_seeds=dataset.sequence_seeds,
            lengths=dataset.lengths,
            conditions=dataset.conditions,
            errors=dataset.errors,
            valid_mask=dataset.valid_mask,
            conditional_mean=dataset.conditional_mean,
            latent_state=dataset.latent_state,
            reference_curvature=dataset.reference_curvature,
            reference_heading=dataset.reference_heading,
            reference_xy=dataset.reference_xy,
            s_grid_m=dataset.s_grid_m,
        )
    os.replace(temporary_path, destination)
    valid_errors = dataset.errors[dataset.valid_mask].astype(np.float64)
    condition_rows = np.concatenate(
        [
            dataset.conditions[sequence_index, : int(length)]
            for sequence_index, length in enumerate(dataset.lengths)
        ],
        axis=0,
    ).astype(np.float64)
    active_latent_states = np.concatenate(
        [
            dataset.latent_state[sequence_index, : int(length)]
            for sequence_index, length in enumerate(dataset.lengths)
        ]
    )
    state_values, state_counts = np.unique(active_latent_states, return_counts=True)
    active_target_count = int(np.sum(dataset.lengths)) * int(len(dataset.s_grid_m))
    return {
        "path": destination.as_posix(),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "sequence_count": int(len(dataset.lengths)),
        "max_sequence_frames": int(dataset.conditions.shape[1]),
        "valid_error_values": int(np.count_nonzero(dataset.valid_mask)),
        "valid_fraction_within_active_frames": float(
            np.count_nonzero(dataset.valid_mask) / active_target_count
        ),
        "error_summary_m": {
            "mean": float(np.mean(valid_errors)),
            "standard_deviation": float(np.std(valid_errors)),
            "absolute_q95": float(np.quantile(np.abs(valid_errors), 0.95)),
            "absolute_q99": float(np.quantile(np.abs(valid_errors), 0.99)),
            "absolute_max": float(np.max(np.abs(valid_errors))),
        },
        "condition_min": dict(zip(FEATURE_NAMES, np.min(condition_rows, axis=0).tolist())),
        "condition_max": dict(zip(FEATURE_NAMES, np.max(condition_rows, axis=0).tolist())),
        "latent_state_counts": {
            str(int(state)): int(count)
            for state, count in zip(state_values, state_counts)
        },
    }


def load_dataset(path: str | Path) -> PaddedDataset:
    """Load and validate a generated split without pickle/object arrays."""

    with np.load(Path(path), allow_pickle=False) as archive:
        dataset = PaddedDataset(
            sequence_ids=archive["sequence_ids"],
            sequence_seeds=archive["sequence_seeds"],
            lengths=archive["lengths"],
            conditions=archive["conditions"],
            errors=archive["errors"],
            valid_mask=archive["valid_mask"],
            conditional_mean=archive["conditional_mean"],
            latent_state=archive["latent_state"],
            reference_curvature=archive["reference_curvature"],
            reference_heading=archive["reference_heading"],
            reference_xy=archive["reference_xy"],
            s_grid_m=archive["s_grid_m"],
        )
    dataset.validate()
    return dataset


def write_manifest(
    output_root: str | Path,
    config: SyntheticDatasetConfig,
    file_records: list[dict[str, Any]],
) -> Path:
    """Write a human- and machine-readable provenance manifest atomically."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": config.schema_version,
        "generator_package": "lane-error-modeling",
        "generator_version": "0.2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "target": "signed reference-normal lateral path error in metres",
        "config": asdict(config),
        "files": file_records,
    }
    destination = root / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".manifest-", suffix=".json", dir=root, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(manifest, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    os.replace(temporary_path, destination)
    return destination
