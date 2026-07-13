"""Orchestration for deterministic sequence and split generation."""

from __future__ import annotations

import numpy as np

from .conditions import generate_condition_sequence
from .config import SyntheticDatasetConfig
from .scenarios import generate_errors
from .schema import PaddedDataset, SequenceSample, pad_samples


_SPLIT_CODES = {"train": 11, "validation": 23, "test": 37}
_SCENARIO_CODES = {
    "conditional_gaussian": 101,
    "latent_autoregressive": 211,
    "nonlinear_heavy_tailed": 307,
}


def _sequence_seed(
    master_seed: int,
    scenario: str,
    split: str,
    sequence_index: int,
) -> int:
    """Derive an order-independent seed from stable integer identifiers."""

    seed_sequence = np.random.SeedSequence(
        [master_seed, _SCENARIO_CODES[scenario], _SPLIT_CODES[split], sequence_index]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def generate_sequence(
    config: SyntheticDatasetConfig,
    scenario: str,
    split: str,
    sequence_index: int,
) -> SequenceSample:
    """Generate and validate one independently reproducible sequence."""

    config.validate()
    if scenario not in config.scenarios:
        raise ValueError(f"scenario {scenario!r} is not enabled by this configuration")
    if split not in _SPLIT_CODES:
        raise ValueError(f"unknown split {split!r}")
    if sequence_index < 0:
        raise ValueError("sequence_index must be non-negative")

    sequence_seed = _sequence_seed(config.master_seed, scenario, split, sequence_index)
    rng = np.random.default_rng(sequence_seed)
    length = int(
        rng.integers(config.min_sequence_frames, config.max_sequence_frames + 1)
    )
    s_grid_m = np.asarray(config.s_grid_m, dtype=np.float64)
    condition_sequence = generate_condition_sequence(
        rng=rng,
        length=length,
        s_grid_m=s_grid_m,
        ranges=config.condition_ranges,
        sample_rate_hz=config.sample_rate_hz,
    )
    scenario_output = generate_errors(
        scenario=scenario,
        rng=rng,
        conditions=condition_sequence.features,
        s_grid_m=s_grid_m,
    )

    marking_quality = condition_sequence.features[:, 4]
    environment_quality = condition_sequence.features[:, 5]
    valid_range_m = np.minimum(
        s_grid_m[-1],
        40.0
        + 80.0
        * (0.60 * marking_quality + 0.40 * environment_quality),
    )
    valid_mask = s_grid_m[None, :] <= valid_range_m[:, None]
    errors = np.where(valid_mask, scenario_output.errors, 0.0)
    conditional_mean = np.where(valid_mask, scenario_output.conditional_mean, 0.0)

    sample = SequenceSample(
        sequence_id=f"{scenario}:{split}:{sequence_index:06d}",
        sequence_seed=sequence_seed,
        scenario=scenario,
        conditions=condition_sequence.features,
        errors=errors,
        valid_mask=valid_mask,
        conditional_mean=conditional_mean,
        latent_state=scenario_output.latent_state,
        reference_curvature=condition_sequence.reference_curvature,
        reference_heading=condition_sequence.reference_heading,
        reference_xy=condition_sequence.reference_xy,
    )
    sample.validate(config.n_stations, config.max_plausible_abs_error_m)
    return sample


def split_size(config: SyntheticDatasetConfig, split: str) -> int:
    """Return the configured number of sequences in a named split."""

    sizes = {
        "train": config.splits.train,
        "validation": config.splits.validation,
        "test": config.splits.test,
    }
    try:
        return sizes[split]
    except KeyError as error:
        raise ValueError(f"unknown split {split!r}") from error


def generate_dataset(
    config: SyntheticDatasetConfig,
    scenario: str,
    split: str,
) -> PaddedDataset:
    """Generate one complete scenario/split dataset in deterministic index order."""

    samples = [
        generate_sequence(config, scenario, split, sequence_index)
        for sequence_index in range(split_size(config, split))
    ]
    return pad_samples(samples, np.asarray(config.s_grid_m, dtype=np.float64))
