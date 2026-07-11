"""Explicit data-generating mechanisms used to validate the three model families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScenarioOutput:
    """Error sequence and oracle variables available only for synthetic validation."""

    errors: NDArray[np.float64]
    conditional_mean: NDArray[np.float64]
    latent_state: NDArray[np.int8]


def _normalized_conditions(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert the six physical features to stable dimensionless values."""

    normalized = np.empty_like(features, dtype=np.float64)
    normalized[:, 0] = (features[:, 0] - 20.0) / 15.0
    normalized[:, 1] = features[:, 1] / 0.01
    normalized[:, 2] = features[:, 2] / 0.02
    normalized[:, 3] = (features[:, 3] - 3.6) / 0.4
    normalized[:, 4] = 1.0 - features[:, 4]
    normalized[:, 5] = 1.0 - features[:, 5]
    return normalized


def _correlation_cholesky(
    s_grid_m: NDArray[np.float64], correlation_length_m: float
) -> NDArray[np.float64]:
    distance = np.abs(s_grid_m[:, None] - s_grid_m[None, :])
    correlation = np.exp(-distance / correlation_length_m)
    correlation.flat[:: len(s_grid_m) + 1] += 1e-8
    return np.linalg.cholesky(correlation)


def _linear_target_mean(
    normalized_conditions: NDArray[np.float64],
    normalized_distance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Conditional mean shared by the Gaussian and latent scenarios."""

    x = normalized_conditions
    r = normalized_distance
    return (
        0.006 * r[None, :]
        + 0.016 * x[:, 0, None] * r[None, :]
        + 0.065 * x[:, 1, None] * r[None, :] ** 1.35
        + 0.025 * x[:, 2, None] * r[None, :]
        + 0.012 * x[:, 3, None] * r[None, :]
        + 0.055 * x[:, 4, None] * r[None, :] ** 1.6
        + 0.035 * x[:, 5, None] * r[None, :] ** 1.4
    )


def generate_conditional_gaussian(
    rng: np.random.Generator,
    conditions: NDArray[np.float64],
    s_grid_m: NDArray[np.float64],
) -> ScenarioOutput:
    """Generate independent-in-time multivariate conditional Gaussian errors."""

    x = _normalized_conditions(conditions)
    r = s_grid_m / s_grid_m[-1]
    conditional_mean = _linear_target_mean(x, r)
    spatial_sigma_m = 0.012 + 0.095 * r**1.35
    cholesky = spatial_sigma_m[:, None] * _correlation_cholesky(
        s_grid_m, correlation_length_m=25.0
    )
    standard_noise = rng.normal(size=conditional_mean.shape)
    errors = conditional_mean + standard_noise @ cholesky.T
    latent_state = np.zeros(len(conditions), dtype=np.int8)
    return ScenarioOutput(errors, conditional_mean, latent_state)


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def generate_latent_autoregressive(
    rng: np.random.Generator,
    conditions: NDArray[np.float64],
    s_grid_m: NDArray[np.float64],
) -> ScenarioOutput:
    """Generate an input-dependent three-regime autoregressive process."""

    length = len(conditions)
    station_count = len(s_grid_m)
    x = _normalized_conditions(conditions)
    r = s_grid_m / s_grid_m[-1]
    base_mean = _linear_target_mean(x, r)

    base_transition = np.array(
        [
            [0.965, 0.028, 0.007],
            [0.075, 0.885, 0.040],
            [0.035, 0.125, 0.840],
        ],
        dtype=np.float64,
    )
    severity_preference = np.array([-1.0, 0.25, 1.15], dtype=np.float64)
    autoregressive_coefficient = np.array([0.45, 0.70, 0.88])
    innovation_sigma_scale = np.array([0.65, 1.25, 2.35])
    correlation_lengths = np.array([28.0, 22.0, 16.0])
    correlation_cholesky = [
        _correlation_cholesky(s_grid_m, value) for value in correlation_lengths
    ]

    errors = np.zeros((length, station_count), dtype=np.float64)
    conditional_mean = np.zeros_like(errors)
    latent_state = np.zeros(length, dtype=np.int8)

    initial_risk = np.clip(
        0.35 * abs(x[0, 1]) + 0.30 * x[0, 4] + 0.25 * x[0, 5], 0.0, 2.5
    )
    latent_state[0] = rng.choice(3, p=_softmax(np.array([1.6, 0.2, -0.8]) + initial_risk * severity_preference))

    for time_index in range(length):
        if time_index > 0:
            risk = np.clip(
                0.25 * max(x[time_index, 0], 0.0)
                + 0.35 * abs(x[time_index, 1])
                + 0.15 * x[time_index, 2]
                + 0.35 * x[time_index, 4]
                + 0.30 * x[time_index, 5],
                0.0,
                3.0,
            )
            previous_state = int(latent_state[time_index - 1])
            logits = np.log(base_transition[previous_state]) + risk * severity_preference
            latent_state[time_index] = rng.choice(3, p=_softmax(logits))

        state = int(latent_state[time_index])
        if state == 0:
            state_target = np.zeros(station_count)
        elif state == 1:
            signed_curvature = np.clip(x[time_index, 1], -1.5, 1.5)
            state_target = signed_curvature * (0.020 * r + 0.085 * r**1.45)
        else:
            signed_curvature = np.clip(x[time_index, 1], -1.5, 1.5)
            state_target = (
                0.035 * np.sin(2.0 * np.pi * r)
                + signed_curvature * (0.025 * r + 0.090 * r**1.7)
                + 0.080 * (x[time_index, 4] + x[time_index, 5]) * r**1.5
            )

        phi = autoregressive_coefficient[state]
        previous_error = errors[time_index - 1] if time_index > 0 else np.zeros(station_count)
        target = base_mean[time_index] + state_target
        conditional_mean[time_index] = phi * previous_error + (1.0 - phi) * target

        sigma = innovation_sigma_scale[state] * (0.012 + 0.070 * r**1.3)
        innovation = sigma * (
            correlation_cholesky[state] @ rng.normal(size=station_count)
        )
        errors[time_index] = conditional_mean[time_index] + innovation

    return ScenarioOutput(errors, conditional_mean, latent_state)


def generate_nonlinear_heavy_tailed(
    rng: np.random.Generator,
    conditions: NDArray[np.float64],
    s_grid_m: NDArray[np.float64],
) -> ScenarioOutput:
    """Generate nonlinear, long-memory, heavy-tailed errors with rare bursts."""

    length = len(conditions)
    station_count = len(s_grid_m)
    x = _normalized_conditions(conditions)
    r = s_grid_m / s_grid_m[-1]
    correlation_cholesky = _correlation_cholesky(s_grid_m, 19.0)
    recurrent_weights = np.array(
        [
            [0.55, 0.80, 0.20, 0.00, 0.70, 0.25],
            [-0.15, 0.45, 0.65, 0.25, 0.30, 0.75],
            [0.35, -0.40, 0.30, -0.20, 0.85, 0.55],
        ],
        dtype=np.float64,
    )

    errors = np.zeros((length, station_count), dtype=np.float64)
    conditional_mean = np.zeros_like(errors)
    latent_state = np.zeros(length, dtype=np.int8)
    memory = np.zeros(3, dtype=np.float64)
    burst_active = False
    burst_amplitude = 0.0
    burst_center = 0.75
    burst_width = 0.18

    for time_index in range(length):
        memory = (
            0.92 * memory
            + 0.08 * np.tanh(recurrent_weights @ x[time_index])
            + rng.normal(0.0, 0.012, size=3)
        )
        risk = np.clip(
            0.20 * max(x[time_index, 0], 0.0)
            + 0.30 * abs(x[time_index, 1])
            + 0.15 * x[time_index, 2]
            + 0.45 * x[time_index, 4]
            + 0.40 * x[time_index, 5],
            0.0,
            3.0,
        )
        if burst_active:
            burst_active = rng.random() < 0.93
            burst_amplitude *= 0.96
        elif rng.random() < 0.002 + 0.012 * min(risk, 2.0):
            burst_active = True
            burst_amplitude = rng.choice(np.array([-1.0, 1.0])) * rng.uniform(0.25, 0.85)
            burst_center = rng.uniform(0.45, 0.95)
            burst_width = rng.uniform(0.08, 0.24)

        latent_state[time_index] = np.int8(burst_active)
        burst_profile = (
            burst_amplitude
            * np.exp(-0.5 * ((r - burst_center) / burst_width) ** 2)
            if burst_active
            else np.zeros(station_count)
        )
        nonlinear_target = (
            0.065 * np.sin(1.4 * x[time_index, 1] + 3.2 * r) * r**1.25
            + 0.12 * x[time_index, 4] * x[time_index, 5] * r**2
            + 0.045 * np.tanh(x[time_index, 0] * x[time_index, 1]) * r
            + 0.055 * memory[0] * r
            + 0.040 * memory[1] * np.sin(np.pi * r)
            + 0.035 * memory[2] * r**2
            + burst_profile
        )
        phi = 0.62 + 0.24 / (1.0 + np.exp(-risk))
        previous_error = errors[time_index - 1] if time_index > 0 else np.zeros(station_count)
        conditional_mean[time_index] = phi * previous_error + (1.0 - phi) * nonlinear_target

        sigma = (
            0.010
            + 0.060 * r**1.3
            + 0.075 * x[time_index, 4] * r
            + 0.055 * x[time_index, 5] * r
        )
        correlated_normal = correlation_cholesky @ rng.normal(size=station_count)
        degrees_of_freedom = 7.0
        student_scale = np.sqrt(
            degrees_of_freedom / rng.chisquare(degrees_of_freedom)
        )
        innovation = sigma * correlated_normal * student_scale
        skew_strength = 0.010 * (x[time_index, 4] + x[time_index, 5])
        innovation += skew_strength * (correlated_normal**2 - 1.0) * r
        errors[time_index] = conditional_mean[time_index] + innovation

    return ScenarioOutput(errors, conditional_mean, latent_state)


SCENARIO_GENERATORS = {
    "conditional_gaussian": generate_conditional_gaussian,
    "latent_autoregressive": generate_latent_autoregressive,
    "nonlinear_heavy_tailed": generate_nonlinear_heavy_tailed,
}


def generate_errors(
    scenario: str,
    rng: np.random.Generator,
    conditions: NDArray[np.float64],
    s_grid_m: NDArray[np.float64],
) -> ScenarioOutput:
    """Dispatch to a named, version-controlled data-generating mechanism."""

    try:
        generator = SCENARIO_GENERATORS[scenario]
    except KeyError as error:
        raise ValueError(f"unsupported scenario {scenario!r}") from error
    return generator(rng, conditions, s_grid_m)
