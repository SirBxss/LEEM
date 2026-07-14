"""Numerically stable inference for time-inhomogeneous hidden Markov models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def logsumexp(
    values: NDArray[np.float64],
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> NDArray[np.float64] | np.float64:
    """Stable NumPy-only log-sum-exp."""

    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    total = np.sum(np.exp(array - finite_maximum), axis=axis, keepdims=True)
    result = finite_maximum + np.log(total)
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    if axis is None and not keepdims:
        return np.float64(np.asarray(result).item())
    return np.asarray(result, dtype=np.float64)


def log_softmax(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return array - np.asarray(logsumexp(array, axis=axis, keepdims=True))


def transition_log_probabilities(
    conditions: NDArray[np.float64],
    transition_weights: NDArray[np.float64],
    *,
    input_dependent: bool,
) -> NDArray[np.float64]:
    """Return ``[T-1, source state, destination state]`` log probabilities."""

    condition_array = np.asarray(conditions, dtype=np.float64)
    weights = np.asarray(transition_weights, dtype=np.float64)
    if condition_array.ndim != 2:
        raise ValueError("conditions must have shape [T, F]")
    if weights.ndim != 3 or weights.shape[0] != weights.shape[1]:
        raise ValueError("transition_weights must have shape [N, N, F+1]")
    if weights.shape[2] != condition_array.shape[1] + 1:
        raise ValueError("transition weights and conditions are incompatible")
    if len(condition_array) < 1:
        raise ValueError("at least one condition frame is required")
    if len(condition_array) == 1:
        return np.empty((0, weights.shape[0], weights.shape[1]), dtype=np.float64)
    design = np.column_stack(
        (
            np.ones(len(condition_array) - 1, dtype=np.float64),
            condition_array[1:],
        )
    )
    if not input_dependent:
        design[:, 1:] = 0.0
    logits = np.einsum("tp,ijp->tij", design, weights, optimize=True)
    return log_softmax(logits, axis=2)


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Posterior state and transition probabilities for one sequence."""

    log_probability: float
    state_probabilities: NDArray[np.float64]
    transition_probabilities: NDArray[np.float64]

    def validate(self) -> None:
        gamma = self.state_probabilities
        xi = self.transition_probabilities
        if gamma.ndim != 2:
            raise ValueError("state_probabilities must have shape [T, N]")
        if xi.shape != (max(len(gamma) - 1, 0), gamma.shape[1], gamma.shape[1]):
            raise ValueError("transition_probabilities have an invalid shape")
        if not np.isfinite(self.log_probability):
            raise ValueError("sequence log probability must be finite")
        if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(xi)):
            raise ValueError("posterior probabilities must be finite")
        if not np.allclose(np.sum(gamma, axis=1), 1.0, atol=1e-8):
            raise ValueError("state posterior rows must sum to one")
        if len(xi):
            if not np.allclose(np.sum(xi, axis=(1, 2)), 1.0, atol=1e-8):
                raise ValueError("transition posterior slices must sum to one")


def forward_backward(
    log_initial_probabilities: NDArray[np.float64],
    log_transition_probabilities: NDArray[np.float64],
    log_emission_probabilities: NDArray[np.float64],
) -> ForwardBackwardResult:
    """Run scaled log-domain forward-backward inference for one sequence."""

    log_initial = np.asarray(log_initial_probabilities, dtype=np.float64)
    log_transition = np.asarray(log_transition_probabilities, dtype=np.float64)
    log_emission = np.asarray(log_emission_probabilities, dtype=np.float64)
    if log_initial.ndim != 1:
        raise ValueError("log_initial_probabilities must have shape [N]")
    if log_emission.ndim != 2 or log_emission.shape[1] != len(log_initial):
        raise ValueError("log_emission_probabilities must have shape [T, N]")
    time_count, state_count = log_emission.shape
    if time_count < 1:
        raise ValueError("at least one emission frame is required")
    if log_transition.shape != (time_count - 1, state_count, state_count):
        raise ValueError(
            "log_transition_probabilities must have shape [T-1, N, N]"
        )
    if not (
        np.all(np.isfinite(log_initial))
        and np.all(np.isfinite(log_transition))
        and np.all(np.isfinite(log_emission))
    ):
        raise ValueError("inference inputs must be finite")

    log_alpha = np.empty((time_count, state_count), dtype=np.float64)
    normalizers = np.empty(time_count, dtype=np.float64)
    first = log_initial + log_emission[0]
    normalizers[0] = float(logsumexp(first))
    log_alpha[0] = first - normalizers[0]
    for time_index in range(1, time_count):
        prediction = logsumexp(
            log_alpha[time_index - 1, :, None]
            + log_transition[time_index - 1],
            axis=0,
        )
        current = np.asarray(prediction) + log_emission[time_index]
        normalizers[time_index] = float(logsumexp(current))
        log_alpha[time_index] = current - normalizers[time_index]

    log_beta = np.zeros_like(log_alpha)
    for time_index in range(time_count - 2, -1, -1):
        log_beta[time_index] = np.asarray(
            logsumexp(
                log_transition[time_index]
                + log_emission[time_index + 1][None, :]
                + log_beta[time_index + 1][None, :],
                axis=1,
            )
        ) - normalizers[time_index + 1]

    log_gamma = log_alpha + log_beta
    log_gamma -= np.asarray(logsumexp(log_gamma, axis=1, keepdims=True))
    gamma = np.exp(log_gamma)
    xi = np.empty(
        (max(time_count - 1, 0), state_count, state_count), dtype=np.float64
    )
    for time_index in range(time_count - 1):
        log_xi = (
            log_alpha[time_index, :, None]
            + log_transition[time_index]
            + log_emission[time_index + 1][None, :]
            + log_beta[time_index + 1][None, :]
        )
        log_xi -= float(logsumexp(log_xi))
        xi[time_index] = np.exp(log_xi)

    result = ForwardBackwardResult(
        log_probability=float(np.sum(normalizers)),
        state_probabilities=gamma,
        transition_probabilities=xi,
    )
    result.validate()
    return result
