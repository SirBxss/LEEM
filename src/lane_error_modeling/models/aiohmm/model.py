"""Multivariate autoregressive input-output hidden Markov model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lane_error_modeling.data.preprocessing import SequenceDataset
from lane_error_modeling.models.base import (
    FitReport,
    ModelCapabilities,
    ProbabilisticSequenceModel,
    SampleResult,
)

from .config import AIOHMMConfig
from .inference import (
    ForwardBackwardResult,
    forward_backward,
    log_softmax,
    transition_log_probabilities,
)


AIOHMM_MODEL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class _AIOHMMState:
    feature_names: tuple[str, ...]
    s_grid_m: NDArray[np.float64]
    initial_probabilities: NDArray[np.float64]
    transition_weights: NDArray[np.float64]
    base_coefficients: NDArray[np.float64]
    autoregressive_coefficients: NDArray[np.float64]
    covariances: NDArray[np.float64]
    state_occupancies: NDArray[np.float64]
    log_likelihood_history: NDArray[np.float64]
    train_sequence_count: int

    @property
    def n_states(self) -> int:
        return len(self.initial_probabilities)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_stations(self) -> int:
        return len(self.s_grid_m)


@dataclass(frozen=True)
class _ExpectationResult:
    sequence_log_probabilities: NDArray[np.float64]
    posteriors: tuple[ForwardBackwardResult, ...]


def _copy_state(state: _AIOHMMState) -> _AIOHMMState:
    return _AIOHMMState(
        feature_names=state.feature_names,
        s_grid_m=state.s_grid_m.copy(),
        initial_probabilities=state.initial_probabilities.copy(),
        transition_weights=state.transition_weights.copy(),
        base_coefficients=state.base_coefficients.copy(),
        autoregressive_coefficients=state.autoregressive_coefficients.copy(),
        covariances=state.covariances.copy(),
        state_occupancies=state.state_occupancies.copy(),
        log_likelihood_history=state.log_likelihood_history.copy(),
        train_sequence_count=state.train_sequence_count,
    )


def _regularize_covariance(
    raw_covariance: NDArray[np.float64],
    *,
    shrinkage: float,
    minimum_eigenvalue: float,
) -> NDArray[np.float64]:
    diagonal = np.diag(np.diag(raw_covariance))
    covariance = (1.0 - shrinkage) * raw_covariance + shrinkage * diagonal
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (
        eigenvectors * np.maximum(eigenvalues, minimum_eigenvalue)
    ) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


class AutoregressiveInputOutputHMM(ProbabilisticSequenceModel):
    r"""Condition-dependent HMM with multivariate autoregressive emissions.

    For hidden state ``z_t`` and standardized condition vector ``x_t``:

    .. math::
        p(z_t=j\mid z_{t-1}=i,x_t)
        =\operatorname{softmax}_j(W_i[1,x_t]),

    .. math::
        y_t\mid z_t=k,x_t,y_{t-1}
        \sim\mathcal N(B_k^T[1,x_t]+d_k\odot y_{t-1},\Sigma_k).

    The autoregressive matrix is diagonal. During likelihood evaluation, an
    unavailable previous station contributes no autoregressive term. Current
    missing dimensions are marginalized from the Gaussian density.
    """

    def __init__(self, config: AIOHMMConfig | None = None) -> None:
        self.config = config or AIOHMMConfig()
        self.config.validate()
        self._state: _AIOHMMState | None = None
        self._cholesky_factors: NDArray[np.float64] | None = None

    @property
    def model_name(self) -> str:
        return "autoregressive_input_output_hmm"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_log_probability=True)

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def initial_probabilities(self) -> NDArray[np.float64]:
        return self._require_state().initial_probabilities.copy()

    @property
    def transition_weights(self) -> NDArray[np.float64]:
        return self._require_state().transition_weights.copy()

    @property
    def base_coefficients(self) -> NDArray[np.float64]:
        return self._require_state().base_coefficients.copy()

    @property
    def autoregressive_coefficients(self) -> NDArray[np.float64]:
        return self._require_state().autoregressive_coefficients.copy()

    @property
    def covariances(self) -> NDArray[np.float64]:
        return self._require_state().covariances.copy()

    @property
    def state_occupancies(self) -> NDArray[np.float64]:
        return self._require_state().state_occupancies.copy()

    @property
    def log_likelihood_history(self) -> NDArray[np.float64]:
        return self._require_state().log_likelihood_history.copy()

    def _require_state(self) -> _AIOHMMState:
        if self._state is None:
            raise RuntimeError("AIOHMM has not been fitted")
        return self._state

    def _validate_state(self, state: _AIOHMMState) -> None:
        self.config.validate()
        n_states = self.config.n_states
        if state.n_states != n_states:
            raise ValueError("state count differs from the AIOHMM configuration")
        if not state.feature_names or len(set(state.feature_names)) != state.n_features:
            raise ValueError("fitted feature names must be non-empty and unique")
        if state.s_grid_m.shape != (state.n_stations,) or not np.all(
            np.diff(state.s_grid_m) > 0.0
        ):
            raise ValueError("fitted look-ahead grid is invalid")
        if state.initial_probabilities.shape != (n_states,):
            raise ValueError("initial probability shape is invalid")
        if np.any(state.initial_probabilities <= 0.0) or not np.isclose(
            np.sum(state.initial_probabilities), 1.0, atol=1e-10
        ):
            raise ValueError("initial probabilities must be positive and sum to one")
        coefficient_count = state.n_features + 1
        if state.transition_weights.shape != (
            n_states,
            n_states,
            coefficient_count,
        ):
            raise ValueError("transition weight shape is invalid")
        if state.base_coefficients.shape != (
            n_states,
            coefficient_count,
            state.n_stations,
        ):
            raise ValueError("base coefficient shape is invalid")
        if state.autoregressive_coefficients.shape != (
            n_states,
            state.n_stations,
        ):
            raise ValueError("autoregressive coefficient shape is invalid")
        if np.max(np.abs(state.autoregressive_coefficients)) > (
            self.config.maximum_absolute_autoregression + 1e-10
        ):
            raise ValueError("autoregressive stability bound is violated")
        if state.covariances.shape != (
            n_states,
            state.n_stations,
            state.n_stations,
        ):
            raise ValueError("state covariance shape is invalid")
        if state.state_occupancies.shape != (n_states,) or np.any(
            state.state_occupancies < 0.0
        ):
            raise ValueError("state occupancies are invalid")
        for covariance in state.covariances:
            if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
                raise ValueError("state covariance must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) < (
                self.config.minimum_eigenvalue - 1e-10
            ):
                raise ValueError("state covariance violates the eigenvalue floor")
        for array in (
            state.s_grid_m,
            state.initial_probabilities,
            state.transition_weights,
            state.base_coefficients,
            state.autoregressive_coefficients,
            state.covariances,
            state.state_occupancies,
            state.log_likelihood_history,
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError("fitted AIOHMM state contains non-finite values")
        if state.train_sequence_count <= 0:
            raise ValueError("train_sequence_count must be positive")

    def _set_state(self, state: _AIOHMMState) -> None:
        self._validate_state(state)
        self._state = state
        self._cholesky_factors = np.stack(
            [np.linalg.cholesky(covariance) for covariance in state.covariances]
        )

    @staticmethod
    def _previous_observed_values(
        errors: NDArray[np.float64], valid_mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        previous = np.zeros_like(errors, dtype=np.float64)
        if len(errors) > 1:
            previous[1:] = np.where(valid_mask[:-1], errors[:-1], 0.0)
        return previous

    def _sequence_emission_log_probabilities(
        self,
        conditions: NDArray[np.float64],
        errors: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        factor_cache: dict[
            tuple[int, bytes],
            tuple[NDArray[np.int64], NDArray[np.float64], float],
        ],
    ) -> NDArray[np.float64]:
        state = self._require_state()
        time_count = len(conditions)
        base_design = np.column_stack(
            (np.ones(time_count, dtype=np.float64), conditions)
        )
        previous = self._previous_observed_values(errors, valid_mask)
        means = np.einsum(
            "tp,spk->tsk", base_design, state.base_coefficients, optimize=True
        )
        means += (
            previous[:, None, :]
            * state.autoregressive_coefficients[None, :, :]
        )
        log_emission = np.zeros((time_count, state.n_states), dtype=np.float64)
        packed_masks = np.packbits(valid_mask, axis=1, bitorder="little")
        unique_masks, groups = np.unique(
            packed_masks, axis=0, return_inverse=True
        )
        del unique_masks
        log_two_pi = float(np.log(2.0 * np.pi))
        for group_index in range(int(np.max(groups)) + 1):
            frame_indices = np.flatnonzero(groups == group_index)
            observed_indices = np.flatnonzero(valid_mask[frame_indices[0]])
            if len(observed_indices) == 0:
                continue
            mask_key = np.packbits(
                valid_mask[frame_indices[0]], bitorder="little"
            ).tobytes()
            for state_index in range(state.n_states):
                key = (state_index, mask_key)
                cached = factor_cache.get(key)
                if cached is None:
                    covariance = state.covariances[state_index][
                        np.ix_(observed_indices, observed_indices)
                    ]
                    cholesky = np.linalg.cholesky(covariance)
                    normalization = (
                        len(observed_indices) * log_two_pi
                        + 2.0 * float(np.sum(np.log(np.diag(cholesky))))
                    )
                    cached = (observed_indices, cholesky, normalization)
                    factor_cache[key] = cached
                cached_indices, cholesky, normalization = cached
                residuals = (
                    errors[frame_indices][:, cached_indices]
                    - means[frame_indices, state_index][:, cached_indices]
                )
                whitened = np.linalg.solve(cholesky, residuals.T)
                log_emission[frame_indices, state_index] = -0.5 * (
                    normalization + np.sum(whitened**2, axis=0)
                )
        return log_emission

    def _expectation(self, dataset: SequenceDataset) -> _ExpectationResult:
        self._validate_compatible_dataset(dataset)
        state = self._require_state()
        log_initial = np.log(state.initial_probabilities)
        factor_cache: dict[
            tuple[int, bytes],
            tuple[NDArray[np.int64], NDArray[np.float64], float],
        ] = {}
        results: list[ForwardBackwardResult] = []
        sequence_log_probabilities = np.empty(
            dataset.n_sequences, dtype=np.float64
        )
        for sequence_index, raw_length in enumerate(dataset.lengths):
            length = int(raw_length)
            conditions = dataset.conditions[sequence_index, :length].astype(
                np.float64
            )
            errors = dataset.errors[sequence_index, :length].astype(np.float64)
            valid_mask = dataset.valid_mask[sequence_index, :length]
            log_emission = self._sequence_emission_log_probabilities(
                conditions, errors, valid_mask, factor_cache
            )
            log_transition = transition_log_probabilities(
                conditions,
                state.transition_weights,
                input_dependent=self.config.input_dependent_transitions,
            )
            result = forward_backward(
                log_initial, log_transition, log_emission
            )
            results.append(result)
            sequence_log_probabilities[sequence_index] = result.log_probability
        return _ExpectationResult(
            sequence_log_probabilities=sequence_log_probabilities,
            posteriors=tuple(results),
        )

    @staticmethod
    def _flatten_training_arrays(
        dataset: SequenceDataset,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.float64],
    ]:
        conditions: list[NDArray[np.float64]] = []
        errors: list[NDArray[np.float64]] = []
        masks: list[NDArray[np.bool_]] = []
        previous: list[NDArray[np.float64]] = []
        for sequence_index, raw_length in enumerate(dataset.lengths):
            length = int(raw_length)
            sequence_errors = dataset.errors[
                sequence_index, :length
            ].astype(np.float64)
            sequence_mask = dataset.valid_mask[sequence_index, :length]
            conditions.append(
                dataset.conditions[sequence_index, :length].astype(np.float64)
            )
            errors.append(sequence_errors)
            masks.append(sequence_mask)
            previous.append(
                AutoregressiveInputOutputHMM._previous_observed_values(
                    sequence_errors, sequence_mask
                )
            )
        return (
            np.concatenate(conditions, axis=0),
            np.concatenate(errors, axis=0),
            np.concatenate(masks, axis=0),
            np.concatenate(previous, axis=0),
        )

    def _optimize_transition_weights(
        self,
        dataset: SequenceDataset,
        transition_posteriors: tuple[NDArray[np.float64], ...],
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        designs: list[NDArray[np.float64]] = []
        transition_counts: list[NDArray[np.float64]] = []
        for sequence_index, raw_length in enumerate(dataset.lengths):
            length = int(raw_length)
            if length <= 1:
                continue
            conditions = dataset.conditions[
                sequence_index, 1:length
            ].astype(np.float64)
            design = np.column_stack(
                (np.ones(len(conditions), dtype=np.float64), conditions)
            )
            if not self.config.input_dependent_transitions:
                design[:, 1:] = 0.0
            designs.append(design)
            transition_counts.append(transition_posteriors[sequence_index])
        if not designs:
            return initial_weights.copy()
        design = np.concatenate(designs, axis=0)
        counts = np.concatenate(transition_counts, axis=0)
        weights = initial_weights.copy()
        beta_one = 0.9
        beta_two = 0.999
        epsilon = 1e-8

        for source_state in range(self.config.n_states):
            source_counts = counts[:, source_state, :]
            source_mass = np.sum(source_counts, axis=1)
            total_mass = float(np.sum(source_mass))
            if total_mass <= np.finfo(np.float64).eps:
                continue
            current = weights[source_state].copy()
            first_moment = np.zeros_like(current)
            second_moment = np.zeros_like(current)
            best = current.copy()
            best_objective = -np.inf
            for step in range(1, self.config.transition_adam_steps + 1):
                logits = design @ current.T
                log_probabilities = log_softmax(logits, axis=1)
                probabilities = np.exp(log_probabilities)
                objective = float(
                    np.sum(source_counts * log_probabilities) / total_mass
                    - 0.5
                    * self.config.transition_l2_penalty
                    * np.sum(current[:, 1:] ** 2)
                )
                if objective > best_objective:
                    best_objective = objective
                    best = current.copy()
                gradient = (
                    (
                        source_counts
                        - source_mass[:, None] * probabilities
                    ).T
                    @ design
                ) / total_mass
                gradient[:, 1:] -= (
                    self.config.transition_l2_penalty * current[:, 1:]
                )
                first_moment = (
                    beta_one * first_moment + (1.0 - beta_one) * gradient
                )
                second_moment = (
                    beta_two * second_moment
                    + (1.0 - beta_two) * gradient**2
                )
                corrected_first = first_moment / (1.0 - beta_one**step)
                corrected_second = second_moment / (1.0 - beta_two**step)
                current += self.config.transition_learning_rate * (
                    corrected_first / (np.sqrt(corrected_second) + epsilon)
                )
                current -= np.mean(current, axis=0, keepdims=True)
                if not self.config.input_dependent_transitions:
                    current[:, 1:] = 0.0
            weights[source_state] = best
        return weights

    def _m_step(
        self,
        dataset: SequenceDataset,
        state_posteriors: tuple[NDArray[np.float64], ...],
        transition_posteriors: tuple[NDArray[np.float64], ...],
        previous_state: _AIOHMMState | None,
    ) -> _AIOHMMState:
        conditions, errors, valid_mask, previous_values = (
            self._flatten_training_arrays(dataset)
        )
        gamma = np.concatenate(state_posteriors, axis=0)
        n_states = self.config.n_states
        station_count = dataset.n_stations
        base_design = np.column_stack(
            (np.ones(len(conditions), dtype=np.float64), conditions)
        )
        coefficient_count = base_design.shape[1]
        base_coefficients = np.empty(
            (n_states, coefficient_count, station_count), dtype=np.float64
        )
        autoregressive_coefficients = np.empty(
            (n_states, station_count), dtype=np.float64
        )
        covariances = np.empty(
            (n_states, station_count, station_count), dtype=np.float64
        )
        ridge = np.eye(coefficient_count + 1, dtype=np.float64)
        ridge[0, 0] = 0.0
        ridge *= self.config.ridge_penalty

        for state_index in range(n_states):
            state_weights = gamma[:, state_index]
            for station_index in range(station_count):
                observed = valid_mask[:, station_index]
                weights = state_weights[observed]
                effective_count = float(np.sum(weights))
                if (
                    effective_count
                    < self.config.minimum_effective_station_observations
                ):
                    raise ValueError(
                        f"state {state_index} station {station_index} has only "
                        f"{effective_count:.3f} effective observations"
                    )
                station_design = np.column_stack(
                    (
                        base_design[observed],
                        previous_values[observed, station_index],
                    )
                )
                targets = errors[observed, station_index]
                weighted_design = station_design * np.sqrt(weights)[:, None]
                weighted_targets = targets * np.sqrt(weights)
                gram = weighted_design.T @ weighted_design + ridge
                right_hand_side = weighted_design.T @ weighted_targets
                try:
                    coefficients = np.linalg.solve(gram, right_hand_side)
                except np.linalg.LinAlgError:
                    coefficients = np.linalg.lstsq(
                        gram, right_hand_side, rcond=None
                    )[0]
                base_coefficients[state_index, :, station_index] = coefficients[
                    :-1
                ]
                autoregressive_coefficients[state_index, station_index] = np.clip(
                    coefficients[-1],
                    -self.config.maximum_absolute_autoregression,
                    self.config.maximum_absolute_autoregression,
                )

            means = base_design @ base_coefficients[state_index]
            means += (
                previous_values
                * autoregressive_coefficients[state_index][None, :]
            )
            residuals = np.where(valid_mask, errors - means, 0.0)
            validity = valid_mask.astype(np.float64)
            weighted_validity = state_weights[:, None] * validity
            pair_weights = validity.T @ weighted_validity
            weighted_residuals = residuals * np.sqrt(state_weights)[:, None]
            cross_products = weighted_residuals.T @ weighted_residuals
            raw_covariance = np.zeros_like(cross_products)
            sufficient = (
                pair_weights
                >= self.config.minimum_effective_pair_observations
            )
            np.divide(
                cross_products,
                pair_weights,
                out=raw_covariance,
                where=sufficient,
            )
            diagonal = np.diag_indices(station_count)
            if np.any(
                pair_weights[diagonal]
                < self.config.minimum_effective_station_observations
            ):
                raise ValueError(
                    f"state {state_index} has insufficient covariance observations"
                )
            raw_covariance[diagonal] = (
                cross_products[diagonal] / pair_weights[diagonal]
            )
            covariances[state_index] = _regularize_covariance(
                raw_covariance,
                shrinkage=self.config.covariance_shrinkage,
                minimum_eigenvalue=self.config.minimum_eigenvalue,
            )

        initial_counts = np.sum(
            np.stack([posterior[0] for posterior in state_posteriors]), axis=0
        )
        initial_counts += self.config.initial_probability_smoothing
        initial_probabilities = initial_counts / np.sum(initial_counts)
        if previous_state is None:
            transition_weights = np.zeros(
                (n_states, n_states, coefficient_count), dtype=np.float64
            )
            transition_counts = np.sum(
                np.concatenate(transition_posteriors, axis=0), axis=0
            )
            transition_counts += self.config.initial_probability_smoothing
            transition_weights[:, :, 0] = np.log(
                transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
            )
            transition_weights[:, :, 0] -= np.mean(
                transition_weights[:, :, 0], axis=1, keepdims=True
            )
        else:
            transition_weights = previous_state.transition_weights
        transition_weights = self._optimize_transition_weights(
            dataset, transition_posteriors, transition_weights
        )
        occupancies = np.sum(gamma, axis=0)
        occupancies /= np.sum(occupancies)
        return _AIOHMMState(
            feature_names=dataset.feature_names,
            s_grid_m=dataset.s_grid_m.astype(np.float64).copy(),
            initial_probabilities=initial_probabilities,
            transition_weights=transition_weights,
            base_coefficients=base_coefficients,
            autoregressive_coefficients=autoregressive_coefficients,
            covariances=covariances,
            state_occupancies=occupancies,
            log_likelihood_history=np.empty(0, dtype=np.float64),
            train_sequence_count=dataset.n_sequences,
        )

    def _initialize(self, dataset: SequenceDataset) -> _AIOHMMState:
        _, errors, valid_mask, _ = self._flatten_training_arrays(dataset)
        observed_counts = np.count_nonzero(valid_mask, axis=1)
        squared_sum = np.sum(np.where(valid_mask, errors**2, 0.0), axis=1)
        scores = np.sqrt(
            squared_sum / np.maximum(observed_counts, 1)
        )
        rng = np.random.default_rng(self.config.initialization_seed)
        score_scale = max(float(np.std(scores)), 1e-6)
        perturbed_scores = scores + rng.normal(
            scale=0.02 * score_scale, size=len(scores)
        )
        thresholds = np.quantile(
            perturbed_scores,
            np.arange(1, self.config.n_states) / self.config.n_states,
        )
        assignments = np.digitize(perturbed_scores, thresholds)
        smoothing = 0.02
        gamma = np.full(
            (len(assignments), self.config.n_states),
            smoothing / self.config.n_states,
            dtype=np.float64,
        )
        gamma[np.arange(len(assignments)), assignments] += 1.0 - smoothing
        state_posteriors: list[NDArray[np.float64]] = []
        transition_posteriors: list[NDArray[np.float64]] = []
        offset = 0
        for raw_length in dataset.lengths:
            length = int(raw_length)
            sequence_gamma = gamma[offset : offset + length]
            state_posteriors.append(sequence_gamma)
            xi = np.zeros(
                (max(length - 1, 0), self.config.n_states, self.config.n_states),
                dtype=np.float64,
            )
            for time_index in range(length - 1):
                xi[time_index] = np.outer(
                    sequence_gamma[time_index], sequence_gamma[time_index + 1]
                )
                xi[time_index] /= np.sum(xi[time_index])
            transition_posteriors.append(xi)
            offset += length
        return self._m_step(
            dataset,
            tuple(state_posteriors),
            tuple(transition_posteriors),
            previous_state=None,
        )

    def fit(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None = None,
    ) -> FitReport:
        """Fit one deterministic generalized-EM run."""

        self.validate_fit_datasets(train_data, validation_data)
        self.config.validate()
        self._set_state(self._initialize(train_data))
        history: list[float] = []
        best_state: _AIOHMMState | None = None
        best_log_probability = -np.inf
        decrease_count = 0
        converged = False
        warnings: list[str] = []

        for iteration in range(self.config.max_em_iterations):
            expectation = self._expectation(train_data)
            total_log_probability = float(
                np.sum(expectation.sequence_log_probabilities)
            )
            history.append(total_log_probability)
            occupancies = np.sum(
                np.concatenate(
                    [
                        posterior.state_probabilities
                        for posterior in expectation.posteriors
                    ],
                    axis=0,
                ),
                axis=0,
            )
            occupancies /= np.sum(occupancies)
            if np.min(occupancies) < self.config.minimum_state_occupancy_fraction:
                warnings.append(
                    "EM stopped after a hidden state fell below the configured "
                    "occupancy floor"
                )
                break
            current = replace(
                self._require_state(),
                state_occupancies=occupancies,
                log_likelihood_history=np.asarray(history, dtype=np.float64),
            )
            self._set_state(current)
            if total_log_probability > best_log_probability:
                best_log_probability = total_log_probability
                best_state = _copy_state(current)

            if len(history) > 1:
                improvement = history[-1] - history[-2]
                relative_improvement = improvement / max(abs(history[-2]), 1.0)
                if improvement < -1e-7:
                    decrease_count += 1
                if (
                    iteration + 1 >= self.config.min_em_iterations
                    and abs(relative_improvement)
                    < self.config.convergence_tolerance
                ):
                    converged = True
                    break
            if iteration + 1 == self.config.max_em_iterations:
                break

            try:
                updated = self._m_step(
                    train_data,
                    tuple(
                        posterior.state_probabilities
                        for posterior in expectation.posteriors
                    ),
                    tuple(
                        posterior.transition_probabilities
                        for posterior in expectation.posteriors
                    ),
                    previous_state=self._require_state(),
                )
            except ValueError as error:
                warnings.append(f"EM stopped during M-step: {error}")
                break
            self._set_state(updated)

        if best_state is None:
            raise ValueError("AIOHMM training failed before a valid EM state was found")
        best_state = replace(
            best_state,
            log_likelihood_history=np.asarray(history, dtype=np.float64),
        )
        self._set_state(best_state)
        if not converged:
            warnings.append("EM did not reach the configured convergence tolerance")
        if decrease_count:
            warnings.append(
                f"training log likelihood decreased on {decrease_count} EM iterations; "
                "the best observed parameter state was retained"
            )

        observed_train = int(np.count_nonzero(train_data.valid_mask))
        train_log_probability = self.log_probability(train_data)
        metrics: dict[str, float] = {
            "train_nll_per_observed_value_standardized": float(
                -np.sum(train_log_probability) / observed_train
            ),
            "em_best_train_log_probability": best_log_probability,
            "em_iteration_count": float(len(history)),
            "em_converged": float(converged),
            "minimum_state_occupancy_fraction": float(
                np.min(best_state.state_occupancies)
            ),
            "maximum_absolute_autoregressive_coefficient": float(
                np.max(np.abs(best_state.autoregressive_coefficients))
            ),
        }
        if validation_data is not None:
            observed_validation = int(
                np.count_nonzero(validation_data.valid_mask)
            )
            metrics["validation_nll_per_observed_value_standardized"] = float(
                -np.sum(self.log_probability(validation_data))
                / observed_validation
            )
        report = FitReport(
            model_name=self.model_name,
            train_sequence_count=train_data.n_sequences,
            validation_sequence_count=(
                validation_data.n_sequences if validation_data is not None else 0
            ),
            metrics=metrics,
            warnings=tuple(warnings),
        )
        report.validate()
        return report

    def _validate_compatible_dataset(self, dataset: SequenceDataset) -> None:
        dataset.validate()
        state = self._require_state()
        if not dataset.standardized:
            raise ValueError("dataset must be standardized")
        if dataset.feature_names != state.feature_names:
            raise ValueError("dataset feature names/order differ from fitted AIOHMM")
        if dataset.s_grid_m.shape != state.s_grid_m.shape or not np.allclose(
            dataset.s_grid_m, state.s_grid_m, rtol=0.0, atol=1e-10
        ):
            raise ValueError("dataset look-ahead grid differs from fitted AIOHMM")

    def log_probability(self, dataset: SequenceDataset) -> NDArray[np.float64]:
        return self._expectation(dataset).sequence_log_probabilities

    def posterior_state_probabilities(
        self, dataset: SequenceDataset
    ) -> tuple[NDArray[np.float64], ...]:
        """Return smoothed posterior state probabilities for diagnostics."""

        return tuple(
            posterior.state_probabilities.copy()
            for posterior in self._expectation(dataset).posteriors
        )

    @staticmethod
    def _draw_categorical_rows(
        probabilities: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random(len(probabilities))
        return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)

    def sample(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None = None,
    ) -> SampleResult:
        """Generate full autoregressive error profiles on every active frame."""

        state = self._require_state()
        condition_array, length_array, mask_array = self.validate_sample_request(
            conditions,
            lengths,
            n_samples=n_samples,
            seed=seed,
            expected_feature_count=state.n_features,
            expected_station_count=state.n_stations,
            valid_mask=valid_mask,
        )
        if self._cholesky_factors is None:
            raise RuntimeError("AIOHMM covariance factors are unavailable")
        rng = np.random.default_rng(seed)
        values = np.zeros(
            (
                n_samples,
                len(length_array),
                condition_array.shape[1],
                state.n_stations,
            ),
            dtype=np.float32,
        )
        initial_probabilities = np.broadcast_to(
            state.initial_probabilities, (n_samples, state.n_states)
        )
        for sequence_index, raw_length in enumerate(length_array):
            length = int(raw_length)
            hidden_states = self._draw_categorical_rows(
                initial_probabilities, rng
            )
            previous = np.zeros((n_samples, state.n_stations), dtype=np.float64)
            for time_index in range(length):
                condition = condition_array[
                    sequence_index, time_index
                ].astype(np.float64)
                design = np.concatenate(([1.0], condition))
                if time_index > 0:
                    transition_design = design.copy()
                    if not self.config.input_dependent_transitions:
                        transition_design[1:] = 0.0
                    selected_weights = state.transition_weights[hidden_states]
                    logits = np.einsum(
                        "snp,p->sn",
                        selected_weights,
                        transition_design,
                        optimize=True,
                    )
                    probabilities = np.exp(log_softmax(logits, axis=1))
                    hidden_states = self._draw_categorical_rows(
                        probabilities, rng
                    )
                means = np.einsum(
                    "spk,p->sk",
                    state.base_coefficients[hidden_states],
                    design,
                    optimize=True,
                )
                means += (
                    state.autoregressive_coefficients[hidden_states] * previous
                )
                generated = np.empty_like(means)
                for state_index in range(state.n_states):
                    selected = hidden_states == state_index
                    count = int(np.count_nonzero(selected))
                    if count == 0:
                        continue
                    noise = rng.standard_normal((count, state.n_stations))
                    generated[selected] = (
                        means[selected]
                        + noise @ self._cholesky_factors[state_index].T
                    )
                values[:, sequence_index, time_index] = generated.astype(
                    np.float32
                )
                previous = generated
        result = SampleResult(
            values=values,
            lengths=length_array,
            s_grid_m=state.s_grid_m.astype(np.float32),
            standardized=True,
            valid_mask=mask_array,
        )
        result.validate()
        return result

    def save(self, path: str | Path) -> Path:
        state = self._require_state()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.stem}-",
                suffix=".npz",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    schema_version=np.asarray(AIOHMM_MODEL_SCHEMA_VERSION),
                    model_name=np.asarray(self.model_name),
                    config_json=np.asarray(
                        json.dumps(self.config.to_dict(), sort_keys=True)
                    ),
                    feature_names=np.asarray(state.feature_names, dtype=np.str_),
                    s_grid_m=state.s_grid_m,
                    initial_probabilities=state.initial_probabilities,
                    transition_weights=state.transition_weights,
                    base_coefficients=state.base_coefficients,
                    autoregressive_coefficients=(
                        state.autoregressive_coefficients
                    ),
                    covariances=state.covariances,
                    state_occupancies=state.state_occupancies,
                    log_likelihood_history=state.log_likelihood_history,
                    train_sequence_count=np.asarray(state.train_sequence_count),
                )
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with np.load(Path(path), allow_pickle=False) as archive:
            schema_version = str(archive["schema_version"].item())
            if schema_version != AIOHMM_MODEL_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported AIOHMM model schema {schema_version!r}"
                )
            model_name = str(archive["model_name"].item())
            if model_name != "autoregressive_input_output_hmm":
                raise ValueError(f"unexpected persisted model name {model_name!r}")
            raw_config = json.loads(str(archive["config_json"].item()))
            if not isinstance(raw_config, dict):
                raise ValueError("persisted AIOHMM configuration must be an object")
            model = cls(AIOHMMConfig.from_dict(raw_config))
            state = _AIOHMMState(
                feature_names=tuple(
                    str(value) for value in archive["feature_names"].tolist()
                ),
                s_grid_m=np.asarray(archive["s_grid_m"], dtype=np.float64),
                initial_probabilities=np.asarray(
                    archive["initial_probabilities"], dtype=np.float64
                ),
                transition_weights=np.asarray(
                    archive["transition_weights"], dtype=np.float64
                ),
                base_coefficients=np.asarray(
                    archive["base_coefficients"], dtype=np.float64
                ),
                autoregressive_coefficients=np.asarray(
                    archive["autoregressive_coefficients"], dtype=np.float64
                ),
                covariances=np.asarray(
                    archive["covariances"], dtype=np.float64
                ),
                state_occupancies=np.asarray(
                    archive["state_occupancies"], dtype=np.float64
                ),
                log_likelihood_history=np.asarray(
                    archive["log_likelihood_history"], dtype=np.float64
                ),
                train_sequence_count=int(archive["train_sequence_count"].item()),
            )
        model._set_state(state)
        return model
