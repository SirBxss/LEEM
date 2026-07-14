"""Masked conditional multivariate Gaussian lane-error baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lane_error_modeling.data.preprocessing.batching import SequenceDataset
from lane_error_modeling.models.base import (
    FitReport,
    ModelCapabilities,
    ProbabilisticSequenceModel,
    SampleResult,
)

from .config import GaussianConfig


GAUSSIAN_MODEL_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class _GaussianState:
    """Fitted arrays and schema metadata kept separate from hyperparameters."""

    feature_names: tuple[str, ...]
    s_grid_m: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    raw_pairwise_covariance: NDArray[np.float64]
    covariance: NDArray[np.float64]
    pair_observation_counts: NDArray[np.int64]
    train_sequence_count: int

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_stations(self) -> int:
        return len(self.s_grid_m)


class ConditionalMultivariateGaussian(ProbabilisticSequenceModel):
    r"""Homoscedastic spatial Gaussian conditioned on the current frame.

    The baseline models

    .. math::
        Y_t \mid X_t \sim \mathcal N(B^\mathsf{T}[1, X_t], \Sigma),

    independently over time conditional on the inputs. Each station's mean
    coefficients are estimated by masked ridge regression. Residual covariance
    entries use every pair's jointly observed frames, followed by diagonal
    shrinkage and an eigenvalue-floor positive-definite projection.
    """

    def __init__(self, config: GaussianConfig | None = None) -> None:
        self.config = config or GaussianConfig()
        self.config.validate()
        self._state: _GaussianState | None = None
        self._cholesky: NDArray[np.float64] | None = None

    @property
    def model_name(self) -> str:
        return "conditional_multivariate_gaussian"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_log_probability=True)

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Return a copy of the fitted ``[intercept + features, stations]`` matrix."""

        return self._require_state().coefficients.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Return a copy of the regularized spatial covariance matrix."""

        return self._require_state().covariance.copy()

    @property
    def raw_pairwise_covariance(self) -> NDArray[np.float64]:
        """Return the pre-shrinkage pairwise residual second-moment matrix."""

        return self._require_state().raw_pairwise_covariance.copy()

    @property
    def pair_observation_counts(self) -> NDArray[np.int64]:
        """Return jointly observed residual counts for every station pair."""

        return self._require_state().pair_observation_counts.copy()

    def _require_state(self) -> _GaussianState:
        if self._state is None:
            raise RuntimeError("Gaussian model has not been fitted")
        return self._state

    def _set_state(self, state: _GaussianState) -> None:
        self._validate_state(state)
        self._state = state
        self._cholesky = np.linalg.cholesky(state.covariance)

    def _validate_state(self, state: _GaussianState) -> None:
        if not state.feature_names or len(set(state.feature_names)) != state.n_features:
            raise ValueError("fitted feature names must be non-empty and unique")
        if state.s_grid_m.shape != (state.n_stations,):
            raise ValueError("invalid fitted look-ahead grid shape")
        if not np.all(np.isfinite(state.s_grid_m)) or not np.all(
            np.diff(state.s_grid_m) > 0
        ):
            raise ValueError("fitted look-ahead grid must be finite and increasing")
        expected_coefficients = (state.n_features + 1, state.n_stations)
        if state.coefficients.shape != expected_coefficients:
            raise ValueError("invalid fitted coefficient shape")
        expected_square = (state.n_stations, state.n_stations)
        for name, matrix in (
            ("raw_pairwise_covariance", state.raw_pairwise_covariance),
            ("covariance", state.covariance),
        ):
            if matrix.shape != expected_square:
                raise ValueError(f"invalid {name} shape")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} contains non-finite values")
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
                raise ValueError(f"{name} must be symmetric")
        if state.pair_observation_counts.shape != expected_square:
            raise ValueError("invalid pair observation count shape")
        if np.any(state.pair_observation_counts < 0):
            raise ValueError("pair observation counts must be non-negative")
        if not np.all(np.isfinite(state.coefficients)):
            raise ValueError("fitted coefficients contain non-finite values")
        if state.train_sequence_count <= 0:
            raise ValueError("train_sequence_count must be positive")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(state.covariance)))
        tolerance = max(1e-12, self.config.minimum_eigenvalue * 1e-7)
        if minimum_eigenvalue < self.config.minimum_eigenvalue - tolerance:
            raise ValueError("regularized covariance violates its eigenvalue floor")

    def fit(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None = None,
    ) -> FitReport:
        """Fit masked linear means and one full residual spatial covariance."""

        self.validate_fit_datasets(train_data, validation_data)
        self.config.validate()
        conditions = train_data.conditions[train_data.time_mask].astype(np.float64)
        errors = train_data.errors[train_data.time_mask].astype(np.float64)
        valid_mask = train_data.valid_mask[train_data.time_mask]
        design = np.column_stack(
            (np.ones(len(conditions), dtype=np.float64), conditions)
        )
        station_count = train_data.n_stations
        coefficient_count = design.shape[1]
        coefficients = np.empty(
            (coefficient_count, station_count), dtype=np.float64
        )
        residuals = np.zeros_like(errors, dtype=np.float64)

        ridge = np.eye(coefficient_count, dtype=np.float64)
        ridge[0, 0] = 0.0
        ridge *= self.config.ridge_penalty
        station_counts = np.count_nonzero(valid_mask, axis=0).astype(np.int64)
        required_station_observations = max(
            self.config.minimum_station_observations,
            coefficient_count + 2,
        )
        for station_index, count in enumerate(station_counts):
            if count < required_station_observations:
                raise ValueError(
                    f"station {station_index} has {count} observations; "
                    f"at least {required_station_observations} are required"
                )
            observed = valid_mask[:, station_index]
            station_design = design[observed]
            station_targets = errors[observed, station_index]
            gram = station_design.T @ station_design + ridge
            right_hand_side = station_design.T @ station_targets
            try:
                station_coefficients = np.linalg.solve(gram, right_hand_side)
            except np.linalg.LinAlgError:
                station_coefficients = np.linalg.lstsq(
                    gram, right_hand_side, rcond=None
                )[0]
            coefficients[:, station_index] = station_coefficients
            residuals[observed, station_index] = (
                station_targets - station_design @ station_coefficients
            )

        observed_as_integer = valid_mask.astype(np.int64)
        pair_counts = observed_as_integer.T @ observed_as_integer
        cross_products = residuals.T @ residuals
        raw_covariance = np.zeros_like(cross_products)
        sufficiently_observed = (
            pair_counts >= self.config.minimum_pair_observations
        )
        np.divide(
            cross_products,
            pair_counts,
            out=raw_covariance,
            where=sufficiently_observed,
        )
        diagonal = np.diag_indices(station_count)
        raw_covariance[diagonal] = (
            cross_products[diagonal] / pair_counts[diagonal]
        )
        insufficient_pair_count = int(
            np.count_nonzero(np.triu(~sufficiently_observed, k=1))
        )
        del observed_as_integer, cross_products, residuals

        diagonal = np.diag(np.diag(raw_covariance))
        shrunk_covariance = (
            (1.0 - self.config.covariance_shrinkage) * raw_covariance
            + self.config.covariance_shrinkage * diagonal
        )
        shrunk_covariance = 0.5 * (shrunk_covariance + shrunk_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk_covariance)
        clipped = eigenvalues < self.config.minimum_eigenvalue
        repaired_eigenvalues = np.maximum(
            eigenvalues, self.config.minimum_eigenvalue
        )
        covariance = (eigenvectors * repaired_eigenvalues) @ eigenvectors.T
        covariance = 0.5 * (covariance + covariance.T)

        state = _GaussianState(
            feature_names=train_data.feature_names,
            s_grid_m=train_data.s_grid_m.astype(np.float64).copy(),
            coefficients=coefficients,
            raw_pairwise_covariance=raw_covariance,
            covariance=covariance,
            pair_observation_counts=pair_counts,
            train_sequence_count=train_data.n_sequences,
        )
        self._set_state(state)

        metrics: dict[str, float] = {
            "train_mean_rmse_standardized": self._masked_mean_rmse(train_data),
            "train_nll_per_observed_value_standardized": self._nll_per_observed_value(
                train_data
            ),
            "covariance_min_eigenvalue": float(np.min(repaired_eigenvalues)),
            "covariance_condition_number": float(
                np.max(repaired_eigenvalues) / np.min(repaired_eigenvalues)
            ),
            "covariance_clipped_eigenvalue_count": float(np.count_nonzero(clipped)),
        }
        if validation_data is not None:
            metrics["validation_mean_rmse_standardized"] = self._masked_mean_rmse(
                validation_data
            )
            if np.any(validation_data.valid_mask):
                metrics[
                    "validation_nll_per_observed_value_standardized"
                ] = self._nll_per_observed_value(validation_data)

        warnings: list[str] = []
        if insufficient_pair_count:
            warnings.append(
                f"{insufficient_pair_count} station pairs had too few joint "
                "observations; their raw off-diagonal covariance was set to zero"
            )
        if np.any(clipped):
            warnings.append(
                f"{int(np.count_nonzero(clipped))} covariance eigenvalues were "
                "raised to the configured numerical floor"
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
        state = self._require_state()
        dataset.validate()
        if not dataset.standardized:
            raise ValueError("dataset must be standardized")
        if dataset.feature_names != state.feature_names:
            raise ValueError("dataset feature names/order differ from the fitted model")
        if dataset.n_features != state.n_features:
            raise ValueError("dataset condition dimension differs from the fitted model")
        if dataset.s_grid_m.shape != state.s_grid_m.shape or not np.allclose(
            dataset.s_grid_m,
            state.s_grid_m,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("dataset look-ahead grid differs from the fitted model")

    def predict_mean(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
    ) -> NDArray[np.float32]:
        """Predict standardized conditional means with zero-valued padding."""

        state = self._require_state()
        condition_array, length_array, _ = self.validate_sample_request(
            conditions,
            lengths,
            n_samples=1,
            seed=0,
            expected_feature_count=state.n_features,
            expected_station_count=state.n_stations,
        )
        time_mask = (
            np.arange(condition_array.shape[1])[None, :] < length_array[:, None]
        )
        mean = np.zeros(
            (*condition_array.shape[:2], state.n_stations), dtype=np.float64
        )
        active_conditions = condition_array[time_mask].astype(np.float64)
        active_design = np.column_stack(
            (np.ones(len(active_conditions), dtype=np.float64), active_conditions)
        )
        mean[time_mask] = active_design @ state.coefficients
        return mean.astype(np.float32)

    def _masked_mean_rmse(self, dataset: SequenceDataset) -> float:
        self._validate_compatible_dataset(dataset)
        if not np.any(dataset.valid_mask):
            raise ValueError("dataset has no observed targets")
        mean = self.predict_mean(dataset.conditions, dataset.lengths)
        residual = dataset.errors[dataset.valid_mask] - mean[dataset.valid_mask]
        return float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))

    def _nll_per_observed_value(self, dataset: SequenceDataset) -> float:
        observed_count = int(np.count_nonzero(dataset.valid_mask))
        if observed_count == 0:
            raise ValueError("dataset has no observed targets")
        return float(-np.sum(self.log_probability(dataset)) / observed_count)

    def log_probability(self, dataset: SequenceDataset) -> NDArray[np.float64]:
        """Return per-sequence log density marginalized to each observed subset."""

        self._validate_compatible_dataset(dataset)
        state = self._require_state()
        mean = self.predict_mean(dataset.conditions, dataset.lengths).astype(np.float64)
        result = np.zeros(dataset.n_sequences, dtype=np.float64)
        log_two_pi = float(np.log(2.0 * np.pi))

        time_mask = dataset.time_mask
        active_masks = dataset.valid_mask[time_mask]
        active_residuals = dataset.errors[time_mask].astype(np.float64) - mean[
            time_mask
        ]
        sequence_indices = np.repeat(
            np.arange(dataset.n_sequences, dtype=np.int64),
            dataset.lengths.astype(np.int64),
        )
        packed_masks = np.packbits(active_masks, axis=1, bitorder="little")
        _, mask_group = np.unique(packed_masks, axis=0, return_inverse=True)

        for group_index in range(int(np.max(mask_group)) + 1):
            frame_indices = np.flatnonzero(mask_group == group_index)
            observed_indices = np.flatnonzero(active_masks[frame_indices[0]])
            if len(observed_indices) == 0:
                continue
            marginal_covariance = state.covariance[
                np.ix_(observed_indices, observed_indices)
            ]
            cholesky = np.linalg.cholesky(marginal_covariance)
            normalization = (
                len(observed_indices) * log_two_pi
                + 2.0 * float(np.sum(np.log(np.diag(cholesky))))
            )
            residuals = active_residuals[frame_indices][:, observed_indices]
            whitened = np.linalg.solve(cholesky, residuals.T)
            frame_log_probabilities = -0.5 * (
                normalization + np.sum(whitened**2, axis=0)
            )
            result += np.bincount(
                sequence_indices[frame_indices],
                weights=frame_log_probabilities,
                minlength=dataset.n_sequences,
            )
        return result

    def sample(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None = None,
    ) -> SampleResult:
        """Draw full spatial profiles; ``valid_mask`` remains evaluation metadata."""

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
        time_mask = (
            np.arange(condition_array.shape[1])[None, :] < length_array[:, None]
        )
        mean = self.predict_mean(condition_array, length_array).astype(np.float64)
        active_frame_count = int(np.count_nonzero(time_mask))
        standard_noise = np.random.default_rng(seed).standard_normal(
            size=(n_samples, active_frame_count, state.n_stations)
        )
        if self._cholesky is None:
            raise RuntimeError("Gaussian model covariance factor is unavailable")
        generated = standard_noise @ self._cholesky.T
        generated += mean[time_mask][None, :, :]
        values = np.zeros(
            (
                n_samples,
                condition_array.shape[0],
                condition_array.shape[1],
                state.n_stations,
            ),
            dtype=np.float32,
        )
        values[:, time_mask, :] = generated.astype(np.float32)
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
        """Atomically persist configuration, schema metadata, and fitted arrays."""

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
                    schema_version=np.asarray(GAUSSIAN_MODEL_SCHEMA_VERSION),
                    model_name=np.asarray(self.model_name),
                    config_json=np.asarray(
                        json.dumps(self.config.to_dict(), sort_keys=True)
                    ),
                    feature_names=np.asarray(state.feature_names, dtype=np.str_),
                    s_grid_m=state.s_grid_m,
                    coefficients=state.coefficients,
                    raw_pairwise_covariance=state.raw_pairwise_covariance,
                    covariance=state.covariance,
                    pair_observation_counts=state.pair_observation_counts,
                    train_sequence_count=np.asarray(state.train_sequence_count),
                )
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a fitted model without enabling pickle deserialization."""

        with np.load(Path(path), allow_pickle=False) as archive:
            schema_version = str(archive["schema_version"].item())
            if schema_version != GAUSSIAN_MODEL_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported Gaussian model schema {schema_version!r}"
                )
            model_name = str(archive["model_name"].item())
            if model_name != "conditional_multivariate_gaussian":
                raise ValueError(f"unexpected persisted model name {model_name!r}")
            raw_config = json.loads(str(archive["config_json"].item()))
            if not isinstance(raw_config, dict):
                raise ValueError("persisted Gaussian configuration must be an object")
            model = cls(config=GaussianConfig.from_dict(raw_config))
            state = _GaussianState(
                feature_names=tuple(
                    str(value) for value in archive["feature_names"].tolist()
                ),
                s_grid_m=np.asarray(archive["s_grid_m"], dtype=np.float64),
                coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                raw_pairwise_covariance=np.asarray(
                    archive["raw_pairwise_covariance"], dtype=np.float64
                ),
                covariance=np.asarray(archive["covariance"], dtype=np.float64),
                pair_observation_counts=np.asarray(
                    archive["pair_observation_counts"], dtype=np.int64
                ),
                train_sequence_count=int(archive["train_sequence_count"].item()),
            )
        model._set_state(state)
        return model
