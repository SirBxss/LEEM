"""Recurrent conditional GAN for variable-length lane-error sequences."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as functional
except ModuleNotFoundError as error:  # pragma: no cover - exercised without extra
    if error.name == "torch":
        raise ModuleNotFoundError(
            "RC-GAN requires PyTorch; install LEEM with the 'rcgan' extra"
        ) from error
    raise

from lane_error_modeling.data.preprocessing.batching import (
    SequenceDataset,
    iter_sequence_batches,
)
from lane_error_modeling.models.base import (
    FitReport,
    ModelCapabilities,
    ProbabilisticSequenceModel,
    SampleResult,
)

from .architecture import (
    RecurrentDiscriminator,
    RecurrentGenerator,
    initialize_paper_weights,
)
from .config import RCGANConfig


RCGAN_MODEL_SCHEMA_VERSION = "1.0.0"
LOW_DIVERSITY_RATIO_WARNING = 0.10
SEVERE_DIVERSITY_RATIO_WARNING = 0.05


@dataclass(frozen=True)
class _RCGANState:
    feature_names: tuple[str, ...]
    s_grid_m: NDArray[np.float64]
    train_sequence_count: int
    training_history: tuple[dict[str, float], ...]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_stations(self) -> int:
        return len(self.s_grid_m)


def _masked_sequence_mean(values: Tensor, frame_mask: Tensor) -> Tensor:
    """Average over time per sequence, then equally over eligible sequences."""

    weights = frame_mask.to(dtype=values.dtype)
    frame_counts = torch.sum(weights, dim=1)
    eligible = frame_counts > 0
    if not bool(torch.any(eligible)):
        raise ValueError("a training batch contains no observed target frame")
    per_sequence = torch.sum(values * weights, dim=1) / torch.clamp_min(
        frame_counts, 1.0
    )
    return torch.mean(per_sequence[eligible])


def _finite_scalar(value: Tensor, *, name: str) -> float:
    """Convert a scalar tensor while failing at the first numerical instability."""

    result = float(value.detach().cpu())
    if not np.isfinite(result):
        raise RuntimeError(f"RC-GAN produced a non-finite {name}")
    return result


class RecurrentConditionalGAN(ProbabilisticSequenceModel):
    r"""Conditional recurrent GAN based on Arnelid et al. (2019).

    At every time step the generator processes independent Gaussian noise in a
    one-layer LSTM and conditions in a deeper LSTM.  Their outputs and a raw
    condition skip connection produce the complete spatial error profile.  The
    recurrent discriminator returns one conditional real/fake logit per frame.
    """

    def __init__(self, config: RCGANConfig | None = None) -> None:
        self.config = config or RCGANConfig()
        self.config.validate()
        self._state: _RCGANState | None = None
        self._generator: RecurrentGenerator | None = None
        self._discriminator: RecurrentDiscriminator | None = None
        self._device = self._resolve_device()

    @property
    def model_name(self) -> str:
        return "recurrent_conditional_gan"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_log_probability=False)

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def training_history(self) -> tuple[dict[str, float], ...]:
        return tuple(dict(record) for record in self._require_state().training_history)

    def architecture_summary(self) -> dict[str, int | float | str]:
        """Return the persisted dimensions and paper-relevant architecture choices."""

        state = self._require_state()
        return {
            "condition_size": state.n_features,
            "target_size": state.n_stations,
            "latent_size": self.config.latent_size,
            "noise_hidden_size": self.config.noise_hidden_size,
            "noise_layers": 1,
            "context_hidden_size": self.config.context_hidden_size,
            "context_layers": self.config.context_layers,
            "discriminator_hidden_size": self.config.discriminator_hidden_size,
            "discriminator_layers": self.config.discriminator_layers,
            "dense_hidden_size": self.config.dense_hidden_size,
            "target_mask_input": "enabled_leem_adaptation",
        }

    def _resolve_device(self) -> torch.device:
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RC-GAN device='cuda' requested but CUDA is unavailable")
        return torch.device(self.config.device)

    def _require_state(self) -> _RCGANState:
        if self._state is None:
            raise RuntimeError("RC-GAN has not been fitted")
        return self._state

    def _require_networks(
        self,
    ) -> tuple[RecurrentGenerator, RecurrentDiscriminator]:
        if self._generator is None or self._discriminator is None:
            raise RuntimeError("RC-GAN networks are unavailable")
        return self._generator, self._discriminator

    def _build_networks(self, condition_size: int, target_size: int) -> None:
        torch.manual_seed(self.config.initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.initialization_seed)
        generator = RecurrentGenerator(
            condition_size=condition_size,
            output_size=target_size,
            config=self.config,
        )
        discriminator = RecurrentDiscriminator(
            condition_size=condition_size,
            target_size=target_size,
            config=self.config,
        )
        initialize_paper_weights(generator)
        initialize_paper_weights(discriminator)
        self._generator = generator.to(self._device)
        self._discriminator = discriminator.to(self._device)

    @staticmethod
    def _batch_tensors(batch, device: torch.device) -> tuple[Tensor, ...]:
        conditions = torch.as_tensor(
            batch.conditions, dtype=torch.float32, device=device
        )
        errors = torch.as_tensor(batch.errors, dtype=torch.float32, device=device)
        target_mask = torch.as_tensor(
            batch.valid_mask, dtype=torch.float32, device=device
        )
        time_mask = torch.arange(
            batch.conditions.shape[1], device=device
        )[None, :] < torch.as_tensor(batch.lengths, device=device)[:, None]
        observed_frame_mask = time_mask & torch.any(target_mask > 0.0, dim=-1)
        return conditions, errors, target_mask, observed_frame_mask

    @staticmethod
    def _bce(logits: Tensor, real: bool) -> Tensor:
        target = torch.ones_like(logits) if real else torch.zeros_like(logits)
        return functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )

    def fit(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None = None,
    ) -> FitReport:
        """Train with the non-saturating conditional GAN objective."""

        self.validate_fit_datasets(train_data, validation_data)
        self.config.validate()
        if not np.any(train_data.valid_mask):
            raise ValueError("train_data contains no observed targets")
        self._build_networks(train_data.n_features, train_data.n_stations)
        generator, discriminator = self._require_networks()
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )
        history: list[dict[str, float]] = []

        for epoch in range(self.config.epochs):
            generator.train()
            discriminator.train()
            generator_loss_sum = 0.0
            discriminator_loss_sum = 0.0
            generator_gradient_norm_sum = 0.0
            discriminator_gradient_norm_sum = 0.0
            generator_gradient_update_count = 0
            discriminator_gradient_update_count = 0
            generator_clipped_count = 0
            discriminator_clipped_count = 0
            update_count = 0
            for batch in iter_sequence_batches(
                train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                seed=self.config.initialization_seed + epoch,
            ):
                conditions, real, target_mask, frame_mask = self._batch_tensors(
                    batch, self._device
                )
                if not bool(torch.any(frame_mask)):
                    continue
                batch_shape = (
                    conditions.shape[0],
                    conditions.shape[1],
                    self.config.latent_size,
                )
                latest_discriminator_loss = torch.zeros((), device=self._device)
                for _ in range(self.config.discriminator_steps):
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    noise = torch.randn(batch_shape, device=self._device)
                    with torch.no_grad():
                        fake = generator(noise, conditions)
                    real_logits = discriminator(real, conditions, target_mask)
                    fake_logits = discriminator(fake, conditions, target_mask)
                    latest_discriminator_loss = _masked_sequence_mean(
                        self._bce(real_logits, True) + self._bce(fake_logits, False),
                        frame_mask,
                    )
                    _finite_scalar(
                        latest_discriminator_loss, name="discriminator loss"
                    )
                    latest_discriminator_loss.backward()
                    discriminator_gradient_norm = nn.utils.clip_grad_norm_(
                        discriminator.parameters(), self.config.gradient_clip_norm
                    )
                    discriminator_gradient_norm_value = _finite_scalar(
                        discriminator_gradient_norm,
                        name="discriminator gradient norm",
                    )
                    discriminator_gradient_norm_sum += (
                        discriminator_gradient_norm_value
                    )
                    discriminator_gradient_update_count += 1
                    discriminator_clipped_count += int(
                        discriminator_gradient_norm_value
                        > self.config.gradient_clip_norm
                    )
                    discriminator_optimizer.step()

                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                latest_generator_loss = torch.zeros((), device=self._device)
                for _ in range(self.config.generator_steps):
                    generator_optimizer.zero_grad(set_to_none=True)
                    noise = torch.randn(batch_shape, device=self._device)
                    fake = generator(noise, conditions)
                    fake_logits = discriminator(fake, conditions, target_mask)
                    latest_generator_loss = _masked_sequence_mean(
                        self._bce(fake_logits, True), frame_mask
                    )
                    _finite_scalar(latest_generator_loss, name="generator loss")
                    latest_generator_loss.backward()
                    generator_gradient_norm = nn.utils.clip_grad_norm_(
                        generator.parameters(), self.config.gradient_clip_norm
                    )
                    generator_gradient_norm_value = _finite_scalar(
                        generator_gradient_norm, name="generator gradient norm"
                    )
                    generator_gradient_norm_sum += generator_gradient_norm_value
                    generator_gradient_update_count += 1
                    generator_clipped_count += int(
                        generator_gradient_norm_value > self.config.gradient_clip_norm
                    )
                    generator_optimizer.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

                discriminator_loss_sum += _finite_scalar(
                    latest_discriminator_loss, name="discriminator loss"
                )
                generator_loss_sum += _finite_scalar(
                    latest_generator_loss, name="generator loss"
                )
                update_count += 1
            if update_count == 0:
                raise ValueError("no train batch contained an observed target frame")
            record = {
                "epoch": float(epoch + 1),
                "train_discriminator_loss": discriminator_loss_sum / update_count,
                "train_generator_loss": generator_loss_sum / update_count,
                "train_discriminator_gradient_norm": (
                    discriminator_gradient_norm_sum
                    / discriminator_gradient_update_count
                ),
                "train_generator_gradient_norm": (
                    generator_gradient_norm_sum / generator_gradient_update_count
                ),
                "train_discriminator_gradient_clipped_fraction": (
                    discriminator_clipped_count
                    / discriminator_gradient_update_count
                ),
                "train_generator_gradient_clipped_fraction": (
                    generator_clipped_count / generator_gradient_update_count
                ),
            }
            if validation_data is not None:
                record.update(
                    self._adversarial_diagnostics(
                        validation_data,
                        seed=self.config.initialization_seed + 10_000 + epoch,
                    )
                )
            diagnostic_data = (
                validation_data if validation_data is not None else train_data
            )
            record.update(
                self._diversity_diagnostics(
                    diagnostic_data,
                    seed=self.config.initialization_seed + 20_000 + epoch,
                )
            )
            history.append(record)

        self._state = _RCGANState(
            feature_names=train_data.feature_names,
            s_grid_m=train_data.s_grid_m.astype(np.float64).copy(),
            train_sequence_count=train_data.n_sequences,
            training_history=tuple(history),
        )
        metrics = dict(history[-1])
        warnings: list[str] = []
        diversity_ratio = metrics["generated_to_observed_std_ratio"]
        if diversity_ratio < SEVERE_DIVERSITY_RATIO_WARNING:
            warnings.append(
                "generated conditional diversity is below 5% of observed target "
                "variability; severe under-dispersion or mode collapse is likely"
            )
        elif diversity_ratio < LOW_DIVERSITY_RATIO_WARNING:
            warnings.append(
                "generated conditional diversity is below 10% of observed target "
                "variability; inspect calibration and tails before a longer run"
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

    def _adversarial_diagnostics(
        self, dataset: SequenceDataset, *, seed: int
    ) -> dict[str, float]:
        generator, discriminator = self._require_networks()
        generator.eval()
        discriminator.eval()
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []
        real_probabilities: list[float] = []
        fake_probabilities: list[float] = []
        random = torch.Generator(device=self._device)
        random.manual_seed(seed)
        with torch.no_grad():
            for batch in iter_sequence_batches(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            ):
                conditions, real, target_mask, frame_mask = self._batch_tensors(
                    batch, self._device
                )
                if not bool(torch.any(frame_mask)):
                    continue
                noise = torch.randn(
                    (
                        conditions.shape[0],
                        conditions.shape[1],
                        self.config.latent_size,
                    ),
                    generator=random,
                    device=self._device,
                )
                fake = generator(noise, conditions)
                real_logits = discriminator(real, conditions, target_mask)
                fake_logits = discriminator(fake, conditions, target_mask)
                discriminator_losses.append(
                    float(
                        _masked_sequence_mean(
                            self._bce(real_logits, True)
                            + self._bce(fake_logits, False),
                            frame_mask,
                        ).cpu()
                    )
                )
                generator_losses.append(
                    float(
                        _masked_sequence_mean(
                            self._bce(fake_logits, True), frame_mask
                        ).cpu()
                    )
                )
                real_probabilities.append(
                    float(
                        _masked_sequence_mean(
                            torch.sigmoid(real_logits), frame_mask
                        ).cpu()
                    )
                )
                fake_probabilities.append(
                    float(
                        _masked_sequence_mean(
                            torch.sigmoid(fake_logits), frame_mask
                        ).cpu()
                    )
                )
        if not generator_losses:
            raise ValueError("validation data contain no observed target frame")
        return {
            "validation_discriminator_loss": float(np.mean(discriminator_losses)),
            "validation_generator_loss": float(np.mean(generator_losses)),
            "validation_discriminator_real_probability": float(
                np.mean(real_probabilities)
            ),
            "validation_discriminator_fake_probability": float(
                np.mean(fake_probabilities)
            ),
        }

    def _diversity_diagnostics(
        self, dataset: SequenceDataset, *, seed: int
    ) -> dict[str, float]:
        """Measure response to latent noise at fixed validation conditions.

        The numerator is the mean pointwise ensemble standard deviation.  It is
        divided by the observed standardized target deviation only to obtain a
        scale-free collapse indicator; it is not a replacement for calibration
        or proper-score evaluation.
        """

        generator, _ = self._require_networks()
        generator.eval()
        random = torch.Generator(device=self._device)
        random.manual_seed(seed)
        generated_standard_deviation_sum = 0.0
        observed_value_count = 0
        with torch.no_grad():
            for batch in iter_sequence_batches(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            ):
                conditions, _, target_mask, frame_mask = self._batch_tensors(
                    batch, self._device
                )
                if not bool(torch.any(frame_mask)):
                    continue
                sample_count = self.config.diagnostic_sample_count
                repeated_conditions = conditions.repeat((sample_count, 1, 1))
                noise = torch.randn(
                    (
                        sample_count * conditions.shape[0],
                        conditions.shape[1],
                        self.config.latent_size,
                    ),
                    generator=random,
                    device=self._device,
                )
                generated = generator(noise, repeated_conditions).reshape(
                    sample_count,
                    conditions.shape[0],
                    conditions.shape[1],
                    target_mask.shape[-1],
                )
                pointwise_standard_deviation = torch.std(
                    generated, dim=0, correction=0
                )
                observed = target_mask > 0.0
                generated_standard_deviation_sum += float(
                    torch.sum(pointwise_standard_deviation[observed]).cpu()
                )
                observed_value_count += int(torch.sum(observed).cpu())
        if observed_value_count == 0:
            raise ValueError("diagnostic data contain no observed target value")
        observed_values = np.asarray(
            dataset.errors[dataset.valid_mask], dtype=np.float64
        )
        observed_standard_deviation = float(np.std(observed_values))
        if not np.isfinite(observed_standard_deviation) or observed_standard_deviation <= 0.0:
            raise ValueError(
                "diagnostic target standard deviation must be finite and positive"
            )
        generated_standard_deviation = (
            generated_standard_deviation_sum / observed_value_count
        )
        return {
            "diagnostic_sample_count": float(self.config.diagnostic_sample_count),
            "generated_mean_standard_deviation_standardized": float(
                generated_standard_deviation
            ),
            "observed_standard_deviation_standardized": observed_standard_deviation,
            "generated_to_observed_std_ratio": float(
                generated_standard_deviation / observed_standard_deviation
            ),
        }

    def _validate_schema(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_] | None]:
        state = self._require_state()
        return self.validate_sample_request(
            conditions,
            lengths,
            n_samples=n_samples,
            seed=seed,
            expected_feature_count=state.n_features,
            expected_station_count=state.n_stations,
            valid_mask=valid_mask,
        )

    def sample(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None = None,
    ) -> SampleResult:
        """Generate complete standardized spatial profiles with local RNG state."""

        condition_array, length_array, mask_array = self._validate_schema(
            conditions,
            lengths,
            n_samples=n_samples,
            seed=seed,
            valid_mask=valid_mask,
        )
        state = self._require_state()
        generator, _ = self._require_networks()
        generator.eval()
        condition_tensor = torch.as_tensor(
            condition_array, dtype=torch.float32, device=self._device
        )
        random = torch.Generator(device=self._device)
        random.manual_seed(seed)
        values = np.zeros(
            (
                n_samples,
                condition_array.shape[0],
                condition_array.shape[1],
                state.n_stations,
            ),
            dtype=np.float32,
        )
        with torch.no_grad():
            for start in range(0, n_samples, self.config.sample_batch_size):
                count = min(self.config.sample_batch_size, n_samples - start)
                repeated_conditions = condition_tensor.repeat((count, 1, 1))
                noise = torch.randn(
                    (
                        count * condition_array.shape[0],
                        condition_array.shape[1],
                        self.config.latent_size,
                    ),
                    generator=random,
                    device=self._device,
                )
                generated = generator(noise, repeated_conditions)
                values[start : start + count] = (
                    generated.reshape(
                        count,
                        condition_array.shape[0],
                        condition_array.shape[1],
                        state.n_stations,
                    )
                    .cpu()
                    .numpy()
                )
        time_mask = (
            np.arange(condition_array.shape[1])[None, :] < length_array[:, None]
        )
        values *= time_mask[None, :, :, None]
        result = SampleResult(
            values=values,
            lengths=length_array,
            s_grid_m=state.s_grid_m.astype(np.float32),
            standardized=True,
            valid_mask=mask_array,
        )
        result.validate()
        return result

    @staticmethod
    def _network_arrays(
        prefix: str, network: nn.Module
    ) -> tuple[dict[str, NDArray[np.generic]], NDArray[np.str_]]:
        arrays: dict[str, NDArray[np.generic]] = {}
        names: list[str] = []
        for index, (name, value) in enumerate(network.state_dict().items()):
            names.append(name)
            arrays[f"{prefix}_{index:04d}"] = value.detach().cpu().numpy()
        return arrays, np.asarray(names, dtype=np.str_)

    @staticmethod
    def _restore_network(
        archive: np.lib.npyio.NpzFile,
        *,
        prefix: str,
        names: NDArray[np.str_],
        network: nn.Module,
    ) -> None:
        state_dict: dict[str, Tensor] = {}
        for index, name in enumerate(names.tolist()):
            key = f"{prefix}_{index:04d}"
            if key not in archive:
                raise ValueError(f"persisted RC-GAN is missing array {key!r}")
            state_dict[str(name)] = torch.from_numpy(np.asarray(archive[key]).copy())
        network.load_state_dict(state_dict, strict=True)

    def save(self, path: str | Path) -> Path:
        """Atomically persist both networks as safe named NumPy arrays."""

        state = self._require_state()
        generator, discriminator = self._require_networks()
        generator_arrays, generator_names = self._network_arrays(
            "generator", generator
        )
        discriminator_arrays, discriminator_names = self._network_arrays(
            "discriminator", discriminator
        )
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
                    schema_version=np.asarray(RCGAN_MODEL_SCHEMA_VERSION),
                    model_name=np.asarray(self.model_name),
                    config_json=np.asarray(
                        json.dumps(self.config.to_dict(), sort_keys=True)
                    ),
                    feature_names=np.asarray(state.feature_names, dtype=np.str_),
                    s_grid_m=state.s_grid_m,
                    train_sequence_count=np.asarray(state.train_sequence_count),
                    training_history_json=np.asarray(
                        json.dumps(state.training_history, sort_keys=True)
                    ),
                    generator_parameter_names=generator_names,
                    discriminator_parameter_names=discriminator_names,
                    **generator_arrays,
                    **discriminator_arrays,
                )
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a fitted model without pickle deserialization."""

        with np.load(Path(path), allow_pickle=False) as archive:
            schema_version = str(archive["schema_version"].item())
            if schema_version != RCGAN_MODEL_SCHEMA_VERSION:
                raise ValueError(f"unsupported RC-GAN model schema {schema_version!r}")
            if str(archive["model_name"].item()) != "recurrent_conditional_gan":
                raise ValueError("persisted artifact is not an LEEM RC-GAN")
            raw_config = json.loads(str(archive["config_json"].item()))
            if not isinstance(raw_config, dict):
                raise ValueError("persisted RC-GAN configuration must be an object")
            model = cls(RCGANConfig.from_dict(raw_config))
            history_raw = json.loads(str(archive["training_history_json"].item()))
            if not isinstance(history_raw, list):
                raise ValueError("persisted RC-GAN history must be a list")
            state = _RCGANState(
                feature_names=tuple(
                    str(value) for value in archive["feature_names"].tolist()
                ),
                s_grid_m=np.asarray(archive["s_grid_m"], dtype=np.float64),
                train_sequence_count=int(archive["train_sequence_count"].item()),
                training_history=tuple(
                    {str(key): float(value) for key, value in record.items()}
                    for record in history_raw
                ),
            )
            if state.train_sequence_count <= 0 or not state.feature_names:
                raise ValueError("persisted RC-GAN schema metadata are invalid")
            if not np.all(np.diff(state.s_grid_m) > 0):
                raise ValueError("persisted RC-GAN look-ahead grid is invalid")
            model._build_networks(state.n_features, state.n_stations)
            generator, discriminator = model._require_networks()
            cls._restore_network(
                archive,
                prefix="generator",
                names=np.asarray(
                    archive["generator_parameter_names"], dtype=np.str_
                ),
                network=generator,
            )
            cls._restore_network(
                archive,
                prefix="discriminator",
                names=np.asarray(
                    archive["discriminator_parameter_names"], dtype=np.str_
                ),
                network=discriminator,
            )
        model._state = state
        generator.eval()
        discriminator.eval()
        return model
