"""Strict configuration for the recurrent conditional GAN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class RCGANConfig:
    """Architecture and training choices for one deterministic RC-GAN restart.

    The 32-dimensional noise, 64-unit recurrent/dense layers, Adam coefficients,
    learning rate, dropout, and four epochs reproduce the settings reported by
    Arnelid et al.  The paper's final conference architecture used two recurrent
    layers in the deep context and discriminator paths.
    """

    latent_size: int = 32
    noise_hidden_size: int = 64
    context_hidden_size: int = 64
    context_layers: int = 2
    discriminator_hidden_size: int = 64
    discriminator_layers: int = 2
    dense_hidden_size: int = 64
    discriminator_dropout: float = 0.05
    leaky_relu_slope: float = 0.2
    epochs: int = 4
    batch_size: int = 1
    learning_rate: float = 1e-5
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    gradient_clip_norm: float = 1.0
    discriminator_steps: int = 1
    generator_steps: int = 1
    initialization_seed: int = 20260717
    sample_batch_size: int = 16
    device: str = "cpu"

    def validate(self) -> None:
        for name in (
            "latent_size",
            "noise_hidden_size",
            "context_hidden_size",
            "context_layers",
            "discriminator_hidden_size",
            "discriminator_layers",
            "dense_hidden_size",
            "epochs",
            "batch_size",
            "discriminator_steps",
            "generator_steps",
            "sample_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "discriminator_dropout",
            "leaky_relu_slope",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.discriminator_dropout < 1.0:
            raise ValueError("discriminator_dropout must lie in [0, 1)")
        if not 0.0 < self.leaky_relu_slope < 1.0:
            raise ValueError("leaky_relu_slope must lie in (0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if (
            isinstance(self.initialization_seed, bool)
            or not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
        ):
            raise ValueError("initialization_seed must be a non-negative integer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RCGANConfig":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown RC-GAN configuration fields: {sorted(unknown)}")
        config = cls(**dict(raw))
        config.validate()
        return config
