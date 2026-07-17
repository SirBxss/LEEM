"""Probabilistic lane-estimation error model families."""

from .base import (
    FitReport,
    ModelCapabilities,
    ProbabilisticSequenceModel,
    SampleResult,
)
from .gaussian import ConditionalMultivariateGaussian, GaussianConfig
from .aiohmm import AIOHMMConfig, AutoregressiveInputOutputHMM
from .rcgan import RCGANConfig

__all__ = [
    "AIOHMMConfig",
    "AutoregressiveInputOutputHMM",
    "FitReport",
    "ConditionalMultivariateGaussian",
    "GaussianConfig",
    "ModelCapabilities",
    "ProbabilisticSequenceModel",
    "RCGANConfig",
    "RecurrentConditionalGAN",
    "SampleResult",
]


def __getattr__(name: str):
    if name != "RecurrentConditionalGAN":
        raise AttributeError(name)
    from .rcgan import RecurrentConditionalGAN

    return RecurrentConditionalGAN
