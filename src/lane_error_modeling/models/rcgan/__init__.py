"""Recurrent conditional GAN model family.

PyTorch remains an optional dependency so importing the NumPy-only Gaussian and
AIOHMM pipelines does not require a deep-learning installation.
"""

from typing import TYPE_CHECKING

from .config import RCGANConfig

if TYPE_CHECKING:
    from .model import RecurrentConditionalGAN


__all__ = ["RCGANConfig", "RecurrentConditionalGAN"]


def __getattr__(name: str):
    if name != "RecurrentConditionalGAN":
        raise AttributeError(name)
    try:
        from .model import RecurrentConditionalGAN
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                "RC-GAN requires PyTorch; install LEEM with the 'rcgan' extra"
            ) from error
        raise
    return RecurrentConditionalGAN
