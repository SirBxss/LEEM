"""Common physical-unit evaluation for Gaussian, AIOHMM, and RC-GAN."""

from .config import (
    AIOHMMExperimentConfig,
    AIOHMMSearchSpace,
    EvaluationConfig,
    GaussianExperimentConfig,
    GaussianSearchSpace,
    RCGANExperimentConfig,
    RCGANSearchSpace,
    RCGANStabilityChecks,
)
from .reference import EvaluationReference
from .result import EvaluationResult
from .finite_sample import (
    central_order_statistic_reference,
    finite_ensemble_interval_metadata,
    linear_quantile_uniform_reference_coverage,
)
from .metrics import evaluate_probabilistic_samples
from .comparison import compare_experiment_results, save_comparison_report
from .migration import add_finite_ensemble_metadata, upgrade_evaluation_tree
from .gaussian_experiment import run_gaussian_experiment
from .aiohmm_experiment import run_aiohmm_experiment

__all__ = [
    "AIOHMMExperimentConfig",
    "AIOHMMSearchSpace",
    "add_finite_ensemble_metadata",
    "EvaluationConfig",
    "EvaluationReference",
    "EvaluationResult",
    "central_order_statistic_reference",
    "compare_experiment_results",
    "evaluate_probabilistic_samples",
    "finite_ensemble_interval_metadata",
    "linear_quantile_uniform_reference_coverage",
    "run_gaussian_experiment",
    "run_aiohmm_experiment",
    "run_rcgan_experiment",
    "save_comparison_report",
    "upgrade_evaluation_tree",
    "GaussianExperimentConfig",
    "GaussianSearchSpace",
    "RCGANExperimentConfig",
    "RCGANSearchSpace",
    "RCGANStabilityChecks",
]


def __getattr__(name: str):
    if name != "run_rcgan_experiment":
        raise AttributeError(name)
    from .rcgan_experiment import run_rcgan_experiment

    return run_rcgan_experiment
