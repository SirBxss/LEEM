from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lane_error_modeling.data.preprocessing import SequenceDataset
from lane_error_modeling.models.rcgan import RCGANConfig


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from lane_error_modeling.models import RecurrentConditionalGAN
    from lane_error_modeling.models.rcgan.architecture import (
        RecurrentDiscriminator,
        RecurrentGenerator,
    )


def _dataset(seed: int, sequence_count: int = 4) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    max_length = 8
    lengths = np.asarray([8, 7, 6, 8][:sequence_count], dtype=np.int32)
    conditions = np.zeros((sequence_count, max_length, 2), dtype=np.float32)
    errors = np.zeros((sequence_count, max_length, 3), dtype=np.float32)
    valid_mask = np.zeros_like(errors, dtype=np.bool_)
    for sequence_index, raw_length in enumerate(lengths):
        length = int(raw_length)
        active_conditions = rng.normal(size=(length, 2))
        active_errors = (
            0.25 * active_conditions[:, :1]
            + rng.normal(scale=0.4, size=(length, 3))
        )
        active_mask = np.ones((length, 3), dtype=np.bool_)
        active_mask[2, 2] = False
        active_mask[4, :] = False
        conditions[sequence_index, :length] = active_conditions
        valid_mask[sequence_index, :length] = active_mask
        errors[sequence_index, :length] = np.where(
            active_mask, active_errors, 0.0
        )
    return SequenceDataset.from_arrays(
        sequence_ids=[f"sequence-{index}" for index in range(sequence_count)],
        conditions=conditions,
        errors=errors,
        valid_mask=valid_mask,
        lengths=lengths,
        feature_names=("condition_a", "condition_b"),
        s_grid_m=np.asarray([0.0, 5.0, 10.0], dtype=np.float32),
        standardized=True,
    )


def _config(**overrides: object) -> RCGANConfig:
    values: dict[str, object] = {
        "latent_size": 4,
        "noise_hidden_size": 6,
        "context_hidden_size": 6,
        "context_layers": 2,
        "discriminator_hidden_size": 6,
        "discriminator_layers": 2,
        "dense_hidden_size": 6,
        "discriminator_dropout": 0.05,
        "leaky_relu_slope": 0.2,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "adam_beta1": 0.5,
        "adam_beta2": 0.999,
        "gradient_clip_norm": 1.0,
        "discriminator_steps": 1,
        "generator_steps": 1,
        "initialization_seed": 31,
        "sample_batch_size": 2,
        "diagnostic_sample_count": 4,
        "device": "cpu",
    }
    values.update(overrides)
    return RCGANConfig.from_dict(values)


class RCGANConfigurationTest(unittest.TestCase):
    def test_configuration_rejects_invalid_values_and_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RCGANConfig(latent_size=0).validate()
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            RCGANConfig(discriminator_dropout=1.0).validate()
        with self.assertRaisesRegex(ValueError, "unknown"):
            RCGANConfig.from_dict({"not_a_parameter": 1})

    def test_discriminator_learning_rate_defaults_and_override(self) -> None:
        shared = RCGANConfig(learning_rate=3e-5)
        asymmetric = RCGANConfig(
            learning_rate=3e-5,
            discriminator_learning_rate=1e-5,
        )
        self.assertEqual(shared.effective_discriminator_learning_rate, 3e-5)
        self.assertEqual(
            asymmetric.effective_discriminator_learning_rate,
            1e-5,
        )
        self.assertEqual(
            RCGANConfig.from_dict(asymmetric.to_dict()),
            asymmetric,
        )
        with self.assertRaisesRegex(ValueError, "discriminator_learning_rate"):
            RCGANConfig(discriminator_learning_rate=0.0).validate()


@unittest.skipUnless(TORCH_AVAILABLE, "RC-GAN tests require the optional PyTorch extra")
class RecurrentConditionalGANTest(unittest.TestCase):
    def test_paper_architecture_has_separate_recurrent_paths_and_valid_shapes(self) -> None:
        config = _config()
        generator = RecurrentGenerator(
            condition_size=2, output_size=3, config=config
        )
        discriminator = RecurrentDiscriminator(
            condition_size=2, target_size=3, config=config
        )
        self.assertIsNot(generator.noise_recurrent, generator.context_recurrent)
        self.assertEqual(generator.noise_recurrent.num_layers, 1)
        self.assertEqual(generator.context_recurrent.num_layers, 2)
        conditions = torch.zeros((2, 7, 2))
        noise = torch.zeros((2, 7, 4))
        targets = generator(noise, conditions)
        logits = discriminator(targets, conditions, torch.ones_like(targets))
        self.assertEqual(tuple(targets.shape), (2, 7, 3))
        self.assertEqual(tuple(logits.shape), (2, 7))

    def test_fit_masked_sequences_and_seeded_sampling(self) -> None:
        dataset = _dataset(10)
        model = RecurrentConditionalGAN(
            _config(discriminator_learning_rate=0.0005)
        )
        report = model.fit(dataset, dataset)
        self.assertTrue(model.is_fitted)
        self.assertFalse(model.capabilities.supports_log_probability)
        self.assertIn("validation_generator_loss", report.metrics)
        self.assertIn("generated_to_observed_std_ratio", report.metrics)
        self.assertIn(
            "validation_discriminator_real_probability", report.metrics
        )
        self.assertIn("train_generator_noise_gradient_norm", report.metrics)
        self.assertIn(
            "train_generator_noise_to_context_gradient_ratio",
            report.metrics,
        )
        self.assertIn(
            "generated_to_observed_std_ratio_station_median",
            report.metrics,
        )
        self.assertGreaterEqual(
            report.metrics["generated_to_observed_std_ratio"], 0.0
        )
        self.assertEqual(len(model.training_history), 1)
        self.assertEqual(
            model.training_history[0]["generator_learning_rate"],
            0.001,
        )
        self.assertEqual(
            model.training_history[0]["discriminator_learning_rate"],
            0.0005,
        )
        first = model.sample(
            dataset.conditions,
            dataset.lengths,
            n_samples=3,
            seed=41,
            valid_mask=dataset.valid_mask,
        )
        second = model.sample(
            dataset.conditions,
            dataset.lengths,
            n_samples=3,
            seed=41,
            valid_mask=dataset.valid_mask,
        )
        third = model.sample(
            dataset.conditions,
            dataset.lengths,
            n_samples=3,
            seed=42,
            valid_mask=dataset.valid_mask,
        )
        np.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(np.array_equal(first.values, third.values))
        self.assertTrue(np.all(first.values[:, ~dataset.time_mask, :] == 0.0))
        unavailable_active = (~dataset.valid_mask) & dataset.time_mask[:, :, None]
        self.assertTrue(np.any(first.values[0][unavailable_active] != 0.0))
        with self.assertRaises(NotImplementedError):
            model.log_probability(dataset)

    def test_safe_persistence_preserves_generated_samples_exactly(self) -> None:
        dataset = _dataset(20)
        model = RecurrentConditionalGAN(_config(initialization_seed=51))
        model.fit(dataset)
        expected = model.sample(
            dataset.conditions, dataset.lengths, n_samples=3, seed=52
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = model.save(Path(temporary_directory) / "rcgan_model.npz")
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(
                    str(archive["model_name"].item()), "recurrent_conditional_gan"
                )
                self.assertEqual(str(archive["schema_version"].item()), "1.1.0")
            restored = RecurrentConditionalGAN.load(path)
        actual = restored.sample(
            dataset.conditions, dataset.lengths, n_samples=3, seed=52
        )
        np.testing.assert_array_equal(expected.values, actual.values)
        self.assertEqual(model.architecture_summary(), restored.architecture_summary())


if __name__ == "__main__":
    unittest.main()
