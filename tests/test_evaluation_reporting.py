from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lane_error_modeling.evaluation.comparison import (
    DEFAULT_COMPARISON_METRICS,
    compare_experiment_results,
    save_comparison_report,
)
from lane_error_modeling.evaluation.finite_sample import (
    central_order_statistic_reference,
    finite_ensemble_interval_metadata,
    linear_quantile_uniform_reference_coverage,
)
from lane_error_modeling.evaluation.migration import (
    add_finite_ensemble_metadata,
    upgrade_evaluation_tree,
)


def _evaluation_payload(model_name: str, scale: float) -> dict[str, object]:
    metrics = {
        metric.key: scale * float(index + 1)
        for index, metric in enumerate(DEFAULT_COMPARISON_METRICS)
    }
    return {
        "schema_version": "1.0.0",
        "model_name": model_name,
        "scenario": "latent_autoregressive",
        "sample_count": 64,
        "sequence_count": 4,
        "observed_value_count": 100,
        "s_grid_m": [0.0, 5.0, 10.0],
        "global_metrics": metrics,
        "interval_metrics": {
            "0.90": {
                "nominal_coverage": 0.9,
                "global_empirical_coverage": 0.87,
            }
        },
        "approximation_metadata": {},
    }


def _write_experiment(root: Path, model_name: str, scale: float) -> None:
    scenario = root / "latent_autoregressive"
    scenario.mkdir(parents=True)
    (scenario / "evaluation.json").write_text(
        json.dumps(_evaluation_payload(model_name, scale)), encoding="utf-8"
    )
    (scenario / "evaluation_reference.json").write_text(
        json.dumps({"schema_version": "1.0.0", "fitted_split": "train"}),
        encoding="utf-8",
    )


class FiniteSampleReportingTest(unittest.TestCase):
    def test_linear_quantile_reference_for_64_draws(self) -> None:
        self.assertAlmostEqual(
            linear_quantile_uniform_reference_coverage(64, 0.50),
            31.5 / 65.0,
        )
        self.assertAlmostEqual(
            linear_quantile_uniform_reference_coverage(64, 0.90),
            56.7 / 65.0,
        )
        self.assertAlmostEqual(
            linear_quantile_uniform_reference_coverage(64, 0.95),
            59.85 / 65.0,
        )

    def test_rank_reference_reports_unattainable_high_coverage(self) -> None:
        attainable = central_order_statistic_reference(64, 0.95)
        self.assertTrue(attainable["meets_nominal_coverage"])
        self.assertEqual(attainable["lower_rank_1_based"], 1)
        self.assertEqual(attainable["upper_rank_1_based"], 63)
        self.assertAlmostEqual(attainable["reference_coverage"], 62.0 / 65.0)

        unattainable = central_order_statistic_reference(64, 0.99)
        self.assertFalse(unattainable["meets_nominal_coverage"])
        self.assertAlmostEqual(unattainable["reference_coverage"], 63.0 / 65.0)

    def test_metadata_does_not_mislabel_reference_as_nominal(self) -> None:
        metadata = finite_ensemble_interval_metadata(
            sample_count=64,
            nominal_coverage=0.9,
            empirical_coverage=0.87,
        )
        self.assertEqual(metadata["quantile_method"], "linear")
        self.assertAlmostEqual(
            metadata["empirical_minus_linear_uniform_reference"],
            0.87 - 56.7 / 65.0,
        )
        self.assertIn("diagnostic only", metadata["interpretation"])

    def test_existing_results_upgrade_without_samples(self) -> None:
        payload = _evaluation_payload("baseline", 1.0)
        upgraded = add_finite_ensemble_metadata(payload)
        self.assertEqual(upgraded["schema_version"], "1.1.0")
        finite = upgraded["interval_metrics"]["0.90"]["finite_ensemble"]
        self.assertEqual(finite["ensemble_sample_count"], 64)
        self.assertNotIn(
            "finite_ensemble", payload["interval_metrics"]["0.90"]
        )

    def test_comparison_requires_same_reference_and_writes_three_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_experiment(baseline, "gaussian", 2.0)
            _write_experiment(candidate, "aiohmm", 1.0)
            report = compare_experiment_results(
                baseline_root=baseline,
                candidate_root=candidate,
            )
            self.assertEqual(report["scenario_count"], 1)
            first = report["rows"][0]
            self.assertEqual(first["candidate_improvement_percent"], 50.0)
            self.assertEqual(first["better_model"], "aiohmm")
            paths = save_comparison_report(report, root / "comparison")
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))

            reference = (
                candidate
                / "latent_autoregressive"
                / "evaluation_reference.json"
            )
            reference.write_text('{"different": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same reference"):
                compare_experiment_results(
                    baseline_root=baseline,
                    candidate_root=candidate,
                )

    def test_tree_upgrade_is_read_only_unless_write_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evaluation = root / "scenario" / "evaluation.json"
            evaluation.parent.mkdir()
            original = json.dumps(_evaluation_payload("baseline", 1.0))
            evaluation.write_text(original, encoding="utf-8")
            self.assertEqual(upgrade_evaluation_tree(root), (evaluation,))
            self.assertEqual(evaluation.read_text(encoding="utf-8"), original)
            upgrade_evaluation_tree(root, write=True)
            upgraded = json.loads(evaluation.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["schema_version"], "1.1.0")


if __name__ == "__main__":
    unittest.main()
