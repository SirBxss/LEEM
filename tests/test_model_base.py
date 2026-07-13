from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from numpy.typing import ArrayLike

from lane_error_modeling.data.preprocessing import SequenceDataset, SequenceStandardizer
from lane_error_modeling.data.synthetic.config import SyntheticDatasetConfig
from lane_error_modeling.data.synthetic.generator import generate_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES
from lane_error_modeling.models import (
    FitReport,
    ModelCapabilities,
    ProbabilisticSequenceModel,
    SampleResult,
)

from test_config import minimal_config_dict


def _datasets() -> tuple[SequenceDataset, SequenceDataset]:
    config = SyntheticDatasetConfig.from_dict(minimal_config_dict())
    padded = generate_dataset(config, "conditional_gaussian", "train")
    raw = SequenceDataset.from_arrays(
        sequence_ids=padded.sequence_ids,
        conditions=padded.conditions,
        errors=padded.errors,
        valid_mask=padded.valid_mask,
        lengths=padded.lengths,
        feature_names=FEATURE_NAMES,
        s_grid_m=padded.s_grid_m,
    )
    standardizer = SequenceStandardizer().fit(
        raw.conditions,
        raw.errors,
        raw.valid_mask,
        raw.lengths,
        split_name="train",
        feature_names=raw.feature_names,
        s_grid_m=raw.s_grid_m,
    )
    return raw, raw.standardized_copy(standardizer)


class _DummyModel(ProbabilisticSequenceModel):
    def __init__(self) -> None:
        self._is_fitted = False
        self._n_features = 0
        self._s_grid_m = np.empty(0, dtype=np.float32)

    @property
    def model_name(self) -> str:
        return "dummy"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_log_probability=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        train_data: SequenceDataset,
        validation_data: SequenceDataset | None = None,
    ) -> FitReport:
        self.validate_fit_datasets(train_data, validation_data)
        self._is_fitted = True
        self._n_features = train_data.n_features
        self._s_grid_m = train_data.s_grid_m
        report = FitReport(
            model_name=self.model_name,
            train_sequence_count=train_data.n_sequences,
            validation_sequence_count=(
                validation_data.n_sequences if validation_data is not None else 0
            ),
            metrics={"dummy_objective": 0.0},
        )
        report.validate()
        return report

    def sample(
        self,
        conditions: ArrayLike,
        lengths: ArrayLike,
        *,
        n_samples: int,
        seed: int,
        valid_mask: ArrayLike | None = None,
    ) -> SampleResult:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        condition_array, length_array, mask_array = self.validate_sample_request(
            conditions,
            lengths,
            n_samples=n_samples,
            seed=seed,
            expected_feature_count=self._n_features,
            expected_station_count=len(self._s_grid_m),
            valid_mask=valid_mask,
        )
        result = SampleResult(
            values=np.zeros(
                (
                    n_samples,
                    condition_array.shape[0],
                    condition_array.shape[1],
                    len(self._s_grid_m),
                ),
                dtype=np.float32,
            ),
            lengths=length_array,
            s_grid_m=self._s_grid_m,
            standardized=True,
            valid_mask=mask_array,
        )
        result.validate()
        return result

    def save(self, path: str | Path) -> Path:
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "_DummyModel":
        del path
        return cls()


class ModelBaseTest(unittest.TestCase):
    def test_fit_rejects_raw_data_and_accepts_standardized_data(self) -> None:
        raw, standardized = _datasets()
        model = _DummyModel()
        with self.assertRaisesRegex(ValueError, "must be standardized"):
            model.fit(raw)
        report = model.fit(standardized)
        self.assertTrue(model.is_fitted)
        self.assertEqual(report.train_sequence_count, standardized.n_sequences)

    def test_sample_contract_and_default_log_probability(self) -> None:
        _, standardized = _datasets()
        model = _DummyModel()
        model.fit(standardized)
        result = model.sample(
            standardized.conditions,
            standardized.lengths,
            n_samples=3,
            seed=5,
            valid_mask=standardized.valid_mask,
        )
        self.assertEqual(
            result.values.shape,
            (
                3,
                standardized.n_sequences,
                standardized.max_length,
                standardized.n_stations,
            ),
        )
        with self.assertRaises(NotImplementedError):
            model.log_probability(standardized)

