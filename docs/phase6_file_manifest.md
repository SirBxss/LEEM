# Phase 6 File Manifest

Copy the archive contents into the LEEM project root while preserving paths.

## New files

- `configs/aiohmm_experiment_smoke.json`
- `configs/aiohmm_experiment_prototype.json`
- `docs/autoregressive_input_output_hmm.md`
- `docs/phase6_smoke_results.md`
- `docs/phase6_file_manifest.md`
- `scripts/run_aiohmm_experiment.py`
- `src/lane_error_modeling/evaluation/aiohmm_experiment.py`
- `src/lane_error_modeling/models/aiohmm/__init__.py`
- `src/lane_error_modeling/models/aiohmm/config.py`
- `src/lane_error_modeling/models/aiohmm/inference.py`
- `src/lane_error_modeling/models/aiohmm/model.py`
- `tests/test_aiohmm_experiment.py`
- `tests/test_aiohmm_inference.py`
- `tests/test_aiohmm_model.py`

## Replace existing files

- `README.md`
- `pyproject.toml`
- `docs/evaluation_protocol.md`
- `docs/preprocessing_and_model_contract.md`
- `src/lane_error_modeling/__init__.py`
- `src/lane_error_modeling/evaluation/__init__.py`
- `src/lane_error_modeling/evaluation/config.py`
- `src/lane_error_modeling/models/__init__.py`

## Verification

From the project root in the PyCharm terminal:

```powershell
python -m pip install -e ".[evaluation]"
python -m unittest discover -s tests -v
python scripts/run_aiohmm_experiment.py `
  --config configs/aiohmm_experiment_smoke.json `
  --output outputs/experiments/aiohmm_smoke
```

Expected unit-test count after integration is 43. The smoke command must finish
with `AIOHMM experiment passed` and create an `experiment_manifest.json`.
