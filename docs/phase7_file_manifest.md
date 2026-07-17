# Phase 7 File Manifest

Copy the archive contents into the LEEM project root while preserving paths.

## New files

- `configs/rcgan_experiment_smoke.json`
- `configs/rcgan_experiment_prototype.json`
- `docs/recurrent_conditional_gan.md`
- `docs/phase7_file_manifest.md`
- `scripts/run_rcgan_experiment.py`
- `src/lane_error_modeling/evaluation/rcgan_experiment.py`
- `src/lane_error_modeling/models/rcgan/__init__.py`
- `src/lane_error_modeling/models/rcgan/architecture.py`
- `src/lane_error_modeling/models/rcgan/config.py`
- `src/lane_error_modeling/models/rcgan/model.py`
- `tests/test_rcgan_experiment.py`
- `tests/test_rcgan_model.py`

## Replace existing files

- `.gitignore`
- `README.md`
- `pyproject.toml`
- `docs/evaluation_protocol.md`
- `docs/preprocessing_and_model_contract.md`
- `src/lane_error_modeling/__init__.py`
- `src/lane_error_modeling/evaluation/__init__.py`
- `src/lane_error_modeling/evaluation/config.py`
- `src/lane_error_modeling/models/__init__.py`

## Verification

From the project root in the PyCharm PowerShell terminal:

```powershell
python -m pip install -e ".[evaluation,rcgan]"
python -m unittest discover -s tests -v
python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_smoke.json `
  --output outputs/experiments/rcgan_smoke
```

Expected unit-test count after integration is 55. The smoke command must finish
with `RC-GAN experiment passed` and create `experiment_manifest.json` with
`status: passed`. Do not begin the long prototype run until both gates pass.
