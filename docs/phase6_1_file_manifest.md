# Phase 6.1 File Manifest and Application Steps

## Files added

- `src/lane_error_modeling/evaluation/finite_sample.py`
- `src/lane_error_modeling/evaluation/comparison.py`
- `src/lane_error_modeling/evaluation/migration.py`
- `scripts/compare_experiments.py`
- `scripts/upgrade_evaluation_results.py`
- `tests/test_evaluation_reporting.py`
- `docs/phase6_1_evaluation_reporting.md`
- `docs/phase6_1_file_manifest.md`

## Files updated

- `.gitignore`
- `README.md`
- `pyproject.toml`
- `src/lane_error_modeling/__init__.py`
- `src/lane_error_modeling/evaluation/__init__.py`
- `src/lane_error_modeling/evaluation/metrics.py`
- `src/lane_error_modeling/evaluation/result.py`
- `tests/test_evaluation.py`
- `docs/evaluation_protocol.md`

## Apply and verify in PowerShell

Extract the changes archive into the LEEM project root and allow the listed
files to be replaced. Then run:

```powershell
python -m pip install -e ".[evaluation]"
python -m unittest discover -s tests -v
```

The expected result is 49 passing tests.

Upgrade the six already committed prototype evaluation files without
retraining:

```powershell
python scripts/upgrade_evaluation_results.py `
  --root results/synthetic `
  --write
```

Create the checked Gaussian-versus-AIOHMM comparison:

```powershell
python scripts/compare_experiments.py `
  --baseline results/synthetic/gaussian_prototype `
  --candidate results/synthetic/aiohmm_prototype `
  --output results/synthetic/gaussian_vs_aiohmm
```

Copy the small fitted models from the local output tree into the curated
result tree. This permits future diagnostics without refitting AIOHMM:

```powershell
Get-ChildItem outputs/experiments/gaussian_prototype `
  -Recurse -Filter gaussian_model.npz | ForEach-Object {
    $scenario = $_.Directory.Name
    Copy-Item $_.FullName `
      "results/synthetic/gaussian_prototype/$scenario/gaussian_model.npz"
  }

Get-ChildItem outputs/experiments/aiohmm_prototype `
  -Recurse -Filter aiohmm_model.npz | ForEach-Object {
    $scenario = $_.Directory.Name
    Copy-Item $_.FullName `
      "results/synthetic/aiohmm_prototype/$scenario/aiohmm_model.npz"
  }
```

Confirm that there is no local-identifier leakage:

```powershell
Get-ChildItem results -Recurse -Include *.json,*.md,*.csv |
  Select-String "q679381|C:/Users"
```

The command should return no matches. Review and push:

```powershell
git status --short
git add .gitignore README.md pyproject.toml src scripts tests docs results
git commit -m "add finite-sample reporting and model comparison"
git push
```

Do not add `outputs/` or raw synthetic datasets.
