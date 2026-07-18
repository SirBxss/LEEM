# Phase 7.1: RC-GAN Stability Pilot

## 1. Purpose

The Phase 7 smoke experiment verified the complete RC-GAN software pipeline, but
its generated conditional standard deviation was only about 2.3--2.7% of the
observed standardized target variation. It also generated no 95th- or
99th-percentile exceedances. These results are evidence of severe
under-dispersion, not evidence that the implementation is scientifically ready.

Phase 7.1 prevents a long three-scenario run from being used merely to reproduce
the same collapsed behaviour at a larger scale.

## 2. Core idea

Training stability and distribution quality are separated:

1. epoch diagnostics detect numerical instability, discriminator domination,
   ineffective latent noise, and persistent gradient clipping;
2. a validation-only diversity guard rejects clearly collapsed candidates;
3. candidates that pass the guard are ranked by the already frozen
   dimension-normalized Energy Score;
4. test data remain unopened until one candidate is selected.

The guard does not replace CRPS, Energy Score, interval coverage, tail metrics,
or dependence metrics. It is only an engineering prerequisite.

## 3. Pilot design

The pilot contains only the conditional-Gaussian scenario because it is the
simplest capability test. The data contain 128 training, 32 validation, and 32
test sequences with 50--120 frames. RC-GAN uses the full 32-dimensional latent
input and 64-unit recurrent/dense architecture.

One fixed initialization seed is used to compare three learning rates:

- paper anchor: $10^{-5}$;
- moderate adaptation: $10^{-4}$;
- stronger adaptation: $3\times10^{-4}$.

Each candidate trains for 12 epochs with batch size four. This is a deliberately
small validation experiment, not the final hyperparameter study.

## 4. Stability metric

For fixed validation conditions, draw $S$ independent noise sequences. The mean
pointwise conditional ensemble deviation is

$$
\bar\sigma_{\mathrm{gen}}=
\operatorname{mean}_{b,t,k}
\operatorname{Std}_{s}(\widehat y^{(s)}_{btk}\mid\mathbf x_{bt}).
$$

The scale-free diagnostic is

$$
r_{\mathrm{div}}=
\frac{\bar\sigma_{\mathrm{gen}}}{\operatorname{Std}(y_{btk})}.
$$

Candidates with $r_{\mathrm{div}}<0.05$ are rejected before Energy-Score ranking.
This conservative threshold targets only obvious collapse. It must be reported as
a preselection safeguard, not as a measure of calibration or model superiority.

## 5. Commands

From the project root in the PyCharm PowerShell terminal:

```powershell
python -m pip install -e ".[evaluation,rcgan]"
python -m unittest discover -s tests -v

generate-lane-error-data `
  --config configs/synthetic_rcgan_pilot.json `
  --output outputs/synthetic_rcgan_pilot

python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot.json `
  --output outputs/experiments/rcgan_pilot
```

## 6. Exit criteria

Phase 7.1 passes only when:

1. all unit tests pass;
2. all recorded losses and gradient norms remain finite;
3. at least one candidate has `stability_gate.passed: true`;
4. the selected candidate has $r_{\mathrm{div}}\geq0.05$;
5. gradient clipping is not active for nearly every update across all late epochs;
6. validation discriminator probabilities do not show permanent perfect
   separation;
7. test data were not accessed during selection;
8. the pilot is described as a stability gate, not a thesis comparison result.

Only after these checks should the pilot-selected learning rate be frozen in
`rcgan_experiment_prototype.json` and the model be run on all three scenarios.

## 7. Files in this sub-phase

New files:

- `configs/synthetic_rcgan_pilot.json`
- `configs/rcgan_experiment_pilot.json`
- `docs/phase7_1_rcgan_stability_pilot.md`

Replaced files:

- `README.md`
- `configs/rcgan_experiment_smoke.json`
- `configs/rcgan_experiment_prototype.json`
- `docs/recurrent_conditional_gan.md`
- `src/lane_error_modeling/evaluation/config.py`
- `src/lane_error_modeling/evaluation/rcgan_experiment.py`
- `src/lane_error_modeling/models/rcgan/config.py`
- `src/lane_error_modeling/models/rcgan/__init__.py`
- `src/lane_error_modeling/models/rcgan/model.py`
- `tests/test_rcgan_experiment.py`
- `tests/test_rcgan_model.py`
