# Phase 7.2: RC-GAN Stabilization and Validation Gate

## 1. Decision from Phase 7.1

Phase 7.1 completed successfully as a computation, but it did not justify the
three-scenario prototype. The selected $10^{-5}$ candidate had a validation
conditional-diversity ratio of $0.0702$. On the held-out synthetic test set its
empirical 90% interval coverage was only $0.0930$, compared with the
32-sample linear-quantile reference of $0.8455$. Its generated 95th-percentile
absolute error was $0.0398$ m, while the observed value was $0.1360$ m, and it
generated no exceedances of the training-derived 95th-percentile threshold.

The larger learning rates were not acceptable alternatives. At the final epoch,
both $10^{-4}$ and $3\times10^{-4}$ clipped every generator update. Their
validation discriminator real/fake probabilities were approximately
$0.891/0.056$ and $0.923/0.016$, respectively. The discriminator had effectively
separated real and generated sequences, while generator loss was close to seven.

The old $r_{\mathrm{div}}\geq0.05$ check was therefore too weak by itself. Phase
7.2 adds a multi-criterion, validation-only gate. This is an engineering
stabilization phase, not a new thesis model and not a final model comparison.

## 2. Reused data and leakage rule

Phase 7.2 reuses the already generated dataset at
`outputs/synthetic_rcgan_pilot`; it does not generate new observations. The
conditional-Gaussian scenario remains the easiest required capability test.

For each declared candidate, the runner:

1. fits the standardizer and evaluation reference on training data only;
2. trains on the training split and records every completed epoch;
3. evaluates all stability criteria on validation data;
4. rejects candidates that fail any enabled criterion;
5. ranks the remaining candidates by validation dimension-normalized Energy
   Score;
6. opens the test split only if at least one candidate passes.

If all candidates are rejected or fail numerically, the runner persists the
standardizer, evaluation reference, complete candidate table, histories,
validation diagnostics, failure result, and experiment manifest. It does not
load, evaluate, or record provenance for the test split.

## 3. Declared training search

All candidates use the full paper-sized LEEM architecture and one fixed
initialization seed. Only the learning rate changes.

| Setting | Phase 7.2 value | Reason |
|---|---:|---|
| Learning rates | $10^{-5}$, $3\times10^{-5}$, $5\times10^{-5}$ | Search between the stable but collapsed anchor and the unstable $10^{-4}$ run |
| Epochs | 4 | Matches the reported paper setting |
| Batch size | 1 | Restores the paper setting and removes the Phase 7.1 runtime adaptation |
| Latent dimension | 32 | Paper-based LEEM prototype architecture |
| Recurrent/dense hidden size | 64 | Full Phase 7 architecture |
| Selection draws | 64 | More stable interval and distribution diagnostics than the 32-draw pilot |
| Diagnostic draws per epoch | 16 | More stable conditional-diversity estimate |
| Gradient clip norm | 1.0 | Persisted LEEM safety adaptation; clipping frequency is explicitly gated |

This is a deliberately small, declared stabilization search. If it fails, the
correct conclusion is that the current adversarial formulation is not ready for
the expensive prototype—not that more learning rates should be tried against
the same validation set without a new protocol.

## 4. Validation-only stability criteria

Let the last $E=3$ completed epochs be the late-training window.

### 4.1 Conditional diversity

For fixed validation conditions and independent latent-noise sequences,

$$
r_{\mathrm{div}}=
\frac{\operatorname{mean}_{b,t,k}
\operatorname{Std}_{s}(\widehat y^{(s)}_{btk}\mid\mathbf x_{bt})}
{\operatorname{Std}(y_{btk})}.
$$

The candidate must satisfy $r_{\mathrm{div}}\geq0.10$. This rejects the Phase
7.1 selected behavior and remains only a collapse guard, not a calibration
claim.

### 4.2 Persistent generator clipping

Let $c_G^{(e)}$ be the fraction of generator updates whose pre-clipping gradient
norm exceeded 1.0 in epoch $e$. The late mean must satisfy

$$
\frac{1}{E}\sum_{e}c_G^{(e)}\leq0.50.
$$

This rejects candidates whose apparent numerical stability is created mainly by
clipping every update.

### 4.3 Discriminator domination

Let $\bar p_{\mathrm{real}}$ and $\bar p_{\mathrm{fake}}$ be the late-epoch means
of the discriminator probabilities on validation real and generated sequences.
The candidate must satisfy

$$
\left|\bar p_{\mathrm{real}}-\bar p_{\mathrm{fake}}\right|\leq0.75.
$$

The threshold detects extreme separation such as the Phase 7.1 high-rate runs.
It does not require an exact equilibrium at 0.5.

### 4.4 Predictive-interval coverage

With 64 validation draws, let $C_{0.90}$ be empirical coverage of the central
90% interval and $C_{0.90}^{\mathrm{ref}}$ the persisted finite-ensemble linear
quantile reference. The candidate must satisfy

$$
\frac{C_{0.90}}{C_{0.90}^{\mathrm{ref}}}\geq0.50.
$$

This is intentionally a permissive readiness gate. Final calibration claims
still use the common evaluator and substantially more evidence.

### 4.5 Tail exceedance

The evaluation reference fixes the absolute-error 95th-percentile threshold
from training data. If $P_{\mathrm{gen}}$ and $P_{\mathrm{obs}}$ are generated
and observed validation exceedance rates, respectively, the candidate must
satisfy

$$
\frac{P_{\mathrm{gen}}}{P_{\mathrm{obs}}}\geq0.20.
$$

When the observed validation exceedance rate is zero, the ratio is undefined
and the enabled gate fails conservatively.

## 5. Selection and persisted evidence

Only candidates passing all five checks enter Energy-Score ranking. The selected
candidate minimizes validation physical-unit, dimension-normalized Energy Score;
the test set never determines architecture, learning rate, stopping, or gate
thresholds.

`model_selection.json` contains, for every candidate:

- the complete configuration and terminal status;
- every completed epoch record;
- fit warnings and final fit metrics;
- the validation global and interval metrics used by the gates;
- each gate value, threshold, pass/fail result, and rejection reasons;
- the validation Energy Score.

If the gate passes, the normal model, test evaluation, scenario result, plots,
and manifest are saved. If it fails, no `rcgan_model.npz`, `evaluation.json`, or
test plots are created because no model was selected and the test remained
unopened.

## 6. Commands

The pilot data already exist locally; do not regenerate them. From the PyCharm
PowerShell terminal at the project root:

```powershell
python -m pip install -e ".[evaluation,rcgan]"
python -m unittest discover -s tests -v

python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot_v2.json `
  --output results/synthetic/rcgan_pilot_v2
```

The command prints `status=passed` and exits with code 0 when at least one
candidate passes. It prints `status=stability_failed` and exits with code 2 when
all candidates are rejected; that is a scientific gate result, not lost output.
In either case, inspect and push the complete result directory:

```powershell
git add results/synthetic/rcgan_pilot_v2
git commit -m "add Phase 7.2 RC-GAN stabilization results"
git push origin main
```

## 7. Exit criteria and next decision

Phase 7.2 passes only when all tests pass, at least one candidate passes every
enabled validation gate, test access occurs only after selection, and all
candidate histories and gate values are persisted.

If it passes, freeze the selected learning rate and paper-faithful batch size in
the prototype configuration before running all three scenarios. If it fails,
do not run the prototype. The next step must be a documented model/training
revision or a thesis-scope decision, made from training/validation evidence—not
repeated test inspection.

## 8. Files in Phase 7.2

New files:

- `configs/rcgan_experiment_pilot_v2.json`
- `docs/phase7_2_rcgan_stabilization.md`

Updated files:

- `README.md`
- `docs/evaluation_protocol.md`
- `docs/recurrent_conditional_gan.md`
- `scripts/run_rcgan_experiment.py`
- `src/lane_error_modeling/evaluation/__init__.py`
- `src/lane_error_modeling/evaluation/config.py`
- `src/lane_error_modeling/evaluation/rcgan_experiment.py`
- `tests/test_rcgan_experiment.py`
