# Common Evaluation and Gaussian Experiment Protocol

## 1. Purpose

This phase fixes the experimental rules before AIOHMM and RC-GAN are implemented. All three model families must later receive the same condition sequences, target masks, physical-unit transformation, sample count, metric definitions, random-seed policy, and test split.

The Gaussian is the first model connected to the evaluator. Its synthetic results validate the protocol and establish baseline behaviour; they are not final evidence about BMW perception errors or planner performance.

## 2. Split discipline

The experiment order is fixed:

1. load and verify the training and validation archives;
2. fit the standardizer on training data only;
3. fit the histogram ranges and tail thresholds on physical-unit training errors only;
4. fit every Gaussian candidate on standardized training data;
5. rank candidates using validation NLL per observed standardized target;
6. freeze the selected configuration and refit it on training data;
7. open the test archive for the first time;
8. generate a fixed number of seeded test samples;
9. inverse-transform samples into metres;
10. compute test metrics once and persist all artifacts.

Test data do not influence preprocessing, histogram ranges, tail thresholds, hyperparameters, or candidate selection. The selection artifact explicitly records `test_data_accessed_during_selection: false`.

The model is not refitted on train plus validation after selection. This keeps the preprocessing and model-training contract strictly train-only across all model families. The final BMW protocol may later introduce a separate train-validation refit stage, but only if it is specified identically for all models before the held-out test is opened.

## 3. Physical-unit evaluation

Models operate on standardized errors for numerical stability, but every common sample metric is evaluated in metres. Let $y_n$ be an observed valid scalar error, $x_n^{(s)}$ one of $S$ generated errors, and

$$
\bar x_n=\frac{1}{S}\sum_{s=1}^{S}x_n^{(s)}.
$$

Padding remains exactly zero after inverse transformation. A generated active profile may contain values at stations where the corresponding real target is unavailable; `valid_mask` decides evaluation availability, not generation availability.

## 4. Predictive-mean errors

The global predictive-mean metrics are

$$
\operatorname{MAE}=\frac{1}{N}\sum_{n=1}^{N}|\bar x_n-y_n|,
$$

$$
\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{n=1}^{N}(\bar x_n-y_n)^2}.
$$

Both are also reported separately at every look-ahead station. These metrics evaluate the centre of the predictive distribution but cannot assess spread, calibration, tails, or sample diversity alone.

## 5. Empirical CRPS

For one scalar observation, the empirical continuous ranked probability score is

$$
\operatorname{CRPS}
=\frac{1}{S}\sum_{s=1}^{S}|x^{(s)}-y|
-\frac{1}{2S^2}\sum_{s=1}^{S}\sum_{r=1}^{S}|x^{(s)}-x^{(r)}|.
$$

Lower is better. CRPS jointly rewards calibration and sharpness and is available for Gaussian, AIOHMM, and RC-GAN samples. The implementation uses the equivalent sorted-sample formula, reducing the pair term from quadratic to sorting complexity. Direct brute-force equality is covered by unit tests.

## 6. Multivariate energy score

For an observed error profile $\mathbf y$ and generated profiles $\mathbf x^{(s)}$ over its valid stations,

$$
\operatorname{ES}
=\frac{1}{S}\sum_s\lVert\mathbf x^{(s)}-\mathbf y\rVert_2
-\frac{1}{2S^2}\sum_s\sum_r
\lVert\mathbf x^{(s)}-\mathbf x^{(r)}\rVert_2.
$$

The raw score and a dimension-normalized version, divided by the square root of the number of valid stations, are reported. The normalized score is more comparable when visible range differs between frames.

For bounded runtime, a deterministic subset of frames is selected using the metric seed. A deterministic subset of generated samples is then used for an exact empirical pair calculation within each selected frame. The effective frame and sample counts are stored in `approximation_metadata`.

## 7. Prediction-interval calibration

For nominal levels 50%, 90%, and 95%, the evaluator reports:

- global empirical coverage;
- global mean interval width in metres;
- station-wise coverage;
- station-wise mean width.

Coverage without width is insufficient: arbitrarily wide intervals can appear calibrated. Calibration and sharpness must be interpreted together.

## 8. Jensen-Shannon distances

The marginal signed-error distribution and the within-sequence first-difference distribution are compared using Jensen-Shannon distance. For discrete histogram probabilities $P$ and $Q$ and $M=(P+Q)/2$,

$$
\operatorname{JSD}(P,Q)=
\sqrt{
\frac{1}{2}D_{\mathrm{KL},2}(P\Vert M)
+\frac{1}{2}D_{\mathrm{KL},2}(Q\Vert M)
}.
$$

Base-2 logarithms give a distance in $[0,1]$. Lower is better.

Histogram boundaries are fitted from the configured lower and upper training quantiles and then frozen. Validation/test values outside this range are clipped into the boundary bins. This avoids selecting bin ranges after seeing test data.

Temporal first differences are valid only when both adjacent frames belong to the same sequence and both station targets are observed:

$$
\Delta y_{i,t,k}=y_{i,t,k}-y_{i,t-1,k}.
$$

Padding boundaries and missing endpoints are excluded.

## 9. Tail diagnostics

Absolute-error thresholds, currently the training 95th and 99th percentiles, are frozen in the evaluation reference. On test, the evaluator reports observed and generated exceedance rates and their absolute difference.

It also reports observed/generated test absolute quantiles and their error. The thresholds support leakage-safe exceedance comparison; test quantiles are descriptive outcomes, not tuning criteria.

## 10. Residual dependence

The predictive mean is subtracted from observed and generated errors before dependence analysis. This separates dependence in stochastic residuals from smooth changes caused by the condition sequence.

Two common diagnostics are reported:

- mean absolute difference between observed and generated station-wise lag-one residual correlations;
- root-mean-square difference between observed and generated residual spatial-correlation matrices.

These diagnostics are especially important for the Gaussian baseline. A small marginal-distribution distance can coexist with a serious failure to reproduce temporal persistence. The latent autoregressive and nonlinear synthetic scenarios are expected to reveal this limitation.

Dependence calculations use deterministic bounded frame/sample subsets and record their effective sizes.

## 11. Gaussian density metric and selection

Gaussian candidates form a Cartesian grid of ridge penalties and covariance-shrinkage values. The selected candidate minimizes

$$
-\frac{1}{N_{\mathrm{val}}}
\log p(\mathbf Y_{\mathrm{val}}\mid\mathbf X_{\mathrm{val}}),
$$

where $N_{\mathrm{val}}$ is the number of observed scalar validation targets. Missing dimensions are marginalized by the Gaussian model.

NLL is persisted as a secondary Gaussian diagnostic. It cannot be a primary three-model comparison metric because RC-GAN has no tractable normalized likelihood.

## 12. Synthetic oracle diagnostics

Synthetic archives contain oracle conditional means that are unavailable for real data and prohibited as model inputs. After test evaluation, they are used only to report analytic conditional-mean MAE/RMSE.

For the matching conditional-Gaussian DGP, the known spatial covariance is reconstructed from the generator definition. Relative covariance Frobenius error and spatial-correlation RMSE are then reported. These values test parameter recovery, not real-world validity.

## 13. Deterministic approximations

Full evaluation can contain hundreds of millions of generated scalar values. The protocol therefore caps:

- marginal distribution values;
- energy-score frames and samples;
- residual-dependence frames and samples.

Every subset is selected from a fixed metric seed. Effective counts are stored with results. MAE, RMSE, CRPS, and interval coverage use all observed test targets; only the explicitly documented expensive distribution/dependence operations are bounded.

Prototype inverse transformation retains float32 generated samples and performs one output allocation. This prevents multiple gigabyte-sized float64 intermediates without changing stored physical units.

## 14. Artifacts and provenance

Each scenario output contains:

| Artifact | Purpose |
|---|---|
| `standardizer.json` | Frozen training-only transform |
| `gaussian_model.npz` | Selected model parameters and configuration |
| `evaluation_reference.json` | Training-derived histogram edges and tails |
| `model_selection.json` | Every candidate and validation score |
| `evaluation.json` | Common sample-based test metrics |
| `scenario_result.json` | Combined fit, density, oracle, and common results |
| `plots/*.png` | Deterministic diagnostic figures |

The top-level `experiment_manifest.json` records source configuration and dataset-manifest checksums, split-file checksums, artifact checksums, seeds, selected candidates, and scenario summaries.

Experiment output directories are rejected when non-empty. `--overwrite` is accepted only when the target contains a marker created by this runner.

## 15. Running the experiments

Install the plotting dependency:

```powershell
python -m pip install -e ".[evaluation]"
```

Run the smoke experiment first:

```powershell
python scripts/run_gaussian_experiment.py `
  --config configs/gaussian_experiment_smoke.json `
  --output outputs/experiments/gaussian_smoke
```

After it passes, generate the prototype once:

```powershell
generate-lane-error-data `
  --config configs/synthetic_prototype.json `
  --output outputs/synthetic_prototype
```

Then run the controlled prototype study:

```powershell
python scripts/run_gaussian_experiment.py `
  --config configs/gaussian_experiment_prototype.json `
  --output outputs/experiments/gaussian_prototype
```

Outputs are ignored by Git. Do not commit generated archives, models, figures, result JSON, BMW signal names, or BMW-derived measurements to the public repository.

## 16. Phase exit criteria

Phase 5 is complete when:

1. all unit tests and prior verification scripts pass;
2. smoke selection and evaluation complete for all three scenarios;
3. result and artifact checksums are persisted;
4. the conditional-Gaussian scenario shows plausible oracle recovery;
5. temporal misspecification is visible on autoregressive/nonlinear scenarios;
6. repeated runs with the same inputs and seeds reproduce numerical metrics;
7. no synthetic result is described as final BMW performance or safety evidence.

After these criteria pass, the next implementation phase is AIOHMM using this unchanged evaluator.
