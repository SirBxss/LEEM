# Conditional Multivariate Gaussian Baseline

## 1. Scientific role

The conditional multivariate Gaussian is the thesis baseline against which AIOHMM and RC-GAN are compared. It is intentionally interpretable and limited: the current condition vector controls the mean error profile, one full covariance matrix represents spatial dependence across look-ahead stations, and frames are conditionally independent over time.

This is stronger than independent station-wise Gaussian noise because it preserves cross-station dependence. It remains a baseline because it has no latent regimes, autoregressive state, nonlinear conditional mapping, condition-dependent covariance, heavy tails, or multimodality.

## 2. Required data

For sequence $i$, time $t$, condition dimension $F$, and station count $K$, fitting requires:

| Quantity | Shape | Use |
|---|---:|---|
| Standardized conditions $\tilde{\mathbf X}_{i,t}$ | $[B,T,F]$ | Conditional mean predictors |
| Standardized errors $\tilde{\mathbf Y}_{i,t}$ | $[B,T,K]$ | Regression targets and covariance residuals |
| Validity mask $\mathbf M_{i,t}$ | $[B,T,K]$ | Excludes unavailable stations and padding |
| Sequence lengths | $[B]$ | Identifies active frames |
| Feature names | $[F]$ | Prevents silent input reordering |
| Look-ahead grid | $[K]$ | Prevents incompatible station mappings |

The common `SequenceStandardizer` must be fitted on the training split only. Validation and test data reuse the frozen transform. Model fitting rejects non-standardized datasets.

## 3. Conditional mean

Let the augmented input be

\[
\mathbf z_{i,t}=
\begin{bmatrix}
1 & \tilde{\mathbf X}_{i,t}^{\mathsf T}
\end{bmatrix}^{\mathsf T}
\in\mathbb R^{F+1}.
\]

The model is

\[
\tilde{\mathbf Y}_{i,t}\mid\tilde{\mathbf X}_{i,t}
\sim
\mathcal N
\left(
\mathbf B^{\mathsf T}\mathbf z_{i,t},
\boldsymbol\Sigma
\right).
\]

Because visible range can vary, station $k$ is fitted only from its observed training set

\[
\mathcal O_k=\{(i,t):M_{i,t,k}=1\}.
\]

The coefficient vector is estimated by masked ridge regression:

\[
\hat{\boldsymbol\beta}_k=
\arg\min_{\boldsymbol\beta}
\sum_{(i,t)\in\mathcal O_k}
\left(
\tilde Y_{i,t,k}-\mathbf z_{i,t}^{\mathsf T}\boldsymbol\beta
\right)^2
+\lambda\lVert\boldsymbol\beta_{1:F}\rVert_2^2.
\]

The intercept is not penalized. Separate masks avoid treating serialized zeros as measurements and avoid discarding near-range observations merely because a far station is unavailable.

## 4. Residual spatial covariance with missing targets

Observed residuals are

\[
r_{i,t,k}=
\tilde Y_{i,t,k}-
\mathbf z_{i,t}^{\mathsf T}\hat{\boldsymbol\beta}_k.
\]

For stations $k$ and $\ell$, define their joint observation set

\[
\mathcal O_{k\ell}=
\{(i,t):M_{i,t,k}=1\land M_{i,t,\ell}=1\}.
\]

The raw covariance entry is the residual cross-second moment

\[
\hat\Sigma^{\mathrm{pair}}_{k\ell}=
\frac{1}{|\mathcal O_{k\ell}|}
\sum_{(i,t)\in\mathcal O_{k\ell}}
r_{i,t,k}r_{i,t,\ell}.
\]

If an off-diagonal pair has fewer than the configured minimum number of joint observations, that raw entry is set to zero and a fit warning is recorded. Diagonal entries require the stricter per-station minimum and are never silently omitted.

Pairwise covariance estimation preserves more data than complete-case deletion, but different entries are estimated from different frame sets. The assembled matrix is therefore not guaranteed to be positive semidefinite. Two documented regularization steps are applied.

First, diagonal shrinkage is used:

\[
\boldsymbol\Sigma^{\mathrm{shrink}}=
(1-\alpha)\hat{\boldsymbol\Sigma}^{\mathrm{pair}}
+\alpha\operatorname{diag}
\left(\hat{\boldsymbol\Sigma}^{\mathrm{pair}}\right).
\]

Second, with eigendecomposition

\[
\boldsymbol\Sigma^{\mathrm{shrink}}
=\mathbf Q\operatorname{diag}(d_1,\ldots,d_K)\mathbf Q^{\mathsf T},
\]

the final covariance is

\[
\hat{\boldsymbol\Sigma}=
\mathbf Q
\operatorname{diag}(\max(d_1,\epsilon),\ldots,\max(d_K,\epsilon))
\mathbf Q^{\mathsf T}.
\]

This projection guarantees a positive-definite covariance for Cholesky sampling and likelihood evaluation. The number of clipped eigenvalues is recorded in the fit report.

## 5. Likelihood under partial observation

For one frame, let $\mathcal V_{i,t}$ contain its observed station indices. Missing dimensions are marginalized, not imputed. The contribution is the Gaussian density of the observed subvector:

\[
\log p
\left(
\tilde{\mathbf Y}_{i,t,\mathcal V}
\mid\tilde{\mathbf X}_{i,t}
\right)
=
-\frac{1}{2}
\left[
|\mathcal V|\log(2\pi)
+\log|\hat{\boldsymbol\Sigma}_{\mathcal V,\mathcal V}|
+\mathbf r_{\mathcal V}^{\mathsf T}
\hat{\boldsymbol\Sigma}_{\mathcal V,\mathcal V}^{-1}
\mathbf r_{\mathcal V}
\right].
\]

Frame contributions are summed to obtain a per-sequence log probability. Cholesky factors are cached for repeated mask patterns. Reported split NLL is divided by the number of observed scalar targets, because sequences and visible ranges differ in length.

NLL remains a secondary diagnostic: RC-GAN has no tractable normalized likelihood, so it cannot be a primary three-model ranking metric.

## 6. Sampling

For each active condition frame:

\[
\tilde{\mathbf Y}^{(s)}_{i,t}=
\hat{\mathbf B}^{\mathsf T}\mathbf z_{i,t}
+\mathbf L\boldsymbol\epsilon^{(s)}_{i,t},
\qquad
\boldsymbol\epsilon^{(s)}_{i,t}\sim\mathcal N(\mathbf 0,\mathbf I),
\]

where $\mathbf L\mathbf L^{\mathsf T}=\hat{\boldsymbol\Sigma}$. The seed fully determines the samples. Sampling returns full station profiles on active frames even where a real target is unavailable; `valid_mask` remains evaluation metadata. Sequence padding is zero.

Samples must be inverse-transformed before physical-unit evaluation. When retaining full generated profiles, inverse transformation should use the time mask to keep padding zero without hiding generated far-range stations.

## 7. Default configuration

| Parameter | Default | Meaning |
|---|---:|---|
| `ridge_penalty` | 0.001 | Stabilizes station-wise mean regressions |
| `covariance_shrinkage` | 0.10 | Weight assigned to the diagonal covariance target |
| `minimum_eigenvalue` | $10^{-6}$ | Positive-definite eigenvalue floor in standardized units |
| `minimum_station_observations` | 32 | Required observations for every station mean/variance |
| `minimum_pair_observations` | 32 | Required overlap for an off-diagonal raw covariance |

The effective station minimum is also required to be at least two more than the number of fitted coefficients, leaving residual information after mean fitting. These are implementation defaults, not final BMW-data hyperparameters. Final values must be chosen using training/validation data only and reported with sensitivity analysis, especially for covariance shrinkage.

## 8. Persisted artifact

The compressed NumPy artifact contains no pickled objects. It records:

- model schema and stable model name;
- complete Gaussian configuration;
- ordered feature names and look-ahead grid;
- fitted coefficient matrix;
- raw pairwise covariance;
- regularized positive-definite covariance;
- pairwise observation counts;
- training sequence count.

The model artifact depends on the separate persisted training standardizer. Both are required for reproducibility and later planner use.

## 9. Verification protocol

Unit tests cover:

1. recovery of known linear coefficients and covariance;
2. finite positive-definite covariance after masked fitting;
3. equality between implemented partial-observation likelihood and direct marginal calculation;
4. deterministic sampling and zero padding;
5. persistence without changed likelihoods or samples;
6. rejection of stations with inadequate data.

The smoke verification fits the same baseline independently to all three synthetic DGPs. Passing all three means only that fitting, likelihood, sampling, masking, and persistence are numerically compatible. It does not mean the Gaussian assumptions fit all scenarios and must not be presented as a final model ranking.

## 10. Evaluation and expected diagnostics

Common cross-model evaluation later uses physical-unit sample metrics. Gaussian-specific diagnostics additionally include:

- masked conditional-mean RMSE;
- per-observed-value NLL;
- covariance eigenvalues and condition number;
- pairwise observation-count matrix;
- standardized residual marginal shape;
- residual temporal autocorrelation;
- residual spatial correlation;
- coverage by station and condition bin.

Persistent residual autocorrelation would motivate AIOHMM/RC-GAN. Heavy tails, multimodality, condition-dependent spread, or systematic coverage errors expose baseline misspecification rather than an implementation failure.

## 11. Assumptions and limitations

- The conditional mean is linear in standardized inputs.
- Spatial covariance is constant across conditions.
- Frames are conditionally independent; sequence grouping is retained for splitting and evaluation but not used in the Gaussian likelihood dynamics.
- The response is unimodal and light-tailed.
- Missing targets are assumed sufficiently explainable by observed conditions for pairwise residual estimation; non-ignorable missingness can bias covariance.
- Ridge, shrinkage, and eigenvalue flooring introduce bias in exchange for numerical stability.
- Per-observed-value NLLs are comparable only when target definition, scaling, masks, and station grid are held fixed.
- Synthetic success cannot establish real BMW sensor behaviour, planner benefit, calibration, or safety performance.

These limitations define the scientific reason for comparing the baseline with AIOHMM and RC-GAN.
