# Preprocessing and Common Probabilistic Model Contract

## 1. Purpose

Gaussian, AIOHMM, and RC-GAN must be compared using the same target definition, conditions, splits, masks, and physical-unit evaluation. This module establishes one model-independent boundary between data construction and statistical modelling.

The shared layer has three responsibilities:

1. represent variable-length sequences without losing missing-target information;
2. fit and persist leakage-safe numerical transformations;
3. enforce a common lifecycle and sampling shape for all model families.

Model-specific preprocessing is avoided unless an architecture mathematically requires it and the difference is reported as part of the method.

## 2. Model-facing sequence data

For \(B\) sequences, maximum padded length \(T\), \(F=6\) conditions, and \(K=21\) look-ahead stations, `SequenceDataset` contains:

| Field | Shape | Definition |
|---|---:|---|
| `sequence_ids` | \([B]\) | Stable independent sequence identities |
| `conditions` | \([B,T,F]\) | Conditional inputs |
| `errors` | \([B,T,K]\) | Lateral error profiles |
| `valid_mask` | \([B,T,K]\) | Observed target entries |
| `lengths` | \([B]\) | Unpadded sequence lengths |
| `feature_names` | \([F]\) metadata | Ordered semantic feature names |
| `s_grid_m` | \([K]\) | Ordered look-ahead stations |

The temporal activity mask is

\[
M^{\mathrm{time}}_{i,t}=\mathbb 1(t<L_i).
\]

The target mask \(M^{\mathrm{target}}_{i,t,k}\) must be false for all padding frames. A stored numerical zero is not automatically an observed zero. Invalid target and padding entries are required to be numerically zero so accidental unmasked use is easier to detect, but every loss and metric must still apply the appropriate mask.

## 3. Train-only condition standardization

The six conditions use different units and scales. For feature \(f\), statistics are fitted over active frames of the training split only:

\[
\mu_f^X=
\frac{1}{N_{\mathrm{active}}}
\sum_{i,t:M^{\mathrm{time}}_{i,t}=1}X_{i,t,f},
\]

\[
(\sigma_f^X)^2=
\frac{1}{N_{\mathrm{active}}}
\sum_{i,t:M^{\mathrm{time}}_{i,t}=1}
(X_{i,t,f}-\mu_f^X)^2.
\]

The transform is

\[
\tilde X_{i,t,f}=\frac{X_{i,t,f}-\mu_f^X}{\sigma_f^X}.
\]

Population standard deviation (`ddof=0`) is used because these values define a deterministic numerical transform rather than an unbiased variance estimator. Validation and test frames never contribute to these statistics.

If a training feature has scale below the configured numerical threshold, its transform scale is set to 1 and its index is recorded in the persisted state. A constant condition should normally trigger a later feature-review decision rather than be silently retained.

## 4. Mask-aware station-wise target standardization

Error magnitude generally changes with look-ahead distance. A single global target scale would cause high-variance distant stations to dominate optimization, whereas independent station scaling preserves a comparable numerical scale.

For station \(k\), let

\[
\mathcal O_k=
\{(i,t):M^{\mathrm{target}}_{i,t,k}=1\}.
\]

Only observed training targets are used:

\[
\mu_k^Y=
\frac{1}{|\mathcal O_k|}
\sum_{(i,t)\in\mathcal O_k}Y_{i,t,k},
\]

\[
(\sigma_k^Y)^2=
\frac{1}{|\mathcal O_k|}
\sum_{(i,t)\in\mathcal O_k}
(Y_{i,t,k}-\mu_k^Y)^2,
\]

\[
\tilde Y_{i,t,k}=
\frac{Y_{i,t,k}-\mu_k^Y}{\sigma_k^Y}
\quad\text{only when}\quad
M^{\mathrm{target}}_{i,t,k}=1.
\]

Invalid values remain zero rather than being transformed. Every station requires at least two observed training targets. Constant station scales are replaced by 1 and explicitly recorded.

For a generated standardized sample \(\tilde Y\), the physical-unit inverse is

\[
Y_{i,t,k}=\sigma_k^Y\tilde Y_{i,t,k}+\mu_k^Y.
\]

All calibration, tail, distance, and planner-level metrics must be computed after this inverse transformation into metres. Reporting only normalized errors would hide the physical importance of look-ahead-dependent uncertainty.

## 5. Leakage protections

The implementation enforces the following protections:

- `SequenceStandardizer.fit` rejects any `split_name` other than `train`.
- Padding is excluded using `lengths`.
- Missing targets are excluded using `valid_mask`.
- Ordered feature names are persisted and checked before transformation.
- The look-ahead grid is persisted and checked numerically.
- Validation and test splits reuse the frozen training transform.
- The persisted JSON records observation counts and constant dimensions.

The explicit `split_name` guard cannot prove that a caller has honestly labelled an array as training data. Dataset provenance and experiment configuration must still record the actual source file and checksum.

## 6. Persisted standardization artifact

The JSON artifact contains:

- schema version;
- fitted split name;
- ordered feature names;
- look-ahead grid;
- condition means, scales, and active-frame count;
- station-wise error means, scales, and valid counts;
- constant feature/station indices;
- numerical minimum-scale threshold.

This file is a model dependency. A saved model is not reproducible or safely deployable without the exact associated standardization artifact.

## 7. Complete-sequence batching

`iter_sequence_batches` shuffles and groups sequence indices, never individual frames. This preserves temporal order and prevents neighbouring frames from being treated as independent examples.

For reproducibility:

- a non-negative seed is mandatory when shuffling;
- the caller should derive a distinct fixed seed for every training epoch;
- the selected source indices are returned with every batch;
- every batch is trimmed to its longest contained sequence;
- `drop_last=True` removes only a final incomplete sequence batch.

Batch trimming reduces padding computation but does not cut or reorder any sequence. Model code receives both `lengths` and `valid_mask` and remains responsible for applying them to likelihoods or losses.

## 8. Common model lifecycle

Every model derives from `ProbabilisticSequenceModel` and implements:

```text
fit(train_data, validation_data) -> FitReport
sample(conditions, lengths, n_samples, seed, valid_mask) -> SampleResult
save(path) -> path
load(path) -> model
```

Density-based models may implement:

```text
log_probability(dataset) -> per-sequence log probabilities
```

RC-GAN does not have a tractable normalized density, so log probability is an optional capability rather than a mandatory comparison metric.

All models must declare:

- whether log probability is available;
- whether missing targets are supported;
- whether variable-length sequences are supported.

The base contract rejects raw, non-standardized fitting data. This ensures the same frozen transform is used across model families.

## 9. Sampling contract

Generated values use the common shape

\[
[S,B,T,K],
\]

where \(S\) is the number of stochastic samples. The result also contains sequence lengths, look-ahead grid, standardized/physical-unit status, and an optional evaluation mask.

Padding frames must be numerically zero. The supplied seed must completely determine stochastic sampling. A model may generate a full path profile beyond the observed evaluation mask; the mask identifies which real targets are available for comparison, not necessarily which synthetic values the model is allowed to generate.

## 10. Fair-comparison implications

The common interface does not make all metrics applicable to every model. The primary cross-model comparison must use sample-based quantities available for all three, while density-based diagnostics remain secondary for Gaussian and AIOHMM.

Standardization is common, but final evaluation is performed in physical units. Training budgets, condition sequences, requested sample counts, and random-seed policies must also be matched.

## 11. Known limitations and next decisions

- Per-station standardization does not remove spatial dependence; covariance must still be modelled.
- The Gaussian baseline now uses pairwise observed residual covariance, diagonal shrinkage, and positive-definite projection instead of complete-case deletion.
- Batching is NumPy-based. RC-GAN will later receive a thin PyTorch adapter without changing the stored data contract.
- The six-feature contract is provisional until BMW signal mapping and real-data feature analysis.
- Standardization parameters will change when the final training split changes; all models must then be retrained.

The next model implementation is AIOHMM. It must preserve the same standardized sequence contract while adding autoregressive emissions and input-dependent latent-state transitions.
