# Phase 6 AIOHMM Smoke Results

## Scope

This record documents the first deterministic end-to-end capability check after
the AIOHMM implementation. It is not a final model ranking and is not evidence
about BMW lane-estimation errors. The smoke data contain only 12/4/4 sequences
per scenario for train/validation/test.

Both models used synthetic manifest SHA-256
`6ee6283427ce88e118bde8dcdfdc25a09e713881868b686972fcaed4784f6438`.
Gaussian configuration SHA-256 was
`c2700065e647db368a2815e12e1967e25ee8bc355364e364d6c5131fc53961a0`;
AIOHMM configuration SHA-256 was
`82630998da24f95ad939a37f435d4e386311aa306533183deb719ea98f4e9ac4`.
Generated experiment directories were intentionally not committed.

All three AIOHMM scenarios selected three states and initialization seed
`20260716` from a validation-only grid of 3/4 states and two deterministic
restarts. Test data were first loaded after selection was frozen.

## Common test metrics

All metrics below are physical-unit sample metrics shared with the Gaussian and
future RC-GAN. Lower is better. Relative change is AIOHMM versus Gaussian; a
negative value is an improvement.

| Scenario | Metric | Gaussian | AIOHMM | Relative change |
|---|---|---:|---:|---:|
| Conditional Gaussian | Predictive-mean RMSE (m) | 0.06091 | 0.06123 | +0.5% |
| Conditional Gaussian | CRPS (m) | 0.03040 | 0.03030 | -0.3% |
| Conditional Gaussian | First-difference JS distance | 0.05060 | 0.05065 | +0.1% |
| Conditional Gaussian | Lag-one residual-correlation MAE | 0.06595 | 0.04565 | -30.8% |
| Conditional Gaussian | Spatial-correlation RMSE | 0.06442 | 0.10605 | +64.6% |
| Latent autoregressive | Predictive-mean RMSE (m) | 0.13284 | 0.12027 | -9.5% |
| Latent autoregressive | CRPS (m) | 0.06336 | 0.05314 | -16.1% |
| Latent autoregressive | Energy score (m) | 0.39434 | 0.32215 | -18.3% |
| Latent autoregressive | First-difference JS distance | 0.36583 | 0.14179 | -61.2% |
| Latent autoregressive | Lag-one residual-correlation MAE | 0.77603 | 0.12934 | -83.3% |
| Latent autoregressive | Spatial-correlation RMSE | 0.20127 | 0.28050 | +39.4% |
| Nonlinear heavy-tailed | Predictive-mean RMSE (m) | 0.10622 | 0.12193 | +14.8% |
| Nonlinear heavy-tailed | CRPS (m) | 0.05401 | 0.06000 | +11.1% |
| Nonlinear heavy-tailed | Energy score (m) | 0.33445 | 0.37134 | +11.0% |
| Nonlinear heavy-tailed | First-difference JS distance | 0.29532 | 0.05619 | -81.0% |
| Nonlinear heavy-tailed | Lag-one residual-correlation MAE | 0.75273 | 0.08314 | -89.0% |
| Nonlinear heavy-tailed | Spatial-correlation RMSE | 0.14090 | 0.21612 | +53.4% |

## State and tail diagnostics

On the latent-autoregressive scenario, the AIOHMM maximum-posterior states match
the synthetic oracle regimes with normalized mutual information 0.968. Posterior
mean entropy is 0.013 nats, state occupancies are approximately 0.566/0.215/0.219,
and fitted AR coefficients range from 0.360 to 0.907. This is the intended
positive-control result: the model recovers a strong temporal regime structure
without receiving oracle labels.

The nonlinear-heavy-tailed scenario has oracle-state NMI only 0.027. One fitted
AR coefficient reaches the configured 0.98 stability bound. The model improves
temporal metrics but worsens predictive score metrics and overestimates the test
99th absolute-error quantile: 0.441 m generated versus 0.331 m observed. This is
consistent with a Gaussian-state, free-running AR approximation struggling with
nonlinear burst/tail behavior.

The latent-autoregressive 99th quantile is underestimated: 0.352 m generated
versus 0.465 m observed. Spatial residual-correlation error is worse than
Gaussian in every scenario. These negative results must remain visible in the
prototype study and final comparison.

## Decision

Phase 6 passes the smoke gate because it demonstrates the intended temporal
capability, remains competitive on the Gaussian control, exposes rather than
hides misspecification, and satisfies the common no-test-leakage protocol.

The next checks are:

1. run the larger prototype data to see whether the smoke conclusions persist;
2. treat state recovery as a synthetic diagnostic, not a final research claim;
3. carry the tail, spatial-correlation, and free-running stability failures into
   RC-GAN requirements and the later BMW data request.
